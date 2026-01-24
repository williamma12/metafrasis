#!/usr/bin/env python3
"""
Make bundled dylibs relocatable using @executable_path

This script:
1. Scans all dylibs in the deps directory
2. Uses otool to find dependencies
3. Uses install_name_tool to rewrite paths to @executable_path

This allows the bundled libraries to be found relative to the
executable location rather than requiring absolute paths.
"""
import os
import re
import subprocess
import sys
from pathlib import Path


def get_dylib_id(dylib_path: Path) -> str | None:
    """Get the install name (ID) of a dylib"""
    result = subprocess.run(
        ["otool", "-D", str(dylib_path)],
        capture_output=True,
        text=True
    )
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        return lines[1].strip()
    return None


def get_dylib_deps(dylib_path: Path) -> list[str]:
    """Get list of dylib dependencies using otool"""
    result = subprocess.run(
        ["otool", "-L", str(dylib_path)],
        capture_output=True,
        text=True
    )
    deps = []
    for line in result.stdout.split('\n')[1:]:  # Skip first line (file name)
        match = re.match(r'\s+(.+\.dylib)', line)
        if match:
            dep_path = match.group(1).split()[0]
            deps.append(dep_path)
    return deps


def is_system_lib(path: str) -> bool:
    """Check if a library is a system library that shouldn't be modified"""
    system_prefixes = [
        '/System/',
        '/usr/lib/',
        '/Library/Frameworks/',
    ]
    return any(path.startswith(prefix) for prefix in system_prefixes)


def make_relocatable(deps_dir: Path):
    """
    Make all dylibs in deps_dir relocatable

    Args:
        deps_dir: Directory containing bin/ and lib/ subdirectories
    """
    lib_dir = deps_dir / "lib"
    bin_dir = deps_dir / "bin"

    if not lib_dir.exists():
        print(f"Warning: lib directory not found at {lib_dir}")
        return

    # Collect all dylibs
    dylibs = list(lib_dir.glob("*.dylib"))
    dylib_names = {d.name for d in dylibs}

    # Also track the original paths -> new names mapping
    original_to_new = {}
    for dylib in dylibs:
        original_id = get_dylib_id(dylib)
        if original_id:
            original_to_new[original_id] = dylib.name
            # Also add common variations
            original_to_new[dylib.name] = dylib.name

    print(f"Found {len(dylibs)} dylibs to process")

    # Process each dylib
    for dylib in dylibs:
        print(f"Processing {dylib.name}...")

        # Change the dylib's own ID to use @executable_path
        new_id = f"@executable_path/../lib/{dylib.name}"
        subprocess.run([
            "install_name_tool", "-id", new_id, str(dylib)
        ], check=True)

        # Rewrite dependencies to use @executable_path
        for dep in get_dylib_deps(dylib):
            if is_system_lib(dep):
                continue

            dep_name = Path(dep).name

            # Check if this dependency is in our lib directory
            if dep_name in dylib_names:
                new_path = f"@executable_path/../lib/{dep_name}"
                print(f"  Rewriting {dep} -> {new_path}")
                subprocess.run([
                    "install_name_tool", "-change",
                    dep, new_path, str(dylib)
                ], check=True)
            elif dep in original_to_new:
                # Handle case where the reference uses a different name
                new_name = original_to_new[dep]
                new_path = f"@executable_path/../lib/{new_name}"
                print(f"  Rewriting {dep} -> {new_path}")
                subprocess.run([
                    "install_name_tool", "-change",
                    dep, new_path, str(dylib)
                ], check=True)

    # Also fix binaries (tesseract, poppler utils)
    if bin_dir.exists():
        for binary in bin_dir.iterdir():
            if binary.is_file() and os.access(binary, os.X_OK):
                print(f"Processing binary {binary.name}...")
                for dep in get_dylib_deps(binary):
                    if is_system_lib(dep):
                        continue

                    dep_name = Path(dep).name

                    if dep_name in dylib_names:
                        new_path = f"@executable_path/../lib/{dep_name}"
                        print(f"  Rewriting {dep} -> {new_path}")
                        subprocess.run([
                            "install_name_tool", "-change",
                            dep, new_path, str(binary)
                        ], check=True)
                    elif dep in original_to_new:
                        new_name = original_to_new[dep]
                        new_path = f"@executable_path/../lib/{new_name}"
                        print(f"  Rewriting {dep} -> {new_path}")
                        subprocess.run([
                            "install_name_tool", "-change",
                            dep, new_path, str(binary)
                        ], check=True)

    print("Done making libraries relocatable")


def verify_relocatable(deps_dir: Path):
    """Verify that libraries are properly relocatable"""
    lib_dir = deps_dir / "lib"
    bin_dir = deps_dir / "bin"

    print("\n=== Verifying relocatable libraries ===")

    all_good = True

    # Check dylibs
    for dylib in lib_dir.glob("*.dylib"):
        deps = get_dylib_deps(dylib)
        for dep in deps:
            if not is_system_lib(dep) and not dep.startswith("@"):
                print(f"WARNING: {dylib.name} has non-relocatable dep: {dep}")
                all_good = False

    # Check binaries
    if bin_dir.exists():
        for binary in bin_dir.iterdir():
            if binary.is_file() and os.access(binary, os.X_OK):
                deps = get_dylib_deps(binary)
                for dep in deps:
                    if not is_system_lib(dep) and not dep.startswith("@"):
                        print(f"WARNING: {binary.name} has non-relocatable dep: {dep}")
                        all_good = False

    if all_good:
        print("All libraries are properly relocatable")
    else:
        print("Some libraries have non-relocatable dependencies")
        return False

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bundle_deps.py <deps_dir> [--verify]")
        print("Example: bundle_deps.py installer/arch/arm64/tesseract")
        sys.exit(1)

    deps_dir = Path(sys.argv[1])

    if not deps_dir.exists():
        print(f"Error: Directory not found: {deps_dir}")
        sys.exit(1)

    if "--verify" in sys.argv:
        success = verify_relocatable(deps_dir)
        sys.exit(0 if success else 1)
    else:
        make_relocatable(deps_dir)
        verify_relocatable(deps_dir)
