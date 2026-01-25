#!/usr/bin/env python3
"""
Generate compile_flags.txt for LSP support in the mps_ctc extension.

This script creates a compile_flags.txt file that helps clangd (and other LSPs)
find PyTorch headers, enabling autocomplete and error checking for TORCH_CHECK,
torch::Tensor, and other PyTorch C++ APIs.

Usage:
    python setup_lsp.py
    # or
    ./setup_lsp.py
"""

import subprocess
import sys
from pathlib import Path


def get_macos_sdk_path() -> str:
    """Get the macOS SDK path using xcrun."""
    try:
        result = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not detect SDK path via xcrun", file=sys.stderr)
        return "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"


def main():
    # Get the directory where this script lives
    script_dir = Path(__file__).parent.resolve()
    csrc_dir = script_dir / "csrc"
    output_file = csrc_dir / "compile_flags.txt"

    # Check that csrc directory exists
    if not csrc_dir.exists():
        print(f"Error: {csrc_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Try to import torch to get include paths
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Run: pip install torch", file=sys.stderr)
        sys.exit(1)

    # Get macOS SDK path
    sdk_path = get_macos_sdk_path()

    # Get PyTorch include directories
    torch_dir = Path(torch.__file__).parent
    torch_include = torch_dir / "include"
    torch_csrc_include = torch_include / "torch" / "csrc" / "api" / "include"

    # Verify paths exist
    if not torch_include.exists():
        print(f"Warning: {torch_include} not found", file=sys.stderr)
    if not torch_csrc_include.exists():
        print(f"Warning: {torch_csrc_include} not found", file=sys.stderr)

    # Build compile flags
    flags = [
        "-xobjective-c++",
        "-std=c++17",
        "-fobjc-arc",
        f"-I{torch_include}",
        f"-I{torch_csrc_include}",
        # macOS SDK headers for Metal/Foundation
        "-isysroot",
        sdk_path,
        # Framework search paths
        f"-F{sdk_path}/System/Library/Frameworks",
    ]

    # Write compile_flags.txt
    with open(output_file, "w") as f:
        f.write("\n".join(flags) + "\n")

    print(f"Created {output_file}")
    print(f"PyTorch include: {torch_include}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"macOS SDK: {sdk_path}")
    print("\nRestart your LSP/editor to pick up the changes.")


if __name__ == "__main__":
    main()
