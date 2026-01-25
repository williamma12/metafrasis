"""
Build script for mps_ctc native extension.

This script compiles the Objective-C++ code and Metal shaders into a
Python extension module that can be imported as `mps_ctc._C`.

=============================================================================
BUILD PROCESS
=============================================================================

1. Metal Shader Compilation (manual step):
   The Metal shaders need to be compiled to a .metallib file first:

   cd mps_ctc/csrc
   xcrun -sdk macosx metal -c ctc_kernels.metal -o ctc_kernels.air
   xcrun -sdk macosx metallib ctc_kernels.air -o ctc_kernels.metallib

   Or you can compile at runtime (slower first run, but simpler).

2. Python Extension Build:
   pip install -e .

   This compiles ctc_mps.mm and links with PyTorch and Metal frameworks.

=============================================================================
USAGE
=============================================================================

From the mps_ctc directory:
    pip install -e .

Or from the root project directory:
    pip install -e ./mps_ctc

After installation:
    from mps_ctc import CTCLossMPS
    criterion = CTCLossMPS(blank=0)

=============================================================================
TROUBLESHOOTING
=============================================================================

1. "Metal.framework not found"
   - Ensure you're on macOS with Xcode command line tools installed
   - Run: xcode-select --install

2. "torch not found"
   - Install PyTorch first: pip install torch

3. "Unsupported compiler"
   - The Objective-C++ code requires clang (default on macOS)
   - Ensure you're using the system compiler, not a conda/brew version

4. "metallib not found at runtime"
   - Compile the Metal shaders first (step 1 above)
   - Or modify ctc_mps.mm to compile from source at runtime
"""

import os
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension


def get_extension():
    """
    Configure the native extension.

    Returns a CppExtension configured for Objective-C++ with Metal framework.
    """
    # Source files
    sources = [
        "csrc/ctc_mps.mm",
    ]

    # Compiler flags
    extra_compile_args = {
        "cxx": [
            "-std=c++17",           # C++17 for modern features
            "-O3",                  # Optimization
            "-fobjc-arc",           # Automatic Reference Counting
        ],
    }

    # Linker flags for Metal and Foundation frameworks
    extra_link_args = [
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "MetalPerformanceShaders",
    ]

    return CppExtension(
        name="mps_ctc._C",
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="objc++",
    )


def compile_metal_shaders():
    """
    Compile Metal shaders to metallib (optional helper).

    This function can be called during build to compile the Metal shaders.
    Alternatively, you can compile them manually or at runtime.
    """
    csrc_dir = Path(__file__).parent / "csrc"
    metal_src = csrc_dir / "ctc_kernels.metal"
    air_file = csrc_dir / "ctc_kernels.air"
    metallib = csrc_dir / "ctc_kernels.metallib"

    if not metal_src.exists():
        print(f"Warning: {metal_src} not found, skipping Metal compilation")
        return

    # Check if metallib needs to be rebuilt
    if metallib.exists():
        if metallib.stat().st_mtime > metal_src.stat().st_mtime:
            print("Metal library is up to date")
            return

    print("Compiling Metal shaders...")

    try:
        # Compile to AIR (Apple Intermediate Representation)
        subprocess.run([
            "xcrun", "-sdk", "macosx", "metal",
            "-c", str(metal_src),
            "-o", str(air_file),
        ], check=True)

        # Link to metallib
        subprocess.run([
            "xcrun", "-sdk", "macosx", "metallib",
            str(air_file),
            "-o", str(metallib),
        ], check=True)

        # Clean up intermediate file
        air_file.unlink()

        print(f"Created {metallib}")

    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to compile Metal shaders: {e}")
        print("You may need to compile manually or use runtime compilation")
    except FileNotFoundError:
        print("Warning: xcrun not found. Install Xcode command line tools.")


# Optionally compile Metal shaders during build
# Uncomment the next line to enable:
# compile_metal_shaders()


setup(
    name="mps_ctc",
    version="0.1.0",
    description="MPS-accelerated CTC Loss for PyTorch",
    author="Your Name",
    packages=["mps_ctc"],
    package_dir={"mps_ctc": "."},
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
