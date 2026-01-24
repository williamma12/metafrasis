#!/bin/bash
# Build universal macOS app for both Intel and Apple Silicon
#
# This script builds the app for both architectures and merges them
# into a single universal binary using lipo.
#
# Note: This requires either:
#   - An Apple Silicon Mac (can build both natively and via Rosetta)
#   - Or two separate Macs (Intel and Apple Silicon) with shared storage
#
# Usage:
#   ./installer/build_universal.sh [version]

set -e

VERSION="${1:-1.0.0}"

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALLER_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$INSTALLER_DIR/output"

echo "=== Building Metafrasis Universal Binary v${VERSION} ==="

# Detect current architecture
CURRENT_ARCH="$(uname -m)"
echo "Current architecture: $CURRENT_ARCH"

# Clean previous builds
rm -rf "$OUTPUT_DIR/arm64" "$OUTPUT_DIR/x86_64" "$OUTPUT_DIR/universal"
mkdir -p "$OUTPUT_DIR"/{arm64,x86_64,universal}

# ============================================
# Step 1: Build arm64 (Apple Silicon) version
# ============================================
echo ""
echo "=== Building arm64 version ==="

if [ "$CURRENT_ARCH" == "arm64" ]; then
    # Native build
    BUILD_VENV_ARM64="$INSTALLER_DIR/.build_venv_arm64"
    python3 -m venv "$BUILD_VENV_ARM64"
    source "$BUILD_VENV_ARM64/bin/activate"

    pip install --upgrade pip
    pip install pyinstaller
    pip install -e "$PROJECT_ROOT"

    # Prepare arm64 dependencies
    mkdir -p "$INSTALLER_DIR/arch/arm64"
    "$INSTALLER_DIR/deps/fetch_tesseract.sh" arm64 "$INSTALLER_DIR/arch/arm64/tesseract"
    "$INSTALLER_DIR/deps/fetch_poppler.sh" arm64 "$INSTALLER_DIR/arch/arm64/poppler"
    python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/arm64/tesseract"
    python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/arm64/poppler"

    # Build
    export TARGET_ARCH="arm64"
    export METAFRASIS_VERSION="$VERSION"
    pyinstaller "$INSTALLER_DIR/config/metafrasis.spec" \
        --distpath "$OUTPUT_DIR/arm64" \
        --workpath "$INSTALLER_DIR/build/arm64" \
        --noconfirm

    deactivate
else
    echo "Warning: Not on Apple Silicon. Skipping native arm64 build."
    echo "You'll need to build arm64 on an Apple Silicon Mac."
fi

# ============================================
# Step 2: Build x86_64 (Intel) version
# ============================================
echo ""
echo "=== Building x86_64 version ==="

if [ "$CURRENT_ARCH" == "arm64" ]; then
    # Build via Rosetta on Apple Silicon
    echo "Building x86_64 via Rosetta..."

    BUILD_VENV_X86="$INSTALLER_DIR/.build_venv_x86_64"
    arch -x86_64 /usr/bin/python3 -m venv "$BUILD_VENV_X86"
    source "$BUILD_VENV_X86/bin/activate"

    arch -x86_64 pip install --upgrade pip
    arch -x86_64 pip install pyinstaller
    arch -x86_64 pip install -e "$PROJECT_ROOT"

    # Prepare x86_64 dependencies (need Intel Homebrew or pre-built binaries)
    mkdir -p "$INSTALLER_DIR/arch/x86_64"

    # Note: On Apple Silicon, Intel Homebrew is at /usr/local
    # You may need to install Tesseract/Poppler there via Rosetta
    if [ -f "/usr/local/bin/tesseract" ]; then
        "$INSTALLER_DIR/deps/fetch_tesseract.sh" x86_64 "$INSTALLER_DIR/arch/x86_64/tesseract"
        "$INSTALLER_DIR/deps/fetch_poppler.sh" x86_64 "$INSTALLER_DIR/arch/x86_64/poppler"
        python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/x86_64/tesseract"
        python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/x86_64/poppler"
    else
        echo "Warning: Intel Tesseract not found at /usr/local/bin/tesseract"
        echo "Install via: arch -x86_64 /bin/bash -c '\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'"
        echo "Then: arch -x86_64 /usr/local/bin/brew install tesseract poppler"
    fi

    # Build
    export TARGET_ARCH="x86_64"
    export METAFRASIS_VERSION="$VERSION"
    arch -x86_64 pyinstaller "$INSTALLER_DIR/config/metafrasis.spec" \
        --distpath "$OUTPUT_DIR/x86_64" \
        --workpath "$INSTALLER_DIR/build/x86_64" \
        --noconfirm

    deactivate
elif [ "$CURRENT_ARCH" == "x86_64" ]; then
    # Native build on Intel
    BUILD_VENV_X86="$INSTALLER_DIR/.build_venv_x86_64"
    python3 -m venv "$BUILD_VENV_X86"
    source "$BUILD_VENV_X86/bin/activate"

    pip install --upgrade pip
    pip install pyinstaller
    pip install -e "$PROJECT_ROOT"

    # Prepare x86_64 dependencies
    mkdir -p "$INSTALLER_DIR/arch/x86_64"
    "$INSTALLER_DIR/deps/fetch_tesseract.sh" x86_64 "$INSTALLER_DIR/arch/x86_64/tesseract"
    "$INSTALLER_DIR/deps/fetch_poppler.sh" x86_64 "$INSTALLER_DIR/arch/x86_64/poppler"
    python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/x86_64/tesseract"
    python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/x86_64/poppler"

    # Build
    export TARGET_ARCH="x86_64"
    export METAFRASIS_VERSION="$VERSION"
    pyinstaller "$INSTALLER_DIR/config/metafrasis.spec" \
        --distpath "$OUTPUT_DIR/x86_64" \
        --workpath "$INSTALLER_DIR/build/x86_64" \
        --noconfirm

    deactivate
fi

# ============================================
# Step 3: Create Universal App Bundle
# ============================================
echo ""
echo "=== Creating Universal Binary ==="

ARM64_APP="$OUTPUT_DIR/arm64/Metafrasis.app"
X86_APP="$OUTPUT_DIR/x86_64/Metafrasis.app"
UNIVERSAL_APP="$OUTPUT_DIR/universal/Metafrasis.app"

# Check if both architectures were built
if [ ! -d "$ARM64_APP" ]; then
    echo "Error: arm64 build not found at $ARM64_APP"
    echo "Universal binary requires both architectures"
    exit 1
fi

if [ ! -d "$X86_APP" ]; then
    echo "Error: x86_64 build not found at $X86_APP"
    echo "Universal binary requires both architectures"
    exit 1
fi

# Use arm64 as base and merge x86_64 binaries
cp -R "$ARM64_APP" "$UNIVERSAL_APP"

# Merge main executable with lipo
MACOS_DIR="$UNIVERSAL_APP/Contents/MacOS"
ARM64_EXE="$ARM64_APP/Contents/MacOS/Metafrasis"
X86_EXE="$X86_APP/Contents/MacOS/Metafrasis"

echo "Merging executables with lipo..."
lipo -create "$ARM64_EXE" "$X86_EXE" -output "$MACOS_DIR/Metafrasis"

# Verify universal binary
echo "Verifying universal binary..."
lipo -archs "$MACOS_DIR/Metafrasis"

# For PyTorch: We can't easily merge the libraries, so we keep both versions
# and select at runtime based on architecture
RESOURCES_DIR="$UNIVERSAL_APP/Contents/Resources"

# Copy x86_64-specific torch to a subfolder
echo "Setting up architecture-specific PyTorch..."
mkdir -p "$RESOURCES_DIR/torch_x86_64"

# Copy x86_64 torch libraries (if they exist in the bundle)
X86_RESOURCES="$X86_APP/Contents/Resources"
if [ -d "$X86_RESOURCES" ]; then
    # Find and copy torch-related files
    find "$X86_RESOURCES" -name "torch*" -type d -exec cp -R {} "$RESOURCES_DIR/torch_x86_64/" \; 2>/dev/null || true
fi

# Copy to main output
cp -R "$UNIVERSAL_APP" "$OUTPUT_DIR/Metafrasis.app"

echo ""
echo "=== Universal build complete ==="
echo "Output: $OUTPUT_DIR/Metafrasis.app"

# Verify architectures
echo ""
echo "Architectures in final binary:"
lipo -archs "$OUTPUT_DIR/Metafrasis.app/Contents/MacOS/Metafrasis"
