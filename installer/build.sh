#!/bin/bash
# Main build script for Metafrasis macOS installer
#
# Usage:
#   ./installer/build.sh [version] [--sign] [--universal]
#
# Examples:
#   ./installer/build.sh                    # Build for current arch only
#   ./installer/build.sh 1.0.0              # Build with specific version
#   ./installer/build.sh 1.0.0 --sign       # Build and sign
#   ./installer/build.sh 1.0.0 --universal  # Build universal binary

set -e

# Parse arguments
VERSION="${1:-1.0.0}"
SIGN=false
UNIVERSAL=false

for arg in "$@"; do
    case $arg in
        --sign)
            SIGN=true
            ;;
        --universal)
            UNIVERSAL=true
            ;;
    esac
done

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALLER_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$INSTALLER_DIR/output"

echo "============================================"
echo "Building Metafrasis v${VERSION} for macOS"
echo "Universal: ${UNIVERSAL}"
echo "Sign: ${SIGN}"
echo "============================================"

# Step 1: Verify prerequisites
echo ""
echo "Step 1: Verifying prerequisites..."

command -v python3 >/dev/null 2>&1 || { echo "Error: Python 3 required"; exit 1; }
command -v brew >/dev/null 2>&1 || { echo "Error: Homebrew required"; exit 1; }

# Check for Tesseract and Poppler
brew list tesseract &>/dev/null || { echo "Installing Tesseract..."; brew install tesseract; }
brew list poppler &>/dev/null || { echo "Installing Poppler..."; brew install poppler; }

# Check for create-dmg
brew list create-dmg &>/dev/null || { echo "Installing create-dmg..."; brew install create-dmg; }

echo "Prerequisites verified"

# Step 2: Set up virtual environment and install dependencies
echo ""
echo "Step 2: Setting up build environment..."

cd "$PROJECT_ROOT"

# Create build venv if it doesn't exist
BUILD_VENV="$INSTALLER_DIR/.build_venv"
if [ ! -d "$BUILD_VENV" ]; then
    python3 -m venv "$BUILD_VENV"
fi

source "$BUILD_VENV/bin/activate"

# Install PyInstaller and project dependencies
pip install --upgrade pip
pip install pyinstaller
pip install -e "$PROJECT_ROOT"

echo "Build environment ready"

# Step 3: Build frontend components (if needed)
echo ""
echo "Step 3: Checking frontend components..."

OCR_VIEWER_BUILD="$PROJECT_ROOT/frontend/ocr_viewer/build"
if [ ! -d "$OCR_VIEWER_BUILD" ]; then
    echo "Building OCR viewer component..."
    if command -v npm >/dev/null 2>&1; then
        cd "$PROJECT_ROOT/frontend/ocr_viewer"
        npm install && npm run build
        cd "$PROJECT_ROOT"
    else
        echo "Warning: npm not found, skipping frontend build"
        echo "React components may not work properly"
    fi
else
    echo "Frontend components already built"
fi

# Step 4: Fetch and prepare dependencies
echo ""
echo "Step 4: Preparing bundled dependencies..."

ARCH="$(uname -m)"

# Create arch directory
mkdir -p "$INSTALLER_DIR/arch/$ARCH"

# Fetch Tesseract
if [ ! -d "$INSTALLER_DIR/arch/$ARCH/tesseract/bin" ]; then
    echo "Fetching Tesseract for $ARCH..."
    chmod +x "$INSTALLER_DIR/deps/fetch_tesseract.sh"
    "$INSTALLER_DIR/deps/fetch_tesseract.sh" "$ARCH" "$INSTALLER_DIR/arch/$ARCH/tesseract"
else
    echo "Tesseract already prepared"
fi

# Fetch Poppler
if [ ! -d "$INSTALLER_DIR/arch/$ARCH/poppler/bin" ]; then
    echo "Fetching Poppler for $ARCH..."
    chmod +x "$INSTALLER_DIR/deps/fetch_poppler.sh"
    "$INSTALLER_DIR/deps/fetch_poppler.sh" "$ARCH" "$INSTALLER_DIR/arch/$ARCH/poppler"
else
    echo "Poppler already prepared"
fi

# Make libraries relocatable
echo "Making libraries relocatable..."
python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/$ARCH/tesseract"
python "$INSTALLER_DIR/deps/bundle_deps.py" "$INSTALLER_DIR/arch/$ARCH/poppler"

# Step 5: Build with PyInstaller
echo ""
echo "Step 5: Building application bundle..."

# Clean previous output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for the spec file
export TARGET_ARCH="$ARCH"
export METAFRASIS_VERSION="$VERSION"

if [ "$SIGN" = true ] && [ -n "$CODESIGN_IDENTITY" ]; then
    export CODESIGN_IDENTITY
fi

# Run PyInstaller
pyinstaller "$INSTALLER_DIR/config/metafrasis.spec" \
    --distpath "$OUTPUT_DIR" \
    --workpath "$INSTALLER_DIR/build" \
    --noconfirm

echo "Application bundle created"

# Step 6: Universal binary (optional)
if [ "$UNIVERSAL" = true ]; then
    echo ""
    echo "Step 6: Creating universal binary..."
    chmod +x "$INSTALLER_DIR/build_universal.sh"
    "$INSTALLER_DIR/build_universal.sh" "$VERSION"
fi

# Step 7: Sign (optional)
if [ "$SIGN" = true ]; then
    echo ""
    echo "Step 7: Signing application..."

    if [ -z "$CODESIGN_IDENTITY" ]; then
        echo "Warning: CODESIGN_IDENTITY not set, skipping signing"
        echo "Set it with: export CODESIGN_IDENTITY='Developer ID Application: Your Name (XXXXXXXXXX)'"
    else
        # Sign all dylibs first
        find "$OUTPUT_DIR/Metafrasis.app" -name "*.dylib" -o -name "*.so" | while read lib; do
            codesign --force --options runtime --sign "$CODESIGN_IDENTITY" "$lib" 2>/dev/null || true
        done

        # Sign the main bundle
        codesign --force --options runtime \
            --entitlements "$INSTALLER_DIR/config/entitlements.plist" \
            --sign "$CODESIGN_IDENTITY" \
            "$OUTPUT_DIR/Metafrasis.app"

        # Verify
        codesign --verify --deep --strict "$OUTPUT_DIR/Metafrasis.app"
        echo "Application signed successfully"
    fi
fi

# Step 8: Create DMG
echo ""
echo "Step 8: Creating DMG installer..."

chmod +x "$INSTALLER_DIR/build_dmg.sh"
"$INSTALLER_DIR/build_dmg.sh" "$VERSION" "$OUTPUT_DIR/Metafrasis.app" "$OUTPUT_DIR/Metafrasis-${VERSION}.dmg"

# Deactivate venv
deactivate

# Done
echo ""
echo "============================================"
echo "Build complete!"
echo ""
echo "Output:"
echo "  App Bundle: $OUTPUT_DIR/Metafrasis.app"
echo "  DMG:        $OUTPUT_DIR/Metafrasis-${VERSION}.dmg"
echo ""
echo "To test the app:"
echo "  open $OUTPUT_DIR/Metafrasis.app"
echo "============================================"
