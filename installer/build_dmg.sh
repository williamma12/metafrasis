#!/bin/bash
# Create DMG installer from app bundle
#
# Uses create-dmg for a professional looking installer with:
# - Custom background
# - Icon positioning
# - Applications folder symlink
#
# Usage:
#   ./installer/build_dmg.sh [version] [app_path] [output_path]

set -e

VERSION="${1:-1.0.0}"
APP_PATH="${2:-./installer/output/Metafrasis.app}"
OUTPUT_PATH="${3:-./installer/output/Metafrasis-${VERSION}.dmg}"

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/resources"

echo "=== Creating DMG installer ==="
echo "Version: $VERSION"
echo "App: $APP_PATH"
echo "Output: $OUTPUT_PATH"

# Check if app exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

# Check if create-dmg is installed
if ! command -v create-dmg &> /dev/null; then
    echo "Installing create-dmg..."
    brew install create-dmg
fi

# Remove existing DMG if present
rm -f "$OUTPUT_PATH"

# Build create-dmg command
CREATE_DMG_ARGS=(
    --volname "Metafrasis"
    --window-pos 200 120
    --window-size 600 400
    --icon-size 128
    --icon "Metafrasis.app" 150 200
    --hide-extension "Metafrasis.app"
    --app-drop-link 450 200
)

# Add icon if available
if [ -f "$RESOURCES_DIR/icon.icns" ]; then
    CREATE_DMG_ARGS+=(--volicon "$RESOURCES_DIR/icon.icns")
fi

# Add background if available
if [ -f "$RESOURCES_DIR/background.png" ]; then
    CREATE_DMG_ARGS+=(--background "$RESOURCES_DIR/background.png")
fi

# Add code signing if identity is set
if [ -n "$CODESIGN_IDENTITY" ]; then
    CREATE_DMG_ARGS+=(--codesign "$CODESIGN_IDENTITY")
fi

# Create DMG
echo "Running create-dmg..."
create-dmg "${CREATE_DMG_ARGS[@]}" "$OUTPUT_PATH" "$APP_PATH"

# Verify DMG
echo ""
echo "Verifying DMG..."
hdiutil verify "$OUTPUT_PATH"

# Show DMG info
echo ""
echo "DMG Info:"
hdiutil imageinfo "$OUTPUT_PATH" | grep -E "(Format|Size|Compressed)"

# Calculate size
DMG_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
echo ""
echo "=== DMG created successfully ==="
echo "Path: $OUTPUT_PATH"
echo "Size: $DMG_SIZE"
echo ""
echo "To test:"
echo "  open $OUTPUT_PATH"
