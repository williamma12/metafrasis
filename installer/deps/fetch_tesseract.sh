#!/bin/bash
# Fetch and prepare Tesseract OCR for bundling
# Supports both arm64 (Apple Silicon) and x86_64 (Intel) architectures

set -e

ARCH="${1:-$(uname -m)}"
OUTPUT_DIR="${2:-./installer/arch/${ARCH}/tesseract}"
TESSDATA_BEST_URL="https://github.com/tesseract-ocr/tessdata_best/raw/main"

echo "=== Fetching Tesseract for ${ARCH} ==="

# Create output directory structure
mkdir -p "${OUTPUT_DIR}"/{bin,lib,share/tessdata}

# Determine Homebrew prefix based on architecture
if [ "${ARCH}" == "arm64" ]; then
    HOMEBREW_PREFIX="/opt/homebrew"
else
    HOMEBREW_PREFIX="/usr/local"
fi

# Check if Tesseract is installed via Homebrew
if [ ! -f "${HOMEBREW_PREFIX}/bin/tesseract" ]; then
    echo "Error: Tesseract not found at ${HOMEBREW_PREFIX}/bin/tesseract"
    echo "Please install with: brew install tesseract"
    exit 1
fi

echo "Copying Tesseract binary..."
cp "${HOMEBREW_PREFIX}/bin/tesseract" "${OUTPUT_DIR}/bin/"

echo "Copying required libraries..."
# Core Tesseract and Leptonica libraries
for lib in tesseract lept; do
    find "${HOMEBREW_PREFIX}/lib" -maxdepth 1 -name "lib${lib}*.dylib" -exec cp {} "${OUTPUT_DIR}/lib/" \; 2>/dev/null || true
done

# Image format libraries required by Leptonica
for lib in png jpeg tiff webp gif openjp2 zstd lz4 jbig; do
    find "${HOMEBREW_PREFIX}/lib" -maxdepth 1 -name "lib${lib}*.dylib" -exec cp {} "${OUTPUT_DIR}/lib/" \; 2>/dev/null || true
done

# Additional dependencies that might be needed
for lib in z bz2; do
    # System libraries - skip these as they're part of macOS
    :
done

echo "Downloading Ancient Greek language data..."
curl -L "${TESSDATA_BEST_URL}/grc.traineddata" -o "${OUTPUT_DIR}/share/tessdata/grc.traineddata"

echo "Downloading English language data (fallback)..."
curl -L "${TESSDATA_BEST_URL}/eng.traineddata" -o "${OUTPUT_DIR}/share/tessdata/eng.traineddata"

echo "Downloading OSD data (orientation/script detection)..."
curl -L "${TESSDATA_BEST_URL}/osd.traineddata" -o "${OUTPUT_DIR}/share/tessdata/osd.traineddata"

echo "=== Tesseract prepared at ${OUTPUT_DIR} ==="
echo "Contents:"
ls -la "${OUTPUT_DIR}/bin/"
ls -la "${OUTPUT_DIR}/lib/"
ls -la "${OUTPUT_DIR}/share/tessdata/"
