#!/bin/bash
# Fetch and prepare Poppler utilities for bundling
# Required for pdf2image (PDF to image conversion)
# Supports both arm64 (Apple Silicon) and x86_64 (Intel) architectures

set -e

ARCH="${1:-$(uname -m)}"
OUTPUT_DIR="${2:-./installer/arch/${ARCH}/poppler}"

echo "=== Fetching Poppler for ${ARCH} ==="

# Create output directory structure
mkdir -p "${OUTPUT_DIR}"/{bin,lib,share}

# Determine Homebrew prefix based on architecture
if [ "${ARCH}" == "arm64" ]; then
    HOMEBREW_PREFIX="/opt/homebrew"
else
    HOMEBREW_PREFIX="/usr/local"
fi

# Check if Poppler is installed via Homebrew
if [ ! -f "${HOMEBREW_PREFIX}/bin/pdftoppm" ]; then
    echo "Error: Poppler not found at ${HOMEBREW_PREFIX}/bin/pdftoppm"
    echo "Please install with: brew install poppler"
    exit 1
fi

echo "Copying Poppler binaries..."
# pdf2image uses pdftoppm and pdfinfo
for bin in pdftoppm pdfinfo pdftotext pdftocairo pdfimages; do
    if [ -f "${HOMEBREW_PREFIX}/bin/${bin}" ]; then
        cp "${HOMEBREW_PREFIX}/bin/${bin}" "${OUTPUT_DIR}/bin/"
    fi
done

echo "Copying Poppler libraries..."
# Core Poppler library
find "${HOMEBREW_PREFIX}/lib" -maxdepth 1 -name "libpoppler*.dylib" -exec cp {} "${OUTPUT_DIR}/lib/" \; 2>/dev/null || true

# Font libraries
for lib in fontconfig freetype; do
    find "${HOMEBREW_PREFIX}/lib" -maxdepth 1 -name "lib${lib}*.dylib" -exec cp {} "${OUTPUT_DIR}/lib/" \; 2>/dev/null || true
done

# Image libraries (some overlap with Tesseract)
for lib in png jpeg tiff openjp2 lcms2; do
    find "${HOMEBREW_PREFIX}/lib" -maxdepth 1 -name "lib${lib}*.dylib" -exec cp {} "${OUTPUT_DIR}/lib/" \; 2>/dev/null || true
done

# Additional dependencies
for lib in expat gettext intl iconv brotli; do
    find "${HOMEBREW_PREFIX}/lib" -maxdepth 1 -name "lib${lib}*.dylib" -exec cp {} "${OUTPUT_DIR}/lib/" \; 2>/dev/null || true
done

# Copy fontconfig configuration if available
if [ -d "${HOMEBREW_PREFIX}/etc/fonts" ]; then
    mkdir -p "${OUTPUT_DIR}/etc"
    cp -R "${HOMEBREW_PREFIX}/etc/fonts" "${OUTPUT_DIR}/etc/" 2>/dev/null || true
fi

echo "=== Poppler prepared at ${OUTPUT_DIR} ==="
echo "Contents:"
ls -la "${OUTPUT_DIR}/bin/"
ls -la "${OUTPUT_DIR}/lib/"
