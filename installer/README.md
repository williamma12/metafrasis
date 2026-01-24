# Metafrasis macOS Installer

This directory contains scripts to build a standalone macOS application bundle (.app) and DMG installer for Metafrasis.

## Quick Start

```bash
# Build for your current architecture (Intel or Apple Silicon)
./installer/build.sh

# The output will be in:
# - installer/output/Metafrasis.app     (the app bundle)
# - installer/output/Metafrasis-1.0.0.dmg  (the DMG installer)
```

## Requirements

Before building, ensure you have:

1. **macOS** (10.15 Catalina or later)
2. **Homebrew** - Install from https://brew.sh
3. **Python 3.11+** - The build script will use your system Python or uv

The build script will automatically install these if missing:
- Tesseract OCR
- Poppler (for PDF support)
- create-dmg (for DMG creation)

## Build Options

### Basic Build (Current Architecture)
```bash
./installer/build.sh
```

### Specify Version
```bash
./installer/build.sh 1.2.0
```

### Build with Code Signing
```bash
export CODESIGN_IDENTITY="Developer ID Application: Your Name (XXXXXXXXXX)"
./installer/build.sh 1.0.0 --sign
```

### Build Universal Binary (Intel + Apple Silicon)
```bash
./installer/build.sh 1.0.0 --universal
```

Note: Universal builds require either:
- An Apple Silicon Mac (can build both via native and Rosetta)
- Or access to both Intel and Apple Silicon Macs

## Directory Structure

```
installer/
├── build.sh              # Main build orchestrator
├── build_universal.sh    # Universal binary builder
├── build_dmg.sh          # DMG creation script
├── wrapper/
│   └── launcher.py       # App entry point (manages Streamlit)
├── config/
│   ├── metafrasis.spec   # PyInstaller configuration
│   └── entitlements.plist # Code signing entitlements
├── deps/
│   ├── fetch_tesseract.sh # Bundles Tesseract
│   ├── fetch_poppler.sh   # Bundles Poppler
│   └── bundle_deps.py     # Makes dylibs relocatable
├── resources/
│   ├── icon.icns         # App icon (you provide)
│   └── background.png    # DMG background (you provide)
├── tests/
│   └── test_app_bundle.py # Verification tests
└── output/               # Build output (gitignored)
    ├── Metafrasis.app
    └── Metafrasis-X.X.X.dmg
```

## Customization

### App Icon
Place your app icon at `installer/resources/icon.icns`.

To create an .icns file from a PNG:
```bash
# Create iconset folder
mkdir MyIcon.iconset

# Add different sizes (16, 32, 64, 128, 256, 512, 1024)
sips -z 16 16 icon.png --out MyIcon.iconset/icon_16x16.png
sips -z 32 32 icon.png --out MyIcon.iconset/icon_16x16@2x.png
# ... repeat for other sizes

# Convert to icns
iconutil -c icns MyIcon.iconset
mv MyIcon.icns installer/resources/icon.icns
```

### DMG Background
Place a 600x400 PNG at `installer/resources/background.png` for a custom DMG background.

## Code Signing (Optional)

Without code signing, users will see a Gatekeeper warning on first launch and need to right-click → Open.

To sign your app:

1. **Get an Apple Developer ID** ($99/year at developer.apple.com)

2. **Find your identity:**
   ```bash
   security find-identity -v -p codesigning
   ```

3. **Set the environment variable:**
   ```bash
   export CODESIGN_IDENTITY="Developer ID Application: Your Name (XXXXXXXXXX)"
   ```

4. **Build with signing:**
   ```bash
   ./installer/build.sh 1.0.0 --sign
   ```

5. **Notarize (recommended for distribution):**
   ```bash
   xcrun notarytool submit installer/output/Metafrasis-1.0.0.dmg \
       --apple-id your@email.com \
       --password @keychain:AC_PASSWORD \
       --team-id XXXXXXXXXX \
       --wait

   xcrun stapler staple installer/output/Metafrasis-1.0.0.dmg
   ```

## Testing the Build

Run the automated tests:
```bash
pytest installer/tests/test_app_bundle.py -v
```

Or test manually:
```bash
# Open the app
open installer/output/Metafrasis.app

# Or mount and test the DMG
open installer/output/Metafrasis-1.0.0.dmg
```

## Troubleshooting

### "App is damaged and can't be opened"
The app isn't signed. Right-click the app and select "Open", then click "Open" in the dialog.

Or remove the quarantine attribute:
```bash
xattr -cr /Applications/Metafrasis.app
```

### Tesseract not found
Ensure Tesseract is installed:
```bash
brew install tesseract
```

### Build fails on Intel Mac
You may need to install Intel versions of dependencies:
```bash
arch -x86_64 brew install tesseract poppler
```

### Large DMG size
The app is ~1GB due to PyTorch. This is expected. The DMG compresses to ~800MB.

## How It Works

1. **launcher.py** is the entry point. It:
   - Sets up environment variables for bundled Tesseract/Poppler
   - Finds a free port
   - Starts Streamlit as a background process
   - Opens the default browser
   - Handles clean shutdown

2. **PyInstaller** bundles Python, all dependencies, and your code into a .app

3. **Tesseract and Poppler** are bundled as binary dependencies with their libraries made relocatable using `@executable_path`

4. **create-dmg** creates the professional-looking DMG installer
