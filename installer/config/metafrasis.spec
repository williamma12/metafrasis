# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Metafrasis macOS application

Bundles:
- Python runtime
- All Python dependencies (including PyTorch, Streamlit, Transformers)
- Application code
- Pre-built React components
- System dependencies (Tesseract, Poppler) as data files

Usage:
    pyinstaller installer/config/metafrasis.spec --distpath installer/output
"""

import os
import sys
from pathlib import Path

# Import PyInstaller utilities
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ============================================
# Configuration
# ============================================

# Project paths - determine from spec file location
SPEC_DIR = Path(SPECPATH)
PROJECT_ROOT = SPEC_DIR.parent.parent
INSTALLER_DIR = PROJECT_ROOT / 'installer'

# Target architecture (set via environment variable or detect)
ARCH = os.environ.get('TARGET_ARCH', 'arm64')

# Version
VERSION = os.environ.get('METAFRASIS_VERSION', '1.0.0')

# Code signing identity (optional)
CODESIGN_IDENTITY = os.environ.get('CODESIGN_IDENTITY', None)

print(f"Building Metafrasis v{VERSION} for {ARCH}")
print(f"Project root: {PROJECT_ROOT}")

# ============================================
# Data Files to Bundle
# ============================================

# Collect Streamlit's data files (templates, static assets, etc.)
streamlit_datas = collect_data_files('streamlit')

# Collect altair data files
altair_datas = collect_data_files('altair')

# Application data files
datas = [
    # Application code packages
    (str(PROJECT_ROOT / 'app'), 'app'),
    (str(PROJECT_ROOT / 'services'), 'services'),
    (str(PROJECT_ROOT / 'utils'), 'utils'),

    # Configuration and entry point
    (str(PROJECT_ROOT / 'config.py'), '.'),
    (str(PROJECT_ROOT / 'app.py'), '.'),

    # Model registry (URLs, not actual weights)
    (str(PROJECT_ROOT / 'models' / 'registry.json'), 'models'),

    # Pre-built React components
    (str(PROJECT_ROOT / 'frontend' / 'ocr_viewer' / 'build'), 'frontend/ocr_viewer/build'),
]

# Add annotation canvas if it exists
annotation_canvas_build = PROJECT_ROOT / 'frontend' / 'annotation_canvas' / 'build'
if annotation_canvas_build.exists():
    datas.append((str(annotation_canvas_build), 'frontend/annotation_canvas/build'))

# Add bundled system dependencies (Tesseract, Poppler)
tesseract_dir = INSTALLER_DIR / 'arch' / ARCH / 'tesseract'
poppler_dir = INSTALLER_DIR / 'arch' / ARCH / 'poppler'

if tesseract_dir.exists():
    datas.append((str(tesseract_dir), 'deps/tesseract'))
else:
    print(f"Warning: Tesseract not found at {tesseract_dir}")
    print("Run: ./installer/deps/fetch_tesseract.sh first")

if poppler_dir.exists():
    datas.append((str(poppler_dir), 'deps/poppler'))
else:
    print(f"Warning: Poppler not found at {poppler_dir}")
    print("Run: ./installer/deps/fetch_poppler.sh first")

# Add Streamlit and altair data
datas.extend(streamlit_datas)
datas.extend(altair_datas)

# ============================================
# Hidden Imports
# ============================================

# Imports that PyInstaller might miss due to dynamic loading
hiddenimports = [
    # Streamlit internals
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.caching',
    'streamlit.runtime.legacy_caching',
    'streamlit.components.v1',
    'streamlit.elements',

    # OCR dependencies
    'pytesseract',
    'PIL',
    'PIL.Image',
    'cv2',
    'pdf2image',

    # PyTorch and ML
    'torch',
    'torch.nn',
    'torch.utils',
    'torchvision',
    'torchvision.transforms',
    'transformers',
    'transformers.models',
    'sentencepiece',
    'huggingface_hub',

    # Data processing
    'numpy',
    'pandas',

    # Web/async
    'altair',
    'tornado',
    'websocket',

    # Application modules
    'app',
    'app.main',
    'app.state',
    'app.pages',
    'app.pages.ocr',
    'app.pages.annotate',
    'app.components',
    'services',
    'services.ocr',
    'services.ocr.base',
    'services.ocr.factory',
    'services.ocr.preprocessing',
    'utils',
]

# Collect all submodules from our packages
hiddenimports.extend(collect_submodules('services'))
hiddenimports.extend(collect_submodules('app'))

# ============================================
# Excluded Modules (to reduce size)
# ============================================

excludes = [
    'tkinter',
    'tcl',
    'tk',
    'matplotlib',  # Large, not needed for core OCR
    'scipy',  # Only needed for optional CRAFT/DB
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'ruff',
]

# ============================================
# Analysis
# ============================================

a = Analysis(
    [str(INSTALLER_DIR / 'wrapper' / 'launcher.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

# ============================================
# Build Steps
# ============================================

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Metafrasis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Don't compress - can cause issues on macOS
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    target_arch=ARCH,
    codesign_identity=CODESIGN_IDENTITY,
    entitlements_file=str(INSTALLER_DIR / 'config' / 'entitlements.plist') if CODESIGN_IDENTITY else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Metafrasis',
)

# ============================================
# macOS App Bundle
# ============================================

app = BUNDLE(
    coll,
    name='Metafrasis.app',
    icon=str(INSTALLER_DIR / 'resources' / 'icon.icns') if (INSTALLER_DIR / 'resources' / 'icon.icns').exists() else None,
    bundle_identifier='io.metafrasis.app',
    info_plist={
        'CFBundleName': 'Metafrasis',
        'CFBundleDisplayName': 'Metafrasis',
        'CFBundleVersion': VERSION,
        'CFBundleShortVersionString': VERSION,
        'CFBundleExecutable': 'Metafrasis',
        'CFBundleIdentifier': 'io.metafrasis.app',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': 'MFRS',
        'LSMinimumSystemVersion': '10.15',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        'LSApplicationCategoryType': 'public.app-category.productivity',
        'NSHumanReadableCopyright': 'Copyright 2024',
        # Privacy descriptions (required for some features)
        'NSDesktopFolderUsageDescription': 'Metafrasis needs access to read and save files.',
        'NSDocumentsFolderUsageDescription': 'Metafrasis needs access to read and save files.',
        'NSDownloadsFolderUsageDescription': 'Metafrasis needs access to save exported files.',
    },
)
