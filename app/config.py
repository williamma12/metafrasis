"""
Application configuration settings for Metafrasis UI and services
"""
import os
from pathlib import Path

# Project directories
# Support bundled mode via environment variable override
PROJECT_ROOT = Path(os.environ.get('METAFRASIS_ROOT', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_DIR = DATA_DIR / "lexicon"

# UI settings
DEFAULT_IMAGE_WIDTH = 400
DEFAULT_TEXT_AREA_HEIGHT = 150

# OCR settings
DEFAULT_OCR_ENGINE = "tesseract"
PDF_DPI = 200  # DPI for PDF to image conversion

# OCR Viewer Configuration
# Defaults to False (development mode) unless explicitly set to 'true' for production releases
# Development mode connects to Vite dev server at http://localhost:5173
# Production mode loads pre-built component from frontend/ocr_viewer/build/
VIEWER_RELEASE_MODE = os.getenv('VIEWER_RELEASE', 'false').lower() == 'true'

# Annotation Canvas Configuration
# Development mode connects to Vite dev server at http://localhost:5174
# Production mode loads pre-built component from frontend/annotation_canvas/build/
ANNOTATION_CANVAS_RELEASE_MODE = os.getenv('ANNOTATION_CANVAS_RELEASE', 'false').lower() == 'true'
