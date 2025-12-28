"""
Configuration settings for Metafrasis application
"""
import json
import os
import torch
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_WEIGHTS_DIR = DATA_DIR / "model_weights"
LEXICON_DIR = DATA_DIR / "lexicon"

# UI settings
DEFAULT_IMAGE_WIDTH = 400
DEFAULT_TEXT_AREA_HEIGHT = 150

# OCR settings
DEFAULT_OCR_ENGINE = "tesseract"
OCR_BATCH_SIZE = 8  # Default batch size for PyTorch engines
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


def get_device() -> str:
    """
    Auto-detect best available device (CUDA GPU or CPU)

    Returns:
        Device string ('cuda' or 'cpu')
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model_registry() -> dict:
    """
    Load model registry from models/registry.json

    Returns:
        Dict with model URLs and metadata
    """
    registry_path = MODELS_DIR / "registry.json"
    if not registry_path.exists():
        return {}

    with open(registry_path, 'r') as f:
        return json.load(f)


def get_model_info(engine_name: str, variant: str = "base") -> dict:
    """
    Get model information from registry

    Args:
        engine_name: Engine name (e.g., 'tesseract', 'trocr')
        variant: Model variant (e.g., 'base', 'large')

    Returns:
        Dict with model URL, type, and description
        Returns None if not found
    """
    registry = get_model_registry()

    if engine_name not in registry:
        return None

    if variant not in registry[engine_name]:
        return None

    return registry[engine_name][variant]
