"""
ML configuration settings for models, training, and inference
"""
import json
import os
import torch
from pathlib import Path

# Project directories
# Support bundled mode via environment variable override
PROJECT_ROOT = Path(os.environ.get('METAFRASIS_ROOT', Path(__file__).parent.parent))
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_WEIGHTS_DIR = DATA_DIR / "model_weights"

# Training and inference settings
OCR_BATCH_SIZE = 8  # Default batch size for PyTorch engines


def get_device() -> str:
    """
    Auto-detect best available device (CUDA GPU, Apple Silicon MPS, or CPU)

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_model_registry() -> dict:
    """
    Load model registry from ml/models/registry.json

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
