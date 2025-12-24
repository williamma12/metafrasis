"""
Configuration settings for Metafrasis application
"""
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_DIR = DATA_DIR / "lexicon"

# UI settings
DEFAULT_IMAGE_WIDTH = 400
DEFAULT_TEXT_AREA_HEIGHT = 150
