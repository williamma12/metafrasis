"""
OCR service for Ancient Greek text recognition

Supports multiple engines with streaming and batch pipelines:
- Streaming: recognize() for single images (low latency)
- Batch: recognize_batch() for multiple images (PyTorch engines optimize this)

PDF conversion is handled separately via preprocessing.pdf_to_images()
"""
from .base import *
from .factory import *
from . import preprocessing
from .viewer import ocr_viewer
from .cache import ImageCache

# Import engines and add to factory dict
try:
    from .engines.tesseract import TesseractEngine
    OCREngineFactory._engines['tesseract'] = TesseractEngine
except ImportError:
    pass  # Tesseract not available

try:
    from .engines.trocr import TrOCREngine
    OCREngineFactory._engines['trocr'] = TrOCREngine
except ImportError:
    pass  # trOCR not available
