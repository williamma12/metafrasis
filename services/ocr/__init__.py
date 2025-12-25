"""
OCR service for Ancient Greek text recognition

Supports multiple engines with streaming and batch pipelines:
- Streaming: recognize() for single images (low latency)
- Batch: recognize_batch() for multiple images (PyTorch engines optimize this)

PDF conversion is handled separately via preprocessing.pdf_to_images()

Architecture:
- Modular: Separate text detection from text recognition
- Detectors: Find text regions in images (e.g., WholeImageDetector, CRAFTDetector)
- Recognizers: Read text from regions (e.g., TrOCRRecognizer, CRNNRecognizer)
- Engines: Compose detectors + recognizers (e.g., PyTorchOCREngine)
"""
from .base import *
from .factory import *
from .types import DetectorType, RecognizerType, EngineType
from . import preprocessing
from .viewer import ocr_viewer
from .cache import ImageCache

# Register monolithic engines
try:
    from .engines.tesseract import TesseractEngine
    OCREngineFactory.register_engine('tesseract', TesseractEngine)
except ImportError:
    pass  # Tesseract not available

# Register detectors
try:
    from .detectors.whole_image import WholeImageDetector
    OCREngineFactory.register_detector('whole_image', WholeImageDetector)
except ImportError:
    pass

try:
    from .detectors.craft import CRAFTDetector
    OCREngineFactory.register_detector('craft', CRAFTDetector)
except ImportError as e:
    print(f"CRAFT detector not available: {e}")

try:
    from .detectors.db import DBDetector
    OCREngineFactory.register_detector('db', DBDetector)
except ImportError as e:
    print(f"DB detector not available: {e}")

# Register recognizers
try:
    from .recognizers.trocr import TrOCRRecognizer
    OCREngineFactory.register_recognizer('trocr', TrOCRRecognizer)
except ImportError:
    pass  # trOCR not available

try:
    from .recognizers.crnn import CRNNRecognizer
    OCREngineFactory.register_recognizer('crnn', CRNNRecognizer)
except ImportError as e:
    print(f"CRNN recognizer not available: {e}")

try:
    from .recognizers.kraken import KrakenRecognizer
    OCREngineFactory.register_recognizer('kraken', KrakenRecognizer)
except ImportError as e:
    print(f"Kraken recognizer not available: {e}")
