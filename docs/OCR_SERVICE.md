# OCR Service Documentation

## Overview

The OCR service provides a modular architecture for optical character recognition with support for both monolithic and composable engines. The architecture separates text **detection** (finding text regions) from text **recognition** (reading text), allowing flexible composition of different detectors and recognizers.

## Architecture

### Modular Pipeline

The OCR service implements a two-stage pipeline for modular engines:

```
Image → TextDetector → TextRegions → TextRecognizer → Words → OCRResult
```

**Components:**
- **TextDetector**: Finds text regions in images
- **TextRegion**: Intermediate representation (bounding box + cropped image)
- **TextRecognizer**: Reads text from cropped regions
- **Word**: Recognized text with bounding box and confidence
- **OCRResult**: Final output with list of words and metadata

### Two Engine Modes

#### 1. Monolithic Engines
Traditional all-in-one engines that handle both detection and recognition internally.

**Example: Tesseract**
```python
from services.ocr.factory import OCREngineFactory

engine = OCREngineFactory.create(engine='tesseract')
result = engine.recognize(image)
```

#### 2. Modular Engines (PyTorchOCREngine)
Compose separate detector and recognizer components for maximum flexibility.

**Example: WholeImage + trOCR**
```python
from services.ocr.factory import OCREngineFactory
from services.ocr.types import DetectorType, RecognizerType

engine = OCREngineFactory.create(
    detector=DetectorType.WHOLE_IMAGE,
    recognizer=RecognizerType.TROCR,
    device='cuda',
    batch_size=8
)
result = engine.recognize(image)
```

## Core Components

### Data Structures

#### BoundingBox
```python
@dataclass
class BoundingBox:
    """Rectangular bounding box"""
    left: int
    top: int
    width: int
    height: int
```

#### TextRegion
```python
@dataclass
class TextRegion:
    """
    A detected text region before recognition

    Serves as the bridge between detection and recognition phases.
    Contains both the location (bbox) and the image content (crop).
    """
    bbox: BoundingBox          # Region coordinates in original image
    crop: Image.Image          # Cropped image of this region
    confidence: float          # Detection confidence (-1.0 if unavailable)
    polygon: Optional[List[tuple]] = None  # Optional polygon for rotated text
```

#### Word
```python
@dataclass
class Word:
    """A recognized word with location and confidence"""
    text: str
    bbox: BoundingBox
    confidence: float  # -1.0 if unavailable
```

#### OCRResult
```python
@dataclass
class OCRResult:
    """Final OCR output"""
    words: List[Word]
    engine_name: str
    processing_time: float
    source: str = ""

    @property
    def text(self) -> str:
        """Concatenated text from all words"""
        return " ".join(word.text for word in self.words)

    @property
    def confidence_stats(self) -> ConfidenceStats:
        """Statistics on word confidences"""
        # Returns mean, std, availability flag
```

### Base Classes

#### OCREngine
```python
from abc import ABC, abstractmethod
from PIL import Image

class OCREngine(ABC):
    """Base class for all OCR engines"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_loaded = False

    @abstractmethod
    def load_model(self):
        """Load the OCR model (lazy loading)"""
        pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult:
        """Run OCR on a single image"""
        pass

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Run OCR on multiple images

        Default implementation processes sequentially.
        Engines can override for optimized batch processing.
        """
        return [self.recognize(img) for img in images]

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier"""
        pass
```

#### TextDetector
```python
class TextDetector(ABC):
    """Base class for text detectors"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.is_loaded = False

    @abstractmethod
    def detect(self, image: Image.Image) -> List[TextRegion]:
        """
        Detect text regions in an image

        Args:
            image: PIL Image

        Returns:
            List of TextRegion objects with bounding boxes and crops
        """
        pass

    def detect_batch(self, images: List[Image.Image]) -> List[List[TextRegion]]:
        """
        Detect text regions in multiple images

        Default implementation processes sequentially.
        Detectors can override for batch optimization.

        Args:
            images: List of PIL Images

        Returns:
            List of lists of TextRegion objects (one list per image)
        """
        return [self.detect(img) for img in images]

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector identifier"""
        pass
```

#### TextRecognizer
```python
class TextRecognizer(ABC):
    """Base class for text recognizers"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.is_loaded = False

    @abstractmethod
    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from multiple regions (batch processing)

        This is the primary method - designed for batch efficiency.
        Single region recognition is just a batch of one.

        Args:
            regions: List of TextRegion objects with cropped images

        Returns:
            List of Word objects with recognized text and confidences
            (one Word per TextRegion, in same order)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Recognizer identifier"""
        pass
```

#### PyTorchOCREngine
```python
class PyTorchOCREngine(OCREngine):
    """
    Composes a detector and recognizer into a complete OCR pipeline

    Implements cross-image batching optimization for maximum GPU utilization.
    """

    def __init__(
        self,
        detector: TextDetector,
        recognizer: TextRecognizer,
        batch_size: int = 8
    ):
        super().__init__()
        self.detector = detector
        self.recognizer = recognizer
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{self.detector.name}_{self.recognizer.name}"

    def load_model(self):
        """Lazy load both detector and recognizer"""
        if not self.is_loaded:
            self.detector.load_model()
            self.recognizer.load_model()
            self.is_loaded = True

    def recognize(self, image: Image.Image) -> OCRResult:
        """Single image OCR pipeline"""
        # 1. Detect regions
        regions = self.detector.detect(image)

        # 2. Recognize text in regions (with internal batching)
        words = self.recognizer.recognize_regions(regions)

        # 3. Return result
        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=0.0  # Calculated during execution
        )

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Batch OCR with cross-image batching optimization

        Optimizes GPU utilization by batching regions across all images,
        not just within each image.
        """
        # See Cross-Image Batching section below for details
```

## Directory Structure

```
services/ocr/
├── __init__.py              # Package initialization, registration
├── base.py                  # OCRResult, OCREngine, Word, BoundingBox, TextRegion
├── types.py                 # DetectorType, RecognizerType, EngineType enums
├── factory.py               # OCREngineFactory with registries
├── preprocessing.py         # PDF conversion, image utilities
├── cache.py                 # ImageCache for temporary storage
│
├── detectors/               # Text detection components
│   ├── __init__.py
│   ├── base.py             # TextDetector abstract base class
│   ├── whole_image.py      # WholeImageDetector (pass-through)
│   ├── craft.py            # CRAFTDetector (character-level detection)
│   └── db.py               # DBDetector (differentiable binarization)
│
├── recognizers/             # Text recognition components
│   ├── __init__.py
│   ├── base.py             # TextRecognizer abstract base class
│   ├── trocr.py            # TrOCRRecognizer (Transformer-based)
│   ├── crnn.py             # CRNNRecognizer (CNN+RNN with CTC)
│   └── kraken.py           # KrakenRecognizer (historical documents)
│
└── engines/                 # OCR engine implementations
    ├── __init__.py
    ├── tesseract.py        # TesseractEngine (monolithic)
    └── pytorch_engine.py   # PyTorchOCREngine (modular composition)
```

## Factory Pattern

### OCREngineFactory

Central registry and factory for creating OCR engines.

```python
from services.ocr.factory import OCREngineFactory
from services.ocr.types import DetectorType, RecognizerType, EngineType

# Monolithic engine creation
engine = OCREngineFactory.create(engine=EngineType.TESSERACT)
engine = OCREngineFactory.create(engine='tesseract')

# Modular engine creation (explicit composition)
engine = OCREngineFactory.create(
    detector=DetectorType.WHOLE_IMAGE,
    recognizer=RecognizerType.TROCR,
    device='cuda',
    batch_size=8
)

# String-based composition (equivalent)
engine = OCREngineFactory.create(
    detector='whole_image',
    recognizer='trocr',
    device='cuda'
)

# Query available components
engines = OCREngineFactory.available_engines()        # ['tesseract']
detectors = OCREngineFactory.available_detectors()    # ['whole_image', 'craft', 'db']
recognizers = OCREngineFactory.available_recognizers()  # ['trocr', 'crnn', 'kraken']
```

### Registration System

New components can be registered dynamically:

```python
# In services/ocr/__init__.py
from .factory import OCREngineFactory

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

# Register recognizers
try:
    from .recognizers.trocr import TrOCRRecognizer
    OCREngineFactory.register_recognizer('trocr', TrOCRRecognizer)
except ImportError:
    pass  # trOCR dependencies not available
```

### Type-Safe Enums

```python
from enum import Enum

class DetectorType(str, Enum):
    """Available text detectors"""
    WHOLE_IMAGE = "whole_image"
    CRAFT = "craft"
    DB = "db"

class RecognizerType(str, Enum):
    """Available text recognizers"""
    TROCR = "trocr"
    CRNN = "crnn"
    KRAKEN = "kraken"

class EngineType(str, Enum):
    """Available monolithic engines"""
    TESSERACT = "tesseract"
```

## Available Components

### Monolithic Engines

| Engine | Type | Speed | Accuracy | GPU | Best For |
|--------|------|-------|----------|-----|----------|
| **Tesseract** | Traditional | Fast | Good | No | Printed text, baseline comparisons |

**Tesseract** uses pytesseract library, supports Ancient Greek (`grc` language code), and provides word-level bounding boxes with confidence scores.

### Detectors

| Detector | Purpose | Output | Use Case |
|----------|---------|--------|----------|
| **WholeImageDetector** | Pass-through for end-to-end models | Single TextRegion covering entire image | Use with recognizers that don't need explicit detection (like trOCR) |
| **CRAFTDetector** | Character-level text detection | Multiple TextRegions (words/characters) | Scene text, documents with complex layouts, multi-region detection |
| **DBDetector** | Real-time text detection | Multiple TextRegions (word boxes) | Fast document/scene text detection, real-time applications |

**CRAFTDetector** features:
- Character Region Awareness For Text detection (CRAFT)
- VGG16-BN backbone with U-Net decoder architecture
- Dual heatmap outputs: region score (text) + affinity score (character linkage)
- Configurable thresholds for detection sensitivity
- Handles rotated and curved text with polygon support
- GPU acceleration with CPU fallback
- Pretrained weights available: MLT (multi-lingual) and ICDAR datasets

**DBDetector** features:
- Differentiable Binarization (DB) for real-time text detection
- ResNet backbone with Feature Pyramid Network (FPN)
- Dual head outputs: probability map + adaptive threshold map
- Differentiable binarization for end-to-end training
- Fast inference: suitable for real-time applications
- Configurable thresholds and unclip ratio for box expansion
- Works well on both document and scene text
- GPU/CPU support with batch processing
- Pretrained weights: ResNet50 (base) and MobileNetV3 (fast)

### Recognizers

| Recognizer | Type | Speed | Accuracy | GPU | Best For |
|------------|------|-------|----------|-----|----------|
| **TrOCRRecognizer** | Transformer | Slow | Excellent | Yes | Handwritten text, fine-tuning, Ancient Greek |
| **CRNNRecognizer** | CNN+RNN | Fast | Good | Optional | Printed text, real-time applications, resource-constrained environments |
| **KrakenRecognizer** | RNN | Medium | Excellent | Optional | Historical documents, Ancient Greek manuscripts, degraded text |

**TrOCRRecognizer** features:
- HuggingFace transformer model (default: `microsoft/trocr-base-handwritten`)
- Batch processing with configurable batch size (default: 8)
- GPU acceleration with automatic CPU fallback
- Uses detection confidence (trOCR doesn't provide character-level confidence)

**CRNNRecognizer** features:
- Convolutional Recurrent Neural Network with CTC decoding
- CNN backbone (7 conv layers) for feature extraction
- Bidirectional LSTM for sequence modeling
- CTC (Connectionist Temporal Classification) for alignment-free decoding
- Fast inference: ~10x faster than transformer models
- Configurable character set (default: digits + lowercase letters)
- Batch processing with configurable batch size (default: 16)
- Works efficiently on both CPU and GPU
- Grayscale or RGB input support
- Pretrained weights available from Deep Text Recognition Benchmark

**KrakenRecognizer** features:
- Specialized for historical document OCR
- Excellent Ancient Greek support (multiple pretrained models)
- Handles degraded, noisy, and low-quality documents
- Supports polytonic Greek with diacritics
- Built-in layout analysis (optional)
- Custom model training support
- Multiple pretrained models: Porson typeface, Medieval Greek, Polytonic Greek
- Wraps the kraken library (install: `pip install kraken`)
- Configurable language/script codes
- Batch processing with configurable batch size (default: 8)

## Cross-Image Batching Optimization

`PyTorchOCREngine` implements an advanced batching strategy that processes regions across all images simultaneously for maximum GPU utilization.

### Traditional Approach (Per-Image)
```
Image 1 → detect → [region1, region2] → recognize → 2 words
Image 2 → detect → [region3, region4] → recognize → 2 words
Image 3 → detect → [region5, region6] → recognize → 2 words

Total: 3 detection calls + 3 recognition calls
```

### Cross-Image Batching Approach
```
All Images → batch detect → [[region1, region2], [region3, region4], [region5, region6]]
                           ↓
                    Flatten to [region1, region2, region3, region4, region5, region6]
                           ↓
              Batch recognize with batch_size=4 → 2 batches: [4 regions, 2 regions]
                           ↓
                    Reassemble to 3 OCRResults

Total: 1 detection call + 2 recognition calls
```

### Benefits
- **Maximizes GPU batch slots**: No wasted capacity from underutilized batches
- **Fewer model forward passes**: More efficient than per-image processing
- **Maintains correctness**: Region counts track which words belong to which image

### Implementation
```python
def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
    """Cross-image batching pipeline"""

    # Step 1: Batch detect all images
    all_regions = self.detector.detect_batch(images)

    # Step 2: Flatten regions across all images
    flattened_regions = []
    region_counts = []  # Track regions per image

    for regions in all_regions:
        flattened_regions.extend(regions)
        region_counts.append(len(regions))

    # Step 3: Batch recognize ALL regions together
    all_words = []
    for i in range(0, len(flattened_regions), self.batch_size):
        batch_regions = flattened_regions[i:i + self.batch_size]
        batch_words = self.recognizer.recognize_regions(batch_regions)
        all_words.extend(batch_words)

    # Step 4: Reassemble words to per-image results
    results = []
    word_idx = 0
    for img_idx, num_regions in enumerate(region_counts):
        image_words = all_words[word_idx:word_idx + num_regions]
        word_idx += num_regions

        result = OCRResult(
            words=image_words,
            engine_name=self.name,
            processing_time=avg_time,
            source=f"image_{img_idx+1}"
        )
        results.append(result)

    return results
```

## Usage Examples

### Basic Usage - Tesseract

```python
from services.ocr.factory import OCREngineFactory
from PIL import Image

# Create engine
engine = OCREngineFactory.create(engine='tesseract')

# Single image
image = Image.open('document.png')
result = engine.recognize(image)

print(f"Text: {result.text}")
print(f"Confidence: {result.confidence_stats.mean:.2%}")
print(f"Words: {len(result.words)}")
```

### Advanced Usage - Modular trOCR

```python
from services.ocr.factory import OCREngineFactory
from services.ocr.types import DetectorType, RecognizerType

# Create modular engine
engine = OCREngineFactory.create(
    detector=DetectorType.WHOLE_IMAGE,
    recognizer=RecognizerType.TROCR,
    device='cuda',
    batch_size=8
)

# Single image
result = engine.recognize(image)

# Batch processing with cross-image optimization
images = [Image.open(f'page_{i}.png') for i in range(10)]
results = engine.recognize_batch(images)

for i, result in enumerate(results):
    print(f"Page {i+1}: {result.text}")
```

### CRAFT + CRNN for Scene Text and Documents

```python
from services.ocr.factory import OCREngineFactory
from services.ocr.types import DetectorType, RecognizerType

# Create CRAFT + CRNN engine for fast, accurate printed text
engine = OCREngineFactory.create(
    detector=DetectorType.CRAFT,
    recognizer=RecognizerType.CRNN,
    device='cuda',
    batch_size=16
)

# Process image with complex layout
image = Image.open('document.png')
result = engine.recognize(image)

# CRAFT detects multiple text regions
print(f"Detected {len(result.words)} text regions")
for word in result.words:
    print(f"'{word.text}' at ({word.bbox.left}, {word.bbox.top})")
```

### DB + Kraken for Ancient Greek Manuscripts

```python
from services.ocr.factory import OCREngineFactory
from services.ocr.types import DetectorType, RecognizerType

# Create DB + Kraken engine for Ancient Greek documents
engine = OCREngineFactory.create(
    detector=DetectorType.DB,
    recognizer=RecognizerType.KRAKEN,
    device='cuda',
    # DB-specific parameters
    thresh=0.3,              # Binary threshold
    box_thresh=0.7,          # Box confidence threshold
    unclip_ratio=1.5,        # Box expansion ratio
    # Kraken-specific parameters
    language='grc',          # Ancient Greek
    recognizer_model_path='models/kraken/kraken_polytonic_greek.mlmodel'
)

# Process Ancient Greek manuscript
manuscript = Image.open('greek_manuscript.jpg')
result = engine.recognize(manuscript)

print(f"Recognized Ancient Greek text:")
print(result.text)
```

### Custom Device and Batch Size

```python
# CPU-only processing
engine = OCREngineFactory.create(
    detector='whole_image',
    recognizer='trocr',
    device='cpu',
    batch_size=4  # Smaller batches for CPU
)

# Large GPU batch processing
engine = OCREngineFactory.create(
    detector='whole_image',
    recognizer='trocr',
    device='cuda',
    batch_size=32  # Larger batches for GPU
)

# CRAFT + CRNN with custom thresholds
engine = OCREngineFactory.create(
    detector='craft',
    recognizer='crnn',
    device='cuda',
    # CRAFT-specific parameters
    text_threshold=0.7,     # Higher = fewer false positives
    link_threshold=0.4,     # Controls character grouping
    canvas_size=1920,       # Max image dimension
    # CRNN-specific parameters
    charset="0123456789abcdefghijklmnopqrstuvwxyz",
    img_height=32,
    img_width=100
)
```

### PDF Processing

```python
from services.ocr.preprocessing import pdf_to_images

# Convert PDF to images
images = pdf_to_images('manuscript.pdf', dpi=300)

# Process all pages
engine = OCREngineFactory.create(
    detector='whole_image',
    recognizer='trocr',
    device='cuda'
)

results = engine.recognize_batch(images)

# Extract text from all pages
full_text = "\n\n".join(r.text for r in results)
```

## Performance Considerations

### Model Loading
- **Lazy loading**: Models load on first `recognize()` call
- **Caching**: Models stay in memory during session
- **Device management**: Auto-detect CUDA/CPU with manual override

### Batch Processing
- **Cross-image batching**: Flattens regions across all images for maximum GPU utilization
- **Configurable batch size**: Tune based on GPU memory (default: 8)
- **Sequential fallback**: If batch processing fails, falls back to sequential

### Memory Management
- **TextRegion crops**: Store PIL Images (memory-efficient)
- **Temporary caching**: ImageCache for intermediate results
- **Auto-cleanup**: Cache clears on deletion or explicit `.clear()`

## Testing

The OCR service has comprehensive test coverage:

```
tests/test_ocr/
├── conftest.py                          # Shared fixtures
├── test_factory.py                      # Factory pattern tests
│
├── test_detectors/
│   ├── test_base.py                    # TextDetector base class
│   ├── test_whole_image.py             # WholeImageDetector
│   └── test_craft.py                   # CRAFTDetector
│
├── test_recognizers/
│   ├── test_base.py                    # TextRecognizer base class
│   ├── test_trocr.py                   # TrOCRRecognizer (mocked)
│   └── test_crnn.py                    # CRNNRecognizer
│
└── test_engines/
    ├── test_tesseract.py               # TesseractEngine
    └── test_pytorch_engine.py          # PyTorchOCREngine composition
```

Run tests:
```bash
# All tests
pytest tests/test_ocr/ -v

# Skip tests requiring pretrained models
pytest tests/test_ocr/ -v -m "not requires_craft and not requires_crnn"

# Only CRAFT tests
pytest tests/test_ocr/test_detectors/test_craft.py -v

# Only CRNN tests
pytest tests/test_ocr/test_recognizers/test_crnn.py -v
```

## Installation

### Base Installation

Install the base OCR dependencies:
```bash
uv sync
```

### Component-Specific Dependencies

Install additional dependencies for specific OCR components:

```bash
# For CRAFT detector
uv sync --extra craft

# For DB detector
uv sync --extra db

# For Kraken recognizer
uv sync --extra kraken

# Install all OCR components
uv sync --extra ocr-full

# With development dependencies
uv sync --extra dev --extra ocr-full
```

**Component Requirements:**
- **CRAFT**: Requires `scipy` for connected component analysis
- **DB**: Requires `pyclipper` for polygon operations
- **Kraken**: Requires `kraken` library (OCR toolkit for historical documents)
- **CRNN**: No additional dependencies (uses base PyTorch)
- **TrOCR**: No additional dependencies (uses base transformers + PyTorch)

## Model Weights

### Downloading Pretrained Weights

CRAFT and CRNN require pretrained model weights for optimal performance. Use the download utility:

```bash
# List available models
python models/download_models.py --list

# Download CRAFT weights (MLT dataset - recommended)
python models/download_models.py --craft base

# Download CRNN weights
python models/download_models.py --crnn base

# Download DB weights
python models/download_models.py --db base

# Download Kraken weights (Ancient Greek)
python models/download_models.py --kraken greek

# Download all models
python models/download_models.py --all
```

**Available model variants:**

**CRAFT:**
- `base` (craft_mlt_25k.pth): Trained on Multi-Lingual Text dataset - recommended for general use
- `icdar` (craft_ic15_20k.pth): Trained on ICDAR 2015 dataset - optimized for scene text

**CRNN:**
- `base` (crnn_vgg_bilstm_ctc.pth): VGG + BiLSTM with CTC, supports digits and lowercase letters

**DB:**
- `base` (db_resnet50.pth): ResNet50 backbone - best accuracy, multilingual
- `mobilenet` (db_mobilenet.pth): MobileNetV3 backbone - faster, mobile-friendly

**Kraken:**
- `greek` (kraken_greek_porson.mlmodel): Ancient Greek Porson typeface
- `greek_medieval` (kraken_medieval_greek.mlmodel): Medieval Greek manuscripts
- `polytonic` (kraken_polytonic_greek.mlmodel): Polytonic Greek with diacritics

**Model Registry:**

Models are registered in `models/registry.json` with download URLs and metadata. Downloaded weights are stored in model-specific subdirectories:
```
models/
├── craft/
│   └── craft_mlt_25k.pth
├── crnn/
│   └── crnn_vgg_bilstm_ctc.pth
├── db/
│   └── db_resnet50.pth
└── kraken/
    ├── kraken_greek_porson.mlmodel
    ├── kraken_medieval_greek.mlmodel
    └── kraken_polytonic_greek.mlmodel
```

### Using Pretrained Weights

```python
# With pretrained weights
engine = OCREngineFactory.create(
    detector='craft',
    recognizer='crnn',
    device='cuda',
    # Specify weight paths (optional - uses defaults if not provided)
    detector_model_path='models/craft/craft_mlt_25k.pth',
    recognizer_model_path='models/crnn/crnn_vgg_bilstm_ctc.pth'
)

# Without weights (random initialization - for testing only)
engine = OCREngineFactory.create(
    detector='craft',
    recognizer='crnn',
    device='cpu'
)
```

## Future Enhancements

### Potential Future Components

All major detector and recognizer types are now implemented. Potential future additions:

**Additional Detectors:**
- **PSENet**: Progressive Scale Expansion Network for arbitrary-shaped text
- **EAST**: Efficient and Accurate Scene Text detector
- **TextSnake**: Detector for curved text

**Additional Recognizers:**
- **ASTER**: Attentional Scene Text Recognizer with rectification
- **SAR**: Show, Attend and Read for irregular text
- **Fine-tuned models**: Custom Ancient Greek models trained on specific manuscripts

### Ensemble Engines
Combine multiple engines with voting strategies:
```python
engine = OCREngineFactory.create_ensemble(
    engines=['tesseract', 'whole_image_trocr'],
    strategy='confidence_weighted'
)
```

## Migration Notes

### From Old Monolithic trOCR
The old monolithic `TrOCREngine` has been replaced with modular components:

**Old (deprecated):**
```python
engine = OCREngineFactory.create(engine='trocr')  # No longer works
```

**New (current):**
```python
engine = OCREngineFactory.create(
    detector='whole_image',
    recognizer='trocr',
    device='cuda'
)
```

The modular approach provides the same functionality with added flexibility for future detector combinations (e.g., CRAFT + trOCR).
