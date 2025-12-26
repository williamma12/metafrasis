# Metafrasis Architecture

## System Overview

Metafrasis is a modular Ancient Greek OCR application with a flexible detector + recognizer architecture, fine-tuning capabilities, and a Streamlit-based user interface.

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Upload → OCR → Transliteration → Translation        │  │
│  │  [Engine Selector] [Process] [View Results]          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ OCR Service  │  │Transliterate │  │  Translation │      │
│  │  (Factory)   │  │   Service    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OCR Engine Layer                          │
│  ┌──────────────┐                                            │
│  │  Monolithic  │  Tesseract                                 │
│  └──────────────┘                                            │
│  ┌──────────────────────────────────────────────────┐        │
│  │  Modular (PyTorchOCREngine)                      │        │
│  │  ┌──────────────┐  ┌──────────────┐             │        │
│  │  │  Detectors   │→ │  Recognizers │             │        │
│  │  │ WholeImage   │  │    trOCR     │             │        │
│  │  │  (CRAFT)     │  │   (CRNN)     │             │        │
│  │  │   (DB)       │  │  (Kraken)    │             │        │
│  │  └──────────────┘  └──────────────┘             │        │
│  └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                External Model Storage                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ HuggingFace  │  │  Direct URLs │  │     S3       │      │
│  │     Hub      │  │              │  │   Storage    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Plugin Architecture
- All OCR engines implement a common `OCREngine` interface
- New engines can be added without modifying existing code
- Runtime engine selection via factory pattern

### 2. Modular Detection + Recognition
- **Text detection** (finding text regions) separated from **text recognition** (reading text)
- Detectors and recognizers are composable via `PyTorchOCREngine`
- Intermediate `TextRegion` representation with crop and bounding box
- Cross-image batching optimization for maximum GPU efficiency
- Mix and match: Any detector can work with any recognizer

### 3. External Model Hosting
- **No models in git** - repository stays lightweight
- All models hosted externally (HuggingFace, S3, direct URLs)
- `models/registry.json` is the single source of truth
- Models downloaded on first use and cached locally

### 4. Modular Services
- OCR, transliteration, and translation are separate services
- Each service can be developed and tested independently
- Clear interfaces between components

### 5. Pipeline Architecture
- Linear data flow: Image → OCR → Transliteration → Translation
- Each stage can be inspected and edited
- State preserved in Streamlit session

## Directory Structure

```
metafrasis/
├── app.py                       # Streamlit application
├── config.py                    # Configuration
├── pyproject.toml               # Dependencies
│
├── services/                    # Business logic
│   ├── ocr/                     # OCR service
│   │   ├── __init__.py         # Registration
│   │   ├── base.py             # OCREngine, Word, BoundingBox, TextRegion, OCRResult
│   │   ├── types.py            # Type-safe enums (DetectorType, RecognizerType, EngineType)
│   │   ├── factory.py          # OCREngineFactory with registries
│   │   ├── preprocessing.py    # PDF conversion, image utilities
│   │   ├── cache.py            # Temporary image caching
│   │   │
│   │   ├── detectors/          # Text detection inference (models in models/)
│   │   │   ├── base.py        # TextDetector abstract base
│   │   │   ├── whole_image.py # WholeImageDetector (pass-through)
│   │   │   ├── craft.py       # CRAFTDetector (uses models.CRAFT)
│   │   │   └── db.py          # DBDetector (uses models.DBNet)
│   │   │
│   │   ├── recognizers/        # Text recognition inference (models in models/)
│   │   │   ├── base.py        # TextRecognizer abstract base
│   │   │   ├── trocr.py       # TrOCRRecognizer (Transformer-based)
│   │   │   ├── crnn.py        # CRNNRecognizer (uses models.CRNN)
│   │   │   ├── ppocr.py       # PPOCRRecognizer (uses models.PPOCRModel)
│   │   │   └── kraken.py      # KrakenRecognizer (library wrapper)
│   │   │
│   │   └── engines/            # OCR engine implementations
│   │       ├── tesseract.py   # TesseractEngine (monolithic)
│   │       └── pytorch_engine.py  # PyTorchOCREngine (modular composition)
│   │
│   ├── transliterate_service.py
│   └── translate_service.py
│
├── training/                    # Training & optimization
│   ├── README.md
│   ├── configs/                # Training configs
│   ├── finetune/               # Fine-tuning scripts
│   ├── benchmarks/             # Engine comparison
│   ├── data/                   # Dataset creation
│   │   ├── vision_annotate.py # Vision model annotation
│   │   ├── annotation_tool.py # Manual review
│   │   └── dataset_builder.py
│   └── notebooks/              # Experiments
│
├── models/                      # Model definitions & management
│   ├── __init__.py             # Package exports
│   ├── registry.json           # Model URLs (IN GIT)
│   ├── download_models.py      # Download script (IN GIT)
│   ├── layers.py               # Shared building blocks (ConvBN, SE, CTC, etc.)
│   ├── backbones/              # Feature extractors
│   │   ├── vgg.py             # VGG16BN (CRAFT)
│   │   ├── resnet.py          # ResNetBackbone (DB)
│   │   ├── mobilenet.py       # MobileNetV3Backbone (PP-OCR)
│   │   └── crnn_cnn.py        # CRNN CNN backbone
│   ├── necks/                  # Feature aggregation
│   │   ├── fpn.py             # Feature Pyramid Network (DB)
│   │   └── sequence.py        # BiLSTM, SequenceEncoder
│   ├── heads/                  # Task-specific outputs
│   │   ├── ctc.py             # CTCHead
│   │   └── db.py              # DBHead
│   ├── composites/             # Full model architectures
│   │   ├── craft.py           # CRAFT detector
│   │   ├── dbnet.py           # DBNet detector
│   │   ├── crnn.py            # CRNN recognizer
│   │   └── ppocr.py           # PPOCRModel recognizer
│   └── [downloaded weights]    # (GITIGNORED)
│
├── data/                        # Datasets (ALL GITIGNORED)
│   ├── raw/                    # Unlabeled images
│   ├── annotated/              # Vision-model annotated
│   ├── reviewed/               # Manually corrected
│   └── datasets/               # Final HF datasets
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # This file
│   └── OCR_SERVICE.md          # Detailed OCR documentation
│
├── utils/                       # Shared utilities
└── tests/                       # Unit tests
    └── test_ocr/               # OCR service tests
        ├── test_factory.py
        ├── test_detectors/
        ├── test_recognizers/
        └── test_engines/
```

## OCR Engine Architecture

### Modular Pipeline

The core innovation is the separation of detection from recognition:

```
Image → TextDetector → TextRegions → TextRecognizer → Words → OCRResult
```

### Base Abstractions

#### OCREngine (Base for all engines)
```python
class OCREngine(ABC):
    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult:
        """Run OCR on a single image"""
        pass

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """Run OCR on multiple images (overrideable for optimization)"""
        return [self.recognize(img) for img in images]

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier"""
        pass
```

#### TextDetector (NEW - Modular Component)
```python
class TextDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> List[TextRegion]:
        """Detect text regions in an image"""
        pass

    def detect_batch(self, images: List[Image.Image]) -> List[List[TextRegion]]:
        """Batch detect (overrideable)"""
        return [self.detect(img) for img in images]
```

#### TextRecognizer (NEW - Modular Component)
```python
class TextRecognizer(ABC):
    @abstractmethod
    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """Recognize text from regions (batch-optimized)"""
        pass
```

#### PyTorchOCREngine (NEW - Composition)
```python
class PyTorchOCREngine(OCREngine):
    def __init__(self, detector: TextDetector, recognizer: TextRecognizer, batch_size: int = 8):
        self.detector = detector
        self.recognizer = recognizer
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{self.detector.name}_{self.recognizer.name}"

    def recognize(self, image: Image.Image) -> OCRResult:
        # 1. Detect text regions
        regions = self.detector.detect(image)

        # 2. Recognize text from regions
        words = self.recognizer.recognize_regions(regions)

        # 3. Return result
        return OCRResult(words=words, engine_name=self.name, ...)

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        # Cross-image batching optimization (see below)
```

### Intermediate Representation: TextRegion

Bridges detection and recognition phases:

```python
@dataclass
class TextRegion:
    bbox: BoundingBox          # Location in original image
    crop: Image.Image          # Cropped region for recognition
    confidence: float          # Detection confidence
    polygon: Optional[List]    # Optional polygon for rotated text
```

### Factory Pattern

Two creation modes with explicit parameters:

```python
from services.ocr.factory import OCREngineFactory
from services.ocr.types import DetectorType, RecognizerType, EngineType

# Monolithic engine
engine = OCREngineFactory.create(engine=EngineType.TESSERACT)

# Modular engine (explicit composition)
engine = OCREngineFactory.create(
    detector=DetectorType.WHOLE_IMAGE,
    recognizer=RecognizerType.TROCR,
    device='cuda',
    batch_size=8
)

# Registration system
OCREngineFactory.register_engine('tesseract', TesseractEngine)
OCREngineFactory.register_detector('whole_image', WholeImageDetector)
OCREngineFactory.register_recognizer('trocr', TrOCRRecognizer)
```

### Available Components

**Monolithic Engines:**
| Engine | Type | Speed | GPU | Best For |
|--------|------|-------|-----|----------|
| **Tesseract** | Traditional | Fast | No | Printed text, baseline |

**Detectors (Modular):**
| Detector | Type | Output | Use Case |
|----------|------|--------|----------|
| **WholeImageDetector** | Pass-through | Single region (entire image) | End-to-end models like trOCR |
| CRAFT (future) | Character-level | Polygons | Scene text, documents |
| DB (future) | Document | Rectangles | Fast printed text |

**Recognizers (Modular):**
| Recognizer | Type | Speed | GPU | Best For |
|------------|------|-------|-----|----------|
| **TrOCRRecognizer** | Transformer | Slow | Yes | Handwritten, Ancient Greek |
| CRNN (future) | CNN+RNN | Fast | Optional | Printed text |
| Kraken (future) | LSTM | Medium | Optional | Historical manuscripts |

### Cross-Image Batching Optimization

`PyTorchOCREngine` flattens regions across all images for maximum GPU utilization:

```
Traditional (per-image):
  Image 1 → detect → recognize → Result 1
  Image 2 → detect → recognize → Result 2
  Image 3 → detect → recognize → Result 3

Cross-image batching:
  All images → batch detect → flatten ALL regions → batch recognize → reassemble

Example: 3 images × 2 regions = 6 regions
  - With batch_size=4: [4 regions batch, 2 regions batch] = 2 forward passes
  - Per-image: 3 forward passes
```

## Model Management

### Registry System

`models/registry.json` (committed to git):
```json
{
  "trocr": {
    "base": {
      "url": "microsoft/trocr-base-handwritten",
      "type": "huggingface"
    },
    "finetuned": {
      "url": "your-username/trocr-ancient-greek-lora",
      "type": "huggingface_adapter"
    }
  }
}
```

### Download on First Use

```python
def get_model_path(engine_name: str, variant: str = "base"):
    registry = load_registry()

    # Check local cache first
    local_path = MODELS_DIR / engine_name / variant
    if local_path.exists():
        return local_path

    # Download from registry URL
    url = registry[engine_name][variant]["url"]
    download_model(url, local_path)
    return local_path
```

## Training Architecture

### Vision Model Annotation

Use large vision models to annotate real images:

```
Raw Images → Vision Model (GPT-4V/Claude) → Annotations → Review → Dataset
```

**Benefits:**
- Real-world image distribution
- Natural degradation and noise
- Faster than manual annotation
- Bootstrap from unlabeled data

### Fine-tuning Workflow

```
Dataset → Fine-tune with LoRA → Upload to HuggingFace → Update Registry
```

**Supported Methods:**
- trOCR: LoRA adapters (~10 MB)
- EasyOCR: Full fine-tuning
- Kraken: Custom model training

### Benchmarking

```
Test Set → Run All Engines → Compute CER/WER → Visualize
```

Metrics tracked:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Confidence scores (when available)
- Processing time

## Data Flow

### Runtime (User Interaction)

```
1. Upload Image/PDF
   ↓
2. Select Engine Mode (UI dropdown)
   ├─ Monolithic: Select engine
   └─ Modular: Select detector + recognizer
   ↓
3. OCR Processing
   ├─ Monolithic (Tesseract):
   │  └─ Direct recognize(image) → OCRResult
   │
   └─ Modular (PyTorchOCREngine):
      ├─ Detector.detect(image) → List[TextRegion]
      ├─ Recognizer.recognize_regions(regions) → List[Word]
      └─ Assemble → OCRResult
   ↓
4. Return OCRResult
   ├─ words: List[Word]
   ├─ text: str (concatenated)
   ├─ confidence_stats: ConfidenceStats
   └─ processing_time: float
   ↓
5. Display & Edit in UI
   ↓
6. Optional: Transliteration
   ↓
7. Optional: Translation
```

### Training (Offline)

```
1. Collect unlabeled images
   ↓
2. Annotate with vision model
   ↓
3. Manual review/correction
   ↓
4. Build HuggingFace dataset
   ↓
5. Fine-tune model (trOCR/EasyOCR/Kraken)
   ↓
6. Upload to HuggingFace/S3
   ↓
7. Update models/registry.json
```

## Configuration Management

### Environment Variables
```bash
# Vision model API keys (for annotation)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### Config Files

- `config.py`: Application settings (OCR batch size, device, etc.)
- `training/configs/*.yaml`: Training configurations
- `models/registry.json`: Model URLs and metadata

## Testing Strategy

```
tests/
├── test_ocr/
│   ├── conftest.py              # Shared fixtures (images, regions, words)
│   ├── test_factory.py          # Factory pattern, registration
│   │
│   ├── test_detectors/
│   │   ├── test_base.py        # TextDetector base class
│   │   └── test_whole_image.py # WholeImageDetector
│   │
│   ├── test_recognizers/
│   │   ├── test_base.py        # TextRecognizer base class
│   │   └── test_trocr.py       # TrOCRRecognizer (mocked)
│   │
│   └── test_engines/
│       ├── test_tesseract.py   # TesseractEngine
│       └── test_pytorch_engine.py  # PyTorchOCREngine composition, batching
│
├── test_training/
│   ├── test_vision_annotate.py
│   └── test_dataset_builder.py
│
└── fixtures/
    ├── test_images/
    └── test_labels/
```

## Performance Considerations

### Model Loading
- **Lazy loading**: Models load on first use (both detector and recognizer)
- **Caching**: Models stay loaded in memory during session
- **GPU management**: Automatic device selection with manual override
- **Modular engines**: Both detector and recognizer lazy load independently

### Image Processing
- **Cross-image batching**: Flattens regions across all images before batching
- **Batch processing support**: Configurable batch size (default: 8)
- **Preprocessing pipeline**: Resize, denoise, binarize
- **Memory-efficient streaming**: For large datasets in training

### Batching Strategy

**Per-Image (Traditional):**
```
Image 1 (2 regions) → batch → 2 words
Image 2 (2 regions) → batch → 2 words
Image 3 (2 regions) → batch → 2 words
Total: 3 recognition calls
```

**Cross-Image (Optimized):**
```
Images 1-3 → detect all → 6 regions total
  → flatten → batch(4) + batch(2) → 6 words
  → reassemble → 3 results
Total: 2 recognition calls (batch_size=4)
```

### API Costs
- Vision model annotation: ~$0.01-0.03 per image
- Budget: ~$100-300 for 10K images
- Can use open-source alternatives (LLaVA, CogVLM)

## Security Considerations

- API keys stored in environment variables (not git)
- Models downloaded from trusted sources only
- Input validation on uploaded images
- No execution of arbitrary code from models

## Future Enhancements

### Near-term (Modular Architecture)
- [ ] CRAFT detector implementation
- [ ] DB detector implementation
- [ ] CRNN recognizer implementation
- [ ] Kraken recognizer implementation
- [ ] Ensemble voting (combine multiple detector+recognizer combinations)

### Long-term
- [ ] Real-time OCR with webcam
- [ ] Mobile app integration
- [ ] Collaborative annotation
- [ ] Active learning loop
- [ ] Multi-language support (Latin, Coptic)
- [ ] Integration with digital libraries (Perseus, TLG)

## Key Architectural Decisions

### Why Separate Detection from Recognition?

**Flexibility**: Mix and match detectors with recognizers
- CRAFT + trOCR for handwritten scene text
- DB + CRNN for fast printed documents
- WholeImage + trOCR for end-to-end (current default)

**GPU Efficiency**: Cross-image batching maximizes utilization
- Flatten all regions across all images
- Single large batch instead of many small batches
- Better GPU memory utilization

**Specialization**: Each component does one thing well
- Detectors focus on finding text
- Recognizers focus on reading text
- Easier to optimize and fine-tune independently

### Why Keep Tesseract Monolithic?

Tesseract is a library wrapper - no need to decompose:
- Already implements both detection and recognition
- CPU-only, no batching benefits
- Provides baseline for comparison
- Fast and lightweight

### Why Cross-Image Batching?

**Problem**: Processing each image separately wastes GPU batch slots
- Image 1 has 2 regions → batch of 2 (underutilized if batch_size=8)
- Image 2 has 2 regions → batch of 2 (underutilized)

**Solution**: Flatten all regions across all images
- Images 1-3 have 6 total regions → batches of 4 and 2 (better utilization)
- Fewer forward passes, better GPU usage

**Trade-off**: Slightly more complex reassembly logic, but significant performance gain
