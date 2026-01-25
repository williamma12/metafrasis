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

### 6. Multi-Platform GPU Acceleration
- **CUDA**: NVIDIA GPU support via PyTorch (automatic detection)
- **MPS**: Apple Metal Performance Shaders for M1/M2/M3 Macs
  - Native CTC loss implementation in `native/mps/ctc/`
  - Future: Custom matmul and kernels for additional speedup
- **CPU**: Automatic fallback for systems without GPU
- All PyTorch models automatically use available accelerator

## Directory Structure

```
metafrasis/
├── app.py                            # Streamlit entry point
├── pyproject.toml                    # Dependencies
│
├── app/                              # Application code
│   ├── config.py                    # App configuration
│   ├── main.py                      # Streamlit main with navigation
│   │
│   ├── backend/                     # Streamlit backend
│   │   ├── pages/                  # Page implementations
│   │   │   ├── ocr.py             # OCR page (455 lines)
│   │   │   └── annotate.py        # Annotation page (455 lines)
│   │   ├── components/             # Reusable UI components
│   │   └── state.py                # Session state management
│   │
│   ├── services/                    # Business logic
│   │   ├── ocr/                    # OCR service
│   │   │   ├── base.py            # OCREngine, Word, BoundingBox, TextRegion, OCRResult
│   │   │   ├── factory.py         # OCREngineFactory with registries
│   │   │   ├── preprocessing.py   # PDF conversion, image utilities
│   │   │   │
│   │   │   ├── detectors/         # Text detection implementations
│   │   │   │   ├── base.py       # TextDetector abstract base
│   │   │   │   ├── whole_image.py # WholeImageDetector (pass-through)
│   │   │   │   ├── craft.py      # CRAFTDetector (character-level)
│   │   │   │   └── db.py         # DBDetector (differentiable binarization)
│   │   │   │
│   │   │   ├── recognizers/       # Text recognition implementations
│   │   │   │   ├── base.py       # TextRecognizer abstract base
│   │   │   │   ├── trocr.py      # TrOCRRecognizer (HuggingFace)
│   │   │   │   ├── crnn.py       # CRNNRecognizer (CTC-based)
│   │   │   │   ├── ppocr.py      # PPOCRRecognizer (Greek, PyTorch)
│   │   │   │   ├── ppocr_onnx.py # PPOCR ONNX (production)
│   │   │   │   └── kraken.py     # KrakenRecognizer (Ancient Greek)
│   │   │   │
│   │   │   └── engines/           # Complete OCR pipelines
│   │   │       ├── tesseract.py  # TesseractEngine (monolithic)
│   │   │       └── pytorch_engine.py  # PyTorchOCREngine (modular)
│   │   │
│   │   └── annotation/             # Annotation service
│   │       ├── models.py          # Point, Region, AnnotatedImage, Dataset (248 lines)
│   │       ├── storage.py         # Load/save, dataset management (392 lines)
│   │       ├── exporter.py        # ZIP export for datasets (221 lines)
│   │       └── canvas.py          # Streamlit component bridge (137 lines)
│   │
│   └── frontend/                    # React/TypeScript components
│       ├── ocr_viewer/             # OCR result visualization (16 tests)
│       └── annotation_canvas/      # Annotation drawing (30 tests)
│
├── ml/                              # Machine learning code
│   ├── config.py                   # ML configuration (device detection, registry)
│   │
│   ├── models/                     # PyTorch model definitions
│   │   ├── registry.json          # Model URLs (committed)
│   │   ├── download_models.py     # Download script (committed)
│   │   ├── layers.py              # Shared layers (ConvBN, SE, CTC, etc.)
│   │   │
│   │   ├── backbones/             # Feature extractors
│   │   │   ├── vgg.py            # VGG16BN (CRAFT)
│   │   │   ├── resnet.py         # ResNetBackbone (DB)
│   │   │   ├── mobilenet.py      # MobileNetV3 (PP-OCR)
│   │   │   └── crnn_cnn.py       # CRNN CNN
│   │   │
│   │   ├── necks/                 # Feature aggregation
│   │   │   ├── fpn.py            # Feature Pyramid Network (DB)
│   │   │   └── sequence.py       # BiLSTM, SequenceEncoder
│   │   │
│   │   ├── heads/                 # Task-specific outputs
│   │   │   ├── ctc.py            # CTCHead (recognition)
│   │   │   └── db.py             # DBHead (detection)
│   │   │
│   │   └── composites/            # Full model architectures
│   │       ├── craft.py          # CRAFT detector
│   │       ├── dbnet.py          # DBNet detector
│   │       ├── crnn.py           # CRNN recognizer
│   │       └── ppocr.py          # PPOCRModel recognizer
│   │
│   └── training/                   # Training infrastructure (~3500 lines)
│       ├── finetune/              # Fine-tuning scripts
│       │   ├── base.py           # BaseTrainer (~570 lines)
│       │   ├── recognizers/
│       │   │   ├── crnn.py       # CTC-based training (136 lines)
│       │   │   ├── ppocr.py      # PP-OCR training
│       │   │   └── trocr.py      # LoRA fine-tuning (337 lines)
│       │   └── detectors/
│       │       ├── craft.py      # CRAFT training
│       │       └── db.py         # DB training
│       │
│       ├── evaluate/              # Evaluation metrics
│       │   └── metrics.py        # CER, WER, Precision, Recall, F1
│       │
│       └── export/                # Model export
│           ├── to_onnx.py        # ONNX conversion (274 lines)
│           └── to_huggingface.py # HuggingFace Hub upload (334 lines)
│
├── native/                          # Platform-specific optimizations
│   └── mps/                        # Metal Performance Shaders (macOS GPU)
│       ├── ctc/                   # CTC loss (Metal-accelerated)
│       │   ├── ctc_loss.py       # Python interface
│       │   ├── csrc/             # Metal/C++ implementation
│       │   └── tests/            # Native extension tests
│       ├── matmul/                # Matrix operations (planned)
│       └── kernels/               # Custom kernels (planned)
│
├── data/                            # All gitignored
│   ├── model_weights/              # Downloaded model weights
│   ├── annotations/                # Annotation datasets
│   └── training/                   # Training datasets
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md             # This file
│   ├── OCR_SERVICE.md              # OCR implementation details
│   ├── ANNOTATION.md               # Annotation tool usage
│   ├── TESTING.md                  # Testing guide
│   └── QUICKSTART.md               # Setup guide
│
└── tests/                           # Comprehensive test suite (246 tests)
    ├── ml/models/                  # ML model tests (148 tests)
    ├── app/                        # Backend tests (52 tests)
    ├── frontend/                   # Component tests (46 tests)
    └── native/                     # Native extension tests
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
| **CRAFT** | Character-level | Polygons | Scene text, documents, manuscripts |
| **DB** | Differentiable Binarization | Rectangles | Fast printed text, documents |

**Recognizers (Modular):**
| Recognizer | Type | Speed | GPU | Best For |
|------------|------|-------|-----|----------|
| **TrOCRRecognizer** | Transformer | Slow | Yes | Handwritten text |
| **CRNNRecognizer** | CNN+RNN+CTC | Fast | Yes (CUDA, MPS, CPU) | Printed text, English |
| **KrakenRecognizer** | LSTM | Medium | Yes (CUDA, MPS, CPU) | Ancient Greek manuscripts |
| **PPOCRRecognizer** | MobileNetV3+BiLSTM | Fast | Yes (CUDA, MPS, CPU) | Greek language text |
| **PPOCRONNXRecognizer** | ONNX Runtime | Very Fast | CPU optimized | Greek language text (production) |

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

## Annotation Service Architecture

The annotation service provides a complete solution for creating and managing text region datasets.

### Data Models

```python
@dataclass
class Point:
    x: float
    y: float

@dataclass
class Region:
    id: str
    type: str  # 'rectangle' or 'polygon'
    points: List[Point]
    text: Optional[str]
    auto_detected: bool  # True if from CRAFT, False if user-drawn
    verified: bool       # User has reviewed this region

@dataclass
class AnnotatedImage:
    image_id: str
    image_path: str
    regions: List[Region]
    created_at: datetime
    updated_at: datetime

@dataclass
class AnnotationDataset:
    dataset_id: str
    name: str
    images: List[AnnotatedImage]
    created_at: datetime
```

### Storage Layer

- **JSON-based persistence**: Each dataset stored as JSON file
- **Image references**: Relative paths to original images
- **Version management**: Migration system for schema changes
- **Atomic operations**: Safe concurrent access

### Export Functionality

- **ZIP export**: Package entire dataset with images and metadata
- **HuggingFace integration**: Export in HF datasets format (planned)
- **COCO/VOC formats**: Support for common annotation formats (planned)

### Frontend Integration

- **React Canvas**: Custom Streamlit component for drawing
- **Drawing modes**: Rectangle, polygon, and select modes
- **Auto-detection**: CRAFT-based region proposal
- **Keyboard shortcuts**: Delete, Escape, etc.

See **[ANNOTATION.md](ANNOTATION.md)** for detailed usage guide.

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

## Training Infrastructure

The project includes comprehensive training infrastructure (~3500 lines of functional code).

### Base Trainer

`ml/training/finetune/base.py` provides a complete training loop with:
- **Training loop**: Epochs, batches, gradient accumulation
- **Evaluation**: Validation metrics, early stopping
- **Checkpointing**: Save best model, resume from checkpoint
- **Logging**: Metrics, learning rate, loss curves
- **Device management**: Automatic GPU selection (CUDA, MPS, CPU)

### Fine-tuning Support

**Recognition Models:**
- **CRNN**: CTC-based training with CTCRecognizerTrainer
- **PP-OCR**: Greek text recognition with gradient accumulation
- **trOCR**: LoRA adapter fine-tuning (memory efficient, ~10 MB adapters)

**Detection Models:**
- **CRAFT**: Character-level detector training
- **DBNet**: Differentiable Binarization detector training

### Evaluation Metrics

`ml/training/evaluate/metrics.py`:
- **Character Error Rate (CER)**: Edit distance at character level
- **Word Error Rate (WER)**: Edit distance at word level
- **Precision/Recall/F1**: For detection tasks
- **Confidence calibration**: Alignment between confidence and accuracy

### Model Export

**ONNX Conversion** (`ml/training/export/to_onnx.py`):
- Convert PyTorch models to ONNX format
- Production deployment with ONNX Runtime
- Optimized inference (CPU-focused)

**HuggingFace Hub** (`ml/training/export/to_huggingface.py`):
- Upload models with model cards
- Version management
- Automatic README generation with usage examples

### Vision Model Annotation (Planned)

Future: Use large vision models to annotate real images:

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
Dataset → Train with BaseTrainer → Evaluate → Export (ONNX/HF) → Update Registry
```

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

Comprehensive test suite with **246 tests, 100% passing**.

### Test Organization

```
tests/
├── ml/models/                  # ML model tests (148 tests)
│   ├── test_layers.py         # Shared layers (ConvBN, SE, CTC, etc.)
│   ├── test_backbones.py      # VGG, ResNet, MobileNetV3
│   ├── test_necks.py          # FPN, BiLSTM, SequenceEncoder
│   ├── test_heads.py          # CTCHead, DBHead
│   ├── test_composites.py     # CRAFT, DBNet, CRNN, PPOCRModel
│   └── test_registry.py       # Model registry and download
│
├── app/                        # Backend tests (52 tests)
│   ├── services/ocr/
│   │   ├── test_factory.py   # Factory pattern, registration
│   │   ├── test_detectors/   # CRAFT, DB, WholeImage
│   │   ├── test_recognizers/ # trOCR, CRNN, Kraken, PP-OCR
│   │   └── test_engines/     # Tesseract, PyTorchOCREngine
│   │
│   └── services/annotation/
│       ├── test_models.py    # Data models
│       ├── test_storage.py   # Load/save, dataset management
│       └── test_exporter.py  # ZIP export
│
├── frontend/                   # Component tests (46 tests, Vitest)
│   ├── ocr_viewer/tests/      # OCR Viewer tests (16 tests)
│   └── annotation_canvas/tests/  # Annotation Canvas tests (30 tests)
│
└── native/                     # Native extension tests
    └── mps/ctc/tests/         # Metal CTC loss tests
```

### Test Infrastructure

- **pytest**: Python backend and ML tests
- **Vitest**: Frontend React component tests
- **Makefile**: Convenient test commands (make test-all, make test-ml, etc.)
- **Coverage reporting**: HTML reports for all test suites
- **Fast execution**: Mock weights, no model downloads

### Running Tests

```bash
# All tests (Python + Frontend)
make test-all              # 246 tests

# Specific test suites
make test-ml               # ML model tests (148 tests)
make test-backend          # Backend tests (52 tests)
make test-frontend         # Frontend tests (46 tests)

# With coverage
make test-coverage-ml
make test-coverage-backend

# Quick tests (skip slow tests)
make test-quick
```

See **[TESTING.md](../TESTING.md)** for complete testing guide.

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

See **[ROADMAP.md](../ROADMAP.md)** for the complete feature roadmap and priorities.

### High Priority
- [ ] EasyOCR engine integration
- [ ] Training UI (Streamlit interface for fine-tuning)
- [ ] Improved Ancient Greek models (fine-tuned on more data)
- [ ] OCR confidence filtering UI
- [ ] Vision model annotation (GPT-4V, Claude integration)

### Medium Priority
- [ ] Ensemble voting (combine multiple detector+recognizer combinations)
- [ ] Region merging and splitting tools
- [ ] Multi-language switching (Ancient Greek, Latin, Coptic)
- [ ] Batch annotation export (COCO, Pascal VOC, YOLO formats)

### Long-term
- [ ] Distributed training (multi-GPU with PyTorch DDP)
- [ ] Mixed precision training (FP16/BF16)
- [ ] CLI tool for batch processing
- [ ] Docker support
- [ ] API server with async job queue
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
