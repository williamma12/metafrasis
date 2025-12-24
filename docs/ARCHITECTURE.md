# Metafrasis Architecture

## System Overview

Metafrasis is a modular Ancient Greek OCR application with support for multiple OCR engines, fine-tuning capabilities, and a Streamlit-based user interface.

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
│                 OCR Engine Plugins                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Tesseract │ │  Kraken  │ │ EasyOCR  │ │  trOCR   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────────────────────────────┐                       │
│  │    Ensemble (Voting)             │                       │
│  └──────────────────────────────────┘                       │
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

### 2. External Model Hosting
- **No models in git** - repository stays lightweight
- All models hosted externally (HuggingFace, S3, direct URLs)
- `models/registry.json` is the single source of truth
- Models downloaded on first use and cached locally

### 3. Modular Services
- OCR, transliteration, and translation are separate services
- Each service can be developed and tested independently
- Clear interfaces between components

### 4. Pipeline Architecture
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
│   │   ├── base.py             # Abstract base class
│   │   ├── factory.py          # Engine factory
│   │   ├── preprocessing.py    # Image preprocessing
│   │   ├── postprocessing.py   # Text postprocessing
│   │   └── engines/            # Engine implementations
│   │       ├── tesseract.py
│   │       ├── kraken.py
│   │       ├── easyocr.py
│   │       ├── trocr.py
│   │       └── ensemble.py
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
├── models/                      # Model management
│   ├── registry.json           # Model URLs (IN GIT)
│   ├── download_models.py      # Download script (IN GIT)
│   └── [downloaded models]     # (GITIGNORED)
│
├── data/                        # Datasets (ALL GITIGNORED)
│   ├── raw/                    # Unlabeled images
│   ├── annotated/              # Vision-model annotated
│   ├── reviewed/               # Manually corrected
│   └── datasets/               # Final HF datasets
│
├── utils/                       # Shared utilities
└── tests/                       # Unit tests
```

## OCR Engine Architecture

### Base Class Pattern

All engines inherit from `OCREngine` abstract base class:

```python
@dataclass
class OCRResult:
    text: str
    confidence: Optional[float] = None  # Only if engine provides it
    bounding_boxes: Optional[List[dict]] = None
    word_confidences: Optional[List[float]] = None
    engine_name: str = ""
    processing_time: float = 0.0

class OCREngine(ABC):
    @abstractmethod
    def load_model(self): pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult: pass

    @property
    @abstractmethod
    def name(self) -> str: pass
```

### Factory Pattern

```python
class OCREngineFactory:
    _engines = {
        "tesseract": TesseractEngine,
        "kraken": KrakenEngine,
        "easyocr": EasyOCREngine,
        "trocr": TrOCREngine,
        "ensemble": EnsembleEngine,
    }

    @classmethod
    def create(cls, engine_name: str, **kwargs) -> OCREngine:
        return cls._engines[engine_name](**kwargs)
```

### Engine Implementations

| Engine | Type | Speed | Accuracy | GPU | Best For |
|--------|------|-------|----------|-----|----------|
| **Tesseract** | Traditional | Fast | Good | No | Printed text, baseline |
| **Kraken** | LSTM | Medium | Very Good | Optional | Manuscripts, historical docs |
| **EasyOCR** | Deep Learning | Medium | Very Good | Yes | General multilingual |
| **trOCR** | Transformer | Slow | Excellent | Yes | Handwritten, fine-tuning |
| **Ensemble** | Voting | Slowest | Best | Depends | Maximum accuracy |

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
1. Upload Image
   ↓
2. Select Engine (UI dropdown)
   ↓
3. OCREngine.recognize(image)
   ↓  - Load model (if needed)
   ↓  - Preprocess image
   ↓  - Run inference
   ↓
4. Return OCRResult
   ↓
5. Display & Edit
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

- `config.py`: Application settings
- `training/configs/*.yaml`: Training configurations
- `models/registry.json`: Model URLs and metadata

## Testing Strategy

```
tests/
├── test_ocr/
│   ├── test_base.py          # Base class tests
│   ├── test_factory.py       # Factory pattern tests
│   ├── test_tesseract.py     # Engine-specific tests
│   └── ...
├── test_training/
│   ├── test_vision_annotate.py
│   └── test_dataset_builder.py
└── fixtures/
    ├── test_images/
    └── test_labels/
```

## Performance Considerations

### Model Loading
- Lazy loading: Models loaded on first use
- Caching: Models stay loaded in memory during session
- GPU management: Automatic device selection

### Image Processing
- Batch processing support (for training)
- Preprocessing pipeline (resize, denoise, binarize)
- Memory-efficient streaming for large datasets

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

- [ ] Real-time OCR with webcam
- [ ] Mobile app integration
- [ ] Collaborative annotation
- [ ] Active learning loop
- [ ] Multi-language support (Latin, Coptic)
- [ ] Integration with digital libraries (Perseus, TLG)
