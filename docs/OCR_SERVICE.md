# Plan: Modular OCR Service Implementation

## Goal
Implement a modular OCR service with plugin architecture supporting Tesseract, Kraken, EasyOCR, trOCR, and ensemble voting.

## Architecture

### Directory Structure
```
services/ocr/
├── __init__.py
├── base.py                  # Abstract base class
├── factory.py               # Engine registry/factory
├── preprocessing.py         # Shared image preprocessing
├── postprocessing.py        # Shared text postprocessing
└── engines/
    ├── __init__.py
    ├── tesseract.py
    ├── kraken.py
    ├── easyocr.py
    ├── trocr.py
    └── ensemble.py
```

## Core Components

### 1. Base Class (`services/ocr/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image

@dataclass
class OCRResult:
    """Standardized result format"""
    text: str
    confidence: float
    bounding_boxes: Optional[List[dict]] = None
    word_confidences: Optional[List[float]] = None
    engine_name: str = ""
    processing_time: float = 0.0

class OCREngine(ABC):
    """Base class for all OCR engines"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_loaded = False

    @abstractmethod
    def load_model(self):
        """Load the OCR model"""
        pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult:
        """Run OCR on image"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        pass

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Optional preprocessing hook"""
        return image
```

### 2. Factory Pattern (`services/ocr/factory.py`)

```python
from typing import Dict, Type
from .base import OCREngine
from .engines import (
    TesseractEngine,
    KrakenEngine,
    EasyOCREngine,
    TrOCREngine,
    EnsembleEngine
)

class OCREngineFactory:
    _engines: Dict[str, Type[OCREngine]] = {
        "tesseract": TesseractEngine,
        "kraken": KrakenEngine,
        "easyocr": EasyOCREngine,
        "trocr": TrOCREngine,
        "ensemble": EnsembleEngine,
    }

    @classmethod
    def create(cls, engine_name: str, **kwargs) -> OCREngine:
        if engine_name not in cls._engines:
            raise ValueError(f"Unknown engine: {engine_name}")
        return cls._engines[engine_name](**kwargs)

    @classmethod
    def available_engines(cls) -> List[str]:
        return list(cls._engines.keys())
```

### 3. Engine Implementations

**Tesseract** (`services/ocr/engines/tesseract.py`):
- Use pytesseract library
- Support custom trained models
- Fast, good baseline

**Kraken** (`services/ocr/engines/kraken.py`):
- Use kraken library
- Specialized for historical documents
- Good for manuscripts

**EasyOCR** (`services/ocr/engines/easyocr.py`):
- Use easyocr library
- Deep learning based
- GPU support optional

**trOCR** (`services/ocr/engines/trocr.py`):
- Use HuggingFace transformers
- State-of-the-art for handwritten text
- Support LoRA adapters

**Ensemble** (`services/ocr/engines/ensemble.py`):
- Runs multiple engines
- Voting strategies: majority, confidence-weighted
- Configurable members

### 4. Streamlit Integration

Update `app.py`:
```python
from services.ocr.factory import OCREngineFactory
import config

# In OCR tab:
engine_name = st.selectbox(
    "Select OCR Engine",
    OCREngineFactory.available_engines()
)

if st.button("🔍 Run OCR"):
    engine = OCREngineFactory.create(engine_name)
    result = engine.recognize(uploaded_file)
    state.ocr_text = result.text
    st.metric("Confidence", f"{result.confidence:.1%}")
```

## Implementation Plan

### Step 1: Core Infrastructure
- Create `services/ocr/` directory
- Implement `base.py` (OCRResult, OCREngine)
- Implement `factory.py` (OCREngineFactory)
- Implement `preprocessing.py` (basic image ops)
- Create `engines/` directory with `__init__.py`

### Step 2: Tesseract Engine (Baseline)
- Implement `engines/tesseract.py`
- Test with sample Greek image
- Integrate with Streamlit app
- Verify end-to-end flow

### Step 3: Additional Engines
- Implement `engines/trocr.py` (most powerful)
- Implement `engines/easyocr.py`
- Implement `engines/kraken.py`
- Test each engine independently

### Step 4: Ensemble Engine
- Implement `engines/ensemble.py`
- Add voting strategies
- Test ensemble vs individual engines

### Step 5: Configuration & Polish
- Update `config.py` with engine settings
- Add model download utilities
- Update `.gitignore` for models
- Add error handling and logging

## Configuration (`config.py`)

```python
import json
from pathlib import Path

# OCR Settings
OCR_ENGINE_DEFAULT = "tesseract"
OCR_ENGINES_ENABLED = ["tesseract", "trocr", "easyocr", "kraken", "ensemble"]

# Ensemble settings
OCR_ENSEMBLE_MEMBERS = ["tesseract", "trocr"]
OCR_ENSEMBLE_STRATEGY = "confidence_weighted"

# Model registry - all models externally hosted
def get_model_registry() -> dict:
    """Load model URLs/paths from registry"""
    registry_path = MODELS_DIR / "registry.json"
    if registry_path.exists():
        return json.loads(registry_path.read_text())
    return {}

def get_model_path(engine_name: str, variant: str = "base") -> str:
    """Get model URL or local path after download"""
    registry = get_model_registry()
    engine_config = registry.get(engine_name, {})

    # Return URL for download or local cached path
    if variant in engine_config:
        url = engine_config[variant]
        # Check if already downloaded
        local_path = MODELS_DIR / engine_name / variant
        if local_path.exists():
            return str(local_path)
        return url
    return None
```

## Dependencies (`pyproject.toml`)

```toml
dependencies = [
    # ... existing ...
    "pytesseract>=0.3.10",
]

[project.optional-dependencies]
ocr-full = [
    "kraken>=5.2.0",
    "easyocr>=1.7.0",
]
```

## `.gitignore` Updates

```
# ALL models - hosted externally
models/*

# Keep only registry and download scripts
!models/registry.json
!models/download_models.py
!models/.gitkeep
```

## Testing

```
tests/test_ocr/
├── test_base.py           # Test OCRResult, base class
├── test_factory.py        # Test factory pattern
├── test_tesseract.py      # Test Tesseract engine
├── test_trocr.py          # Test trOCR engine
├── test_easyocr.py        # Test EasyOCR engine
├── test_kraken.py         # Test Kraken engine
├── test_ensemble.py       # Test ensemble voting
└── fixtures/
    └── test_greek_image.png
```

## Success Criteria

- ✅ User can select OCR engine from dropdown
- ✅ Each engine returns consistent OCRResult
- ✅ Ensemble mode combines multiple engines
- ✅ Models are gitignored but downloadable
- ✅ Easy to add new engines without modifying existing code

## Next Steps After Implementation

1. Benchmark engines on Ancient Greek test set
2. Implement training/fine-tuning infrastructure (separate plan)
3. Add confidence visualization in UI
4. Optimize preprocessing for Ancient Greek manuscripts
