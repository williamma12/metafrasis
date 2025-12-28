# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metafrasis is an Ancient Greek OCR, transliteration, and translation application with a modular plugin architecture. The project uses Streamlit for the UI and supports multiple OCR engines (Tesseract, Kraken, EasyOCR, trOCR, Ensemble) with fine-tuning capabilities.

**Key Design Principle**: All models are hosted externally (HuggingFace, S3, direct URLs). The repository contains NO model files - only `models/registry.json` which maps model names to their external URLs. Models are downloaded on first use and cached locally.

## Development Commands

### Package Management
This project uses `uv` for dependency management:
```bash
# Install all dependencies
uv sync

# Install with optional dependencies
uv sync --extra dev
uv sync --extra ocr-full
uv sync --extra training

# Run commands in the virtual environment
uv run <command>
```

### Running the Application
```bash
# Start the Streamlit UI
uv run streamlit run app.py
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_ocr/test_factory.py

# Run with coverage
uv run pytest --cov=services --cov=utils

# Run single test function
uv run pytest tests/test_ocr/test_base.py::test_ocr_result
```

### Code Quality
```bash
# Run linter (ruff)
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Architecture

### Plugin Architecture for OCR Engines

All OCR engines implement the `OCREngine` abstract base class defined in `services/ocr/base.py`:

```python
class OCREngine(ABC):
    @abstractmethod
    def load_model(self): pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult: pass

    @property
    @abstractmethod
    def name(self) -> str: pass
```

New engines are registered in `OCREngineFactory` (`services/ocr/factory.py`). To add a new engine:
1. Create a new class in `services/ocr/engines/` that inherits from `OCREngine`
2. Implement the required abstract methods
3. Register it in `OCREngineFactory._engines` dictionary

### Model Management System

**Critical**: Models are NEVER committed to git. The workflow is:

1. `models/registry.json` (committed) contains URLs and metadata
2. Engine loads model using `config.get_model_path(engine_name, variant)`
3. If not cached locally, model is downloaded from URL in registry
4. Downloaded weights are cached in `data/model_weights/` (gitignored)

When adding a new model:
1. Host it externally (HuggingFace Hub recommended)
2. Add entry to `models/registry.json` with URL and type
3. Engine will download on first use to `data/model_weights/<model_type>/`

### Data Flow

**Runtime Pipeline**:
```
Image Upload → OCR Engine Selection → OCR → Transliteration (optional) → Translation (optional)
```

**Training Pipeline** (when implemented):
```
Raw Images → Vision Model Annotation (GPT-4V/Claude) → Manual Review → Dataset → Fine-tuning → Upload to HuggingFace → Update registry.json
```

### State Management

The Streamlit app uses a dataclass-based session state pattern (`app.py`):
- `AppState` dataclass defines all session state fields
- State is initialized once and persists across Streamlit reruns
- Access via `st.session_state.state`

## Directory Structure

```
metafrasis/
├── app.py                       # Streamlit UI - main entry point
├── config.py                    # Configuration (paths, model registry access)
├── services/
│   └── ocr/                     # OCR service (planned)
│       ├── base.py             # Abstract OCREngine class, OCRResult dataclass
│       ├── factory.py          # OCREngineFactory - engine registry
│       ├── preprocessing.py    # Shared image preprocessing
│       └── engines/            # Concrete engine implementations
├── utils/                       # Shared utilities
├── models/                      # PyTorch model definitions (code only)
│   ├── __init__.py             # Package exports
│   ├── registry.json           # Model URLs (committed)
│   ├── download_models.py      # Download script (committed)
│   ├── layers.py               # Shared building blocks
│   ├── backbones/              # Feature extractors (VGG, ResNet, MobileNet)
│   ├── necks/                  # Feature aggregation (FPN, BiLSTM)
│   ├── heads/                  # Task outputs (CTC, DB head)
│   └── composites/             # Full models (CRAFT, DBNet, CRNN, PPOCRModel)
├── data/                        # All gitignored
│   ├── model_weights/          # Downloaded model weights
│   ├── raw/                    # Unlabeled images
│   ├── annotated/              # Vision-model annotated
│   ├── reviewed/               # Manually corrected
│   └── datasets/               # HuggingFace datasets
└── training/                    # Training infrastructure (planned)
    ├── finetune/               # Fine-tuning scripts (trOCR LoRA, etc.)
    ├── benchmarks/             # Engine comparison tools
    └── data/                   # Dataset creation (vision annotation)
```

## Key Configuration

### Python Version
Requires Python 3.11+

### Ruff Settings
- Line length: 100 characters
- Target: Python 3.11
- Selected rules: E, F, I, N, W

### Test Configuration
- Test files: `test_*.py`
- Test functions: `test_*`
- Test directory: `tests/`

## Training & Fine-tuning

### Vision Model Annotation Philosophy
Instead of rendering synthetic fonts, use vision models (GPT-4V, Claude, LLaVA) to annotate real manuscript images. This provides:
- Real-world image distribution with natural degradation
- Faster than manual annotation
- Can bootstrap from unlabeled data

### Fine-tuning Methods by Engine
- **trOCR**: LoRA adapters (~10 MB) - preferred method
- **EasyOCR**: Full fine-tuning
- **Kraken**: Custom model training

### After Fine-tuning
1. Upload adapter/model to HuggingFace or S3
2. Update `models/registry.json` with new URL
3. No model files committed to git

## Important Patterns

### Adding a New OCR Engine
1. Create `services/ocr/engines/your_engine.py`
2. Inherit from `OCREngine` base class
3. Implement `load_model()`, `recognize()`, and `name` property
4. Register in `OCREngineFactory._engines`
5. Add model URL to `models/registry.json` if needed

### Model Loading Pattern
All engines should check for cached models before downloading:
```python
model_path = config.get_model_path(self.name, variant)
if not Path(model_path).exists():
    # Download from URL in registry
```

### Streamlit Session State Pattern
Use the centralized `AppState` dataclass:
```python
state = st.session_state.state
state.ocr_text = "result"  # Persists across reruns
```

## Current Status

The project is under active development:
- ✅ Project structure and configuration
- ✅ Streamlit UI pipeline flow
- ⏳ OCR engines (planned - see docs/OCR_SERVICE.md)
- ⏳ Training infrastructure (planned - see docs/TRAINING.md)
- ⏳ Transliteration & translation services (planned)

## Additional Documentation

- **README.md**: Project overview and quick start
- **docs/ARCHITECTURE.md**: Complete architecture details, model management, testing strategy
- **docs/OCR_SERVICE.md**: OCR service implementation plan with code samples
- **docs/TRAINING.md**: Training infrastructure plan, vision annotation, benchmarking
- **docs/QUICKSTART.md**: Installation and getting started guide
