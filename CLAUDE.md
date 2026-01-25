# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metafrasis is an Ancient Greek OCR and annotation platform with a modular plugin architecture. The project uses Streamlit for the UI and supports multiple OCR engines (Tesseract, Kraken, PP-OCR, CRAFT+recognizers) with comprehensive fine-tuning infrastructure.

**Key Design Principles**:
- All models are hosted externally (HuggingFace, S3, direct URLs). The repository contains NO model files - only `ml/models/registry.json` which maps model names to their external URLs. Models are downloaded on first use and cached locally.
- Clean separation between application code (`app/`), machine learning code (`ml/`), and native extensions (`native/`).

## Project Structure

The codebase is organized into three main directories:

### `app/` - Application Code
Contains all user-facing application code:
- **`backend/`**: Streamlit UI (pages, components, state management)
- **`services/`**: Business logic (OCR, annotation services)
- **`frontend/`**: React/TypeScript components for rich UI
- **`config.py`**: Application configuration (UI settings, paths)

### `ml/` - Machine Learning Code
Contains all ML-related code:
- **`models/`**: PyTorch model definitions, registry, and download scripts
- **`training/`**: Fine-tuning, evaluation, and export infrastructure
- **`config.py`**: ML configuration (device detection, model registry access)

### `native/` - Native Extensions
Contains platform-specific optimizations:
- **`mps/ctc/`**: Metal Performance Shaders CTC loss (macOS GPU acceleration)
- **`mps/matmul/`**: Future matrix operations
- **`mps/kernels/`**: Future custom kernels

This structure enables:
- Clear separation of concerns
- Independent development workflows
- Future microservices or package distribution
- Easy addition of new native bindings (CUDA, ROCm, etc.)

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

**Quick Start:**
```bash
# Show all available test commands
make help

# Run all tests (Python + Frontend)
make test-all

# Run specific test suites
make test-ml              # ML model tests (148 tests)
make test-backend         # Backend/service tests (52 tests)
make test-frontend        # All frontend tests (46 tests)
make test-ocr-viewer      # Just OCR Viewer (16 tests)
make test-annotation-canvas  # Just Annotation Canvas (30 tests)

# Quick tests (skip slow tests)
make test-quick

# Coverage reports
make test-coverage-ml
make test-coverage-backend
```

**Manual Testing (pytest):**
```bash
# Run all Python tests
uv run pytest

# Run specific test directory
uv run pytest tests/ml/models/
uv run pytest tests/app/

# Run specific test file
uv run pytest tests/ml/models/test_layers.py

# Run with coverage
uv run pytest --cov=app --cov=ml --cov=native

# Run single test function
uv run pytest tests/ml/models/test_layers.py::TestConvBNLayer::test_output_shape_with_stride_1

# Skip slow tests
uv run pytest -m "not slow"

# Run only MPS-specific tests
uv run pytest -m requires_mps
```

**Manual Testing (Frontend):**
```bash
# OCR Viewer
cd app/frontend/ocr_viewer
npm test              # Interactive mode
npm test -- --run     # Run once and exit
npm run test:coverage # With coverage

# Annotation Canvas
cd app/frontend/annotation_canvas
npm test -- --run
```

**Test Suite Summary:**
- **Total Tests**: 246 (100% passing)
  - ML Models: 148 tests
  - Backend: 52 tests
  - Frontend: 46 tests (16 OCR Viewer + 30 Annotation Canvas)

See **[docs/TESTING.md](docs/TESTING.md)** for complete testing documentation.

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

All OCR engines implement the `OCREngine` abstract base class defined in `app/services/ocr/base.py`:

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

New engines are registered in `OCREngineFactory` (`app/services/ocr/factory.py`). To add a new engine:
1. Create a new class in `app/services/ocr/engines/` that inherits from `OCREngine`
2. Implement the required abstract methods
3. Register it in `OCREngineFactory._engines` dictionary

### Model Management System

**Critical**: Models are NEVER committed to git. The workflow is:

1. `ml/models/registry.json` (committed) contains URLs and metadata
2. Engine loads model using ML config's model registry functions
3. If not cached locally, model is downloaded from URL in registry
4. Downloaded weights are cached in `data/model_weights/` (gitignored)

When adding a new model:
1. Host it externally (HuggingFace Hub recommended)
2. Add entry to `ml/models/registry.json` with URL and type
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
├── Makefile                     # Test automation (17 targets)
├── app/                         # Application code
│   ├── config.py               # App configuration (UI, OCR settings)
│   ├── main.py                 # Streamlit main entry with navigation
│   ├── backend/                # Streamlit backend
│   │   ├── pages/             # Page implementations (OCR, Annotate)
│   │   ├── components/        # Reusable UI components
│   │   └── state.py           # Session state management
│   ├── services/               # Business logic
│   │   ├── ocr/               # OCR service
│   │   │   ├── base.py       # Abstract OCREngine class, OCRResult
│   │   │   ├── factory.py    # OCREngineFactory - engine registry
│   │   │   ├── detectors/    # Text detection implementations
│   │   │   ├── recognizers/  # Text recognition implementations
│   │   │   └── engines/      # Complete OCR pipelines
│   │   └── annotation/        # Annotation service
│   └── frontend/               # React/TypeScript components
│       ├── ocr_viewer/        # OCR result visualization
│       │   ├── src/           # React components
│       │   └── tests/         # Frontend tests (16 tests)
│       └── annotation_canvas/ # Annotation drawing interface
│           ├── src/           # React components
│           └── tests/         # Frontend tests (30 tests)
├── ml/                          # Machine learning code
│   ├── config.py               # ML configuration (models, device, registry)
│   ├── models/                 # PyTorch model definitions (code only)
│   │   ├── registry.json      # Model URLs (committed)
│   │   ├── download_models.py # Download script
│   │   ├── layers.py          # Shared building blocks
│   │   ├── backbones/         # Feature extractors (VGG, ResNet, MobileNet)
│   │   ├── necks/             # Feature aggregation (FPN, BiLSTM)
│   │   ├── heads/             # Task outputs (CTC, DB head)
│   │   └── composites/        # Full models (CRAFT, DBNet, CRNN, PPOCRModel)
│   └── training/               # Training infrastructure
│       ├── finetune/          # Fine-tuning scripts
│       ├── evaluate/          # Evaluation metrics
│       └── export/            # Model export utilities
├── native/                      # Native extensions
│   └── mps/                    # Metal Performance Shaders bindings
│       ├── ctc/               # CTC loss (Metal-accelerated)
│       │   ├── ctc_loss.py   # Python interface
│       │   ├── csrc/         # Metal/C++ implementation
│       │   └── tests/        # Native extension tests
│       ├── matmul/            # Matrix operations (future)
│       └── kernels/           # Custom kernels (future)
├── tests/                       # Test suite
│   ├── ml/models/              # ML model tests (148 tests)
│   │   ├── test_layers.py     # ConvBNLayer, SEModule, BasicBlock, CTCDecoder
│   │   ├── test_backbones.py  # VGG16BN, ResNet, MobileNetV3, CRNNCNN
│   │   ├── test_necks.py      # FPN, BiLSTM, SequenceEncoder
│   │   ├── test_heads.py      # CTCHead, DBHead
│   │   ├── test_composites.py # CRAFT, DBNet, CRNN, PPOCRModel
│   │   ├── test_registry.py   # Model registry validation
│   │   └── conftest.py        # Shared fixtures
│   └── app/                    # Backend tests (52 tests)
│       ├── services/ocr/
│       └── services/annotation/
├── installer/                   # macOS application bundler
├── data/                        # All gitignored
│   ├── model_weights/          # Downloaded model weights
│   ├── annotations/            # Annotation data
│   └── training/               # Training datasets
└── docs/                        # Documentation
```

## Key Configuration

### Python Version
Requires Python 3.11+

### Ruff Settings
- Line length: 100 characters
- Target: Python 3.11
- Selected rules: E, F, I, N, W

### Test Configuration

**Python Tests (pytest):**
- Test files: `test_*.py`
- Test functions: `test_*`
- Test directories: `tests/ml/models/`, `tests/app/`, `native/`
- Test markers:
  - `slow`: Long-running tests (skip with `-m "not slow"`)
  - `requires_craft`: Requires CRAFT model weights
  - `requires_crnn`: Requires CRNN model weights
  - `requires_tesseract`: Requires Tesseract installation
  - `requires_mps`: Requires Apple Metal Performance Shaders
  - `requires_native`: Requires native extension compilation

**Frontend Tests (vitest):**
- Test files: `*.test.tsx`, `*.test.ts`
- Test directories: `app/frontend/ocr_viewer/tests/`, `app/frontend/annotation_canvas/tests/`
- Frameworks: Vitest, React Testing Library
- Mock setup: `tests/setup.ts` in each component directory

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
1. Create `app/services/ocr/engines/your_engine.py`
2. Inherit from `OCREngine` base class
3. Implement `load_model()`, `recognize()`, and `name` property
4. Register in `OCREngineFactory._engines`
5. Add model URL to `ml/models/registry.json` if needed

### Model Loading Pattern
All engines should use the ML config's model registry:
```python
from ml.config import get_model_info, MODEL_WEIGHTS_DIR

model_info = get_model_info(self.name, variant)
model_path = MODEL_WEIGHTS_DIR / model_info['type'] / f"{self.name}_{variant}.pth"
if not model_path.exists():
    # Download from URL in registry
```

### Streamlit Session State Pattern
Use the centralized `AppState` dataclass:
```python
state = st.session_state.state
state.ocr_text = "result"  # Persists across reruns
```

## Current Status

**Production-Ready Core Features**

✅ **Implemented**:
- Project structure with clean separation (app/, ml/, native/)
- Streamlit UI with OCR and Annotation pages
- ML model implementations (CRAFT, DBNet, CRNN, PPOCRModel)
- OCR engines: Tesseract, Kraken, PP-OCR (PyTorch + ONNX), CRAFT+recognizers
- Text detectors: CRAFT, DB (Differentiable Binarization), WholeImage
- Text recognizers: Kraken (Ancient Greek), PP-OCR (Greek), trOCR, CRNN
- Annotation service (models, storage, export, React canvas)
- Training infrastructure (~3500 lines):
  - Fine-tuning scripts for CRNN, PP-OCR, trOCR (LoRA), CRAFT, DBNet
  - Evaluation metrics (CER, WER, Precision, Recall, F1)
  - ONNX export and HuggingFace Hub upload
- Frontend components (OCRViewer, AnnotationCanvas with full interactivity)
- Comprehensive test suite (246 tests, 100% passing)
- Test automation infrastructure (Makefile)
- Model registry and external hosting system
- GPU acceleration (CUDA, MPS, CPU)
- Native MPS extensions (CTC loss for macOS)

See **[ROADMAP.md](ROADMAP.md)** for planned features and development timeline.

## Additional Documentation

- **README.md**: Project overview and quick start
- **ROADMAP.md**: Planned features and development timeline
- **docs/QUICKSTART.md**: Complete installation and setup guide
- **docs/ARCHITECTURE.md**: System architecture, model management, testing strategy
- **docs/OCR_SERVICE.md**: OCR engine implementation details and usage
- **docs/ANNOTATION.md**: Annotation tool usage and data formats
- **docs/TESTING.md**: Comprehensive testing guide (organization, commands, best practices)
