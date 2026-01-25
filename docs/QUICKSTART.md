# Metafrasis Quick Start Guide

## What is Metafrasis?

Metafrasis is an Ancient Greek OCR, transliteration, and translation application. It provides a modular architecture that supports multiple OCR engines with streaming and batch processing pipelines.

## Current Status

**Testing Infrastructure Complete**

Currently implemented:
- ✅ **Multiple OCR Engines**:
  - Tesseract (Ancient Greek support)
  - Kraken (Ancient Greek manuscripts)
  - PP-OCR (Greek language, PyTorch and ONNX)
  - CRAFT + recognizers (modular detection + recognition)
  - DB detector (Differentiable Binarization)
- ✅ **GPU Acceleration**:
  - CUDA (NVIDIA GPUs)
  - MPS (Apple Metal Performance Shaders for M1/M2/M3)
  - CPU fallback
- ✅ **Interactive OCR Viewer** (custom React component)
  - Click-to-toggle individual bounding boxes
  - Hover tooltips for hidden words
  - Image navigation (First/Prev/Next/Last)
  - Toggle between original and annotated views
- ✅ **Interactive Annotation Canvas** (custom React component)
  - Rectangle and polygon drawing modes
  - Select mode with region editing
  - Keyboard shortcuts (Delete, Escape)
  - Auto-detected vs user-created regions
- ✅ **ML Model Implementations**:
  - CRAFT (character-level text detection)
  - DBNet (Differentiable Binarization detection)
  - CRNN (recognition with CTC loss)
  - PPOCRModel (Greek recognition)
- ✅ **Training Infrastructure**:
  - Fine-tuning scripts for CRNN, PP-OCR, trOCR (LoRA), CRAFT, DBNet
  - Evaluation metrics (CER, WER, Precision, Recall, F1)
  - ONNX export and HuggingFace Hub upload
- ✅ **Comprehensive Test Suite** (246 tests, 100% passing)
  - 148 ML model tests
  - 52 backend tests
  - 46 frontend tests
- ✅ Streamlit UI with multi-file upload
- ✅ Streaming and batch processing pipelines
- ✅ PDF support with automatic page conversion
- ✅ Word-level bounding boxes and confidence statistics
- ✅ Model registry and external model hosting
- ✅ Automated setup script

See **[ROADMAP.md](../ROADMAP.md)** for planned features.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- macOS or Linux (Windows: manual installation required)

### Automated Setup (Recommended)

The setup script handles everything automatically:

```bash
# Clone repository
git clone https://github.com/yourusername/metafrasis.git
cd metafrasis

# Basic installation (OCR engines only)
./scripts/setup.sh

# With development tools (pytest, ruff, etc.)
./scripts/setup.sh --dev

# Show help
./scripts/setup.sh --help
```

**What the script does:**
1. Detects your OS (macOS or Linux)
2. Installs Tesseract OCR with Ancient Greek language data
3. Installs poppler-utils (for PDF support)
4. Installs uv package manager (if needed)
5. Installs all Python dependencies

### Manual Setup

If you prefer manual installation or are on Windows:

#### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install tesseract tesseract-lang poppler

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# For development
uv sync --extra dev
```

#### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-grc poppler-utils

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# For development
uv sync --extra dev
```

#### Windows

1. Install [Python 3.11+](https://www.python.org/downloads/)
2. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki):
   - Download and install Tesseract-OCR
   - Download Ancient Greek language data (grc.traineddata)
3. Install [poppler](https://github.com/oschwartz10612/poppler-windows/releases)
4. Install uv: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
5. Install Python dependencies: `uv sync`

## Running the Application

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Using the Application

### 1. Select OCR Engine

In the sidebar, choose your OCR engine:
- **Tesseract**: Fast, CPU-only, good for printed text
- **Kraken (Ancient Greek)**: Excellent for Ancient Greek manuscripts
- **PP-OCR (Greek)**: Greek language support with high accuracy
- **CRAFT + Recognizer**: Modular pipeline with custom detectors
  - Detectors: CRAFT, DB (Differentiable Binarization)
  - Recognizers: Kraken, PP-OCR, trOCR, CRNN

### 2. Choose Processing Pipeline

- **Streaming (Sequential)**: Processes images one at a time (lower latency)
- **Batch (Parallel)**: Processes all images together (faster for multiple files, optimized for GPU)

### 3. Upload Files

Upload one or more:
- Images: PNG, JPG, JPEG
- PDFs: Automatically converted to images

### 4. Run OCR

Click "Run OCR" and view results:
- Extracted text for each page/image
- Word-level confidence scores
- Bounding boxes (Tesseract only)
- Processing time and device info

## OCR Engines

### Tesseract

| Feature | Details |
|---------|---------|
| Type | Traditional OCR (CPU) |
| Speed | Fast (~0.5-2s per image) |
| Ancient Greek | ✅ Yes (grc language data) |
| Bounding Boxes | ✅ Yes (word-level) |
| Confidence | ✅ Yes (per-word) |
| GPU Support | ❌ No |
| Best For | Printed books, clean scans |

### Kraken (Ancient Greek)

| Feature | Details |
|---------|---------|
| Type | Specialized OCR (PyTorch) |
| Speed | Medium (~2-4s per image) |
| Ancient Greek | ✅ Excellent (trained on manuscripts) |
| Bounding Boxes | ✅ Yes (word-level) |
| Confidence | ✅ Yes |
| GPU Support | ✅ Yes (CUDA, MPS, CPU) |
| Best For | Ancient Greek manuscripts, polytonic text |

### PP-OCR (Greek)

| Feature | Details |
|---------|---------|
| Type | PaddleOCR pipeline (PyTorch) |
| Speed | Medium-Fast (GPU: ~1-2s, CPU: ~3-5s) |
| Ancient Greek | ✅ Yes (Greek language support) |
| Bounding Boxes | ✅ Yes (via DB detector) |
| Confidence | ✅ Yes |
| GPU Support | ✅ Yes (CUDA, MPS, CPU) |
| Best For | Greek text with high accuracy requirements |

### CRAFT + Recognizers

| Feature | Details |
|---------|---------|
| Type | Modular detector + recognizer |
| Speed | Medium-Slow (depends on recognizer) |
| Ancient Greek | ✅ Yes (with Kraken or PP-OCR recognizer) |
| Bounding Boxes | ✅ Yes (character and word-level via CRAFT) |
| Confidence | ✅ Yes |
| GPU Support | ✅ Yes (CUDA, MPS, CPU) |
| Customizable | ✅ Choose detector (CRAFT, DB) and recognizer |
| Best For | Complex layouts, mixed scripts, fine-grained control |

## Model Management

All models are hosted externally and registered in `ml/models/registry.json`:
- **Tesseract**: Ancient Greek traineddata from tessdata_best
- **Kraken**: Ancient Greek models from Kraken model zoo
- **PP-OCR**: Greek models (PyTorch and ONNX versions)
- **CRAFT**: Text detector models (Google Drive, HuggingFace)
- **DB**: Differentiable Binarization detector
- **CRNN**: Recognition models from HuggingFace

Models are automatically downloaded on first use and cached locally in `data/model_weights/` (gitignored).

**GPU Acceleration**:
- ✅ **CUDA**: NVIDIA GPU support (automatic detection)
- ✅ **MPS**: Apple Metal Performance Shaders for M1/M2/M3 Macs
- ✅ **CPU**: Automatic fallback for systems without GPU

## Project Architecture

```
metafrasis/
├── app.py                           # Streamlit UI entry point
├── app/                             # Application code
│   ├── backend/pages/              # OCR and Annotation pages
│   ├── services/ocr/               # OCR service
│   │   ├── base.py                # OCREngine, OCRResult, Word, BoundingBox
│   │   ├── factory.py             # OCREngineFactory
│   │   ├── detectors/             # Text detection (CRAFT, DB, whole_image)
│   │   ├── recognizers/           # Text recognition (Kraken, PP-OCR, trOCR, CRNN)
│   │   └── engines/               # Complete OCR pipelines (Tesseract, PyTorch)
│   ├── services/annotation/        # Annotation backend
│   └── frontend/                   # React components (OCR Viewer, Annotation Canvas)
├── ml/                              # Machine learning code
│   ├── models/                     # PyTorch model definitions
│   │   ├── registry.json          # Model URLs (committed)
│   │   ├── backbones/             # VGG16BN, ResNet, MobileNetV3
│   │   ├── necks/                 # FPN, BiLSTM
│   │   ├── heads/                 # CTCHead, DBHead
│   │   └── composites/            # CRAFT, DBNet, CRNN, PPOCRModel
│   └── training/                   # Fine-tuning infrastructure
│       ├── finetune/              # Trainer classes for all models
│       ├── evaluate/              # Metrics (CER, WER, F1)
│       └── export/                # ONNX and HuggingFace export
├── native/                          # Platform-specific optimizations
│   └── mps/                        # Metal Performance Shaders (macOS GPU)
│       ├── ctc/                   # CTC loss implementation
│       ├── matmul/                # Matrix operations (planned)
│       └── kernels/               # Custom kernels (planned)
├── tests/                           # Comprehensive test suite (246 tests)
│   ├── ml/models/                 # ML model tests (148 tests)
│   ├── app/                       # Backend tests (52 tests)
│   ├── frontend/                  # Component tests (46 tests)
│   └── native/                    # Native extension tests
├── data/                            # All gitignored
│   ├── model_weights/             # Downloaded model weights
│   ├── annotations/               # Annotation data
│   └── training/                  # Training datasets
└── docs/                           # Documentation
```

## Development

### Running Tests

The project includes comprehensive tests covering ML models, backend services, and frontend components.

**Quick Start (Makefile):**
```bash
# Show all available test commands
make help

# Run all tests (Python + Frontend) - 246 tests
make test-all

# Run specific test suites
make test-ml              # ML model tests (148 tests)
make test-backend         # Backend/service tests (52 tests)
make test-frontend        # All frontend tests (46 tests)

# Individual frontend components
make test-ocr-viewer           # OCR Viewer (16 tests)
make test-annotation-canvas    # Annotation Canvas (30 tests)

# Quick tests (skip slow tests)
make test-quick

# Coverage reports
make test-coverage-ml
make test-coverage-backend
```

**Manual Testing (Python/pytest):**
```bash
# Run all Python tests
uv run pytest

# Run specific test directories
uv run pytest tests/ml/models/ -v      # ML model tests
uv run pytest tests/app/ -v            # Backend tests

# Run specific test file
uv run pytest tests/ml/models/test_layers.py -v
uv run pytest tests/app/services/test_ocr.py -v

# Run with coverage
uv run pytest tests/ml/models/ --cov=ml.models --cov-report=html

# Skip slow tests
uv run pytest -m "not slow"
```

**Manual Testing (Frontend/vitest):**
```bash
# OCR Viewer
cd app/frontend/ocr_viewer
npm test              # Interactive mode
npm test -- --run     # Run once and exit
npm run test:coverage # With coverage

# Annotation Canvas
cd app/frontend/annotation_canvas
npm test -- --run
npm run test:ui       # Interactive UI mode
```

**Test Coverage Summary:**

| Category | Tests | Coverage |
|----------|-------|----------|
| ML Models | 148 | Layers, backbones, necks, heads, composites, registry |
| Backend | 52 | Pages, services, components |
| OCR Viewer | 16 | Rendering, interactivity, layout, edge cases |
| Annotation Canvas | 30 | Rectangle/polygon/select modes, keyboard shortcuts |
| **Total** | **246** | **100% passing** |

See **[docs/TESTING.md](TESTING.md)** for complete testing documentation including:
- Test organization and structure
- Writing new tests (best practices)
- Troubleshooting guide
- CI/CD integration

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .
```

### Interactive OCR Viewer Development

The OCR viewer is a custom React component with two modes:

**Development Mode (Default):**
```bash
# Terminal 1: Start Vite dev server
cd frontend/ocr_viewer && npm run dev

# Terminal 2: Run Streamlit (connects to dev server automatically)
uv run streamlit run app.py
```

**Production Mode:**
```bash
# Run with pre-built component
VIEWER_RELEASE=true uv run streamlit run app.py
```

## Building Releases

To create a production release:

```bash
./scripts/build-release.sh
```

The release script will:
1. ✅ Check for uncommitted changes (fails if found)
2. ✅ Run all tests (fails if any fail)
3. ✅ Build the frontend component
4. ✅ Create a timestamp-based git tag (e.g., `v2024.12.24`)
5. ✅ Ask if you want to push the tag to remote

After the script completes, run in production mode:
```bash
VIEWER_RELEASE=true uv run streamlit run app.py
```

**Version Format:**
- Tags use format: `vYYYY.MM.DD` (e.g., `v2024.12.24`)
- Multiple releases on same day: `vYYYY.MM.DD.N` (e.g., `v2024.12.24.2`)

**Requirements:**
- No uncommitted changes
- All tests must pass
- Node.js, npm, and uv installed

## Building macOS Installer

Create a standalone .app bundle and DMG installer for non-technical users:

```bash
# Build for your current architecture (Intel or Apple Silicon)
./installer/build.sh

# Build with specific version
./installer/build.sh 1.0.0

# Build universal binary (Intel + Apple Silicon)
./installer/build.sh 1.0.0 --universal

# Build with code signing (requires Apple Developer ID)
export CODESIGN_IDENTITY="Developer ID Application: Your Name (XXXXXXXXXX)"
./installer/build.sh 1.0.0 --sign
```

**Output:**
- `installer/output/Metafrasis.app` - The standalone application
- `installer/output/Metafrasis-1.0.0.dmg` - DMG installer for distribution

**What gets bundled:**
- Python runtime and all dependencies
- Tesseract OCR with Ancient Greek language data
- Poppler (for PDF support)
- Pre-built React components
- PyTorch (for trOCR engine)

**Size:** ~1GB for single architecture, ~1.8GB for universal binary

**Note:** Without code signing, users will need to right-click → Open on first launch to bypass Gatekeeper.

See [installer/README.md](../installer/README.md) for detailed documentation.

## Troubleshooting

### Tesseract Not Found

Make sure Tesseract is installed and in your PATH:
```bash
tesseract --version
```

### Ancient Greek Language Data Missing

Download manually:
```bash
# macOS
brew install tesseract-lang

# Linux
sudo apt install tesseract-ocr-grc
```

### GPU Not Detected

Check PyTorch CUDA support:
```python
import torch
print(torch.cuda.is_available())  # Should be True for GPU
```

### PDF Conversion Fails

Make sure poppler is installed:
```bash
# macOS
brew install poppler

# Linux
sudo apt install poppler-utils
```

### macOS Installer Issues

**"App is damaged and can't be opened"**

The app isn't signed. Right-click the app and select "Open", then click "Open" in the dialog. Or remove the quarantine attribute:
```bash
xattr -cr /Applications/Metafrasis.app
```

**Build fails with missing dependencies**

Ensure Tesseract and Poppler are installed:
```bash
brew install tesseract poppler create-dmg
```

## Next Steps

- **Training UI**: Streamlit interface for fine-tuning models (training infrastructure fully implemented)
- **EasyOCR Integration**: Additional OCR engine with multilingual support
- **Improved Ancient Greek Models**: Fine-tuned models for better manuscript accuracy
- **Transliteration & Translation**: Post-OCR processing services

See **[ROADMAP.md](../ROADMAP.md)** for the complete feature roadmap and development timeline.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture
- **[OCR_SERVICE.md](OCR_SERVICE.md)** - OCR implementation details
- **[ANNOTATION.md](ANNOTATION.md)** - Annotation tool usage and data formats
- **[TESTING.md](TESTING.md)** - Comprehensive testing guide
- **[ROADMAP.md](../ROADMAP.md)** - Planned features and development timeline
- **[CLAUDE.md](../CLAUDE.md)** - Guide for Claude Code development

## Contributing

This project is in active development. Contributions welcome!

## License

TBD
