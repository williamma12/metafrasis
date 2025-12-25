# Metafrasis Quick Start Guide

## What is Metafrasis?

Metafrasis is an Ancient Greek OCR, transliteration, and translation application. It provides a modular architecture that supports multiple OCR engines with streaming and batch processing pipelines.

## Current Status

**Interactive OCR Viewer Complete**

Currently implemented:
- ✅ Tesseract OCR engine (CPU baseline)
- ✅ trOCR engine (PyTorch transformer with GPU support)
- ✅ Interactive OCR viewer (custom React component)
  - Click-to-toggle individual bounding boxes
  - Hover tooltips for hidden words
  - Image navigation (First/Prev/Next/Last)
  - Toggle between original and annotated views
- ✅ Streamlit UI with multi-file upload
- ✅ Streaming and batch processing pipelines
- ✅ PDF support with automatic page conversion
- ✅ Word-level bounding boxes and confidence statistics
- ✅ Image disk caching (temporary storage)
- ✅ Model registry and external model hosting
- ✅ Automated setup script (includes Node.js/npm)

Coming soon:
- ⏳ EasyOCR and Kraken engines
- ⏳ Training infrastructure
- ⏳ Transliteration & translation services

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
./setup.sh

# With development tools (pytest, ruff, etc.)
./setup.sh --dev

# Show help
./setup.sh --help
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
- **trOCR**: Slower, GPU/CPU, excellent for handwritten text

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
| Accuracy | Good for printed text |
| Bounding Boxes | ✅ Yes (word-level) |
| Confidence | ✅ Yes (per-word) |
| Best For | Printed books, clean scans |

### trOCR

| Feature | Details |
|---------|---------|
| Type | Transformer (PyTorch) |
| Speed | Medium-Slow (GPU: ~1-2s, CPU: ~5-10s) |
| Accuracy | Excellent for handwritten |
| Bounding Boxes | ❌ No (end-to-end model) |
| Confidence | ❌ No |
| GPU Support | ✅ Yes (automatic detection) |
| Batch Processing | ✅ Optimized |
| Best For | Handwritten manuscripts |

## Model Management

All models are hosted externally:
- **Tesseract**: Ancient Greek traineddata from tessdata_best
- **trOCR**: `microsoft/trocr-base-handwritten` from HuggingFace Hub

Models are automatically downloaded on first use and cached locally in `models/` (gitignored).

## Project Architecture

```
metafrasis/
├── app.py                      # Streamlit UI
├── config.py                   # Configuration
├── setup.sh                    # Automated setup script
├── services/
│   └── ocr/                    # OCR service
│       ├── base.py            # OCREngine, OCRResult, Word, BoundingBox
│       ├── factory.py         # OCREngineFactory
│       ├── preprocessing.py   # Image utilities + pdf_to_images()
│       └── engines/
│           ├── tesseract.py   # Tesseract engine
│           └── trocr.py       # trOCR engine
├── models/
│   ├── registry.json          # Model URLs (committed)
│   └── [cached models]        # Downloaded models (gitignored)
└── docs/                      # Documentation
```

## Development

### Running Tests

The project includes comprehensive tests for the OCR service:

```bash
# Install dev dependencies
./setup.sh --dev

# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run only OCR service tests
uv run pytest tests/test_ocr/ -v

# Run specific test file
uv run pytest tests/test_ocr/test_base.py -v
uv run pytest tests/test_ocr/test_factory.py -v
uv run pytest tests/test_ocr/test_tesseract.py -v
uv run pytest tests/test_ocr/test_cache.py -v
uv run pytest tests/test_ocr/test_preprocessing.py -v

# Run with coverage report
uv run pytest --cov=services --cov=utils

# Run tests and skip slow tests
uv run pytest -m "not slow"
```

**Test Coverage:**
- `test_base.py` - Base classes (BoundingBox, Word, OCRResult, confidence statistics)
- `test_factory.py` - OCREngineFactory (engine registration and creation)
- `test_tesseract.py` - Tesseract engine (recognition, bounding boxes, confidence)
- `test_cache.py` - ImageCache (caching, retrieval, deduplication)
- `test_preprocessing.py` - Preprocessing utilities (PDF to images)
- `conftest.py` - Shared fixtures (sample images, mock data, test helpers)

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

## Next Steps

- **Fine-tuning**: Coming soon - train custom trOCR models on your data
- **More Engines**: EasyOCR and Kraken support planned
- **Translation**: Ancient Greek to modern languages
- **Transliteration**: Greek ↔ Latin script conversion

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture
- **[OCR_SERVICE.md](OCR_SERVICE.md)** - OCR implementation details
- **[TRAINING.md](TRAINING.md)** - Training infrastructure plan
- **[CLAUDE.md](../CLAUDE.md)** - Guide for Claude Code development

## Contributing

This project is in active development. Contributions welcome!

## License

TBD
