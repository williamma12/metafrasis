# Metafrasis

**Ancient Greek OCR, Transliteration, and Translation**

A modular Python application for processing Ancient Greek texts with support for multiple OCR engines, fine-tuning, and vision-model-based dataset creation.

## Features

- 🔍 **Multiple OCR Engines**: Tesseract (implemented), trOCR (implemented), EasyOCR (planned), Kraken (planned)
- 🎨 **Interactive OCR Viewer**: Custom React component with click-to-toggle bounding boxes, hover tooltips, and image navigation
- ✏️ **Interactive Annotation Canvas**: Custom React component with rectangle/polygon drawing, region editing, and keyboard shortcuts
- 📄 **PDF Support**: Process entire PDFs with automatic page conversion
- ⚡ **Dual Pipelines**: Streaming (sequential) or Batch (parallel) processing
- 🖥️ **GPU Acceleration**: Automatic GPU detection with CPU fallback
- 📊 **Rich Results**: Word-level bounding boxes and confidence statistics (mean, std)
- 🗂️ **Image Caching**: Disk-based temporary storage for processed images
- 📦 **External Model Hosting**: No models in git, all hosted externally
- 🧪 **Comprehensive Testing**: 246 tests (100% passing) covering ML models, backend, and frontend
- 🛠️ **Test Automation**: Makefile-based test infrastructure with coverage reporting
- 🎯 **Fine-tuning Support**: LoRA adapter support (planned)
- 🔄 **Transliteration**: Greek ↔ Latin script conversion (planned)
- 🌍 **Translation**: Ancient Greek to modern languages (planned)

## Quick Start

### Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/metafrasis.git
cd metafrasis

# Run automated setup (installs Tesseract, poppler, and Python deps)
./setup.sh

# For development
./setup.sh --dev
```

### Manual Setup

```bash
# Install system dependencies
# macOS:
brew install tesseract tesseract-lang poppler

# Linux:
sudo apt install tesseract-ocr tesseract-ocr-grc poppler-utils

# Install Python dependencies
uv sync
```

### Run the Application

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501`

See **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for detailed setup and testing instructions.

## Architecture

Metafrasis uses a clean plugin architecture where OCR engines implement a simple interface:

```python
# Streaming: Process single images
result = engine.recognize(image)

# Batch: Process multiple images (optimized for PyTorch)
results = engine.recognize_batch(images)

# PDF: Convert to images, then batch process
images = pdf_to_images(pdf_path)
results = engine.recognize_batch(images)
```

**Key Design Decisions:**
- **Engines process images only** - PDF conversion is a separate utility
- **Models hosted externally** - HuggingFace Hub, downloaded on first use
- **Simple factory pattern** - No complex registration, just a dict
- **Structured results** - Word objects with text, confidence, and bounding boxes

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for complete architecture details.

## Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Installation and getting started
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design
- **[OCR_SERVICE.md](docs/OCR_SERVICE.md)** - OCR service implementation plan
- **[TRAINING.md](docs/TRAINING.md)** - Training and optimization plan

## Project Status

**Current Phase: Testing Infrastructure Complete**

- ✅ Core OCR infrastructure (base classes, factory, preprocessing)
- ✅ Tesseract engine (CPU baseline with bounding boxes)
- ✅ trOCR engine (PyTorch transformer with batch optimization)
- ✅ Interactive OCR viewer (custom React component)
  - Click-to-toggle individual bounding boxes
  - Hover tooltips for hidden words (text + confidence)
  - Image navigation (First/Prev/Next/Last)
  - Toggle between original and annotated views
- ✅ Interactive annotation canvas (custom React component)
  - Rectangle and polygon drawing modes
  - Select mode with region editing
  - Keyboard shortcuts (Delete, Escape)
  - Auto-detected vs user-created regions
- ✅ ML model implementations (CRAFT, DBNet, CRNN, PPOCRModel)
- ✅ Comprehensive test suite (246 tests, 100% passing)
  - 148 ML model tests (layers, backbones, necks, heads, composites)
  - 52 backend/service tests
  - 46 frontend tests (OCR Viewer + Annotation Canvas)
- ✅ Test automation infrastructure (Makefile, docs/TESTING.md)
- ✅ Image disk caching (temporary storage during sessions)
- ✅ Streamlit UI (multi-file upload, streaming/batch pipelines)
- ✅ PDF support (automatic page conversion)
- ✅ Automated setup script (with Node.js/npm installation)
- ✅ Model registry system
- ⏳ EasyOCR engine (planned)
- ⏳ Kraken engine (planned)
- ⏳ Training infrastructure (planned)
- ⏳ Transliteration & translation services (planned)

## Testing

The project includes a comprehensive test suite covering all components:

```bash
# Run all tests (Python + Frontend) - 246 tests
make test-all

# Run specific test suites
make test-ml              # ML models (148 tests)
make test-backend         # Backend (52 tests)
make test-frontend        # Frontend (46 tests)

# Coverage reports
make test-coverage-ml
make test-coverage-backend

# Show all test commands
make help
```

**Test Coverage:**
- ✅ **ML Models** (148 tests): Layers, backbones, necks, heads, composites
- ✅ **Backend** (52 tests): Pages, services, components
- ✅ **Frontend** (46 tests): OCR Viewer, Annotation Canvas

See **[docs/TESTING.md](docs/TESTING.md)** for complete testing documentation.

## Development

Built with:
- Python 3.11+
- Streamlit for UI
- React + TypeScript for interactive components
- uv for package management
- HuggingFace for model hosting
- PyTorch for ML models
- Vitest for frontend testing

### Frontend Component Development

The interactive OCR viewer is a custom Streamlit component built with React and TypeScript.

**Development Mode (Default):**
1. Start the Vite dev server: `cd frontend/ocr_viewer && npm run dev`
2. In another terminal, run Streamlit: `uv run streamlit run app.py`
3. Changes to React code will hot-reload automatically
4. The app connects to the Vite dev server at `http://localhost:5173`

**Production Mode:**
- Run with environment variable: `VIEWER_RELEASE=true uv run streamlit run app.py`
- Loads pre-built component from `frontend/ocr_viewer/build/`

**Building a Release:**
Use the automated release script:
```bash
./scripts/build-release.sh
```

The script will:
- Check for uncommitted changes (fails if found)
- Run tests (fails if they fail)
- Build the frontend component
- Create a timestamp-based git tag (YYYY.MM.DD format)
- Ask if you want to push the tag to remote

Then run in production mode:
```bash
VIEWER_RELEASE=true uv run streamlit run app.py
```

**Component Features:**
- Click individual bounding boxes to toggle visibility
- Hover over hidden boxes to see word text and confidence
- Navigate between images with First/Prev/Next/Last buttons
- Toggle between original and annotated views

## License

TBD
