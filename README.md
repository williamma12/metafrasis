# Metafrasis

**Ancient Greek OCR and Annotation Platform**

A modular Python application for processing Ancient Greek texts with support for multiple OCR engines, interactive annotation, and model fine-tuning capabilities.

---

## Features

### OCR Processing
- 🔍 **Multiple OCR Engines**: Tesseract, Kraken (Ancient Greek), PP-OCR (Greek), CRAFT+CRNN, CRAFT+trOCR
- 📄 **PDF Support**: Automatic page conversion and batch processing
- ⚡ **Dual Pipelines**: Streaming (sequential) or Batch (parallel) processing
- 🖥️ **GPU Acceleration**: Automatic device detection (CUDA, MPS, CPU)
- 📊 **Rich Results**: Word-level bounding boxes and confidence statistics
- 🎨 **Interactive Viewer**: Custom React component with click-to-toggle regions

### Dataset Annotation
- ✏️ **Interactive Canvas**: Rectangle and polygon drawing modes
- 🤖 **Auto-Detection**: CRAFT-based text region detection
- 💾 **Dataset Management**: Create, load, and export annotation datasets
- 📦 **Export Formats**: ZIP archives with JSON metadata and images

### Model Training
- 🎯 **Fine-Tuning**: CRNN, PP-OCR, trOCR (LoRA), CRAFT, DBNet trainers
- 📈 **Evaluation Metrics**: CER, WER, Precision, Recall, F1-score
- 🔄 **Model Export**: ONNX conversion and HuggingFace Hub upload
- 🧪 **Comprehensive Testing**: 246 tests covering ML models, backend, and frontend

### Infrastructure
- 🗂️ **Model Registry**: External model hosting (HuggingFace, Google Drive, direct URLs)
- 🔌 **Plugin Architecture**: Easy addition of new OCR engines
- ⚡ **Multi-Platform GPU**: CUDA (NVIDIA), MPS (Apple Silicon), CPU fallback
- 🧰 **Test Automation**: Makefile-based test infrastructure with coverage reporting
- 📚 **Complete Documentation**: Setup guides, API docs, and development guides

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/metafrasis.git
cd metafrasis

# Automated setup (macOS/Linux)
./scripts/setup.sh

# Or install manually
brew install tesseract tesseract-lang poppler  # macOS
# sudo apt install tesseract-ocr tesseract-ocr-grc poppler-utils  # Linux

uv sync
```

### Run the Application

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501`

See **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for detailed setup instructions and system requirements.

---

## Usage

### OCR Processing

1. Navigate to the **OCR** page
2. Select your OCR engine:
   - **Tesseract**: Fast, CPU-only (Ancient Greek support)
   - **Kraken**: Excellent for Ancient Greek manuscripts
   - **PP-OCR**: Greek language support with high accuracy
   - **CRAFT + Recognizer**: Modular pipeline with custom detectors
3. Upload images or PDFs
4. Click "Run OCR" and view results

### Dataset Annotation

1. Navigate to the **Annotate** page
2. Create a new dataset or load existing
3. Upload images to annotate
4. Use drawing tools:
   - **Rectangle**: Click and drag
   - **Polygon**: Click vertices, double-click to finish
   - **Auto-detect**: CRAFT-based region detection
5. Label each region with text
6. Export dataset as ZIP for training

### Model Fine-Tuning

Fine-tune models using Python scripts:

```bash
# Fine-tune CRNN recognizer
python -m ml.training.finetune.recognizers.crnn ml/training/configs/crnn.yaml

# Fine-tune trOCR with LoRA adapters
python -m ml.training.finetune.recognizers.trocr ml/training/configs/trocr_lora.yaml

# Fine-tune CRAFT detector
python -m ml.training.finetune.detectors.craft ml/training/configs/craft.yaml
```

---

## Testing

Run the comprehensive test suite:

```bash
# All tests (Python + Frontend)
make test-all

# Specific test suites
make test-ml              # ML models (148 tests)
make test-backend         # Backend (52 tests)
make test-frontend        # Frontend (46 tests)

# Coverage reports
make test-coverage-ml
```

See **[docs/TESTING.md](docs/TESTING.md)** for testing guide and best practices.

---

## Architecture

Metafrasis uses a clean, modular architecture with platform-specific optimizations:

```
app/                    # Streamlit UI and services
├── backend/pages/      # OCR and Annotation pages
├── services/ocr/       # OCR engines and pipelines
├── services/annotation/# Annotation backend
└── frontend/           # React components (OCR Viewer, Annotation Canvas)

ml/                     # Machine learning code
├── models/             # PyTorch model definitions
│   ├── composites/     # CRAFT, DBNet, CRNN, PPOCRModel
│   ├── backbones/      # VGG16BN, ResNet, MobileNetV3
│   ├── necks/          # FPN, BiLSTM
│   └── heads/          # CTCHead, DBHead
└── training/           # Fine-tuning infrastructure

native/                 # Platform-specific optimizations
└── mps/                # Metal Performance Shaders (macOS GPU acceleration)
    ├── ctc/            # CTC loss implementation
    ├── matmul/         # Matrix operations (planned)
    └── kernels/        # Custom kernels (planned)

tests/                  # Comprehensive test suite (246 tests)
├── ml/models/          # ML model tests (148 tests)
├── app/                # Backend tests (52 tests)
├── frontend/           # Component tests (46 tests)
└── native/             # Native extension tests
```

**GPU Acceleration Support**:
- ✅ **CUDA**: NVIDIA GPU support via PyTorch
- ✅ **MPS**: Apple Metal Performance Shaders for M1/M2/M3 Macs
- ✅ **CPU**: Automatic fallback for systems without GPU

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for complete architecture details.

---

## Documentation

### Getting Started
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Complete installation and setup guide
- **[TESTING.md](docs/TESTING.md)** - Testing guide and best practices

### Architecture & Design
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design decisions
- **[OCR_SERVICE.md](docs/OCR_SERVICE.md)** - OCR engine implementation details
- **[ANNOTATION.md](docs/ANNOTATION.md)** - Annotation tool usage and data formats

### Development
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code
- **[ROADMAP.md](ROADMAP.md)** - Planned features and future development

---

## Project Status

**Current Phase**: Production-Ready Core Features

✅ **Implemented**:
- OCR processing with multiple engines (Tesseract, Kraken, PP-OCR, CRAFT+recognizers)
- Interactive annotation tool with auto-detection and export
- ML model implementations (CRAFT, DBNet, CRNN, PPOCRModel)
- Training infrastructure (fine-tuning scripts for all models)
- Comprehensive test suite (246 tests, 100% passing)
- Frontend components (OCR Viewer, Annotation Canvas)
- Model registry and external hosting system

See **[ROADMAP.md](ROADMAP.md)** for planned features and development timeline.

---

## Tech Stack

- **Python 3.11+**: Core application language
- **Streamlit**: Web UI framework
- **React + TypeScript**: Interactive frontend components
- **PyTorch**: ML model implementations
- **Vitest**: Frontend testing framework
- **uv**: Package management
- **HuggingFace**: Model hosting and distribution

---

## Contributing

Contributions welcome! See **[ROADMAP.md](ROADMAP.md)** for planned features and high-impact starter tasks.

**Quick setup for development**:

```bash
./scripts/setup.sh --dev  # Install with development dependencies
make test-all             # Run full test suite
```

---

## License

TBD

---

## Acknowledgments

Built with support from:
- **HuggingFace Transformers**: Pre-trained models and training infrastructure
- **PaddleOCR**: Greek OCR models and PP-OCR architecture
- **Kraken OCR**: Ancient Greek manuscript recognition
- **CRAFT**: Character-level text detection
- **Streamlit**: Web UI framework

---

**Questions or feedback?** Open an issue or start a discussion.
