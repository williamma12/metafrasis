# Metafrasis Quick Start Guide

## What is Metafrasis?

Metafrasis is an Ancient Greek OCR, transliteration, and translation application. It provides a modular architecture that supports multiple OCR engines and includes tools for fine-tuning models on your own data.

## Current Status

This project is under active development. Currently implemented:
- ✅ Basic Streamlit UI with pipeline flow
- ✅ Project structure and configuration
- ⏳ OCR engines (coming soon)
- ⏳ Training infrastructure (coming soon)

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/metafrasis.git
cd metafrasis

# Install dependencies
uv sync
```

## Running the Application

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Project Architecture

### Planned OCR Engines

| Engine | Type | Best For | Status |
|--------|------|----------|--------|
| **Tesseract** | Traditional | Printed text, baseline | Planned |
| **Kraken** | LSTM | Manuscripts, historical docs | Planned |
| **EasyOCR** | Deep Learning | General multilingual | Planned |
| **trOCR** | Transformer | Handwritten, fine-tuning | Planned |
| **Ensemble** | Voting | Maximum accuracy | Planned |

### Model Management

All models are hosted externally (HuggingFace Hub, S3, etc.):
- **No models in git** - repository stays lightweight
- `models/registry.json` contains URLs to all models
- Models downloaded on first use and cached locally

### Training Approach

**Vision Model Annotation:**
Use large vision models (GPT-4V, Claude, LLaVA) to annotate real Greek manuscript images instead of rendering synthetic fonts. This provides:
- Real-world image distribution
- Natural degradation and noise
- Faster than manual annotation

**Fine-tuning:**
- trOCR: LoRA adapters (~10 MB)
- EasyOCR: Full fine-tuning
- Kraken: Custom model training

## Directory Structure

```
metafrasis/
├── app.py                   # Streamlit UI
├── config.py                # Configuration
├── services/
│   ├── ocr/                # OCR engines (planned)
│   ├── transliterate_service.py (planned)
│   └── translate_service.py (planned)
├── training/               # Training tools (planned)
├── models/
│   ├── registry.json      # Model URLs
│   └── download_models.py # Download script
├── data/                  # Datasets (gitignored)
└── docs/                  # Documentation
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture
- **OCR Service Plan** - `.claude/plans/ocr-service-plan.md`
- **Training Plan** - `.claude/plans/training-plan.md`

## Development Roadmap

### Phase 1: OCR Infrastructure
- [ ] Implement base OCR engine class
- [ ] Add Tesseract engine
- [ ] Add trOCR engine
- [ ] Add EasyOCR engine
- [ ] Add Kraken engine
- [ ] Implement ensemble voting

### Phase 2: Training Tools
- [ ] Vision model annotation script
- [ ] Manual annotation/review tool
- [ ] Dataset builder
- [ ] trOCR fine-tuning script
- [ ] Benchmarking tools

### Phase 3: Services
- [ ] Transliteration service
- [ ] Translation service
- [ ] Lexicon integration

## Contributing

This project is in early development. Contributions welcome!

## License

TBD
