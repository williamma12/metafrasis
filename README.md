# Metafrasis

**Ancient Greek OCR, Transliteration, and Translation**

A modular Python application for processing Ancient Greek texts with support for multiple OCR engines, fine-tuning, and vision-model-based dataset creation.

## Features

- 🔍 **Multiple OCR Engines**: Tesseract, Kraken, EasyOCR, trOCR, and Ensemble
- 🎯 **Fine-tuning Support**: Train custom models with LoRA on your own data
- 🤖 **Vision Model Annotation**: Use GPT-4V/Claude to annotate real manuscript images
- 🔄 **Transliteration**: Greek ↔ Latin script conversion
- 🌍 **Translation**: Ancient Greek to modern languages
- 📦 **External Model Hosting**: No models in git, all hosted externally

## Quick Start

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

See **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for detailed setup instructions.

## Architecture

Metafrasis uses a plugin architecture where all OCR engines implement a common interface. Models are hosted externally (HuggingFace, S3) and downloaded on first use.

```
Streamlit UI → OCR Service (Factory) → OCR Engines (Plugins) → External Models
```

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for complete architecture details.

## Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Installation and getting started
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design
- **OCR Service Plan** - `.claude/plans/ocr-service-plan.md`
- **Training Plan** - `.claude/plans/training-plan.md`

## Project Status

🚧 **Under Active Development**

- ✅ Project structure and configuration
- ✅ Streamlit UI pipeline
- ⏳ OCR engines (in progress)
- ⏳ Training infrastructure (planned)
- ⏳ Transliteration & translation services (planned)

## Development

Built with:
- Python 3.11+
- Streamlit for UI
- uv for package management
- HuggingFace for model hosting

## License

TBD
