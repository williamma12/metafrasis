# Metafrasis

**Ancient Greek OCR, Transliteration, and Translation Application**

A self-contained Python application for processing Ancient Greek texts through optical character recognition, script transliteration, and translation. Designed to run entirely offline on MacBooks and be easily distributed as a standalone package.

## Overview

Metafrasis provides three core capabilities:
1. **OCR**: Extract Ancient Greek text from images (manuscripts, printed books, inscriptions)
2. **Transliteration**: Convert between Greek script and romanized forms (Beta Code, standard schemes)
3. **Translation**: Translate Ancient Greek to modern languages using local ML models and lexicon lookup

## Architecture

### System Design

```
┌─────────────────────────────────────────────────┐
│         Streamlit Web Interface                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │   OCR    │ │Transliter│ │Translate │        │
│  │   Tab    │ │   Tab    │ │   Tab    │        │
│  └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│              Service Layer                      │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ OCR Service  │  │Transliterate │            │
│  │  (Tesseract) │  │   Service    │            │
│  └──────────────┘  └──────────────┘            │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Translation │  │ Preprocessing│            │
│  │   Service    │  │   Pipeline   │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│          Data & Models Layer                    │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Tesseract  │  │  Local ML    │            │
│  │ Ancient Greek│  │   Models     │            │
│  │  Language    │  │  (MarianMT)  │            │
│  └──────────────┘  └──────────────┘            │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Perseus    │  │   SQLite     │            │
│  │   Lexicon    │  │  (Optional)  │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
```

### Core Components

#### 1. Streamlit Frontend
- Single-page application with tabbed interface
- File uploader for images (PNG, JPG, PDF)
- Text display and editing areas with Greek font support
- Export functionality (TXT, PDF)
- Real-time processing feedback

#### 2. OCR Service (`services/ocr_service.py`)
- Ancient Greek text extraction from images
- Image preprocessing pipeline
- Confidence scoring and text extraction

#### 3. Transliteration Service (`services/transliterate_service.py`)
- Greek ↔ Latin conversion with polytonic mark handling

#### 4. Translation Service (`services/translate_service.py`)
- Ancient Greek to English translation
- Lexicon integration for word-level lookup

### Tech Stack

- **UI Framework**: Streamlit
- **OCR Engine**: Tesseract 5.x
- **Image Processing**: Pillow, OpenCV
- **ML Framework**: PyTorch (CPU)
- **NLP Models**: Hugging Face Transformers
- **Database**: SQLite3 (optional)
- **Packaging**: PyInstaller
- **Python**: 3.11+

## Project Structure

```
metafrasis/
├── app.py                       # Streamlit main application
├── requirements.txt             # Python dependencies
├── setup.py                     # PyInstaller configuration
├── config.py                    # Application settings
├── services/                    # Core business logic
│   ├── ocr_service.py
│   ├── transliterate_service.py
│   ├── translate_service.py
│   └── preprocessing.py
├── models/                      # ML models
│   └── download_models.py
├── data/                        # Static data
│   └── lexicon/
├── utils/                       # Utilities
│   ├── text_processing.py
│   └── export.py
└── tests/                       # Unit tests
    ├── test_ocr_service.py
    ├── test_transliterate_service.py
    └── test_translate_service.py
```

## Installation

### For Development

1. **Install uv** (if you don't have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/metafrasis.git
   cd metafrasis
   ```

3. **Install dependencies** (uv handles venv automatically):
   ```bash
   uv sync
   ```

4. **Install Tesseract OCR**:
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr tesseract-ocr-grc`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Download language models** (first run):
   ```bash
   uv run python models/download_models.py
   ```

6. **Run the application**:
   ```bash
   uv run streamlit run app.py
   ```

   Or use uvx for a one-liner (no install):
   ```bash
   uvx --from . streamlit run app.py
   ```

### For End Users (Packaged App)

*Coming soon: Standalone .app bundle for macOS*

## Usage

- **OCR Tab**: Upload images containing Ancient Greek text
- **Transliteration Tab**: Convert between Greek and Latin scripts
- **Translation Tab**: Translate Ancient Greek to English with lexicon support

## Development Roadmap

- [x] Project structure
- [x] Documentation
- [ ] OCR service implementation
- [ ] Transliteration service
- [ ] Translation service
- [ ] Streamlit UI
- [ ] Packaging for distribution

## License

TBD
