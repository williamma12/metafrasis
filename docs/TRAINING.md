# Training & Optimization Plan

## Overview
Infrastructure for fine-tuning OCR models on Ancient Greek text using vision-model-generated annotations, benchmarking engines, and model management.

## Key Principles
1. **All models externally hosted** - Nothing in git, download on-demand
2. **Vision model annotation** - Use GPT-4V/Claude to annotate real images
3. **Modular training** - Each engine has its own fine-tuning script
4. **Comprehensive benchmarking** - Compare all engines systematically

## Directory Structure

```
training/
├── README.md                    # Training documentation
├── __init__.py
│
├── configs/                     # Training configurations
│   ├── trocr_lora.yaml         # trOCR + LoRA config
│   ├── easyocr.yaml            # EasyOCR config
│   └── kraken.yaml             # Kraken config
│
├── finetune/                    # Fine-tuning scripts
│   ├── __init__.py
│   ├── finetune_trocr.py       # trOCR + LoRA fine-tuning
│   ├── finetune_easyocr.py     # EasyOCR fine-tuning
│   ├── finetune_kraken.py      # Kraken fine-tuning
│   └── train_utils.py          # Shared utilities
│
├── benchmarks/                  # Engine comparison
│   ├── __init__.py
│   ├── compare_engines.py      # Run all engines, compare results
│   ├── evaluate.py             # Compute CER, WER metrics
│   └── visualize.py            # Plot comparisons
│
├── data/                        # Dataset creation
│   ├── __init__.py
│   ├── vision_annotate.py      # Vision model annotation (GPT-4V, Claude)
│   ├── annotation_tool.py      # Manual review/correction (Streamlit)
│   └── dataset_builder.py      # Build HuggingFace datasets
│
└── notebooks/                   # Jupyter experiments
    ├── exploratory_analysis.ipynb
    └── model_comparison.ipynb

models/
├── registry.json                # Model URLs and metadata (IN GIT)
├── download_models.py           # Download script (IN GIT)
└── .gitkeep                     # (IN GIT)
# Everything else in models/ is gitignored

data/                            # All gitignored
├── raw/                         # Raw unlabeled images
├── annotated/                   # Vision-model annotated
├── reviewed/                    # Manually reviewed/corrected
└── datasets/                    # Final HuggingFace datasets
```

## Vision Model Annotation Workflow

### Philosophy
Use large vision models to annotate **real** manuscript/book images instead of rendering fonts. This provides:
- Real-world image distribution
- Natural degradation, noise, layouts
- Faster than manual annotation
- Bootstrap from unlabeled data

### Supported Vision Models

**Commercial:**
- OpenAI GPT-4V / GPT-4o (best quality)
- Anthropic Claude 3 Opus/Sonnet
- Google Gemini Pro Vision

**Open Source:**
- LLaVA 1.6 (good quality, free)
- CogVLM (specialized for OCR tasks)
- Qwen-VL (multilingual)

## Model Registry

**`models/registry.json`** (committed to git) contains URLs and metadata:

```json
{
  "tesseract": {
    "base": {
      "url": "https://github.com/tesseract-ocr/tessdata_best/raw/main/grc.traineddata",
      "type": "traineddata"
    }
  },
  "kraken": {
    "base": {
      "url": "https://zenodo.org/record/.../greek_best.mlmodel",
      "type": "mlmodel"
    },
    "finetuned": {
      "url": "https://huggingface.co/your-username/kraken-ancient-greek",
      "type": "mlmodel"
    }
  },
  "trocr": {
    "base": {
      "url": "microsoft/trocr-base-handwritten",
      "type": "huggingface"
    },
    "finetuned": {
      "url": "your-username/trocr-ancient-greek-lora",
      "type": "huggingface_adapter"
    }
  }
}
```

## Dependencies

```toml
[project.optional-dependencies]
training = [
    # Fine-tuning
    "peft>=0.8.0",              # LoRA adapters
    "accelerate>=0.26.0",       # Distributed training
    "transformers>=4.37.0",     # HuggingFace models

    # Vision model APIs
    "openai>=1.10.0",           # GPT-4V
    "anthropic>=0.18.0",        # Claude
    "google-generativeai>=0.3.0", # Gemini

    # Metrics and evaluation
    "jiwer>=3.0.0",             # WER/CER
    "edit-distance>=1.0.0",     # Levenshtein

    # Experiment tracking - manual with markdown + matplotlib

    # Data processing
    "datasets>=2.16.0",         # HuggingFace datasets
    "albumentations>=1.3.0",    # Image augmentation

    # Visualization
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
]
```

## .gitignore

```
# ALL models - hosted externally
models/*

# Keep only registry and download script
!models/registry.json
!models/download_models.py
!models/.gitkeep

# All training data
data/

# Training outputs (temporary)
training/outputs/
training/logs/
training/checkpoints/

# Keep experiment results (markdown + plots)
!training/results/
```

## Success Criteria

- ✅ Can annotate 1000 images with GPT-4V in < 1 hour
- ✅ Annotation tool allows review/correction
- ✅ trOCR fine-tuning with LoRA works end-to-end
- ✅ Adapters automatically uploaded to HuggingFace
- ✅ Benchmark script compares all engines
- ✅ No models stored in git - all external
- ✅ `models/registry.json` is single source of truth
- ✅ Experiments tracked with markdown docs and matplotlib plots
