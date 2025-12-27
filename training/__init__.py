"""
Training infrastructure for OCR models.

This module provides tools for:
- Converting annotation datasets to training-ready formats
- Fine-tuning detectors (CRAFT, DB) and recognizers (CRNN, PP-OCR, trOCR)
- Evaluating model performance
- Exporting models to ONNX and HuggingFace Hub
"""

from training.data import (
    DetectorDataset,
    CRAFTDataset,
    DBDataset,
    RecognizerDataset,
)

__all__ = [
    "DetectorDataset",
    "CRAFTDataset",
    "DBDataset",
    "RecognizerDataset",
]
