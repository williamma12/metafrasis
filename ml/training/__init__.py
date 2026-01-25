"""
Training infrastructure for OCR models.

This module provides tools for:
- Fine-tuning detectors (CRAFT, DB) and recognizers (CRNN, PP-OCR, trOCR)
- Evaluating model performance
- Exporting models to ONNX

Usage:
    python -m training.finetune.detectors.craft training/configs/craft.yaml
    python -m training.finetune.detectors.db training/configs/db.yaml
    python -m training.finetune.recognizers.crnn training/configs/crnn.yaml
"""

from ml.training.finetune.base import BaseTrainer
from ml.training.finetune.detectors.base import DetectorTrainer
from ml.training.finetune.recognizers.base import RecognizerTrainer

__all__ = [
    "BaseTrainer",
    "DetectorTrainer",
    "RecognizerTrainer",
]
