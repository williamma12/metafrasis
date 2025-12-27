"""Recognizer trainers for text recognition models."""

from .base import (
    RecognizerTrainer,
    CTCRecognizerTrainer,
    TransformerRecognizerTrainer,
    RecognizerDataset,
    collate_fn,
)
from .crnn import CRNNTrainer
from .ppocr import PPOCRTrainer
from .trocr import TrOCRTrainer, TrOCRDataset

__all__ = [
    "RecognizerTrainer",
    "CTCRecognizerTrainer",
    "TransformerRecognizerTrainer",
    "RecognizerDataset",
    "collate_fn",
    "CRNNTrainer",
    "PPOCRTrainer",
    "TrOCRTrainer",
    "TrOCRDataset",
]
