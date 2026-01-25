"""
Fine-tuning infrastructure for OCR models.

Provides base classes and concrete trainers for:
- Detectors: CRAFT, DBNet
- Recognizers: CRNN, PP-OCR, trOCR

Example usage:
    from training.finetune import CRNNTrainer

    config = {"data_dir": "data/annotations/my_dataset", "output_dir": "data/training/crnn"}
    trainer = CRNNTrainer(config)
    results = trainer.train()
"""

from ml.training.finetune.base import BaseTrainer
from ml.training.finetune.utils import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    EarlyStopping,
    LRScheduler,
    MetricTracker,
    count_parameters,
    get_device,
)
from ml.training.finetune.detectors import (
    DetectorTrainer,
    CRAFTTrainer,
    DBTrainer,
)
from ml.training.finetune.recognizers import (
    RecognizerTrainer,
    CTCRecognizerTrainer,
    TransformerRecognizerTrainer,
    CRNNTrainer,
    PPOCRTrainer,
    TrOCRTrainer,
)

__all__ = [
    # Base classes
    "BaseTrainer",
    "DetectorTrainer",
    "RecognizerTrainer",
    "CTCRecognizerTrainer",
    "TransformerRecognizerTrainer",
    # Detector trainers
    "CRAFTTrainer",
    "DBTrainer",
    # Recognizer trainers
    "CRNNTrainer",
    "PPOCRTrainer",
    "TrOCRTrainer",
    # Utilities
    "load_config",
    "save_checkpoint",
    "load_checkpoint",
    "setup_logging",
    "EarlyStopping",
    "LRScheduler",
    "MetricTracker",
    "count_parameters",
    "get_device",
]
