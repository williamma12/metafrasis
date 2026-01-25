"""
Head networks for task-specific outputs

- CTCHead: CTC prediction head for text recognition
- DBHead: Differentiable binarization head for text detection
"""
from .ctc import CTCHead
from .db import DBHead

__all__ = [
    "CTCHead",
    "DBHead",
]
