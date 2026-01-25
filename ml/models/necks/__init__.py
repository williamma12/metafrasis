"""
Neck networks for feature aggregation

- FPN: Feature Pyramid Network for multi-scale fusion (DB)
- BidirectionalLSTM: Bidirectional LSTM for sequence modeling
- SequenceEncoder: Stacked BiLSTM encoder
"""
from .fpn import FPN
from .sequence import BidirectionalLSTM, SequenceEncoder

__all__ = [
    "FPN",
    "BidirectionalLSTM",
    "SequenceEncoder",
]
