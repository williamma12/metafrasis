"""
Shared neural network layers for OCR recognizers

DEPRECATED: This module re-exports from models.layers for backward compatibility.
New code should import directly from models.

Example:
    # Old (deprecated):
    from services.ocr.recognizers.layers import BidirectionalLSTM, CTCDecoder

    # New (preferred):
    from models import BidirectionalLSTM, CTCDecoder
"""
# Re-export everything from models.layers for backward compatibility
from models.layers import (
    ConvBNLayer,
    SEModule,
    BasicBlock,
    ResidualUnit,
    CTCDecoder,
    CHARSETS,
    get_charset,
    make_divisible,
)

from models.necks import (
    BidirectionalLSTM,
    SequenceEncoder,
)

from models.heads import (
    CTCHead,
)

__all__ = [
    "ConvBNLayer",
    "SEModule",
    "BasicBlock",
    "ResidualUnit",
    "CTCDecoder",
    "CHARSETS",
    "get_charset",
    "make_divisible",
    "BidirectionalLSTM",
    "SequenceEncoder",
    "CTCHead",
]
