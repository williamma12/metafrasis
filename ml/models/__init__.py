"""
PyTorch model definitions for OCR

Organized by architecture:
- backbones: Feature extractors (VGG, ResNet, MobileNet)
- necks: Feature aggregation (FPN, BiLSTM)
- heads: Task-specific outputs (CTC, DB)
- composites: Full models (CRAFT, DBNet, CRNN, PPOCRModel)
- layers: Shared building blocks
"""
from .layers import (
    ConvBNLayer,
    SEModule,
    BasicBlock,
    ResidualUnit,
    CTCDecoder,
    CHARSETS,
    get_charset,
)

from .backbones import (
    VGG16BN,
    ResNetBackbone,
    MobileNetV3Backbone,
    CRNNCNN,
)

from .necks import (
    FPN,
    BidirectionalLSTM,
    SequenceEncoder,
)

from .heads import (
    CTCHead,
    DBHead,
)

from .composites import (
    CRAFT,
    DBNet,
    CRNN,
    PPOCRModel,
)

__all__ = [
    # Layers
    "ConvBNLayer",
    "SEModule",
    "BasicBlock",
    "ResidualUnit",
    "CTCDecoder",
    "CHARSETS",
    "get_charset",
    # Backbones
    "VGG16BN",
    "ResNetBackbone",
    "MobileNetV3Backbone",
    "CRNNCNN",
    # Necks
    "FPN",
    "BidirectionalLSTM",
    "SequenceEncoder",
    # Heads
    "CTCHead",
    "DBHead",
    # Composites
    "CRAFT",
    "DBNet",
    "CRNN",
    "PPOCRModel",
]
