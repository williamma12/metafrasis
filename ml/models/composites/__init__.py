"""
Composite models that assemble backbones, necks, and heads

- CRAFT: Character Region Awareness For Text detection
- DBNet: Differentiable Binarization Network for text detection
- CRNN: Convolutional Recurrent Neural Network for text recognition
- PPOCRModel: PaddleOCR-style recognition model
"""
from .craft import CRAFT
from .dbnet import DBNet
from .crnn import CRNN
from .ppocr import PPOCRModel

__all__ = [
    "CRAFT",
    "DBNet",
    "CRNN",
    "PPOCRModel",
]
