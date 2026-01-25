"""
Backbone networks for feature extraction

- VGG16BN: VGG16 with batch normalization (CRAFT)
- ResNetBackbone: ResNet-18 for document detection (DB)
- MobileNetV3Backbone: MobileNetV3-small for recognition (PP-OCR)
- CRNNCNN: CNN backbone for CRNN text recognition
"""
from .vgg import VGG16BN
from .resnet import ResNetBackbone
from .mobilenet import MobileNetV3Backbone
from .crnn_cnn import CRNNCNN

__all__ = [
    "VGG16BN",
    "ResNetBackbone",
    "MobileNetV3Backbone",
    "CRNNCNN",
]
