"""
Text detection models

Detectors find text regions in images but don't recognize the text.
They output TextRegion objects with bounding boxes and cropped images.
"""
from .base import TextDetector, TextRegion
