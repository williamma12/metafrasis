"""
Text recognition models

Recognizers read text from image regions but don't detect them.
They take TextRegion objects and return Word objects with recognized text.
"""
from .base import TextRecognizer
