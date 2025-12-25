"""
Type definitions for OCR service
"""
from enum import Enum


class DetectorType(str, Enum):
    """Available text detectors"""
    WHOLE_IMAGE = "whole_image"
    CRAFT = "craft"
    DB = "db"


class RecognizerType(str, Enum):
    """Available text recognizers"""
    TROCR = "trocr"
    CRNN = "crnn"
    KRAKEN = "kraken"


class EngineType(str, Enum):
    """Available monolithic engines"""
    TESSERACT = "tesseract"
