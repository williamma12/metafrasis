"""
Type definitions for OCR service
"""
from enum import Enum


class DetectorType(str, Enum):
    """Available text detectors"""
    WHOLE_IMAGE = "whole_image"
    # CRAFT = "craft"  # To be added
    # DB = "db"  # To be added


class RecognizerType(str, Enum):
    """Available text recognizers"""
    TROCR = "trocr"
    # CRNN = "crnn"  # To be added
    # KRAKEN = "kraken"  # To be added


class EngineType(str, Enum):
    """Available monolithic engines"""
    TESSERACT = "tesseract"
