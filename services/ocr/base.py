"""
Base classes for OCR engines

OCR engines process PIL Images only - PDF conversion is handled separately
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
import numpy as np


# Constants
DEFAULT_CONFIDENCE = -1.0  # Sentinel value for unavailable confidence


@dataclass
class BoundingBox:
    """Bounding box coordinates for text regions"""
    left: int
    top: int
    width: int
    height: int


@dataclass
class Word:
    """Single recognized word with bounding box and confidence"""
    text: str
    bbox: BoundingBox
    confidence: float = DEFAULT_CONFIDENCE


@dataclass
class ConfidenceStats:
    """Confidence statistics computed from word-level confidences"""
    mean: float
    std: float
    available: bool  # False if confidence not available (all words have DEFAULT_CONFIDENCE)


@dataclass
class OCRResult:
    """Standardized OCR result format"""
    words: List[Word]  # List of recognized words with bboxes and confidences
    engine_name: str
    processing_time: float
    source: str = ""  # Optional: source identifier (e.g., "page_1"), set by caller

    @property
    def confidence_stats(self) -> ConfidenceStats:
        """
        Compute confidence statistics from word confidences

        Returns:
            ConfidenceStats with mean, std, and availability
        """
        if not self.words:
            return ConfidenceStats(mean=DEFAULT_CONFIDENCE, std=0.0, available=False)

        # Extract confidences
        confidences = [w.confidence for w in self.words]

        # Check if any valid confidences are available
        valid_confidences = [c for c in confidences if c >= 0]

        if not valid_confidences:
            # All confidences are DEFAULT_CONFIDENCE (unavailable)
            return ConfidenceStats(mean=DEFAULT_CONFIDENCE, std=0.0, available=False)

        # Compute statistics
        mean = float(np.mean(valid_confidences))
        std = float(np.std(valid_confidences))

        return ConfidenceStats(mean=mean, std=std, available=True)


class OCREngine(ABC):
    """
    Abstract base class for all OCR engines

    OCR engines process PIL Images only. PDF conversion is handled
    by separate utilities in preprocessing.py
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize OCR engine

        Args:
            model_path: Path to model file or HuggingFace model ID
            **kwargs: Engine-specific configuration
        """
        self.model_path = model_path
        self.is_loaded = False
        self.config = kwargs

    @abstractmethod
    def load_model(self):
        """Load the OCR model (called lazily on first use)"""
        pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult:
        """
        Streaming pipeline: Process a single image

        Args:
            image: PIL Image object

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Batch pipeline: Process multiple images

        Default implementation processes sequentially. PyTorch engines should
        override this for true batched inference (much faster).

        Args:
            images: List of PIL Image objects

        Returns:
            List of OCRResult objects
        """
        # Default: sequential processing
        results = []
        for i, image in enumerate(images):
            result = self.recognize(image)
            if not result.source:
                result.source = f"image_{i+1}"
            results.append(result)
        return results

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name (e.g., 'tesseract', 'trocr')"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', loaded={self.is_loaded})"
