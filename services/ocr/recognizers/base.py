"""
Base class for text recognition models
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from ..base import TextRegion, Word


class TextRecognizer(ABC):
    """
    Abstract base class for text recognition models

    Recognizers read text from image regions but don't detect them.
    They take TextRegion objects (with cropped images) and return Word objects.

    Design: Optimized for batch processing (many regions is the common case).
    Single region is just a batch of size 1.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = None, **kwargs):
        """
        Initialize recognizer

        Args:
            model_path: Path to recognition model or HuggingFace ID
            device: 'cuda', 'cpu', or None for auto-detect
            **kwargs: Recognizer-specific configuration
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.is_loaded = False
        self.config = kwargs

    @abstractmethod
    def load_model(self):
        """Load the recognition model"""
        pass

    @abstractmethod
    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from multiple regions (batch processing)

        Args:
            regions: List of TextRegion objects with cropped images

        Returns:
            List of Word objects with recognized text and confidences
            (one Word per TextRegion, in same order)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Recognizer name (e.g., 'trocr', 'crnn', 'kraken')"""
        pass

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Setup PyTorch device

        Args:
            device: 'cuda', 'cpu', or None for auto-detect

        Returns:
            torch.device object
        """
        if device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
