"""
Base class for text detection models
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image
import torch

from ..base import TextRegion


class TextDetector(ABC):
    """
    Abstract base class for text detection models

    Detectors find text regions in images but don't read the text.
    They return TextRegion objects with bounding boxes and cropped images.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = None, **kwargs):
        """
        Initialize detector

        Args:
            model_path: Path to detection model weights
            device: 'cuda', 'cpu', or None for auto-detect
            **kwargs: Detector-specific configuration
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.is_loaded = False
        self.config = kwargs

    @abstractmethod
    def load_model(self):
        """Load the detection model"""
        pass

    @abstractmethod
    def detect(self, image: Image.Image) -> List[TextRegion]:
        """
        Detect text regions in an image

        Args:
            image: PIL Image

        Returns:
            List of TextRegion objects with bounding boxes and cropped images
        """
        pass

    def detect_batch(self, images: List[Image.Image]) -> List[List[TextRegion]]:
        """
        Batch detection (default: sequential, override for batching)

        Args:
            images: List of PIL Images

        Returns:
            List of lists of TextRegion objects (one list per image)
        """
        return [self.detect(img) for img in images]

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name (e.g., 'craft', 'db', 'whole_image')"""
        pass

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Setup PyTorch device

        Args:
            device: 'cuda', 'cpu', 'mps', 'auto', or None for auto-detect

        Returns:
            torch.device object
        """
        if device is None or device == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
