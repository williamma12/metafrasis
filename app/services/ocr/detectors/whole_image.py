"""
WholeImageDetector - Pass-through detector for end-to-end models
"""
from typing import List
from PIL import Image

from ..base import BoundingBox, TextRegion
from .base import TextDetector


class WholeImageDetector(TextDetector):
    """
    Pass-through detector that treats entire image as one text region

    Use with end-to-end recognizers like trOCR that don't need detection.
    Always returns a single TextRegion containing the whole image.
    """

    def __init__(self, **kwargs):
        """
        Initialize WholeImage detector

        No model needed - this is just a pass-through.
        """
        # No model path needed, device is ignored (no computation)
        # Remove device from kwargs if present to avoid confusion
        kwargs.pop('device', None)
        super().__init__(model_path=None, **kwargs)
        self.is_loaded = True  # Always ready

    def load_model(self):
        """No model to load"""
        self.is_loaded = True

    def detect(self, image: Image.Image) -> List[TextRegion]:
        """
        Return entire image as single region

        Args:
            image: PIL Image

        Returns:
            List with single TextRegion containing the whole image
        """
        width, height = image.size

        bbox = BoundingBox(
            left=0,
            top=0,
            width=width,
            height=height
        )

        region = TextRegion(
            bbox=bbox,
            crop=image,  # Entire image
            confidence=1.0  # No actual detection, perfect "confidence"
        )

        return [region]

    def detect_batch(self, images: List[Image.Image]) -> List[List[TextRegion]]:
        """
        Batch detect - just wrap each image

        Args:
            images: List of PIL Images

        Returns:
            List of lists, each containing single TextRegion
        """
        return [[self.detect(img)[0]] for img in images]

    @property
    def name(self) -> str:
        return "whole_image"
