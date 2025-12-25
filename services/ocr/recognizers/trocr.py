"""
trOCR Recognizer - Transformer-based text recognition
"""
from typing import List
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..base import TextRegion, Word
from .base import TextRecognizer


class TrOCRRecognizer(TextRecognizer):
    """
    trOCR transformer recognizer

    Recognizes text from cropped image regions.
    Excellent for handwritten text and fine-tuning.

    Note: trOCR doesn't provide per-character confidence, so we use
    the detection confidence from the TextRegion.
    """

    def __init__(
        self,
        model_path: str = "microsoft/trocr-base-handwritten",
        batch_size: int = 8,
        device: str = None,
        **kwargs
    ):
        """
        Initialize trOCR recognizer

        Args:
            model_path: HuggingFace model ID or local path
            batch_size: Internal batch size for processing (default: 8)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional configuration
        """
        super().__init__(model_path=model_path, device=device, **kwargs)
        self.batch_size = batch_size
        self.processor = None
        self.model = None

    def load_model(self):
        """Load trOCR model and processor from HuggingFace"""
        print(f"Loading trOCR recognizer: {self.model_path}")
        print(f"Using device: {self.device}")

        self.processor = TrOCRProcessor.from_pretrained(self.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)

        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        print(f"trOCR recognizer loaded successfully on {self.device}")

    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from multiple regions (batch processing)

        Args:
            regions: List of TextRegion objects with cropped images

        Returns:
            List of Word objects with recognized text
            (one Word per TextRegion, in same order)
        """
        if not self.is_loaded:
            self.load_model()

        if not regions:
            return []

        words = []

        # Process in batches for GPU efficiency
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]

            # Extract cropped images
            batch_crops = [r.crop for r in batch_regions]

            # Prepare batch
            pixel_values = self.processor(
                batch_crops,
                return_tensors="pt",
                padding=True
            ).pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text for batch
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            # Decode all outputs
            texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            # Create Words with original bounding boxes
            for region, text in zip(batch_regions, texts):
                word = Word(
                    text=text,
                    bbox=region.bbox,
                    confidence=region.confidence  # Use detection confidence
                )
                words.append(word)

        return words

    @property
    def name(self) -> str:
        return "trocr"
