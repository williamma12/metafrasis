"""
trOCR Engine - Transformer-based OCR

State-of-the-art OCR using Microsoft's TrOCR (Vision Encoder-Decoder)
- PyTorch-based
- GPU acceleration (with CPU fallback)
- Optimized batch processing
- Excellent for handwritten text

Note: trOCR is an end-to-end model that doesn't provide bounding boxes or confidence scores.
For text detection + recognition, use Tesseract or combine with a separate detector.
"""
import time
from typing import List
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..base import OCREngine, OCRResult, Word, BoundingBox


class TrOCREngine(OCREngine):
    """trOCR engine for handwritten and printed text recognition"""

    def __init__(
        self,
        model_path: str = "microsoft/trocr-base-handwritten",
        batch_size: int = 8,
        device: str = None,
        **kwargs
    ):
        """
        Initialize trOCR engine

        Args:
            model_path: HuggingFace model ID or local path
            batch_size: Batch size for batch processing (default: 8)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional configuration
        """
        super().__init__(model_path=model_path, **kwargs)
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.processor = None
        self.model = None

    def load_model(self):
        """Load trOCR model and processor from HuggingFace"""
        try:
            print(f"Loading trOCR model: {self.model_path}")
            print(f"Using device: {self.device}")

            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            print(f"trOCR model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load trOCR model '{self.model_path}'. "
                f"Error: {e}"
            )

    def recognize(self, image: Image.Image) -> OCRResult:
        """
        Process a single image with trOCR

        Args:
            image: PIL Image object

        Returns:
            OCRResult with recognized text

        Note: trOCR doesn't provide bounding boxes or confidence scores
        """
        # Lazy load
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        # Prepare image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # Decode
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        processing_time = time.time() - start_time

        # trOCR doesn't provide bounding boxes or per-word confidences
        # Create a single "word" for the entire text
        word = Word(
            text=text,
            bbox=BoundingBox(left=0, top=0, width=0, height=0)  # No bbox available
        )

        return OCRResult(
            words=[word] if text.strip() else [],
            engine_name=self.name,
            processing_time=processing_time,
        )

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Batch pipeline: Process multiple images with optimized batching

        This overrides the default sequential processing with true batched inference
        for much better performance on GPU.

        Args:
            images: List of PIL Image objects

        Returns:
            List of OCRResult objects
        """
        # Lazy load
        if not self.is_loaded:
            self.load_model()

        results = []
        total_images = len(images)

        # Process in batches
        for i in range(0, total_images, self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_start_time = time.time()

            # Prepare batch
            pixel_values = self.processor(
                batch,
                return_tensors="pt",
                padding=True
            ).pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text for batch
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            # Decode all outputs
            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Create OCRResults
            batch_time = time.time() - batch_start_time
            per_image_time = batch_time / len(batch)

            for j, text in enumerate(texts):
                word = Word(
                    text=text,
                    bbox=BoundingBox(left=0, top=0, width=0, height=0)
                )

                result = OCRResult(
                    words=[word] if text.strip() else [],
                    engine_name=self.name,
                    processing_time=per_image_time,
                    source=f"image_{i+j+1}"
                )
                results.append(result)

        return results

    @property
    def name(self) -> str:
        return "trocr"
