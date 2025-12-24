"""
Tesseract OCR Engine

Traditional OCR engine using Tesseract. Good baseline for comparison.
- CPU-only (not PyTorch)
- Fast execution
- Ancient Greek language support ('grc')
- Uses default sequential batch processing (inherited from base class)
"""
import time
import numpy as np
from PIL import Image
import pytesseract
from ..base import OCREngine, OCRResult, Word, BoundingBox


class TesseractEngine(OCREngine):
    """Tesseract OCR engine for Ancient Greek text"""

    def __init__(self, lang: str = 'grc', **kwargs):
        """
        Initialize Tesseract engine

        Args:
            lang: Tesseract language code (default: 'grc' for Ancient Greek)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.lang = lang

    def load_model(self):
        """
        Load Tesseract model

        For Tesseract, this just verifies that the language data is available
        """
        try:
            # Test that Tesseract is installed and language is available
            pytesseract.get_languages()
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Tesseract not available or language '{self.lang}' not installed. "
                f"Install Tesseract and download language data. Error: {e}"
            )

    def recognize(self, image: Image.Image) -> OCRResult:
        """
        Process a single image with Tesseract

        Args:
            image: PIL Image object

        Returns:
            OCRResult with words, bounding boxes, and confidences
        """
        # Lazy load
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        # Run Tesseract OCR with detailed data
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            output_type=pytesseract.Output.DICT
        )

        # Extract words with bounding boxes and confidences
        n_boxes = len(data['text'])
        words = []

        for i in range(n_boxes):
            # Only include boxes with text and valid confidence
            if data['text'][i].strip() and data['conf'][i] != '-1':
                bbox = BoundingBox(
                    left=data['left'][i],
                    top=data['top'][i],
                    width=data['width'][i],
                    height=data['height'][i]
                )

                conf = float(data['conf'][i]) / 100.0  # Convert 0-100 to 0-1

                word = Word(
                    text=data['text'][i],
                    confidence=conf,
                    bbox=bbox
                )

                words.append(word)

        processing_time = time.time() - start_time

        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=processing_time,
        )

    @property
    def name(self) -> str:
        return "tesseract"
