"""
PyTorch OCR Engine - Composes detector + recognizer
"""
import time
from typing import List
from PIL import Image

from ..base import OCREngine, OCRResult
from ..detectors.base import TextDetector
from ..recognizers.base import TextRecognizer


class PyTorchOCREngine(OCREngine):
    """
    Modular OCR engine that composes a detector and recognizer

    Handles the pipeline: image → detect regions → recognize text → OCRResult

    Examples:
        # WholeImage + trOCR (end-to-end like old trOCR)
        engine = PyTorchOCREngine(
            detector=WholeImageDetector(),
            recognizer=TrOCRRecognizer(device='cuda')
        )

        # CRAFT + trOCR (detection + recognition)
        engine = PyTorchOCREngine(
            detector=CRAFTDetector(device='cuda'),
            recognizer=TrOCRRecognizer(device='cuda')
        )
    """

    def __init__(
        self,
        detector: TextDetector,
        recognizer: TextRecognizer,
        batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize PyTorch OCR engine

        Args:
            detector: TextDetector instance (finds text regions)
            recognizer: TextRecognizer instance (reads text from regions)
            batch_size: Batch size for recognition (default: 8)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.detector = detector
        self.recognizer = recognizer
        self.batch_size = batch_size

    def load_model(self):
        """Load both detector and recognizer models"""
        if not self.detector.is_loaded:
            self.detector.load_model()
        if not self.recognizer.is_loaded:
            self.recognizer.load_model()
        self.is_loaded = True

    def recognize(self, image: Image.Image) -> OCRResult:
        """
        Process a single image through detection → recognition pipeline

        Args:
            image: PIL Image

        Returns:
            OCRResult with words, bounding boxes, and confidences
        """
        # Lazy load
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        # Step 1: Detect text regions
        regions = self.detector.detect(image)

        # Step 2: Recognize text in regions (with batching)
        words = []
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]
            batch_words = self.recognizer.recognize_regions(batch_regions)
            words.extend(batch_words)

        processing_time = time.time() - start_time

        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=processing_time
        )

    def recognize_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Batch pipeline: Process multiple images

        Optimizes by:
        1. Batch detecting all images
        2. Batch recognizing ALL regions across ALL images (maximum GPU utilization)
        3. Reassembling results per image

        Args:
            images: List of PIL Images

        Returns:
            List of OCRResult objects (one per input image)
        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        # Step 1: Batch detect all images
        all_regions = self.detector.detect_batch(images)

        # Step 2: Flatten all regions across all images for cross-image batching
        flattened_regions = []
        region_counts = []  # Track how many regions per image

        for regions in all_regions:
            flattened_regions.extend(regions)
            region_counts.append(len(regions))

        # Step 3: Batch recognize ALL regions across ALL images
        all_words = []
        for i in range(0, len(flattened_regions), self.batch_size):
            batch_regions = flattened_regions[i:i + self.batch_size]
            batch_words = self.recognizer.recognize_regions(batch_regions)
            all_words.extend(batch_words)

        # Step 4: Reassemble words back to per-image results
        results = []
        word_idx = 0

        for img_idx, num_regions in enumerate(region_counts):
            # Extract words for this image
            image_words = all_words[word_idx:word_idx + num_regions]
            word_idx += num_regions

            result = OCRResult(
                words=image_words,
                engine_name=self.name,
                processing_time=0.0,  # Individual timing not meaningful in batch
                source=f"image_{img_idx+1}"
            )
            results.append(result)

        # Total processing time for all images
        total_time = time.time() - start_time
        avg_time = total_time / len(images) if images else 0.0

        # Update processing time to average per image
        for result in results:
            result.processing_time = avg_time

        return results

    @property
    def name(self) -> str:
        """Engine name: detector_recognizer format"""
        return f"{self.detector.name}_{self.recognizer.name}"
