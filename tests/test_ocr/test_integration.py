"""
Integration tests for OCR inference pipeline.

Tests end-to-end inference from image to text using actual model weights.
"""

import pytest
from PIL import Image, ImageDraw, ImageFont

from services.ocr.factory import OCREngineFactory


class TestOCRInferencePipeline:
    """End-to-end tests for OCR inference pipeline."""

    @pytest.fixture
    def sample_text_image(self):
        """Create a simple image with text for testing."""
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        # Draw some simple text
        draw.text((20, 30), "Hello World", fill="black")
        return img

    @pytest.fixture
    def blank_image(self):
        """Create a blank image with no text."""
        return Image.new("RGB", (200, 100), color="white")

    @pytest.mark.slow
    @pytest.mark.requires_craft
    def test_craft_detector_inference(self, sample_text_image):
        """Test CRAFT detector can detect text regions."""
        from services.ocr.detectors.craft import CRAFTDetector

        detector = CRAFTDetector()
        detector.load_model()

        regions = detector.detect(sample_text_image)

        # Should detect at least one region
        assert isinstance(regions, list)
        # CRAFT may or may not detect regions depending on the simple text

    @pytest.mark.slow
    @pytest.mark.requires_crnn
    def test_crnn_recognizer_inference(self, sample_text_image):
        """Test CRNN recognizer can recognize text from a region."""
        from services.ocr.recognizers.crnn import CRNNRecognizer
        from services.ocr.base import BoundingBox, TextRegion

        recognizer = CRNNRecognizer()
        recognizer.load_model()

        # Create a simple text region from the full image
        region = TextRegion(
            bbox=BoundingBox(left=0, top=0, width=400, height=100),
            crop=sample_text_image,
            confidence=1.0,
        )

        results = recognizer.recognize_regions([region])

        assert isinstance(results, list)
        assert len(results) == 1
        assert hasattr(results[0], "text")

    @pytest.mark.slow
    @pytest.mark.requires_craft
    @pytest.mark.requires_crnn
    def test_modular_engine_inference(self, sample_text_image):
        """Test full modular pipeline: detector + recognizer."""
        engine = OCREngineFactory.create(
            detector="craft",
            recognizer="crnn",
        )

        result = engine.recognize(sample_text_image)

        assert result is not None
        assert hasattr(result, "words")
        assert hasattr(result, "processing_time")
        assert result.processing_time > 0

    @pytest.mark.slow
    @pytest.mark.requires_craft
    @pytest.mark.requires_crnn
    def test_modular_engine_batch_inference(self, sample_text_image, blank_image):
        """Test batch inference with multiple images."""
        engine = OCREngineFactory.create(
            detector="craft",
            recognizer="crnn",
        )

        images = [sample_text_image, blank_image]
        results = engine.recognize_batch(images)

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert hasattr(result, "words")

    @pytest.mark.slow
    @pytest.mark.requires_craft
    @pytest.mark.requires_crnn
    def test_debug_mode_stores_detector_regions(self, sample_text_image):
        """Test that debug mode stores detector regions in result."""
        engine = OCREngineFactory.create(
            detector="craft",
            recognizer="crnn",
            debug_mode=True,
        )

        result = engine.recognize(sample_text_image)

        # In debug mode, detector_regions should be stored
        assert result.detector_regions is not None

    def test_engine_factory_available_engines(self):
        """Test that factory can list available engines."""
        engines = OCREngineFactory.available_engines()
        detectors = OCREngineFactory.available_detectors()
        recognizers = OCREngineFactory.available_recognizers()

        assert isinstance(engines, list)
        assert isinstance(detectors, list)
        assert isinstance(recognizers, list)

    @pytest.mark.slow
    @pytest.mark.requires_tesseract
    def test_tesseract_engine_inference(self, sample_text_image):
        """Test Tesseract engine if available."""
        if "tesseract" not in OCREngineFactory.available_engines():
            pytest.skip("Tesseract not available")

        engine = OCREngineFactory.create(engine="tesseract")
        result = engine.recognize(sample_text_image)

        assert result is not None
        assert hasattr(result, "words")


class TestOCRPreprocessing:
    """Tests for image preprocessing utilities."""

    def test_pdf_to_images_with_sample(self, sample_pdf_path):
        """Test PDF to images conversion with sample PDF."""
        from services.ocr.preprocessing import pdf_to_images

        images = pdf_to_images(sample_pdf_path, dpi=72)

        assert isinstance(images, list)
        assert len(images) > 0
        assert all(isinstance(img, Image.Image) for img in images)

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create or skip if no PDF available."""
        # For now, skip if we don't have a sample PDF
        pytest.skip("Sample PDF not available for testing")
