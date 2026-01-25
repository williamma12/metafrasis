"""
Tests for PyTorchOCREngine
"""
import pytest
from PIL import Image
from unittest.mock import MagicMock
from app.services.ocr.engines.pytorch_engine import PyTorchOCREngine
from app.services.ocr.base import TextRegion, Word, BoundingBox, OCRResult


class MockDetector:
    """Mock detector for testing"""
    def __init__(self):
        self.is_loaded = False
        self.name = "mock_detector"

    def load_model(self):
        self.is_loaded = True

    def detect(self, image):
        # Return 2 regions for each image
        width, height = image.size
        regions = [
            TextRegion(
                bbox=BoundingBox(0, 0, width//2, height),
                crop=image.crop((0, 0, width//2, height)),
                confidence=0.9
            ),
            TextRegion(
                bbox=BoundingBox(width//2, 0, width//2, height),
                crop=image.crop((width//2, 0, width, height)),
                confidence=0.95
            )
        ]
        return regions

    def detect_batch(self, images):
        return [self.detect(img) for img in images]


class MockRecognizer:
    """Mock recognizer for testing"""
    def __init__(self):
        self.is_loaded = False
        self.name = "mock_recognizer"

    def load_model(self):
        self.is_loaded = True

    def recognize_regions(self, regions):
        # Return one word per region
        words = []
        for i, region in enumerate(regions):
            word = Word(
                text=f"word{i}",
                bbox=region.bbox,
                confidence=region.confidence
            )
            words.append(word)
        return words


class TestPyTorchOCREngine:
    """Tests for PyTorchOCREngine"""

    def test_engine_initialization(self):
        """Test engine initializes with detector and recognizer"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(
            detector=detector,
            recognizer=recognizer,
            batch_size=8
        )

        assert engine.detector is detector
        assert engine.recognizer is recognizer
        assert engine.batch_size == 8
        assert not engine.is_loaded

    def test_engine_name(self):
        """Test engine name is detector_recognizer format"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer)

        assert engine.name == "mock_detector_mock_recognizer"

    def test_load_model(self):
        """Test load_model loads both detector and recognizer"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer)
        engine.load_model()

        assert detector.is_loaded
        assert recognizer.is_loaded
        assert engine.is_loaded

    def test_recognize_single_image(self, sample_image):
        """Test recognizing a single image"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer)
        result = engine.recognize(sample_image)

        assert isinstance(result, OCRResult)
        assert len(result.words) == 2  # MockDetector returns 2 regions
        assert result.engine_name == "mock_detector_mock_recognizer"
        assert result.processing_time > 0

    def test_recognize_lazy_loads(self, sample_image):
        """Test recognize lazy loads models"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer)

        assert not detector.is_loaded
        assert not recognizer.is_loaded

        result = engine.recognize(sample_image)

        assert detector.is_loaded
        assert recognizer.is_loaded

    def test_recognize_batch(self, sample_images_batch):
        """Test batch recognition"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer, batch_size=4)
        results = engine.recognize_batch(sample_images_batch)

        assert len(results) == len(sample_images_batch)
        assert all(isinstance(r, OCRResult) for r in results)
        # Each image produces 2 regions -> 2 words
        assert all(len(r.words) == 2 for r in results)

    def test_recognize_batch_cross_image_batching(self):
        """Test that batch processing batches across all images"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        # Track calls to recognize_regions
        original_recognize = recognizer.recognize_regions
        recognize_calls = []

        def track_recognize(regions):
            recognize_calls.append(len(regions))
            return original_recognize(regions)

        recognizer.recognize_regions = track_recognize

        engine = PyTorchOCREngine(detector, recognizer, batch_size=4)

        # 3 images * 2 regions each = 6 total regions
        # With batch_size=4, should be 2 batches: [4, 2]
        images = [Image.new('RGB', (100, 100)) for _ in range(3)]
        results = engine.recognize_batch(images)

        assert len(results) == 3
        # Check that batching happened (number of calls should be less than total regions)
        assert len(recognize_calls) == 2  # 2 batches
        assert recognize_calls[0] == 4  # First batch
        assert recognize_calls[1] == 2  # Second batch

    def test_recognize_batch_reassembles_correctly(self):
        """Test that words are correctly reassembled per image"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer, batch_size=10)

        images = [Image.new('RGB', (100, 100)) for _ in range(3)]
        results = engine.recognize_batch(images)

        # Each image should have exactly 2 words (from 2 regions)
        assert all(len(r.words) == 2 for r in results)

        # Words should be word0, word1 for first image, word2, word3 for second, etc.
        assert results[0].words[0].text == "word0"
        assert results[0].words[1].text == "word1"
        assert results[1].words[0].text == "word2"
        assert results[1].words[1].text == "word3"
        assert results[2].words[0].text == "word4"
        assert results[2].words[1].text == "word5"

    def test_recognize_batch_sets_source(self):
        """Test that batch processing sets source names"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer)

        images = [Image.new('RGB', (100, 100)) for _ in range(3)]
        results = engine.recognize_batch(images)

        assert results[0].source == "image_1"
        assert results[1].source == "image_2"
        assert results[2].source == "image_3"

    def test_recognize_preserves_bboxes(self, sample_image):
        """Test that bounding boxes are preserved from detection to recognition"""
        detector = MockDetector()
        recognizer = MockRecognizer()

        engine = PyTorchOCREngine(detector, recognizer)
        result = engine.recognize(sample_image)

        # Verify bboxes match what detector returned
        width, height = sample_image.size
        assert result.words[0].bbox.left == 0
        assert result.words[0].bbox.width == width // 2

        assert result.words[1].bbox.left == width // 2
        assert result.words[1].bbox.width == width // 2
