"""
Tests for TextRecognizer base class
"""
import pytest
from PIL import Image
from app.services.ocr.recognizers.base import TextRecognizer
from app.services.ocr.base import TextRegion, Word, BoundingBox


class MockRecognizer(TextRecognizer):
    """Mock recognizer for testing"""

    def load_model(self):
        self.is_loaded = True

    def recognize_regions(self, regions):
        # Return mock words
        words = []
        for region in regions:
            word = Word(
                text="mock_text",
                bbox=region.bbox,
                confidence=0.95
            )
            words.append(word)
        return words

    @property
    def name(self):
        return "mock_recognizer"


class TestTextRecognizerBase:
    """Tests for TextRecognizer base class"""

    def test_recognizer_initialization_default_device(self):
        """Test recognizer initializes with default device"""
        recognizer = MockRecognizer()

        assert recognizer.model_path is None
        assert recognizer.device is not None
        assert not recognizer.is_loaded

    def test_recognizer_initialization_with_cuda(self):
        """Test recognizer with CUDA device"""
        recognizer = MockRecognizer(device='cuda')

        assert str(recognizer.device) == 'cuda'

    def test_recognizer_initialization_with_cpu(self):
        """Test recognizer with CPU device"""
        recognizer = MockRecognizer(device='cpu')

        assert str(recognizer.device) == 'cpu'

    def test_recognizer_initialization_with_model_path(self):
        """Test recognizer with custom model path"""
        recognizer = MockRecognizer(model_path='model/path')

        assert recognizer.model_path == 'model/path'

    def test_recognizer_name_property(self):
        """Test recognizer name property"""
        recognizer = MockRecognizer()

        assert recognizer.name == "mock_recognizer"

    def test_recognize_regions_returns_words(self):
        """Test recognize_regions returns list of Word objects"""
        recognizer = MockRecognizer()
        recognizer.load_model()

        # Create mock regions
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        crop = Image.new('RGB', (100, 50), color='white')
        region = TextRegion(bbox=bbox, crop=crop, confidence=1.0)

        words = recognizer.recognize_regions([region])

        assert isinstance(words, list)
        assert len(words) == 1
        assert isinstance(words[0], Word)
        assert words[0].text == "mock_text"

    def test_recognize_regions_batch(self):
        """Test recognize_regions handles multiple regions"""
        recognizer = MockRecognizer()
        recognizer.load_model()

        # Create multiple mock regions
        regions = []
        for i in range(5):
            bbox = BoundingBox(left=i*10, top=20, width=100, height=50)
            crop = Image.new('RGB', (100, 50), color='white')
            region = TextRegion(bbox=bbox, crop=crop, confidence=1.0)
            regions.append(region)

        words = recognizer.recognize_regions(regions)

        assert len(words) == 5
        assert all(isinstance(w, Word) for w in words)

    def test_recognize_regions_empty_list(self):
        """Test recognize_regions with empty list"""
        recognizer = MockRecognizer()
        recognizer.load_model()

        words = recognizer.recognize_regions([])

        assert isinstance(words, list)
        assert len(words) == 0

    def test_recognize_regions_preserves_bbox(self):
        """Test that recognized words preserve original bboxes"""
        recognizer = MockRecognizer()
        recognizer.load_model()

        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        crop = Image.new('RGB', (100, 50), color='white')
        region = TextRegion(bbox=bbox, crop=crop, confidence=1.0)

        words = recognizer.recognize_regions([region])

        assert words[0].bbox == bbox
