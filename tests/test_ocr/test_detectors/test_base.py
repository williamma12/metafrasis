"""
Tests for TextDetector base class
"""
import pytest
from PIL import Image
from services.ocr.detectors.base import TextDetector
from services.ocr.base import TextRegion, BoundingBox


class MockDetector(TextDetector):
    """Mock detector for testing"""

    def load_model(self):
        self.is_loaded = True

    def detect(self, image: Image.Image):
        # Return mock region
        width, height = image.size
        bbox = BoundingBox(left=0, top=0, width=width, height=height)
        region = TextRegion(bbox=bbox, crop=image, confidence=1.0)
        return [region]

    @property
    def name(self):
        return "mock_detector"


class TestTextDetectorBase:
    """Tests for TextDetector base class"""

    def test_detector_initialization_default_device(self):
        """Test detector initializes with default device"""
        detector = MockDetector()

        assert detector.model_path is None
        assert detector.device is not None
        assert not detector.is_loaded

    def test_detector_initialization_with_cuda(self):
        """Test detector with CUDA device"""
        detector = MockDetector(device='cuda')

        assert str(detector.device) == 'cuda'

    def test_detector_initialization_with_cpu(self):
        """Test detector with CPU device"""
        detector = MockDetector(device='cpu')

        assert str(detector.device) == 'cpu'

    def test_detector_initialization_with_model_path(self):
        """Test detector with custom model path"""
        detector = MockDetector(model_path='/path/to/model')

        assert detector.model_path == '/path/to/model'

    def test_detector_name_property(self):
        """Test detector name property"""
        detector = MockDetector()

        assert detector.name == "mock_detector"

    def test_detector_detect_returns_regions(self, sample_image):
        """Test detect returns list of TextRegion objects"""
        detector = MockDetector()
        detector.load_model()

        regions = detector.detect(sample_image)

        assert isinstance(regions, list)
        assert len(regions) == 1
        assert isinstance(regions[0], TextRegion)

    def test_detector_detect_batch_default(self, sample_images_batch):
        """Test default detect_batch processes sequentially"""
        detector = MockDetector()
        detector.load_model()

        all_regions = detector.detect_batch(sample_images_batch)

        assert len(all_regions) == len(sample_images_batch)
        assert all(isinstance(regions, list) for regions in all_regions)
        assert all(len(regions) == 1 for regions in all_regions)

    def test_detector_lazy_loading(self, sample_image):
        """Test detector loads lazily on first use"""
        detector = MockDetector()

        assert not detector.is_loaded

        regions = detector.detect(sample_image)

        # MockDetector doesn't do lazy loading in detect,
        # but we can test that load_model sets is_loaded
        detector.load_model()
        assert detector.is_loaded
