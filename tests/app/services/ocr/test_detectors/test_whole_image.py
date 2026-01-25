"""
Tests for WholeImageDetector
"""
import pytest
from PIL import Image
from app.services.ocr.detectors.whole_image import WholeImageDetector
from app.services.ocr.base import TextRegion, BoundingBox


class TestWholeImageDetector:
    """Tests for WholeImageDetector"""

    def test_detector_initialization(self):
        """Test WholeImageDetector initializes ready to use"""
        detector = WholeImageDetector()

        assert detector.is_loaded is True
        assert detector.name == "whole_image"

    def test_load_model_no_op(self):
        """Test load_model is a no-op"""
        detector = WholeImageDetector()

        detector.load_model()

        assert detector.is_loaded is True

    def test_detect_returns_single_region(self, sample_image):
        """Test detect returns entire image as single region"""
        detector = WholeImageDetector()

        regions = detector.detect(sample_image)

        assert len(regions) == 1
        assert isinstance(regions[0], TextRegion)

    def test_detect_region_covers_whole_image(self, sample_image):
        """Test region bbox covers entire image"""
        detector = WholeImageDetector()
        width, height = sample_image.size

        regions = detector.detect(sample_image)
        region = regions[0]

        assert region.bbox.left == 0
        assert region.bbox.top == 0
        assert region.bbox.width == width
        assert region.bbox.height == height

    def test_detect_region_crop_is_original(self, sample_image):
        """Test region crop is the original image"""
        detector = WholeImageDetector()

        regions = detector.detect(sample_image)
        region = regions[0]

        assert region.crop == sample_image

    def test_detect_confidence_is_one(self, sample_image):
        """Test detection confidence is 1.0 (no actual detection)"""
        detector = WholeImageDetector()

        regions = detector.detect(sample_image)
        region = regions[0]

        assert region.confidence == 1.0

    def test_detect_batch(self, sample_images_batch):
        """Test batch detection returns one region per image"""
        detector = WholeImageDetector()

        all_regions = detector.detect_batch(sample_images_batch)

        assert len(all_regions) == len(sample_images_batch)
        assert all(len(regions) == 1 for regions in all_regions)

    def test_detect_different_sizes(self):
        """Test detector handles different image sizes"""
        detector = WholeImageDetector()

        small_img = Image.new('RGB', (100, 100), color='red')
        large_img = Image.new('RGB', (1000, 500), color='blue')

        small_regions = detector.detect(small_img)
        large_regions = detector.detect(large_img)

        assert small_regions[0].bbox.width == 100
        assert small_regions[0].bbox.height == 100

        assert large_regions[0].bbox.width == 1000
        assert large_regions[0].bbox.height == 500
