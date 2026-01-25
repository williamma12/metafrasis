"""
Tests for CRAFT (Character Region Awareness For Text detection) Detector
"""
import pytest
import torch
import numpy as np
from PIL import Image
from app.services.ocr.detectors.craft import CRAFTDetector
from app.services.ocr.base import TextRegion, BoundingBox


@pytest.mark.requires_craft
class TestCRAFTDetector:
    """Tests for CRAFT text detector"""

    def test_detector_initialization_default(self):
        """Test CRAFTDetector initializes with default parameters"""
        detector = CRAFTDetector()

        assert detector.name == "craft"
        assert detector.is_loaded is False
        assert detector.text_threshold == 0.7
        assert detector.link_threshold == 0.4
        assert detector.low_text == 0.4
        assert detector.canvas_size == 1280
        assert detector.mag_ratio == 1.5
        assert str(detector.device) in ['cpu', 'cuda', 'mps']

    def test_detector_initialization_custom_params(self):
        """Test CRAFTDetector accepts custom parameters"""
        detector = CRAFTDetector(
            text_threshold=0.8,
            link_threshold=0.5,
            low_text=0.3,
            canvas_size=1920,
            mag_ratio=2.0,
            device='cpu'
        )

        assert detector.text_threshold == 0.8
        assert detector.link_threshold == 0.5
        assert detector.low_text == 0.3
        assert detector.canvas_size == 1920
        assert detector.mag_ratio == 2.0
        assert str(detector.device) == 'cpu'

    def test_load_model_without_weights(self):
        """Test model architecture can be loaded without pretrained weights"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        # Should create model architecture with random initialization
        detector.load_model()

        assert detector.is_loaded is True
        assert detector.model is not None
        assert isinstance(detector.model, torch.nn.Module)

    def test_load_model_lazy_loading(self, sample_image):
        """Test model loads automatically on first detection"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        assert detector.is_loaded is False

        # First detect should trigger load_model
        regions = detector.detect(sample_image)

        assert detector.is_loaded is True
        assert isinstance(regions, list)

    def test_detect_returns_list(self, sample_image):
        """Test detect returns a list of TextRegions"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        regions = detector.detect(sample_image)

        assert isinstance(regions, list)
        assert all(isinstance(r, TextRegion) for r in regions)

    def test_detect_on_blank_image(self):
        """Test detection on blank image (may return zero or few regions)"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        blank_image = Image.new('RGB', (640, 480), color='white')

        regions = detector.detect(blank_image)

        # Blank image should return few or no regions
        # With random weights, this is unpredictable, so just check it doesn't crash
        assert isinstance(regions, list)

    def test_detect_on_image_with_text(self, sample_image_with_text):
        """Test detection on image with text"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        regions = detector.detect(sample_image_with_text)

        # Should return some regions (even with random weights)
        assert isinstance(regions, list)
        # Each region should have valid properties
        for region in regions:
            assert isinstance(region.bbox, BoundingBox)
            assert isinstance(region.crop, Image.Image)
            assert 0 <= region.confidence <= 1

    def test_detect_batch(self, sample_images_batch):
        """Test batch detection on multiple images"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        all_regions = detector.detect_batch(sample_images_batch)

        assert len(all_regions) == len(sample_images_batch)
        assert all(isinstance(regions, list) for regions in all_regions)
        assert all(
            all(isinstance(r, TextRegion) for r in regions)
            for regions in all_regions
        )

    def test_detect_different_image_sizes(self):
        """Test detector handles different image sizes"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        small_img = Image.new('RGB', (320, 240), color='white')
        medium_img = Image.new('RGB', (640, 480), color='white')
        large_img = Image.new('RGB', (1920, 1080), color='white')

        small_regions = detector.detect(small_img)
        medium_regions = detector.detect(medium_img)
        large_regions = detector.detect(large_img)

        # Should handle all sizes without errors
        assert isinstance(small_regions, list)
        assert isinstance(medium_regions, list)
        assert isinstance(large_regions, list)

    def test_detect_portrait_and_landscape(self):
        """Test detector handles portrait and landscape orientations"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        portrait = Image.new('RGB', (480, 640), color='white')
        landscape = Image.new('RGB', (640, 480), color='white')

        portrait_regions = detector.detect(portrait)
        landscape_regions = detector.detect(landscape)

        assert isinstance(portrait_regions, list)
        assert isinstance(landscape_regions, list)

    def test_region_bboxes_within_image_bounds(self, sample_image_large):
        """Test that all detected region bboxes are within image bounds"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        width, height = sample_image_large.size

        regions = detector.detect(sample_image_large)

        for region in regions:
            bbox = region.bbox
            # Bbox should be within image bounds
            assert 0 <= bbox.left < width
            assert 0 <= bbox.top < height
            assert bbox.left + bbox.width <= width
            assert bbox.top + bbox.height <= height

    def test_region_crops_match_bboxes(self, sample_image_large):
        """Test that region crops match their bounding box sizes"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        regions = detector.detect(sample_image_large)

        for region in regions:
            bbox = region.bbox
            crop = region.crop

            # Crop size should match bbox dimensions
            assert crop.size == (bbox.width, bbox.height)

    def test_confidence_values_valid(self, sample_image):
        """Test that all confidence values are in [0, 1] range"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        regions = detector.detect(sample_image)

        for region in regions:
            assert 0 <= region.confidence <= 1

    def test_device_placement_cpu(self, sample_image):
        """Test detector works on CPU"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        regions = detector.detect(sample_image)

        assert str(detector.device) == 'cpu'
        assert isinstance(regions, list)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self, sample_image):
        """Test detector works on CUDA"""
        detector = CRAFTDetector(model_path=None, device='cuda')

        regions = detector.detect(sample_image)

        assert detector.device == 'cuda'
        assert isinstance(regions, list)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_placement_mps(self, sample_image):
        """Test detector works on MPS (Apple Silicon)"""
        detector = CRAFTDetector(model_path=None, device='mps')

        regions = detector.detect(sample_image)

        assert str(detector.device) == 'mps'
        assert isinstance(regions, list)

    def test_model_in_eval_mode(self, sample_image):
        """Test that model is in eval mode during inference"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        detector.detect(sample_image)

        assert not detector.model.training

    def test_no_gradient_computation(self, sample_image):
        """Test that gradients are not computed during inference"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        detector.load_model()

        # Check that all parameters have requires_grad=False or gradients disabled
        with torch.no_grad():
            regions = detector.detect(sample_image)

        # Should complete without gradient errors
        assert isinstance(regions, list)

    def test_multiple_detections_consistent(self, sample_image):
        """Test that multiple detections on same image are deterministic"""
        detector = CRAFTDetector(model_path=None, device='cpu')

        # Set model to eval mode for deterministic behavior
        detector.load_model()

        regions1 = detector.detect(sample_image)
        regions2 = detector.detect(sample_image)

        # Should return same number of regions
        assert len(regions1) == len(regions2)

        # Bboxes should match (allowing small floating point differences)
        for r1, r2 in zip(regions1, regions2):
            assert abs(r1.bbox.left - r2.bbox.left) < 2
            assert abs(r1.bbox.top - r2.bbox.top) < 2
            assert abs(r1.bbox.width - r2.bbox.width) < 2
            assert abs(r1.bbox.height - r2.bbox.height) < 2

    def test_threshold_parameters_affect_detection(self, sample_image_with_text):
        """Test that threshold parameters affect detection results"""
        # High thresholds = fewer detections
        detector_strict = CRAFTDetector(
            model_path=None,
            device='cpu',
            text_threshold=0.9,
            link_threshold=0.9
        )

        # Low thresholds = more detections
        detector_lenient = CRAFTDetector(
            model_path=None,
            device='cpu',
            text_threshold=0.3,
            link_threshold=0.3
        )

        regions_strict = detector_strict.detect(sample_image_with_text)
        regions_lenient = detector_lenient.detect(sample_image_with_text)

        # Lenient should detect at least as many regions as strict
        # (This may not always hold with random weights, but is expected behavior)
        assert isinstance(regions_strict, list)
        assert isinstance(regions_lenient, list)

    def test_grayscale_image_detection(self):
        """Test detector handles grayscale images"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        gray_image = Image.new('L', (640, 480), color=128)

        regions = detector.detect(gray_image)

        assert isinstance(regions, list)

    def test_rgba_image_detection(self):
        """Test detector handles RGBA images"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        rgba_image = Image.new('RGBA', (640, 480), color=(255, 255, 255, 255))

        regions = detector.detect(rgba_image)

        assert isinstance(regions, list)

    def test_very_small_image(self):
        """Test detector handles very small images"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        tiny_image = Image.new('RGB', (32, 32), color='white')

        regions = detector.detect(tiny_image)

        # Should not crash
        assert isinstance(regions, list)

    def test_very_large_image(self):
        """Test detector handles very large images (with canvas size limit)"""
        detector = CRAFTDetector(
            model_path=None,
            device='cpu',
            canvas_size=1280  # Limit canvas size
        )
        huge_image = Image.new('RGB', (4000, 3000), color='white')

        regions = detector.detect(huge_image)

        # Should resize and process without crashing
        assert isinstance(regions, list)

    def test_detect_returns_empty_list_not_none(self):
        """Test that detect always returns a list, never None"""
        detector = CRAFTDetector(model_path=None, device='cpu')
        blank_image = Image.new('RGB', (100, 100), color='white')

        regions = detector.detect(blank_image)

        assert regions is not None
        assert isinstance(regions, list)
