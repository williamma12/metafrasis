"""
Tests for CRNN (Convolutional Recurrent Neural Network) Recognizer
"""
import pytest
import torch
import numpy as np
from PIL import Image
from services.ocr.recognizers.crnn import CRNNRecognizer, CRNN
from services.ocr.base import TextRegion, Word, BoundingBox


@pytest.mark.requires_crnn
class TestCRNNRecognizer:
    """Tests for CRNN text recognizer"""

    def test_recognizer_initialization_default(self):
        """Test CRNNRecognizer initializes with default parameters"""
        recognizer = CRNNRecognizer()

        assert recognizer.name == "crnn"
        assert recognizer.is_loaded is False
        assert recognizer.batch_size == 16
        assert recognizer.charset == "0123456789abcdefghijklmnopqrstuvwxyz"
        assert recognizer.num_classes == 37  # 36 chars + blank
        assert recognizer.blank_token == 36
        assert recognizer.img_height == 32
        assert recognizer.img_width == 100
        assert recognizer.num_channels == 1
        assert str(recognizer.device) in ['cpu', 'cuda', 'mps']

    def test_recognizer_initialization_custom_params(self):
        """Test CRNNRecognizer accepts custom parameters"""
        custom_charset = "abc123"
        recognizer = CRNNRecognizer(
            model_path="/path/to/weights.pth",
            device='cpu',
            batch_size=32,
            charset=custom_charset,
            img_height=64,
            img_width=200,
            num_channels=3
        )

        assert recognizer.model_path == "/path/to/weights.pth"
        assert recognizer.batch_size == 32
        assert recognizer.charset == custom_charset
        assert recognizer.num_classes == 7  # 6 chars + blank
        assert recognizer.blank_token == 6
        assert recognizer.img_height == 64
        assert recognizer.img_width == 200
        assert recognizer.num_channels == 3
        assert str(recognizer.device) == 'cpu'

    def test_recognizer_name(self):
        """Test recognizer name property"""
        recognizer = CRNNRecognizer()

        assert recognizer.name == "crnn"

    def test_charset_mappings(self):
        """Test character-to-index and index-to-character mappings"""
        charset = "abc"
        recognizer = CRNNRecognizer(charset=charset)

        # Check char_to_idx
        assert recognizer.char_to_idx['a'] == 0
        assert recognizer.char_to_idx['b'] == 1
        assert recognizer.char_to_idx['c'] == 2

        # Check idx_to_char
        assert recognizer.idx_to_char[0] == 'a'
        assert recognizer.idx_to_char[1] == 'b'
        assert recognizer.idx_to_char[2] == 'c'

    def test_load_model_without_weights(self):
        """Test model architecture can be loaded without pretrained weights"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        recognizer.load_model()

        assert recognizer.is_loaded is True
        assert recognizer.model is not None
        assert isinstance(recognizer.model, CRNN)
        assert not recognizer.model.training  # Should be in eval mode

    def test_load_model_lazy_loading(self, sample_text_region):
        """Test model loads automatically on first recognition"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        assert recognizer.is_loaded is False

        # First recognize should trigger load_model
        words = recognizer.recognize_regions([sample_text_region])

        assert recognizer.is_loaded is True
        assert isinstance(words, list)

    def test_recognize_regions_single(self):
        """Test recognizing a single region"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        crop = Image.new('RGB', (100, 50), color='white')
        region = TextRegion(bbox=bbox, crop=crop, confidence=0.9)

        words = recognizer.recognize_regions([region])

        assert len(words) == 1
        assert isinstance(words[0], Word)
        assert isinstance(words[0].text, str)
        assert words[0].bbox == bbox
        assert words[0].confidence == 0.9  # Uses detection confidence

    def test_recognize_regions_batch(self):
        """Test batch recognition of multiple regions"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu', batch_size=8)

        # Create multiple regions
        regions = []
        for i in range(5):
            bbox = BoundingBox(left=i*10, top=20, width=100, height=50)
            crop = Image.new('RGB', (100, 50), color='white')
            region = TextRegion(bbox=bbox, crop=crop, confidence=0.8 + i*0.05)
            regions.append(region)

        words = recognizer.recognize_regions(regions)

        assert len(words) == 5
        assert all(isinstance(w, Word) for w in words)
        assert all(isinstance(w.text, str) for w in words)

        # Verify confidences come from regions
        assert words[0].confidence == 0.8
        assert words[1].confidence == pytest.approx(0.85, abs=0.01)
        assert words[2].confidence == pytest.approx(0.9, abs=0.01)
        assert words[3].confidence == pytest.approx(0.95, abs=0.01)
        assert words[4].confidence == pytest.approx(1.0, abs=0.01)

    def test_recognize_regions_empty(self):
        """Test recognition with empty region list"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        words = recognizer.recognize_regions([])

        assert len(words) == 0
        assert isinstance(words, list)

    def test_recognize_regions_large_batch(self):
        """Test that large batches are split correctly"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            batch_size=4
        )

        # Create 10 regions (should be split into batches of 4, 4, 2)
        regions = []
        for i in range(10):
            bbox = BoundingBox(left=i*10, top=20, width=100, height=50)
            crop = Image.new('RGB', (100, 50), color='white')
            region = TextRegion(bbox=bbox, crop=crop, confidence=1.0)
            regions.append(region)

        words = recognizer.recognize_regions(regions)

        # Should get all 10 words back
        assert len(words) == 10
        assert all(isinstance(w, Word) for w in words)

    def test_image_preprocessing_grayscale(self):
        """Test preprocessing converts to grayscale when num_channels=1"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            num_channels=1,
            img_height=32,
            img_width=100
        )
        recognizer.load_model()

        # RGB image
        rgb_image = Image.new('RGB', (200, 64), color=(255, 128, 0))

        # Preprocess
        tensor = recognizer._preprocess_image(rgb_image)

        # Should be grayscale: [1, 32, 100]
        assert tensor.shape == (1, 32, 100)

    def test_image_preprocessing_rgb(self):
        """Test preprocessing keeps RGB when num_channels=3"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            num_channels=3,
            img_height=32,
            img_width=100
        )
        recognizer.load_model()

        # RGB image
        rgb_image = Image.new('RGB', (200, 64), color=(255, 128, 0))

        # Preprocess
        tensor = recognizer._preprocess_image(rgb_image)

        # Should be RGB: [3, 32, 100]
        assert tensor.shape == (3, 32, 100)

    def test_image_preprocessing_resize_height(self):
        """Test preprocessing resizes to target height"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            img_height=32,
            img_width=200
        )
        recognizer.load_model()

        # Tall image
        image = Image.new('RGB', (100, 64), color='white')

        tensor = recognizer._preprocess_image(image)

        # Height should be 32
        assert tensor.shape[1] == 32

    def test_image_preprocessing_padding(self):
        """Test preprocessing pads narrow images"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            img_height=32,
            img_width=200
        )
        recognizer.load_model()

        # Narrow image (will need padding)
        image = Image.new('RGB', (50, 32), color='white')

        tensor = recognizer._preprocess_image(image)

        # Width should be padded to 200
        assert tensor.shape[2] == 200

    def test_image_preprocessing_cropping(self):
        """Test preprocessing crops wide images"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            img_height=32,
            img_width=100
        )
        recognizer.load_model()

        # Very wide image (will need cropping)
        image = Image.new('RGB', (500, 32), color='white')

        tensor = recognizer._preprocess_image(image)

        # Width should be cropped to 100
        assert tensor.shape[2] == 100

    def test_ctc_decode_simple(self):
        """Test CTC decoding with simple sequence"""
        recognizer = CRNNRecognizer(
            charset="abc",
            device='cpu'
        )

        # Sequence: blank, a, a, blank, b, b, b, blank, c
        # Should decode to: "abc"
        indices = np.array([3, 0, 0, 3, 1, 1, 1, 3, 2])

        text = recognizer._ctc_decode(indices)

        assert text == "abc"

    def test_ctc_decode_with_blanks(self):
        """Test CTC decoding removes blank tokens"""
        recognizer = CRNNRecognizer(charset="ab", device='cpu')

        # Sequence: blank, blank, a, blank, blank
        # Should decode to: "a"
        blank = recognizer.blank_token
        indices = np.array([blank, blank, 0, blank, blank])

        text = recognizer._ctc_decode(indices)

        assert text == "a"

    def test_ctc_decode_repeated_characters(self):
        """Test CTC decoding merges repeated characters"""
        recognizer = CRNNRecognizer(charset="ab", device='cpu')

        # Sequence: a, a, a, b, b
        # Should decode to: "ab"
        indices = np.array([0, 0, 0, 1, 1])

        text = recognizer._ctc_decode(indices)

        assert text == "ab"

    def test_ctc_decode_blank_separation(self):
        """Test CTC decoding: blanks allow same character to repeat"""
        recognizer = CRNNRecognizer(charset="ab", device='cpu')
        blank = recognizer.blank_token

        # Sequence: a, blank, a
        # Should decode to: "aa" (blank separates the two 'a's)
        indices = np.array([0, blank, 0])

        text = recognizer._ctc_decode(indices)

        assert text == "aa"

    def test_ctc_decode_all_blanks(self):
        """Test CTC decoding with all blank tokens returns empty string"""
        recognizer = CRNNRecognizer(charset="abc", device='cpu')
        blank = recognizer.blank_token

        indices = np.array([blank, blank, blank, blank])

        text = recognizer._ctc_decode(indices)

        assert text == ""

    def test_ctc_decode_empty_sequence(self):
        """Test CTC decoding with empty sequence"""
        recognizer = CRNNRecognizer(charset="abc", device='cpu')

        indices = np.array([])

        text = recognizer._ctc_decode(indices)

        assert text == ""

    def test_device_placement_cpu(self, sample_text_region):
        """Test recognizer works on CPU"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        words = recognizer.recognize_regions([sample_text_region])

        assert str(recognizer.device) == 'cpu'
        assert isinstance(words, list)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self, sample_text_region):
        """Test recognizer works on CUDA"""
        recognizer = CRNNRecognizer(model_path=None, device='cuda')

        words = recognizer.recognize_regions([sample_text_region])

        assert recognizer.device == 'cuda'
        assert isinstance(words, list)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_placement_mps(self, sample_text_region):
        """Test recognizer works on MPS (Apple Silicon)"""
        recognizer = CRNNRecognizer(model_path=None, device='mps')

        words = recognizer.recognize_regions([sample_text_region])

        assert str(recognizer.device) == 'mps'
        assert isinstance(words, list)

    def test_model_in_eval_mode(self, sample_text_region):
        """Test that model is in eval mode during inference"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        recognizer.recognize_regions([sample_text_region])

        assert not recognizer.model.training

    def test_no_gradient_computation(self, sample_text_region):
        """Test that gradients are not computed during inference"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')
        recognizer.load_model()

        # Should complete without gradient errors
        with torch.no_grad():
            words = recognizer.recognize_regions([sample_text_region])

        assert isinstance(words, list)

    def test_different_image_sizes_in_batch(self):
        """Test batch processing with different sized images"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        # Different sized regions
        regions = [
            TextRegion(
                bbox=BoundingBox(left=0, top=0, width=100, height=50),
                crop=Image.new('RGB', (100, 50), color='white'),
                confidence=1.0
            ),
            TextRegion(
                bbox=BoundingBox(left=0, top=0, width=200, height=100),
                crop=Image.new('RGB', (200, 100), color='white'),
                confidence=1.0
            ),
            TextRegion(
                bbox=BoundingBox(left=0, top=0, width=50, height=25),
                crop=Image.new('RGB', (50, 25), color='white'),
                confidence=1.0
            ),
        ]

        words = recognizer.recognize_regions(regions)

        # Should handle all sizes
        assert len(words) == 3

    def test_grayscale_region_images(self):
        """Test recognition on grayscale region images"""
        recognizer = CRNNRecognizer(
            model_path=None,
            device='cpu',
            num_channels=1
        )

        gray_crop = Image.new('L', (100, 50), color=128)
        region = TextRegion(
            bbox=BoundingBox(left=0, top=0, width=100, height=50),
            crop=gray_crop,
            confidence=1.0
        )

        words = recognizer.recognize_regions([region])

        assert len(words) == 1
        assert isinstance(words[0].text, str)

    def test_very_narrow_region(self):
        """Test recognition on very narrow text region"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        narrow_crop = Image.new('RGB', (10, 32), color='white')
        region = TextRegion(
            bbox=BoundingBox(left=0, top=0, width=10, height=32),
            crop=narrow_crop,
            confidence=1.0
        )

        words = recognizer.recognize_regions([region])

        assert len(words) == 1

    def test_very_wide_region(self):
        """Test recognition on very wide text region"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        wide_crop = Image.new('RGB', (500, 32), color='white')
        region = TextRegion(
            bbox=BoundingBox(left=0, top=0, width=500, height=32),
            crop=wide_crop,
            confidence=1.0
        )

        words = recognizer.recognize_regions([region])

        assert len(words) == 1

    def test_very_tall_region(self):
        """Test recognition on very tall text region"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        tall_crop = Image.new('RGB', (100, 200), color='white')
        region = TextRegion(
            bbox=BoundingBox(left=0, top=0, width=100, height=200),
            crop=tall_crop,
            confidence=1.0
        )

        words = recognizer.recognize_regions([region])

        # Should resize to target height
        assert len(words) == 1

    def test_tiny_region(self):
        """Test recognition on tiny text region"""
        recognizer = CRNNRecognizer(model_path=None, device='cpu')

        tiny_crop = Image.new('RGB', (8, 8), color='white')
        region = TextRegion(
            bbox=BoundingBox(left=0, top=0, width=8, height=8),
            crop=tiny_crop,
            confidence=1.0
        )

        words = recognizer.recognize_regions([region])

        # Should handle upscaling
        assert len(words) == 1

    def test_decode_batch_shape(self):
        """Test batch decoding with correct tensor shapes"""
        recognizer = CRNNRecognizer(charset="abc", device='cpu')

        # Create mock predictions: [T=10, B=2, num_classes=4]
        T, B, num_classes = 10, 2, 4
        preds = torch.randn(T, B, num_classes)

        texts = recognizer._decode_batch(preds)

        assert len(texts) == B
        assert all(isinstance(t, str) for t in texts)

    def test_batch_size_attribute_exists(self):
        """Test that batch_size attribute is set correctly"""
        recognizer = CRNNRecognizer(batch_size=32)

        # This was the bug that was fixed
        assert hasattr(recognizer, 'batch_size')
        assert recognizer.batch_size == 32

    def test_model_architecture_correct_num_classes(self):
        """Test that model is created with correct number of output classes"""
        charset = "abc"
        recognizer = CRNNRecognizer(
            charset=charset,
            device='cpu'
        )
        recognizer.load_model()

        # Model should have num_classes = len(charset) + 1 (for blank)
        assert recognizer.model.num_classes == 4  # 3 chars + blank
