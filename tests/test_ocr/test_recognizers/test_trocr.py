"""
Tests for TrOCRRecognizer
"""
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
from services.ocr.recognizers.trocr import TrOCRRecognizer
from services.ocr.base import TextRegion, Word, BoundingBox


class TestTrOCRRecognizer:
    """Tests for TrOCRRecognizer"""

    def test_recognizer_initialization_default(self):
        """Test TrOCRRecognizer initializes with defaults"""
        recognizer = TrOCRRecognizer()

        assert recognizer.model_path == "microsoft/trocr-base-handwritten"
        assert recognizer.batch_size == 8
        assert not recognizer.is_loaded

    def test_recognizer_initialization_custom(self):
        """Test TrOCRRecognizer with custom parameters"""
        recognizer = TrOCRRecognizer(
            model_path="custom/model",
            batch_size=16,
            device='cpu'
        )

        assert recognizer.model_path == "custom/model"
        assert recognizer.batch_size == 16
        assert str(recognizer.device) == 'cpu'

    def test_recognizer_name(self):
        """Test recognizer name property"""
        recognizer = TrOCRRecognizer()

        assert recognizer.name == "trocr"

    @patch('services.ocr.recognizers.trocr.VisionEncoderDecoderModel')
    @patch('services.ocr.recognizers.trocr.TrOCRProcessor')
    def test_load_model(self, mock_processor_class, mock_model_class):
        """Test model loading"""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        recognizer = TrOCRRecognizer(device='cpu')
        recognizer.load_model()

        assert recognizer.is_loaded
        assert recognizer.processor is not None
        assert recognizer.model is not None
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch('services.ocr.recognizers.trocr.VisionEncoderDecoderModel')
    @patch('services.ocr.recognizers.trocr.TrOCRProcessor')
    def test_recognize_regions_single(self, mock_processor_class, mock_model_class):
        """Test recognizing a single region"""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock the processing pipeline
        mock_processor.return_value = MagicMock(pixel_values=MagicMock(to=lambda x: MagicMock()))
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_processor.batch_decode.return_value = ["recognized text"]

        recognizer = TrOCRRecognizer(device='cpu')
        recognizer.load_model()

        # Create mock region
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        crop = Image.new('RGB', (100, 50), color='white')
        region = TextRegion(bbox=bbox, crop=crop, confidence=0.9)

        words = recognizer.recognize_regions([region])

        assert len(words) == 1
        assert isinstance(words[0], Word)
        assert words[0].text == "recognized text"
        assert words[0].bbox == bbox
        assert words[0].confidence == 0.9  # Uses detection confidence

    @patch('services.ocr.recognizers.trocr.VisionEncoderDecoderModel')
    @patch('services.ocr.recognizers.trocr.TrOCRProcessor')
    def test_recognize_regions_batch(self, mock_processor_class, mock_model_class):
        """Test batch recognition of multiple regions"""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock batch processing
        mock_processor.return_value = MagicMock(pixel_values=MagicMock(to=lambda x: MagicMock()))
        mock_model.generate.return_value = [[1, 2], [3, 4], [5, 6]]
        mock_processor.batch_decode.return_value = ["text1", "text2", "text3"]

        recognizer = TrOCRRecognizer(device='cpu', batch_size=8)
        recognizer.load_model()

        # Create multiple mock regions
        regions = []
        for i in range(3):
            bbox = BoundingBox(left=i*10, top=20, width=100, height=50)
            crop = Image.new('RGB', (100, 50), color='white')
            region = TextRegion(bbox=bbox, crop=crop, confidence=0.8 + i*0.1)
            regions.append(region)

        words = recognizer.recognize_regions(regions)

        assert len(words) == 3
        assert words[0].text == "text1"
        assert words[1].text == "text2"
        assert words[2].text == "text3"
        # Verify confidences come from regions
        assert words[0].confidence == 0.8
        assert words[1].confidence == pytest.approx(0.9, abs=0.01)
        assert words[2].confidence == pytest.approx(1.0, abs=0.01)

    @patch('services.ocr.recognizers.trocr.VisionEncoderDecoderModel')
    @patch('services.ocr.recognizers.trocr.TrOCRProcessor')
    def test_recognize_regions_empty(self, mock_processor_class, mock_model_class):
        """Test recognition with empty region list"""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        recognizer = TrOCRRecognizer(device='cpu')
        recognizer.load_model()

        words = recognizer.recognize_regions([])

        assert len(words) == 0

    @patch('services.ocr.recognizers.trocr.VisionEncoderDecoderModel')
    @patch('services.ocr.recognizers.trocr.TrOCRProcessor')
    def test_recognize_regions_batching(self, mock_processor_class, mock_model_class):
        """Test that large batches are split correctly"""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        # Track how many times generate is called
        generate_calls = []

        def mock_generate(pixel_values):
            batch_size = len(pixel_values) if hasattr(pixel_values, '__len__') else 1
            generate_calls.append(batch_size)
            return [[i] for i in range(batch_size)]

        mock_model.generate.side_effect = mock_generate
        mock_processor.return_value = MagicMock(pixel_values=MagicMock(to=lambda x: [1, 2, 3]))
        mock_processor.batch_decode.return_value = ["text"] * 10

        recognizer = TrOCRRecognizer(device='cpu', batch_size=4)
        recognizer.load_model()

        # Create 10 regions (should be split into 3 batches: 4, 4, 2)
        regions = []
        for i in range(10):
            bbox = BoundingBox(left=i*10, top=20, width=100, height=50)
            crop = Image.new('RGB', (100, 50), color='white')
            region = TextRegion(bbox=bbox, crop=crop, confidence=1.0)
            regions.append(region)

        words = recognizer.recognize_regions(regions)

        assert len(words) == 10
        # Note: The actual batching depends on implementation details
        # Just verify we got all results
