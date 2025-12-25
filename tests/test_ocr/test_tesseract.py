"""
Tests for Tesseract OCR Engine
"""
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
from services.ocr.engines.tesseract import TesseractEngine
from services.ocr.base import OCRResult, Word, BoundingBox


class TestTesseractEngine:
    """Tests for TesseractEngine"""

    def test_engine_initialization_default_language(self):
        """Test TesseractEngine initializes with default Ancient Greek language"""
        engine = TesseractEngine()

        assert engine.lang == 'grc'
        assert engine.name == 'tesseract'
        assert not engine.is_loaded

    def test_engine_initialization_custom_language(self):
        """Test TesseractEngine with custom language"""
        engine = TesseractEngine(lang='eng')

        assert engine.lang == 'eng'
        assert engine.name == 'tesseract'

    def test_name_property(self):
        """Test engine name property"""
        engine = TesseractEngine()
        assert engine.name == 'tesseract'

    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_load_model_success(self, mock_get_languages):
        """Test successful model loading"""
        mock_get_languages.return_value = ['eng', 'grc']

        engine = TesseractEngine()
        engine.load_model()

        assert engine.is_loaded is True
        mock_get_languages.assert_called_once()

    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_load_model_tesseract_not_available(self, mock_get_languages):
        """Test that load_model raises error when Tesseract not available"""
        mock_get_languages.side_effect = Exception("Tesseract not found")

        engine = TesseractEngine()

        with pytest.raises(RuntimeError, match="Tesseract not available"):
            engine.load_model()

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_simple_text(self, mock_get_languages, mock_image_to_data):
        """Test recognition of simple text"""
        # Setup mocks
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': ['hello', 'world'],
            'conf': ['95', '87'],
            'left': [10, 60],
            'top': [20, 20],
            'width': [40, 50],
            'height': [30, 30]
        }

        # Create engine and test image
        engine = TesseractEngine()
        test_image = Image.new('RGB', (100, 100), color='white')

        # Recognize
        result = engine.recognize(test_image)

        # Assertions
        assert isinstance(result, OCRResult)
        assert len(result.words) == 2
        assert result.engine_name == 'tesseract'
        assert result.processing_time > 0

        # Check first word
        assert result.words[0].text == 'hello'
        assert result.words[0].confidence == pytest.approx(0.95, abs=0.01)
        assert result.words[0].bbox.left == 10
        assert result.words[0].bbox.top == 20
        assert result.words[0].bbox.width == 40
        assert result.words[0].bbox.height == 30

        # Check second word
        assert result.words[1].text == 'world'
        assert result.words[1].confidence == pytest.approx(0.87, abs=0.01)

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_filters_empty_text(self, mock_get_languages, mock_image_to_data):
        """Test that recognition filters out empty text boxes"""
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': ['hello', '', '  ', 'world'],
            'conf': ['95', '80', '75', '87'],
            'left': [10, 30, 50, 70],
            'top': [20, 20, 20, 20],
            'width': [40, 20, 30, 50],
            'height': [30, 30, 30, 30]
        }

        engine = TesseractEngine()
        test_image = Image.new('RGB', (100, 100), color='white')
        result = engine.recognize(test_image)

        # Should only have 2 words (empty and whitespace-only filtered out)
        assert len(result.words) == 2
        assert result.words[0].text == 'hello'
        assert result.words[1].text == 'world'

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_filters_invalid_confidence(self, mock_get_languages, mock_image_to_data):
        """Test that recognition filters out boxes with invalid confidence (-1)"""
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': ['hello', 'invalid', 'world'],
            'conf': ['95', '-1', '87'],  # -1 indicates invalid
            'left': [10, 30, 50],
            'top': [20, 20, 20],
            'width': [40, 20, 30],
            'height': [30, 30, 30]
        }

        engine = TesseractEngine()
        test_image = Image.new('RGB', (100, 100), color='white')
        result = engine.recognize(test_image)

        # Should only have 2 words (invalid confidence filtered out)
        assert len(result.words) == 2
        assert result.words[0].text == 'hello'
        assert result.words[1].text == 'world'

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_confidence_conversion(self, mock_get_languages, mock_image_to_data):
        """Test that confidence scores are converted from 0-100 to 0-1"""
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': ['test'],
            'conf': ['100'],  # 100% confidence
            'left': [10],
            'top': [20],
            'width': [40],
            'height': [30]
        }

        engine = TesseractEngine()
        test_image = Image.new('RGB', (100, 100), color='white')
        result = engine.recognize(test_image)

        # Should convert 100 to 1.0
        assert result.words[0].confidence == pytest.approx(1.0, abs=0.01)

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_lazy_loads_model(self, mock_get_languages, mock_image_to_data):
        """Test that recognize() lazy loads the model if not already loaded"""
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': ['hello'],
            'conf': ['95'],
            'left': [10],
            'top': [20],
            'width': [40],
            'height': [30]
        }

        engine = TesseractEngine()
        assert not engine.is_loaded

        test_image = Image.new('RGB', (100, 100), color='white')
        result = engine.recognize(test_image)

        # Model should be loaded after first recognition
        assert engine.is_loaded
        assert len(result.words) == 1

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_with_custom_language(self, mock_get_languages, mock_image_to_data):
        """Test recognition with custom language parameter"""
        mock_get_languages.return_value = ['eng', 'grc']
        mock_image_to_data.return_value = {
            'text': ['hello'],
            'conf': ['95'],
            'left': [10],
            'top': [20],
            'width': [40],
            'height': [30]
        }

        # Create engine with English language
        engine = TesseractEngine(lang='eng')
        test_image = Image.new('RGB', (100, 100), color='white')
        result = engine.recognize(test_image)

        # Verify pytesseract was called with correct language
        mock_image_to_data.assert_called_once()
        call_kwargs = mock_image_to_data.call_args[1]
        assert call_kwargs['lang'] == 'eng'

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_empty_result(self, mock_get_languages, mock_image_to_data):
        """Test recognition when no text is detected"""
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': [],
            'conf': [],
            'left': [],
            'top': [],
            'width': [],
            'height': []
        }

        engine = TesseractEngine()
        test_image = Image.new('RGB', (100, 100), color='white')
        result = engine.recognize(test_image)

        assert len(result.words) == 0
        assert result.engine_name == 'tesseract'
        assert isinstance(result, OCRResult)

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_batch(self, mock_get_languages, mock_image_to_data):
        """Test batch recognition (inherited from base class)"""
        mock_get_languages.return_value = ['grc']

        # Mock different results for each image
        mock_image_to_data.side_effect = [
            {
                'text': ['hello'],
                'conf': ['95'],
                'left': [10],
                'top': [20],
                'width': [40],
                'height': [30]
            },
            {
                'text': ['world'],
                'conf': ['87'],
                'left': [15],
                'top': [25],
                'width': [45],
                'height': [35]
            }
        ]

        engine = TesseractEngine()
        images = [
            Image.new('RGB', (100, 100), color='white'),
            Image.new('RGB', (100, 100), color='white')
        ]

        results = engine.recognize_batch(images)

        assert len(results) == 2
        assert all(isinstance(r, OCRResult) for r in results)
        assert results[0].words[0].text == 'hello'
        assert results[1].words[0].text == 'world'

    @patch('services.ocr.engines.tesseract.pytesseract.image_to_data')
    @patch('services.ocr.engines.tesseract.pytesseract.get_languages')
    def test_recognize_creates_correct_bounding_boxes(self, mock_get_languages, mock_image_to_data):
        """Test that bounding boxes are created correctly from Tesseract data"""
        mock_get_languages.return_value = ['grc']
        mock_image_to_data.return_value = {
            'text': ['test'],
            'conf': ['90'],
            'left': [100],
            'top': [200],
            'width': [300],
            'height': [400]
        }

        engine = TesseractEngine()
        test_image = Image.new('RGB', (1000, 1000), color='white')
        result = engine.recognize(test_image)

        bbox = result.words[0].bbox
        assert isinstance(bbox, BoundingBox)
        assert bbox.left == 100
        assert bbox.top == 200
        assert bbox.width == 300
        assert bbox.height == 400
