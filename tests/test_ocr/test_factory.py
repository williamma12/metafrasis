"""
Tests for OCR Engine Factory
"""
import pytest
from PIL import Image
from app.services.ocr.factory import OCREngineFactory
from app.services.ocr.base import OCREngine, OCRResult
from app.services.ocr.types import DetectorType, RecognizerType, EngineType


class TestOCREngineFactory:
    """Tests for OCREngineFactory"""

    def test_available_engines(self):
        """Test that available_engines returns list of engine names"""
        engines = OCREngineFactory.available_engines()

        assert isinstance(engines, list)
        assert len(engines) > 0

    def test_available_engines_contains_tesseract(self):
        """Test that tesseract is in available engines"""
        engines = OCREngineFactory.available_engines()
        assert 'tesseract' in engines

    def test_available_detectors(self):
        """Test that available_detectors returns list"""
        detectors = OCREngineFactory.available_detectors()

        assert isinstance(detectors, list)
        assert 'whole_image' in detectors

    def test_available_recognizers(self):
        """Test that available_recognizers returns list"""
        recognizers = OCREngineFactory.available_recognizers()

        assert isinstance(recognizers, list)
        assert 'trocr' in recognizers

    def test_create_monolithic_engine_with_string(self):
        """Test creating monolithic engine with string"""
        engine = OCREngineFactory.create(engine='tesseract')

        assert engine is not None
        assert engine.name == 'tesseract'
        assert isinstance(engine, OCREngine)

    def test_create_monolithic_engine_with_enum(self):
        """Test creating monolithic engine with enum"""
        engine = OCREngineFactory.create(engine=EngineType.TESSERACT)

        assert engine is not None
        assert engine.name == 'tesseract'

    def test_create_composed_engine_with_strings(self):
        """Test creating composed engine with string args"""
        engine = OCREngineFactory.create(
            detector='whole_image',
            recognizer='trocr',
            device='cpu'
        )

        assert engine is not None
        assert engine.name == "whole_image_trocr"
        assert isinstance(engine, OCREngine)

    def test_create_composed_engine_with_enums(self):
        """Test creating composed engine with enum args"""
        engine = OCREngineFactory.create(
            detector=DetectorType.WHOLE_IMAGE,
            recognizer=RecognizerType.TROCR,
            device='cpu'
        )

        assert engine is not None
        assert engine.name == "whole_image_trocr"

    def test_create_with_both_engine_and_detector_raises(self):
        """Test that specifying both engine and detector raises error"""
        with pytest.raises(ValueError, match="Cannot specify both"):
            OCREngineFactory.create(
                engine='tesseract',
                detector='whole_image',
                recognizer='trocr'
            )

    def test_create_with_neither_raises(self):
        """Test that specifying neither engine nor detector/recognizer raises"""
        with pytest.raises(ValueError, match="Must specify either"):
            OCREngineFactory.create()

    def test_create_with_detector_only_raises(self):
        """Test that detector without recognizer raises error"""
        with pytest.raises(ValueError, match="Must specify either"):
            OCREngineFactory.create(detector='whole_image')

    def test_create_with_recognizer_only_raises(self):
        """Test that recognizer without detector raises error"""
        with pytest.raises(ValueError, match="Must specify either"):
            OCREngineFactory.create(recognizer='trocr')

    def test_create_with_invalid_engine_name(self):
        """Test that creating with invalid engine name raises ValueError"""
        with pytest.raises(ValueError, match="Unknown engine"):
            OCREngineFactory.create(engine='nonexistent_engine')

    def test_create_with_invalid_detector_name(self):
        """Test that creating with invalid detector raises ValueError"""
        with pytest.raises(ValueError, match="Unknown detector"):
            OCREngineFactory.create(
                detector='nonexistent_detector',
                recognizer='trocr'
            )

    def test_create_with_invalid_recognizer_name(self):
        """Test that creating with invalid recognizer raises ValueError"""
        with pytest.raises(ValueError, match="Unknown recognizer"):
            OCREngineFactory.create(
                detector='whole_image',
                recognizer='nonexistent_recognizer'
            )

    def test_register_engine(self):
        """Test registering a new engine"""
        class CustomEngine(OCREngine):
            def load_model(self):
                pass

            def recognize(self, image):
                pass

            @property
            def name(self):
                return "custom"

        OCREngineFactory.register_engine('custom', CustomEngine)

        assert 'custom' in OCREngineFactory.available_engines()

        engine = OCREngineFactory.create(engine='custom')
        assert engine.name == "custom"

    def test_error_message_shows_available_options(self):
        """Test that error messages show available options"""
        with pytest.raises(ValueError) as exc_info:
            OCREngineFactory.create(engine='invalid_engine')

        error_message = str(exc_info.value)
        assert "invalid_engine" in error_message
        assert "Available engines:" in error_message
