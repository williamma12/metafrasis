"""
Tests for OCR preprocessing utilities
"""
import pytest
from pathlib import Path
from PIL import Image
from services.ocr import preprocessing


class TestPreprocessing:
    """Tests for preprocessing utilities"""

    def test_pdf_to_images_function_exists(self):
        """Test that pdf_to_images function exists"""
        assert hasattr(preprocessing, 'pdf_to_images')
        assert callable(preprocessing.pdf_to_images)

    def test_pdf_to_images_with_invalid_path(self):
        """Test that pdf_to_images raises error with invalid path"""
        with pytest.raises(Exception):  # Could be FileNotFoundError or PDFPageCountError
            preprocessing.pdf_to_images("nonexistent.pdf")

    def test_pdf_to_images_with_non_pdf_file(self, tmp_path):
        """Test that pdf_to_images handles non-PDF files appropriately"""
        # Create a temporary text file
        text_file = tmp_path / "not_a_pdf.txt"
        text_file.write_text("This is not a PDF")

        with pytest.raises(Exception):  # Should raise some kind of error
            preprocessing.pdf_to_images(str(text_file))

    @pytest.mark.skipif(
        not Path("tests/test_ocr/fixtures/sample.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_pdf_to_images_with_valid_pdf(self):
        """Test pdf_to_images with a valid PDF file"""
        pdf_path = "tests/test_ocr/fixtures/sample.pdf"
        images = preprocessing.pdf_to_images(pdf_path)

        assert isinstance(images, list)
        assert len(images) > 0
        assert all(isinstance(img, Image.Image) for img in images)

    def test_pdf_to_images_accepts_dpi_parameter(self):
        """Test that pdf_to_images accepts dpi parameter"""
        import inspect
        sig = inspect.signature(preprocessing.pdf_to_images)

        assert 'dpi' in sig.parameters
