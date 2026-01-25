"""
Shared pytest fixtures for OCR tests
"""
import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from app.services.ocr.base import BoundingBox, Word, OCRResult, ConfidenceStats, TextRegion


@pytest.fixture
def sample_image():
    """Create a simple test image (100x100 red)"""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_image_large():
    """Create a larger test image (500x500 blue)"""
    return Image.new('RGB', (500, 500), color='blue')


@pytest.fixture
def sample_image_with_text():
    """Create a test image with simple text"""
    # Create white background
    img = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(img)

    # Add simple text (using default font)
    draw.text((10, 40), "Hello World", fill='black')

    return img


@pytest.fixture
def sample_images_batch():
    """Create a batch of test images with different colors"""
    return [
        Image.new('RGB', (100, 100), color='red'),
        Image.new('RGB', (100, 100), color='green'),
        Image.new('RGB', (100, 100), color='blue'),
    ]


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box"""
    return BoundingBox(left=10, top=20, width=100, height=50)


@pytest.fixture
def sample_text_region(sample_bbox, sample_image):
    """Create a sample TextRegion"""
    return TextRegion(
        bbox=sample_bbox,
        crop=sample_image,
        confidence=0.95
    )


@pytest.fixture
def sample_text_regions(sample_image):
    """Create multiple sample TextRegions"""
    regions = []
    for i in range(3):
        bbox = BoundingBox(left=i*30, top=20, width=100, height=50)
        crop = sample_image.crop((i*30, 20, i*30+100, 70))
        region = TextRegion(bbox=bbox, crop=crop, confidence=0.9 + i*0.05)
        regions.append(region)
    return regions


@pytest.fixture
def sample_word(sample_bbox):
    """Create a sample Word with bounding box"""
    return Word(text="test", bbox=sample_bbox, confidence=0.95)


@pytest.fixture
def sample_words(sample_bbox):
    """Create a list of sample Words"""
    return [
        Word(text="hello", bbox=sample_bbox, confidence=0.95),
        Word(text="world", bbox=sample_bbox, confidence=0.87),
        Word(text="test", bbox=sample_bbox, confidence=0.92),
    ]


@pytest.fixture
def sample_ocr_result(sample_words):
    """Create a sample OCRResult"""
    return OCRResult(
        words=sample_words,
        engine_name="test_engine",
        processing_time=1.5,
        source="test.png"
    )


@pytest.fixture
def sample_ocr_result_empty():
    """Create an empty OCRResult (no words detected)"""
    return OCRResult(
        words=[],
        engine_name="test_engine",
        processing_time=0.5,
        source="empty.png"
    )


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_path(fixtures_dir):
    """Return path to sample PDF if it exists, otherwise skip test"""
    pdf_path = fixtures_dir / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not available")
    return str(pdf_path)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory"""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_tesseract_data():
    """Mock data structure returned by pytesseract.image_to_data"""
    return {
        'text': ['hello', 'world', 'test'],
        'conf': ['95', '87', '92'],
        'left': [10, 60, 120],
        'top': [20, 20, 20],
        'width': [40, 50, 35],
        'height': [30, 30, 30]
    }


@pytest.fixture
def mock_tesseract_data_with_invalid():
    """Mock Tesseract data with invalid/empty entries"""
    return {
        'text': ['hello', '', '  ', 'world', 'invalid'],
        'conf': ['95', '80', '75', '87', '-1'],  # -1 indicates invalid
        'left': [10, 30, 50, 70, 90],
        'top': [20, 20, 20, 20, 20],
        'width': [40, 20, 30, 50, 40],
        'height': [30, 30, 30, 30, 30]
    }


@pytest.fixture
def mock_tesseract_data_empty():
    """Mock empty Tesseract data (no text detected)"""
    return {
        'text': [],
        'conf': [],
        'left': [],
        'top': [],
        'width': [],
        'height': []
    }


@pytest.fixture
def confidence_values_valid():
    """Sample valid confidence values for testing statistics"""
    return [0.90, 0.85, 0.95, 0.88, 0.92]


@pytest.fixture
def confidence_values_mixed():
    """Sample mixed confidence values (valid and default)"""
    from app.services.ocr.base import DEFAULT_CONFIDENCE
    return [0.90, DEFAULT_CONFIDENCE, 0.85, DEFAULT_CONFIDENCE, 0.95]


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Automatically cleanup temporary files after each test"""
    yield
    # Cleanup happens automatically with tmp_path fixture


# Markers for conditional test execution
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_tesseract: marks tests that require Tesseract installation"
    )
    config.addinivalue_line(
        "markers", "requires_trocr: marks tests that require trOCR dependencies"
    )
    config.addinivalue_line(
        "markers", "requires_pdf: marks tests that require sample PDF file"
    )
    config.addinivalue_line(
        "markers", "requires_craft: marks tests that require CRAFT dependencies"
    )
    config.addinivalue_line(
        "markers", "requires_crnn: marks tests that require CRNN dependencies"
    )
