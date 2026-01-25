"""
Shared pytest fixtures for backend page tests
"""
import pytest
from unittest.mock import MagicMock, Mock
from PIL import Image

from app.backend.state import AppState, AnnotationState
from app.services.ocr.base import OCRResult, Word, BoundingBox
from app.services.ocr.cache import ImageCache
from app.services.annotation.models import AnnotationDataset, AnnotatedImage, Region


# Constants for test data
TEST_IMAGE_WIDTH = 400
TEST_IMAGE_HEIGHT_SMALL = 100
TEST_IMAGE_HEIGHT_LARGE = 200
TEST_BBOX_LEFT = 10
TEST_BBOX_TOP = 10
TEST_BBOX_WIDTH = 50
TEST_BBOX_HEIGHT = 20

# Column mocking constants
EXTRA_COLUMN_BUFFER = 5  # Number of extra columns beyond expected count


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit module for UI testing."""
    mock_st = MagicMock()

    # Mock sidebar
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.header = MagicMock()
    mock_st.sidebar.radio = MagicMock(return_value="Monolithic Engine")
    mock_st.sidebar.selectbox = MagicMock(return_value="tesseract")
    mock_st.sidebar.checkbox = MagicMock(return_value=False)
    mock_st.sidebar.info = MagicMock()
    mock_st.sidebar.error = MagicMock()
    mock_st.sidebar.warning = MagicMock()
    mock_st.sidebar.divider = MagicMock()
    mock_st.sidebar.markdown = MagicMock()
    mock_st.sidebar.caption = MagicMock()
    mock_st.sidebar.download_button = MagicMock()
    mock_st.sidebar.button = MagicMock(return_value=False)
    mock_st.sidebar.text_area = MagicMock(return_value="")
    mock_st.sidebar.text_input = MagicMock(return_value="")
    mock_st.sidebar.subheader = MagicMock()

    # Mock main UI elements
    mock_st.file_uploader = MagicMock(return_value=None)
    mock_st.info = MagicMock()
    mock_st.success = MagicMock()
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.button = MagicMock(return_value=False)
    mock_st.toggle = MagicMock(return_value=False)
    mock_st.columns = MagicMock()
    mock_st.divider = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.checkbox = MagicMock(return_value=False)
    mock_st.image = MagicMock()
    mock_st.text_area = MagicMock()
    mock_st.text = MagicMock()
    mock_st.rerun = MagicMock()

    # Mock expander context manager
    mock_expander = MagicMock()
    mock_expander.__enter__ = Mock(return_value=mock_st)
    mock_expander.__exit__ = Mock(return_value=None)
    mock_st.expander = MagicMock(return_value=mock_expander)
    mock_st.sidebar.expander = MagicMock(return_value=mock_expander)

    # Mock session state
    mock_st.session_state = MagicMock()
    mock_st.session_state.current_image_idx = 0

    return mock_st


@pytest.fixture
def mock_streamlit_columns():
    """
    Factory fixture that creates column mocks with extra columns beyond expected.

    Usage:
        columns = mock_streamlit_columns(5)  # Creates 5 + EXTRA_COLUMN_BUFFER columns

    Tests can verify:
    1. Expected buttons/content exists (via label-based helpers)
    2. Extra columns beyond expected count are unused

    To change the buffer size, update EXTRA_COLUMN_BUFFER constant.
    """
    def _create_columns(expected_count):
        cols = []
        for _ in range(expected_count + EXTRA_COLUMN_BUFFER):
            col = MagicMock()
            # Ensure methods return sensible defaults
            col.button = MagicMock(return_value=False)
            col.markdown = MagicMock()
            col.toggle = MagicMock(return_value=False)
            col.metric = MagicMock()
            col.text = MagicMock()
            col.checkbox = MagicMock(return_value=False)
            cols.append(col)
        return cols
    return _create_columns


@pytest.fixture
def sample_ocr_result_with_words():
    """Create OCR result with multiple words."""
    words = [
        Word(text="Hello", bbox=BoundingBox(TEST_BBOX_LEFT, TEST_BBOX_TOP, TEST_BBOX_WIDTH, TEST_BBOX_HEIGHT)),
        Word(text="World", bbox=BoundingBox(70, TEST_BBOX_TOP, TEST_BBOX_WIDTH, TEST_BBOX_HEIGHT)),
    ]
    return OCRResult(
        words=words,
        engine_name="test_engine",
        processing_time=0.5,
        source="test1.png"
    )


@pytest.fixture
def sample_ocr_result_single_word():
    """Create OCR result with single word."""
    words = [
        Word(text="Test", bbox=BoundingBox(TEST_BBOX_LEFT, TEST_BBOX_TOP, 40, TEST_BBOX_HEIGHT)),
    ]
    return OCRResult(
        words=words,
        engine_name="test_engine",
        processing_time=0.3,
        source="test2.png"
    )


@pytest.fixture
def sample_ocr_results(sample_ocr_result_with_words, sample_ocr_result_single_word):
    """Create list of OCR results."""
    return [sample_ocr_result_with_words, sample_ocr_result_single_word]


@pytest.fixture
def sample_test_image():
    """Create simple test image."""
    return Image.new("RGB", (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT_SMALL), color="white")


@pytest.fixture
def sample_test_image_large():
    """Create larger test image."""
    return Image.new("RGB", (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT_LARGE), color="white")


@pytest.fixture
def sample_test_images(sample_test_image, sample_test_image_large):
    """Create list of test images."""
    blank = Image.new("RGB", (200, TEST_IMAGE_HEIGHT_SMALL), color="white")
    return [sample_test_image, sample_test_image_large, blank]


@pytest.fixture
def app_state_with_ocr_results(sample_ocr_results, sample_test_images):
    """Create AppState with OCR results and image cache."""
    state = AppState()
    state.ocr_results = sample_ocr_results
    state.image_cache = ImageCache()
    state.image_cache.add_images(sample_test_images[:2])  # Match number of results
    return state


@pytest.fixture
def app_state_empty():
    """Create empty AppState."""
    return AppState()


@pytest.fixture
def annotation_state_empty():
    """Create empty AnnotationState."""
    return AnnotationState()


@pytest.fixture
def annotation_state_with_dataset(sample_annotation_dataset):
    """Create AnnotationState with loaded dataset."""
    state = AnnotationState()
    state.dataset = sample_annotation_dataset
    state.current_image_idx = 0
    state.drawing_mode = "select"
    state.selected_region_id = None
    state.auto_detect_enabled = False
    state.unsaved_changes = False
    return state


@pytest.fixture
def sample_annotation_dataset():
    """Create sample annotation dataset with images and regions."""
    dataset = AnnotationDataset(name="test_dataset", version="1.0")

    # Add image with regions
    img = AnnotatedImage(
        image_path="images/test.png",
        width=TEST_IMAGE_WIDTH,
        height=TEST_IMAGE_HEIGHT_LARGE,
    )
    img.add_region(Region.from_bbox(TEST_BBOX_LEFT, TEST_BBOX_TOP, TEST_BBOX_WIDTH, 30, text="Region 1", verified=True))
    img.add_region(Region.from_bbox(70, TEST_BBOX_TOP, TEST_BBOX_WIDTH, 30, text="Region 2", verified=False))

    dataset.add_image(img)
    return dataset


@pytest.fixture
def sample_annotation_dataset_empty():
    """Create empty annotation dataset."""
    return AnnotationDataset(name="empty_dataset", version="1.0")


@pytest.fixture
def sample_annotated_image():
    """Create sample annotated image without regions."""
    return AnnotatedImage(
        image_path="images/test.png",
        width=TEST_IMAGE_WIDTH,
        height=TEST_IMAGE_HEIGHT_SMALL,
    )


@pytest.fixture
def sample_annotated_image_with_regions():
    """Create sample annotated image with regions."""
    img = AnnotatedImage(
        image_path="images/test.png",
        width=TEST_IMAGE_WIDTH,
        height=TEST_IMAGE_HEIGHT_SMALL,
    )
    img.add_region(Region.from_bbox(TEST_BBOX_LEFT, TEST_BBOX_TOP, TEST_BBOX_WIDTH, 30, text="Test region", verified=False))
    return img


@pytest.fixture
def mock_ocr_factory_tesseract():
    """Mock OCREngineFactory with only Tesseract available."""
    mock = MagicMock()
    mock.available_engines.return_value = ["tesseract"]
    mock.available_detectors.return_value = []
    mock.available_recognizers.return_value = []
    return mock


@pytest.fixture
def mock_ocr_factory_modular():
    """Mock OCREngineFactory with modular components."""
    mock = MagicMock()
    mock.available_engines.return_value = []
    mock.available_detectors.return_value = ["craft", "db"]
    mock.available_recognizers.return_value = ["crnn", "trocr"]
    return mock


@pytest.fixture
def mock_ocr_factory_full():
    """Mock OCREngineFactory with both monolithic and modular options."""
    mock = MagicMock()
    mock.available_engines.return_value = ["tesseract"]
    mock.available_detectors.return_value = ["craft", "db"]
    mock.available_recognizers.return_value = ["crnn", "trocr"]
    return mock


@pytest.fixture
def mock_uploaded_file(sample_test_image):
    """Mock Streamlit uploaded file."""
    import io

    mock_file = MagicMock()
    mock_file.type = "image/png"
    mock_file.name = "test.png"

    # Create BytesIO buffer
    buf = io.BytesIO()
    sample_test_image.save(buf, format='PNG')
    buf.seek(0)

    mock_file.read = buf.read
    mock_file.seek = buf.seek
    mock_file.tell = buf.tell

    return mock_file


@pytest.fixture
def mock_uploaded_files(sample_test_images):
    """Mock multiple Streamlit uploaded files."""
    import io

    files = []
    for i, img in enumerate(sample_test_images):
        mock_file = MagicMock()
        mock_file.type = "image/png"
        mock_file.name = f"test_{i}.png"

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        mock_file.read = buf.read
        mock_file.seek = buf.seek
        mock_file.tell = buf.tell

        files.append(mock_file)

    return files


# Helper functions for verifying UI components by label


def find_button_by_label(mock_st, label):
    """
    Find button call by its label in mock Streamlit button calls.

    Args:
        mock_st: Mock streamlit object
        label: Button label to search for

    Returns:
        Call args if found, or None if not found
    """
    if not mock_st.button.called:
        return None

    # Check all button calls
    for call in mock_st.button.call_args_list:
        if call[0][0] == label:  # First positional arg is the label
            return call
    return None


def get_button_kwarg(button_call, key, default=None):
    """
    Extract button keyword argument from button call args.

    Args:
        button_call: Button call args from find_button_by_label
        key: Keyword argument name
        default: Default value if not found

    Returns:
        Value of keyword argument or default
    """
    if not button_call:
        return default
    return button_call[1].get(key, default)


def is_button_disabled(button_call):
    """Check if button call has disabled=True."""
    return get_button_kwarg(button_call, "disabled", default=False)


def is_button_primary(button_call):
    """Check if button call has type='primary'."""
    return get_button_kwarg(button_call, "type") == "primary"


def get_button_type(button_call):
    """Get button type (primary/secondary)."""
    return get_button_kwarg(button_call, "type", default="secondary")


def find_text_call(mock_st, text, method="markdown"):
    """
    Find method call containing specific text.

    Args:
        mock_st: Mock streamlit object
        text: Text to search for
        method: Method name to check (markdown, text, etc.)

    Returns:
        True if found, False otherwise
    """
    method_mock = getattr(mock_st, method, None)
    if not method_mock or not method_mock.called:
        return False

    for call in method_mock.call_args_list:
        call_args = str(call)
        if text in call_args:
            return True
    return False


def count_used_columns(columns):
    """
    Count how many columns have been used (had any method called).

    Args:
        columns: List of column mocks

    Returns:
        Number of columns that had any method called
    """
    used_count = 0
    for col in columns:
        # Check common UI methods
        if (col.button.called or col.markdown.called or col.toggle.called or
            col.metric.called or col.text.called or col.checkbox.called):
            used_count += 1
    return used_count


def verify_columns_unused(columns, start_index):
    """
    Verify that columns from start_index onward are unused.

    Args:
        columns: List of column mocks
        start_index: Index from which columns should be unused

    Returns:
        True if all columns from start_index are unused, False otherwise
    """
    for i in range(start_index, len(columns)):
        col = columns[i]
        if (col.button.called or col.markdown.called or col.toggle.called or
            col.metric.called or col.text.called or col.checkbox.called):
            return False
    return True
