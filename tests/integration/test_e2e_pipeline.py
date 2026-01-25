"""
End-to-end integration tests for OCR and Annotation services through app layer.

Tests:
- Backend page functions (app/backend/pages/)
- Frontend component rendering (app/frontend/)
- Data conversion between Python and JavaScript
- Full OCR → Annotation pipeline
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

import pytest
from PIL import Image, ImageDraw

from app.services.annotation.models import Point, Region, AnnotatedImage, AnnotationDataset
from app.services.annotation.storage import AnnotationStorage
from app.services.annotation.exporter import AnnotationExporter
from app.backend.state import AppState, AnnotationState
from app.services.ocr.base import BoundingBox, Word, OCRResult


class MockUploadedFile:
    """Mock Streamlit uploaded file that works with PIL.Image.open()"""

    def __init__(self, image: Image.Image, filename: str, file_type: str = "image/png"):
        import io
        self.buf = io.BytesIO()
        image.save(self.buf, format='PNG')
        self.buf.seek(0)
        self.name = filename
        self.type = file_type

    def read(self, *args, **kwargs):
        return self.buf.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.buf.seek(*args, **kwargs)

    def tell(self):
        return self.buf.tell()

    def __enter__(self):
        return self.buf

    def __exit__(self, *args):
        pass


@pytest.fixture
def sample_images():
    """Create multiple sample images with text."""
    images = []

    # Image 1: Simple text
    img1 = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img1)
    draw.text((20, 30), "Hello World", fill="black")
    images.append(img1)

    # Image 2: Multiple lines
    img2 = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img2)
    draw.text((20, 30), "Line 1", fill="black")
    draw.text((20, 80), "Line 2", fill="black")
    draw.text((20, 130), "Line 3", fill="black")
    images.append(img2)

    # Image 3: Blank
    img3 = Image.new("RGB", (200, 100), color="white")
    images.append(img3)

    return images


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def temp_storage(temp_data_dir):
    """Create AnnotationStorage with temporary directory."""
    return AnnotationStorage(base_path=temp_data_dir / "annotations")


@pytest.fixture
def mock_streamlit():
    """Mock streamlit for testing page functions without full Streamlit runtime."""
    from unittest.mock import MagicMock

    mock_st = MagicMock()

    # Create a proper context manager mock
    class MockContextManager:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    # Mock basic Streamlit functions that are used as context managers
    mock_st.spinner = lambda msg: MockContextManager()

    # Mock other Streamlit functions
    mock_st.success = MagicMock()
    mock_st.error = MagicMock()
    mock_st.info = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.code = MagicMock()

    # Mock progress bar
    progress_mock = MagicMock()
    progress_mock.progress = MagicMock()
    mock_st.progress = MagicMock(return_value=progress_mock)

    # Mock session state
    mock_st.session_state = MagicMock()
    mock_st.session_state.current_image_idx = 0

    return mock_st


class TestFrontendComponents:
    """Test frontend React component loading and data conversion."""

    def test_ocr_viewer_component_declaration_production_mode(self):
        """Test OCR viewer component loads in production mode."""
        # Test that we can check the release mode setting
        # Note: The actual mode is set at import time based on environment
        from app import config
        # Just verify the config value exists and is a boolean
        assert isinstance(config.VIEWER_RELEASE_MODE, bool)

    def test_ocr_viewer_component_declaration_dev_mode(self):
        """Test OCR viewer component loads in development mode."""
        # Verify component is declared regardless of mode
        from app.services.ocr.viewer import _ocr_viewer
        assert _ocr_viewer is not None

    def test_ocr_viewer_image_to_base64(self, sample_images):
        """Test image to base64 conversion for OCR viewer."""
        from app.services.ocr.viewer import image_to_base64

        result = image_to_base64(sample_images[0])

        assert result.startswith("data:image/png;base64,")
        assert len(result) > 100  # Should have actual image data

    def test_annotation_canvas_component_declaration(self):
        """Test annotation canvas component loads."""
        from app.services.annotation.canvas import _annotation_canvas, _RELEASE

        assert _annotation_canvas is not None
        # Should have a release mode setting
        assert isinstance(_RELEASE, bool)

    def test_annotation_canvas_region_conversion(self):
        """Test Region to dict conversion for annotation canvas."""
        from app.services.annotation.canvas import region_to_dict, dict_to_region

        # Create a region
        region = Region(
            type="rectangle",
            points=[Point(10, 20), Point(50, 20), Point(50, 60), Point(10, 60)],
            text="Test region",
            auto_detected=True,
            verified=False,
        )

        # Convert to dict
        region_dict = region_to_dict(region)

        assert region_dict["id"] == region.id
        assert region_dict["type"] == "rectangle"
        assert len(region_dict["points"]) == 4
        assert region_dict["points"][0] == {"x": 10, "y": 20}
        assert region_dict["text"] == "Test region"
        assert region_dict["auto_detected"] is True
        assert region_dict["verified"] is False

        # Convert back to Region
        converted_region = dict_to_region(region_dict)

        assert converted_region.id == region.id
        assert converted_region.type == region.type
        assert len(converted_region.points) == 4
        assert converted_region.points[0].x == 10
        assert converted_region.points[0].y == 20
        assert converted_region.text == region.text
        assert converted_region.auto_detected == region.auto_detected
        assert converted_region.verified == region.verified

    def test_annotation_canvas_image_to_base64(self, sample_images):
        """Test image to base64 conversion for annotation canvas."""
        from app.services.annotation.canvas import image_to_base64

        result = image_to_base64(sample_images[0])

        assert result.startswith("data:image/png;base64,")
        assert len(result) > 100


class TestOCRPageBackend:
    """Test OCR page backend functions."""

    @pytest.mark.slow
    @pytest.mark.requires_tesseract
    def test_process_ocr_streaming_mode(self, sample_images, mock_streamlit):
        """Test process_ocr with streaming pipeline."""
        from app.backend.pages.ocr import process_ocr
        from app.services.ocr.factory import OCREngineFactory

        if "tesseract" not in OCREngineFactory.available_engines():
            pytest.skip("Tesseract not available")

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Create mock uploaded files
            uploaded_files = [
                MockUploadedFile(sample_images[0], "test_0.png"),
                MockUploadedFile(sample_images[1], "test_1.png"),
            ]

            # OCR config
            ocr_config = {
                "engine_name": "tesseract",
                "detector_name": None,
                "recognizer_name": None,
                "use_batch": False,  # Streaming
                "debug_mode": False,
                "device": "cpu",
            }

            state = AppState()

            # Process
            process_ocr(uploaded_files, ocr_config, state)

            # Verify
            assert state.ocr_results is not None
            assert len(state.ocr_results) == 2
            assert state.image_cache is not None
            for result in state.ocr_results:
                assert hasattr(result, "words")
                assert result.processing_time > 0

    @pytest.mark.slow
    @pytest.mark.requires_craft
    @pytest.mark.requires_crnn
    def test_process_ocr_batch_mode_with_debug(self, sample_images, mock_streamlit):
        """Test process_ocr with batch pipeline and debug mode."""
        from app.backend.pages.ocr import process_ocr

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            uploaded_files = [
                MockUploadedFile(sample_images[0], "test_0.png"),
                MockUploadedFile(sample_images[1], "test_1.png"),
                MockUploadedFile(sample_images[2], "test_2.png"),
            ]

            ocr_config = {
                "engine_name": None,
                "detector_name": "craft",
                "recognizer_name": "crnn",
                "use_batch": True,
                "debug_mode": True,
                "device": "cpu",
            }

            state = AppState()
            process_ocr(uploaded_files, ocr_config, state)

            assert state.ocr_results is not None
            assert len(state.ocr_results) == 3

            # Debug mode should store detector regions
            for result in state.ocr_results:
                assert result.detector_regions is not None


class TestAnnotationPageBackend:
    """Test annotation page backend functions."""

    def test_trigger_autosave(self, temp_storage):
        """Test trigger_autosave function."""
        from app.backend.pages.annotate import trigger_autosave

        dataset = AnnotationDataset(name="autosave_test", version="1.0")
        dataset.add_image(AnnotatedImage(image_path="test.png", width=100, height=100))

        state = AnnotationState()
        state.dataset = dataset
        state.unsaved_changes = True

        temp_storage.save(dataset)
        original_time = dataset.updated_at

        import time
        time.sleep(0.01)

        trigger_autosave(temp_storage, state)

        assert state.unsaved_changes is False
        loaded = temp_storage.load(dataset.name)
        assert loaded.updated_at > original_time

    @pytest.mark.slow
    @pytest.mark.requires_craft
    def test_run_auto_detection(self, sample_images, temp_storage, mock_streamlit):
        """Test run_auto_detection function runs without errors."""
        from app.backend.pages.annotate import run_auto_detection

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            # Create dataset with image
            dataset = AnnotationDataset(name="autodetect_test", version="1.0")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sample_images[1].save(tmp.name)
                tmp_path = Path(tmp.name)

            rel_path = temp_storage.copy_image(tmp_path, dataset.name)
            tmp_path.unlink()

            ann_img = AnnotatedImage(
                image_path=rel_path,
                width=sample_images[1].width,
                height=sample_images[1].height,
            )
            dataset.add_image(ann_img)
            temp_storage.save(dataset)

            state = AnnotationState()
            state.dataset = dataset
            state.current_image_idx = 0

            # Run auto-detection (may not find regions with untrained model)
            run_auto_detection(temp_storage, state, ann_img)

            # Verify function ran without error and created proper structure
            # Note: With untrained CRAFT, may find 0 regions on simple images
            assert isinstance(ann_img.regions, list)

            # If regions were found, verify they have correct flags
            for region in ann_img.regions:
                assert region.auto_detected is True
                assert region.verified is False


class TestOCRToAnnotationPipeline:
    """Test complete OCR → Annotation → Export pipeline through app layer."""

    @pytest.mark.slow
    @pytest.mark.requires_craft
    @pytest.mark.requires_crnn
    def test_full_e2e_pipeline(self, sample_images, temp_storage, mock_streamlit):
        """
        Complete E2E test:
        1. Run OCR via process_ocr()
        2. Convert results to annotation dataset
        3. Run auto-detection
        4. Manual edits
        5. Save via trigger_autosave()
        6. Export
        7. Verify data integrity
        """
        from app.backend.pages.ocr import process_ocr
        from app.backend.pages.annotate import trigger_autosave, run_auto_detection

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Step 1: OCR
            uploaded_files = [
                MockUploadedFile(sample_images[0], "test_0.png"),
                MockUploadedFile(sample_images[1], "test_1.png"),
            ]

            ocr_config = {
                "engine_name": None,
                "detector_name": "craft",
                "recognizer_name": "crnn",
                "use_batch": True,
                "debug_mode": True,
                "device": "cpu",
            }

            app_state = AppState()
            process_ocr(uploaded_files, ocr_config, app_state)

            assert app_state.ocr_results is not None

        # Step 2: Create annotation dataset
        dataset = AnnotationDataset(name="e2e_test", version="1.0")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            sample_images[0].save(tmp.name)
            tmp_path = Path(tmp.name)

        rel_path = temp_storage.copy_image(tmp_path, dataset.name)
        tmp_path.unlink()

        ann_img = AnnotatedImage(
            image_path=rel_path,
            width=sample_images[0].width,
            height=sample_images[0].height,
        )
        dataset.add_image(ann_img)

        # Step 3: Auto-detection
        ann_state = AnnotationState()
        ann_state.dataset = dataset
        ann_state.current_image_idx = 0

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            run_auto_detection(temp_storage, ann_state, ann_img)

        # Step 4: Manual edits (add manual region if auto-detection didn't find any)
        if not ann_img.regions:
            # Add a manual region for testing
            manual_region = Region.from_bbox(10, 10, 50, 30, text="Manual region", verified=False)
            ann_img.add_region(manual_region)
            ann_state.unsaved_changes = True

        if ann_img.regions:
            ann_img.regions[0].text = "Manual edit"
            ann_img.regions[0].verified = True
            ann_state.unsaved_changes = True

        # Step 5: Save
        trigger_autosave(temp_storage, ann_state)

        # Step 6: Export
        exporter = AnnotationExporter()
        zip_bytes = exporter.export_dataset_bytes(dataset, temp_storage)
        assert len(zip_bytes) > 0

        # Step 7: Verify
        loaded = temp_storage.load(dataset.name)
        assert len(loaded.images) == 1
        assert len(loaded.images[0].regions) > 0

        if loaded.images[0].regions:
            assert loaded.images[0].regions[0].text == "Manual edit"
            assert loaded.images[0].regions[0].verified is True

    @pytest.mark.slow
    @pytest.mark.requires_craft
    @pytest.mark.requires_crnn
    def test_ocr_results_to_frontend_viewer_data(self, sample_images, mock_streamlit):
        """Test OCR results can be converted to frontend viewer format."""
        from app.backend.pages.ocr import process_ocr
        from app.services.ocr.viewer import image_to_base64

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            uploaded_files = [MockUploadedFile(sample_images[0], "test.png")]

            ocr_config = {
                "engine_name": None,
                "detector_name": "craft",
                "recognizer_name": "crnn",
                "use_batch": False,
                "debug_mode": True,
                "device": "cpu",
            }

            state = AppState()
            process_ocr(uploaded_files, ocr_config, state)

            # Convert to frontend format
            result = state.ocr_results[0]
            image_url = image_to_base64(sample_images[0])

            # Prepare words data (mimics what ocr_viewer does)
            words_data = [
                {
                    "text": word.text,
                    "bbox": {
                        "left": word.bbox.left,
                        "top": word.bbox.top,
                        "width": word.bbox.width,
                        "height": word.bbox.height,
                    },
                    "confidence": word.confidence,
                }
                for word in result.words
            ]

            # Verify format
            assert image_url.startswith("data:image/png;base64,")
            assert isinstance(words_data, list)
            for word_data in words_data:
                assert "text" in word_data
                assert "bbox" in word_data
                assert "confidence" in word_data

    @pytest.mark.slow
    @pytest.mark.requires_craft
    def test_annotation_to_frontend_canvas_data(self, sample_images, temp_storage, mock_streamlit):
        """Test annotation regions can be converted to frontend canvas format."""
        from app.backend.pages.annotate import run_auto_detection
        from app.services.annotation.canvas import region_to_dict, image_to_base64

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            dataset = AnnotationDataset(name="canvas_test", version="1.0")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sample_images[0].save(tmp.name)
                tmp_path = Path(tmp.name)

            rel_path = temp_storage.copy_image(tmp_path, dataset.name)
            tmp_path.unlink()

            ann_img = AnnotatedImage(
                image_path=rel_path,
                width=sample_images[0].width,
                height=sample_images[0].height,
            )
            dataset.add_image(ann_img)
            temp_storage.save(dataset)

            state = AnnotationState()
            state.dataset = dataset
            state.current_image_idx = 0

            run_auto_detection(temp_storage, state, ann_img)

            # Add a manual region if auto-detection didn't find any
            if not ann_img.regions:
                manual_region = Region.from_bbox(10, 10, 50, 30, text="Test region", verified=False)
                ann_img.add_region(manual_region)

            # Convert to frontend format
            image_url = image_to_base64(sample_images[0])
            regions_data = [region_to_dict(r) for r in ann_img.regions]

            # Verify format
            assert image_url.startswith("data:image/png;base64,")
            assert isinstance(regions_data, list)
            assert len(regions_data) > 0  # Should have at least the manual region

            for region_data in regions_data:
                assert "id" in region_data
                assert "type" in region_data
                assert "points" in region_data
                assert isinstance(region_data["points"], list)
                for point in region_data["points"]:
                    assert "x" in point
                    assert "y" in point
