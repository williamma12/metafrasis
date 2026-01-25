"""
Tests for Annotation page UI rendering functions.

Tests the Streamlit UI components in app/backend/pages/annotate.py
"""

from unittest.mock import patch
from pathlib import Path
import tempfile
import pytest
from PIL import Image

from app.services.annotation.storage import AnnotationStorage


@pytest.fixture
def temp_annotation_storage(tmp_path):
    """Create temporary annotation storage."""
    storage_path = tmp_path / "annotations"
    storage_path.mkdir()
    return AnnotationStorage(base_path=storage_path)


class TestRenderDatasetSidebar:
    """Tests for render_dataset_sidebar() function."""

    def test_renders_header(self, mock_streamlit, temp_annotation_storage, annotation_state_empty):
        """Test that sidebar header is rendered."""
        from app.backend.pages.annotate import render_dataset_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_dataset_sidebar(temp_annotation_storage, annotation_state_empty)

            mock_streamlit.sidebar.header.assert_called_once_with("Dataset")

    def test_create_new_dataset_expander(self, mock_streamlit, temp_annotation_storage, annotation_state_empty):
        """Test that create dataset expander is shown."""
        from app.backend.pages.annotate import render_dataset_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_dataset_sidebar(temp_annotation_storage, annotation_state_empty)

            mock_streamlit.sidebar.expander.assert_called()

    def test_load_dataset_dropdown_with_existing_datasets(self, mock_streamlit, temp_annotation_storage, sample_annotation_dataset, annotation_state_empty):
        """Test that dataset dropdown is shown when datasets exist."""
        from app.backend.pages.annotate import render_dataset_sidebar

        # Save a dataset
        temp_annotation_storage.save(sample_annotation_dataset)

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_dataset_sidebar(temp_annotation_storage, annotation_state_empty)

            # Should show selectbox
            mock_streamlit.sidebar.selectbox.assert_called()

    def test_displays_dataset_stats(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset):
        """Test that dataset statistics are displayed."""
        from app.backend.pages.annotate import render_dataset_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_dataset_sidebar(temp_annotation_storage, annotation_state_with_dataset)

            # Should display stats via caption
            mock_streamlit.sidebar.caption.assert_called()

    def test_download_button_for_loaded_dataset(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset):
        """Test that download button is shown for loaded dataset."""
        from app.backend.pages.annotate import render_dataset_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_dataset_sidebar(temp_annotation_storage, annotation_state_with_dataset)

            # Should show download button
            mock_streamlit.sidebar.download_button.assert_called()


class TestRenderImageUpload:
    """Tests for render_image_upload() function."""

    def test_returns_early_if_no_dataset(self, mock_streamlit, temp_annotation_storage, annotation_state_empty):
        """Test that function returns early if no dataset loaded."""
        from app.backend.pages.annotate import render_image_upload

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_image_upload(temp_annotation_storage, annotation_state_empty)

            # Should not render file uploader
            mock_streamlit.file_uploader.assert_not_called()

    def test_renders_subheader_and_uploader(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset):
        """Test that subheader and file uploader are rendered."""
        from app.backend.pages.annotate import render_image_upload

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_image_upload(temp_annotation_storage, annotation_state_with_dataset)

            mock_streamlit.subheader.assert_called_once_with("Add Images")
            mock_streamlit.file_uploader.assert_called_once()

    def test_file_uploader_accepts_image_types(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset):
        """Test that file uploader accepts image types."""
        from app.backend.pages.annotate import render_image_upload

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_image_upload(temp_annotation_storage, annotation_state_with_dataset)

            # Check file uploader call
            call_kwargs = mock_streamlit.file_uploader.call_args[1]
            assert "png" in call_kwargs["type"]
            assert "jpg" in call_kwargs["type"]
            assert "jpeg" in call_kwargs["type"]


class TestRenderImageNavigation:
    """Tests for render_image_navigation() function."""

    def test_returns_early_if_no_dataset(self, mock_streamlit, temp_annotation_storage, annotation_state_empty):
        """Test that function returns early if no dataset."""
        from app.backend.pages.annotate import render_image_navigation

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_image_navigation(temp_annotation_storage, annotation_state_empty)

            # Should not render columns
            mock_streamlit.columns.assert_not_called()

    def test_returns_early_if_no_images(self, mock_streamlit, temp_annotation_storage, sample_annotation_dataset_empty):
        """Test that function returns early if dataset has no images."""
        from app.backend.pages.annotate import render_image_navigation
        from app.backend.state import AnnotationState

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            state = AnnotationState()
            state.dataset = sample_annotation_dataset_empty

            render_image_navigation(temp_annotation_storage, state)

            # Should not render columns
            mock_streamlit.columns.assert_not_called()

    def test_renders_all_navigation_buttons(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that all navigation buttons are rendered."""
        from app.backend.pages.annotate import render_image_navigation
        from .conftest import find_button_by_label, verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(6)
            mock_streamlit.columns.return_value = columns[:6]

            render_image_navigation(temp_annotation_storage, annotation_state_with_dataset)

            # Should create 6 columns
            mock_streamlit.columns.assert_called_once_with([1, 1, 2, 1, 1, 1])

            # Verify all expected buttons exist
            assert find_button_by_label(mock_streamlit, "First") is not None
            assert find_button_by_label(mock_streamlit, "Prev") is not None
            assert find_button_by_label(mock_streamlit, "Next") is not None
            assert find_button_by_label(mock_streamlit, "Last") is not None
            assert find_button_by_label(mock_streamlit, "Delete") is not None

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 6)

    def test_first_prev_disabled_at_start(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that First/Prev buttons are disabled at first image."""
        from app.backend.pages.annotate import render_image_navigation
        from .conftest import find_button_by_label, is_button_disabled, verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(6)
            annotation_state_with_dataset.current_image_idx = 0
            mock_streamlit.columns.return_value = columns[:6]

            render_image_navigation(temp_annotation_storage, annotation_state_with_dataset)

            # First and Prev should be disabled
            first_btn = find_button_by_label(mock_streamlit, "First")
            prev_btn = find_button_by_label(mock_streamlit, "Prev")

            assert is_button_disabled(first_btn) is True
            assert is_button_disabled(prev_btn) is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 6)

    def test_displays_current_position(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that current position is displayed."""
        from app.backend.pages.annotate import render_image_navigation
        from .conftest import find_text_call, verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(6)
            annotation_state_with_dataset.current_image_idx = 0
            mock_streamlit.columns.return_value = columns[:6]

            render_image_navigation(temp_annotation_storage, annotation_state_with_dataset)

            # Check position display exists
            assert find_text_call(mock_streamlit, "Image 1 of 1", method="markdown") is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 6)


class TestRenderDrawingToolbar:
    """Tests for render_drawing_toolbar() function."""

    def test_renders_header(self, mock_streamlit, annotation_state_empty, mock_streamlit_columns):
        """Test that toolbar header is rendered."""
        from app.backend.pages.annotate import render_drawing_toolbar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(4)
            mock_streamlit.columns.return_value = columns[:4]

            render_drawing_toolbar(annotation_state_empty)

            mock_streamlit.markdown.assert_called()
            call_args = str(mock_streamlit.markdown.call_args)
            assert "Drawing Tools" in call_args

    def test_renders_mode_buttons(self, mock_streamlit, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that all mode buttons are rendered."""
        from app.backend.pages.annotate import render_drawing_toolbar
        from .conftest import find_button_by_label, verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(4)
            mock_streamlit.columns.return_value = columns[:4]

            render_drawing_toolbar(annotation_state_with_dataset)

            # Should create 4 columns
            mock_streamlit.columns.assert_called_once_with(4)

            # Verify all mode buttons exist
            assert find_button_by_label(mock_streamlit, "Select") is not None
            assert find_button_by_label(mock_streamlit, "Rectangle") is not None
            assert find_button_by_label(mock_streamlit, "Polygon") is not None

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 4)

    def test_select_button_primary_when_active(self, mock_streamlit, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that select button is primary when in select mode."""
        from app.backend.pages.annotate import render_drawing_toolbar
        from .conftest import find_button_by_label, is_button_primary, verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(4)
            annotation_state_with_dataset.drawing_mode = "select"
            mock_streamlit.columns.return_value = columns[:4]

            render_drawing_toolbar(annotation_state_with_dataset)

            # Select button should be primary
            select_btn = find_button_by_label(mock_streamlit, "Select")
            assert is_button_primary(select_btn) is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 4)

    def test_rectangle_button_primary_when_active(self, mock_streamlit, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that rectangle button is primary when in rectangle mode."""
        from app.backend.pages.annotate import render_drawing_toolbar
        from .conftest import find_button_by_label, is_button_primary, verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(4)
            annotation_state_with_dataset.drawing_mode = "rectangle"
            mock_streamlit.columns.return_value = columns[:4]

            render_drawing_toolbar(annotation_state_with_dataset)

            # Rectangle button should be primary
            rect_btn = find_button_by_label(mock_streamlit, "Rectangle")
            assert is_button_primary(rect_btn) is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 4)

    def test_auto_detect_toggle(self, mock_streamlit, annotation_state_with_dataset, mock_streamlit_columns):
        """Test that auto-detect toggle is rendered."""
        from app.backend.pages.annotate import render_drawing_toolbar
        from .conftest import verify_columns_unused

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            columns = mock_streamlit_columns(4)
            mock_streamlit.columns.return_value = columns[:4]

            render_drawing_toolbar(annotation_state_with_dataset)

            # Auto-detect toggle should be rendered
            mock_streamlit.toggle.assert_called_once()
            call_args = mock_streamlit.toggle.call_args[0]
            assert call_args[0] == "Auto-detect"

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 4)


class TestRenderRegionSidebar:
    """Tests for render_region_sidebar() function."""

    def test_renders_header(self, mock_streamlit, temp_annotation_storage, annotation_state_empty, sample_annotated_image):
        """Test that regions header is rendered."""
        from app.backend.pages.annotate import render_region_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_region_sidebar(temp_annotation_storage, annotation_state_empty, sample_annotated_image)

            mock_streamlit.sidebar.divider.assert_called()
            mock_streamlit.sidebar.header.assert_called_with("Regions")

    def test_shows_info_when_no_regions(self, mock_streamlit, temp_annotation_storage, annotation_state_empty, sample_annotated_image):
        """Test that info message is shown when no regions."""
        from app.backend.pages.annotate import render_region_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_region_sidebar(temp_annotation_storage, annotation_state_empty, sample_annotated_image)

            mock_streamlit.sidebar.info.assert_called()

    def test_renders_region_list(self, mock_streamlit, temp_annotation_storage, annotation_state_empty, sample_annotated_image_with_regions):
        """Test that region list is rendered."""
        from app.backend.pages.annotate import render_region_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            render_region_sidebar(temp_annotation_storage, annotation_state_empty, sample_annotated_image_with_regions)

            # Should render button for region
            mock_streamlit.sidebar.button.assert_called()

    def test_shows_region_editor_when_selected(self, mock_streamlit, temp_annotation_storage, annotation_state_empty, sample_annotated_image_with_regions):
        """Test that region editor is shown when region is selected."""
        from app.backend.pages.annotate import render_region_sidebar

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            annotation_state_empty.selected_region_id = sample_annotated_image_with_regions.regions[0].id

            render_region_sidebar(temp_annotation_storage, annotation_state_empty, sample_annotated_image_with_regions)

            # Should show editor
            mock_streamlit.sidebar.subheader.assert_called_with("Edit Region")
            mock_streamlit.sidebar.text_area.assert_called()
            mock_streamlit.sidebar.checkbox.assert_called()


class TestRenderAnnotationCanvas:
    """Tests for render_annotation_canvas() function."""

    def test_loads_image_from_storage(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset, sample_test_image):
        """Test that image is loaded from storage."""
        from app.backend.pages.annotate import render_annotation_canvas

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            # Create real image file
            dataset = annotation_state_with_dataset.dataset

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sample_test_image.save(tmp.name)
                tmp_path = Path(tmp.name)

            rel_path = temp_annotation_storage.copy_image(tmp_path, dataset.name)
            tmp_path.unlink()

            current_image = dataset.images[0]
            current_image.image_path = rel_path

            temp_annotation_storage.save(dataset)

            with patch("app.backend.pages.annotate.annotation_canvas") as mock_canvas:
                mock_canvas.return_value = None

                render_annotation_canvas(temp_annotation_storage, annotation_state_with_dataset, current_image)

                # Canvas should be called
                mock_canvas.assert_called_once()

    def test_shows_error_if_image_not_found(self, mock_streamlit, temp_annotation_storage, annotation_state_empty, sample_annotated_image):
        """Test that error is shown if image file not found."""
        from app.backend.pages.annotate import render_annotation_canvas
        from app.services.annotation.models import AnnotationDataset

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            annotation_state_empty.dataset = AnnotationDataset(name="test", version="1.0")
            sample_annotated_image.image_path = "nonexistent.png"

            render_annotation_canvas(temp_annotation_storage, annotation_state_empty, sample_annotated_image)

            # Should show error
            mock_streamlit.error.assert_called()

    def test_calls_annotation_canvas_with_correct_params(self, mock_streamlit, temp_annotation_storage, annotation_state_with_dataset, sample_test_image):
        """Test that annotation_canvas component is called with correct params."""
        from app.backend.pages.annotate import render_annotation_canvas

        with patch("app.backend.pages.annotate.st", mock_streamlit):
            dataset = annotation_state_with_dataset.dataset

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sample_test_image.save(tmp.name)
                tmp_path = Path(tmp.name)

            rel_path = temp_annotation_storage.copy_image(tmp_path, dataset.name)
            tmp_path.unlink()

            current_image = dataset.images[0]
            current_image.image_path = rel_path
            temp_annotation_storage.save(dataset)

            annotation_state_with_dataset.drawing_mode = "rectangle"
            annotation_state_with_dataset.selected_region_id = None

            with patch("app.backend.pages.annotate.annotation_canvas") as mock_canvas:
                mock_canvas.return_value = None

                render_annotation_canvas(temp_annotation_storage, annotation_state_with_dataset, current_image)

                # Verify canvas called with correct params
                call_kwargs = mock_canvas.call_args[1]
                assert call_kwargs["drawing_mode"] == "rectangle"
                assert call_kwargs["selected_region_id"] is None
                assert isinstance(call_kwargs["regions"], list)
