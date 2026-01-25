"""
Tests for OCR page UI rendering functions.

Tests the Streamlit UI components in app/backend/pages/ocr.py
"""

from unittest.mock import patch
import pytest


class TestRenderOCRSidebar:
    """Tests for render_ocr_sidebar() function."""

    def test_renders_header(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test that sidebar header is rendered."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                render_ocr_sidebar()

                mock_streamlit.sidebar.header.assert_called_once_with("OCR Settings")

    def test_monolithic_engine_mode(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test monolithic engine mode selection."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                mock_streamlit.sidebar.radio.return_value = "Monolithic Engine"
                mock_streamlit.sidebar.selectbox.return_value = "tesseract"

                config = render_ocr_sidebar()

                assert config["engine_name"] == "tesseract"
                assert config["detector_name"] is None
                assert config["recognizer_name"] is None

    def test_modular_engine_mode(self, mock_streamlit, mock_ocr_factory_modular):
        """Test modular detector+recognizer mode selection."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_modular):
                mock_streamlit.sidebar.radio.return_value = "Modular (Detector + Recognizer)"
                mock_streamlit.sidebar.selectbox.side_effect = ["craft", "crnn"]

                config = render_ocr_sidebar()

                assert config["engine_name"] is None
                assert config["detector_name"] == "craft"
                assert config["recognizer_name"] == "crnn"

    def test_streaming_pipeline_selected(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test streaming pipeline selection."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                mock_streamlit.sidebar.radio.side_effect = [
                    "Monolithic Engine",
                    "Streaming (Sequential)",
                ]

                config = render_ocr_sidebar()

                assert config["use_batch"] is False

    def test_batch_pipeline_selected(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test batch pipeline selection."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                mock_streamlit.sidebar.radio.side_effect = [
                    "Monolithic Engine",
                    "Batch (Parallel)",
                ]

                config = render_ocr_sidebar()

                assert config["use_batch"] is True

    def test_debug_mode_enabled(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test debug mode checkbox enabled."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                mock_streamlit.sidebar.checkbox.return_value = True

                config = render_ocr_sidebar()

                assert config["debug_mode"] is True

    def test_debug_mode_disabled(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test debug mode checkbox disabled."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                mock_streamlit.sidebar.checkbox.return_value = False

                config = render_ocr_sidebar()

                assert config["debug_mode"] is False

    def test_device_info_cpu(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test that CPU device info is displayed."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                with patch("app.backend.pages.ocr.config.get_device", return_value="cpu"):
                    config = render_ocr_sidebar()

                    assert config["device"] == "cpu"
                    mock_streamlit.sidebar.info.assert_called_with("Device: CPU")

    def test_device_info_mps(self, mock_streamlit, mock_ocr_factory_tesseract):
        """Test that MPS device info is displayed."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            with patch("app.backend.pages.ocr.OCREngineFactory", mock_ocr_factory_tesseract):
                with patch("app.backend.pages.ocr.config.get_device", return_value="mps"):
                    config = render_ocr_sidebar()

                    assert config["device"] == "mps"
                    mock_streamlit.sidebar.info.assert_called_with("Device: MPS")

    def test_no_engines_available_shows_error(self, mock_streamlit):
        """Test error when no engines are available."""
        from app.backend.pages.ocr import render_ocr_sidebar

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Configure st.stop() to raise SystemExit like the real Streamlit does
            mock_streamlit.stop.side_effect = SystemExit

            with patch("app.backend.pages.ocr.OCREngineFactory") as mock_factory:
                mock_factory.available_engines.return_value = []
                mock_factory.available_detectors.return_value = []
                mock_factory.available_recognizers.return_value = []

                with pytest.raises(SystemExit):  # st.stop() raises SystemExit
                    render_ocr_sidebar()

                mock_streamlit.error.assert_called_once()


class TestRenderFileUploader:
    """Tests for render_file_uploader() function."""

    def test_renders_file_uploader(self, mock_streamlit):
        """Test that file uploader is rendered."""
        from app.backend.pages.ocr import render_file_uploader

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            mock_streamlit.file_uploader.return_value = None

            result = render_file_uploader()

            mock_streamlit.file_uploader.assert_called_once()
            assert result is None

    def test_shows_upload_count(self, mock_streamlit, mock_uploaded_files):
        """Test that upload count is displayed."""
        from app.backend.pages.ocr import render_file_uploader

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            mock_streamlit.file_uploader.return_value = mock_uploaded_files

            result = render_file_uploader()

            mock_streamlit.info.assert_called_once()
            assert "3 file(s) uploaded" in str(mock_streamlit.info.call_args)

    def test_shows_image_previews_in_expander(self, mock_streamlit, mock_uploaded_file):
        """Test that image previews are shown in expander."""
        from app.backend.pages.ocr import render_file_uploader

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            mock_streamlit.file_uploader.return_value = [mock_uploaded_file]

            result = render_file_uploader()

            mock_streamlit.expander.assert_called_once()

    def test_accepts_image_types(self, mock_streamlit):
        """Test that file uploader accepts image types."""
        from app.backend.pages.ocr import render_file_uploader

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            render_file_uploader()

            call_kwargs = mock_streamlit.file_uploader.call_args[1]
            assert "png" in call_kwargs["type"]
            assert "jpg" in call_kwargs["type"]
            assert "jpeg" in call_kwargs["type"]
            assert "pdf" in call_kwargs["type"]


class TestRenderNavigation:
    """Tests for render_navigation() function."""

    def test_renders_all_navigation_buttons(self, mock_streamlit, mock_streamlit_columns):
        """Test that all navigation buttons are rendered."""
        from app.backend.pages.ocr import render_navigation
        from .conftest import find_button_by_label, verify_columns_unused

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            columns = mock_streamlit_columns(5)
            mock_streamlit.session_state.current_image_idx = 1
            mock_streamlit.columns.return_value = columns[:5]

            render_navigation(3)

            mock_streamlit.columns.assert_called_once_with([1, 1, 2, 1, 1])

            # Verify all expected buttons exist
            assert find_button_by_label(mock_streamlit, "First") is not None
            assert find_button_by_label(mock_streamlit, "Prev") is not None
            assert find_button_by_label(mock_streamlit, "Next") is not None
            assert find_button_by_label(mock_streamlit, "Last") is not None

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 5)

    def test_first_prev_disabled_at_start(self, mock_streamlit, mock_streamlit_columns):
        """Test that First/Prev buttons are disabled at first image."""
        from app.backend.pages.ocr import render_navigation
        from .conftest import find_button_by_label, is_button_disabled, verify_columns_unused

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            columns = mock_streamlit_columns(5)
            mock_streamlit.session_state.current_image_idx = 0
            mock_streamlit.columns.return_value = columns[:5]

            render_navigation(3)

            # First and Prev should be disabled
            first_btn = find_button_by_label(mock_streamlit, "First")
            prev_btn = find_button_by_label(mock_streamlit, "Prev")

            assert is_button_disabled(first_btn) is True
            assert is_button_disabled(prev_btn) is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 5)

    def test_next_last_disabled_at_end(self, mock_streamlit, mock_streamlit_columns):
        """Test that Next/Last buttons are disabled at last image."""
        from app.backend.pages.ocr import render_navigation
        from .conftest import find_button_by_label, is_button_disabled, verify_columns_unused

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            columns = mock_streamlit_columns(5)
            mock_streamlit.session_state.current_image_idx = 2
            mock_streamlit.columns.return_value = columns[:5]

            render_navigation(3)

            # Next and Last should be disabled
            next_btn = find_button_by_label(mock_streamlit, "Next")
            last_btn = find_button_by_label(mock_streamlit, "Last")

            assert is_button_disabled(next_btn) is True
            assert is_button_disabled(last_btn) is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 5)

    def test_all_buttons_enabled_in_middle(self, mock_streamlit, mock_streamlit_columns):
        """Test that all buttons are enabled when in middle of list."""
        from app.backend.pages.ocr import render_navigation
        from .conftest import find_button_by_label, is_button_disabled, verify_columns_unused

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            columns = mock_streamlit_columns(5)
            mock_streamlit.session_state.current_image_idx = 2
            mock_streamlit.columns.return_value = columns[:5]

            render_navigation(5)

            # All buttons should be enabled
            first_btn = find_button_by_label(mock_streamlit, "First")
            prev_btn = find_button_by_label(mock_streamlit, "Prev")
            next_btn = find_button_by_label(mock_streamlit, "Next")
            last_btn = find_button_by_label(mock_streamlit, "Last")

            assert is_button_disabled(first_btn) is False
            assert is_button_disabled(prev_btn) is False
            assert is_button_disabled(next_btn) is False
            assert is_button_disabled(last_btn) is False

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 5)

    def test_displays_current_position(self, mock_streamlit, mock_streamlit_columns):
        """Test that current position is displayed correctly."""
        from app.backend.pages.ocr import render_navigation
        from .conftest import find_text_call, verify_columns_unused

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            columns = mock_streamlit_columns(5)
            mock_streamlit.session_state.current_image_idx = 2
            mock_streamlit.columns.return_value = columns[:5]

            render_navigation(5)

            # Check that position display exists
            assert find_text_call(mock_streamlit, "Image 3 of 5", method="markdown") is True

            # Verify buffer columns are unused
            assert verify_columns_unused(columns, 5)


class TestRenderResultDisplay:
    """Tests for render_result_display() function."""

    def test_returns_early_if_no_results(self, mock_streamlit, app_state_empty):
        """Test that function returns early if no OCR results."""
        from app.backend.pages.ocr import render_result_display

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            render_result_display(app_state_empty)

            # Should not render anything
            mock_streamlit.divider.assert_not_called()
            mock_streamlit.subheader.assert_not_called()

    def test_displays_summary_metrics(self, mock_streamlit, app_state_with_ocr_results, mock_streamlit_columns):
        """Test that summary metrics are displayed."""
        from app.backend.pages.ocr import render_result_display
        from .conftest import verify_columns_unused

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # render_result_display calls st.columns multiple times:
            # 1. st.columns(3) for summary metrics
            # 2. render_navigation -> st.columns([1,1,2,1,1]) for navigation
            # 3. st.columns(4) for metadata
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)

            mock_streamlit.columns.side_effect = [
                summary_cols[:3],  # Summary metrics
                nav_cols[:5],      # Navigation
                meta_cols[:4],     # Metadata
            ]

            render_result_display(app_state_with_ocr_results)

            # Verify metrics displayed (called on st.metric, not col.metric)
            assert mock_streamlit.metric.call_count >= 3

            # Check for specific metrics
            metric_calls = [str(call) for call in mock_streamlit.metric.call_args_list]
            assert any("Total Images" in str(call) for call in metric_calls)
            assert any("Total Words" in str(call) for call in metric_calls)
            assert any("Total Time" in str(call) for call in metric_calls)

            # Verify buffer columns are unused
            assert verify_columns_unused(summary_cols, 3)

    def test_displays_current_image_result(self, mock_streamlit, app_state_with_ocr_results, mock_streamlit_columns):
        """Test that current image result is displayed."""
        from app.backend.pages.ocr import render_result_display

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Setup columns mocks for multiple st.columns() calls
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)
            mock_streamlit.columns.side_effect = [summary_cols[:3], nav_cols[:5], meta_cols[:4]]

            render_result_display(app_state_with_ocr_results)

            # Should display image source
            markdown_calls = [str(call) for call in mock_streamlit.markdown.call_args_list]
            assert any("test1.png" in str(call) for call in markdown_calls)

    def test_shows_original_image_checkbox(self, mock_streamlit, app_state_with_ocr_results, mock_streamlit_columns):
        """Test that original image checkbox is shown."""
        from app.backend.pages.ocr import render_result_display

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Setup columns mocks for multiple st.columns() calls
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)
            mock_streamlit.columns.side_effect = [summary_cols[:3], nav_cols[:5], meta_cols[:4]]

            render_result_display(app_state_with_ocr_results)

            # Checkbox should be called
            mock_streamlit.checkbox.assert_called()

    def test_displays_extracted_text(self, mock_streamlit, app_state_with_ocr_results, mock_streamlit_columns):
        """Test that extracted text is displayed in text area."""
        from app.backend.pages.ocr import render_result_display

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Setup columns mocks for multiple st.columns() calls
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)
            mock_streamlit.columns.side_effect = [summary_cols[:3], nav_cols[:5], meta_cols[:4]]

            render_result_display(app_state_with_ocr_results)

            # Text area should be called with extracted text
            mock_streamlit.text_area.assert_called()
            call_args = mock_streamlit.text_area.call_args
            assert "Hello World" in call_args[0][1]

    def test_shows_word_details_expander(self, mock_streamlit, app_state_with_ocr_results, mock_streamlit_columns):
        """Test that word details expander is rendered."""
        from app.backend.pages.ocr import render_result_display

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Setup columns mocks for multiple st.columns() calls
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)
            mock_streamlit.columns.side_effect = [summary_cols[:3], nav_cols[:5], meta_cols[:4]]

            render_result_display(app_state_with_ocr_results)

            # Expander should be called
            mock_streamlit.expander.assert_called()

    def test_displays_future_feature_placeholders(self, mock_streamlit, app_state_with_ocr_results, mock_streamlit_columns):
        """Test that future feature placeholders are shown."""
        from app.backend.pages.ocr import render_result_display

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Setup columns mocks for multiple st.columns() calls
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)
            mock_streamlit.columns.side_effect = [summary_cols[:3], nav_cols[:5], meta_cols[:4]]

            render_result_display(app_state_with_ocr_results)

            # Should show placeholders for transliteration and translation
            info_calls = [str(call) for call in mock_streamlit.info.call_args_list]
            assert any("Transliteration" in str(call) or "transliteration" in str(call).lower() for call in info_calls)
            assert any("Translation" in str(call) or "translation" in str(call).lower() for call in info_calls)

    def test_handles_missing_image_cache(self, mock_streamlit, sample_ocr_results, mock_streamlit_columns):
        """Test that missing image cache is handled gracefully."""
        from app.backend.pages.ocr import render_result_display
        from app.backend.state import AppState
        from unittest.mock import MagicMock

        with patch("app.backend.pages.ocr.st", mock_streamlit):
            # Setup columns mocks for multiple st.columns() calls
            summary_cols = mock_streamlit_columns(3)
            nav_cols = mock_streamlit_columns(5)
            meta_cols = mock_streamlit_columns(4)
            mock_streamlit.columns.side_effect = [summary_cols[:3], nav_cols[:5], meta_cols[:4]]

            state = AppState()
            state.ocr_results = sample_ocr_results
            # Mock image cache that returns None for get_image
            state.image_cache = MagicMock()
            state.image_cache.get_image.return_value = None

            # Should not crash - displays error and returns early
            render_result_display(state)

            # Should show error
            mock_streamlit.error.assert_called()
