"""
OCR Page - Main OCR processing and result display

Handles:
- Engine/detector/recognizer selection
- File upload and processing
- Result visualization with OCR viewer
"""
import tempfile
import traceback

import streamlit as st
from PIL import Image

from app.services.ocr import (
    OCREngineFactory,
    preprocessing,
    DEFAULT_CONFIDENCE,
    ImageCache,
    ocr_viewer,
)
from app import config
from app.backend.state import AppState


def render_ocr_sidebar() -> dict:
    """Render OCR settings sidebar and return configuration"""
    st.sidebar.header("OCR Settings")

    # Engine selection mode
    available_engines = OCREngineFactory.available_engines()
    available_detectors = OCREngineFactory.available_detectors()
    available_recognizers = OCREngineFactory.available_recognizers()

    # Check if we have any engines
    if not available_engines and not (available_detectors and available_recognizers):
        st.error("No OCR engines available! Check your installation.")
        st.stop()

    # Engine mode selection
    engine_mode = st.sidebar.radio(
        "Engine Mode",
        ["Monolithic Engine", "Modular (Detector + Recognizer)"],
        help="Monolithic: Single engine (e.g., Tesseract). Modular: Combine detector + recognizer."
    )

    # Warning for modular mode
    if engine_mode == "Modular (Detector + Recognizer)":
        st.sidebar.warning(
            "**Experimental**: Modular mode is under development. "
            "Detection + recognition may have reduced accuracy compared to "
            "end-to-end models. Use for testing only."
        )

    engine_name = None
    detector_name = None
    recognizer_name = None

    if engine_mode == "Monolithic Engine":
        if not available_engines:
            st.sidebar.error("No monolithic engines available. Use modular mode.")
            st.stop()

        engine_name = st.sidebar.selectbox(
            "Select Engine",
            available_engines,
            index=0
        )
    else:
        # Modular mode
        if not available_detectors:
            st.sidebar.error("No detectors available!")
            st.stop()
        if not available_recognizers:
            st.sidebar.error("No recognizers available!")
            st.stop()

        detector_name = st.sidebar.selectbox(
            "Select Detector",
            available_detectors,
            help="Detector finds text regions in images"
        )

        recognizer_name = st.sidebar.selectbox(
            "Select Recognizer",
            available_recognizers,
            help="Recognizer reads text from detected regions"
        )

    # Pipeline selection
    pipeline = st.sidebar.radio(
        "Processing Pipeline",
        ["Streaming (Sequential)", "Batch (Parallel)"],
        help="Streaming: Process one at a time. Batch: Process all together (faster for multiple files)"
    )
    use_batch = "Batch" in pipeline

    # Debug mode
    debug_mode = st.sidebar.checkbox(
        "Debug Mode",
        value=False,
        help="Store and visualize detector regions (before recognition). Uses more memory."
    )

    # Show device info
    device = config.get_device()
    st.sidebar.info(f"Device: {device.upper()}")

    return {
        "engine_name": engine_name,
        "detector_name": detector_name,
        "recognizer_name": recognizer_name,
        "use_batch": use_batch,
        "debug_mode": debug_mode,
        "device": device,
    }


def render_file_uploader():
    """Render file uploader and return uploaded files"""
    uploaded_files = st.file_uploader(
        "Upload image(s) or PDF(s)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) uploaded")

        # Show previews
        with st.expander("View Uploaded Files"):
            for i, file in enumerate(uploaded_files):
                if "image" in file.type:
                    st.image(file, caption=f"{i+1}. {file.name}", width=200)
                else:
                    st.text(f"{i+1}. {file.name} (PDF)")

    return uploaded_files


def process_ocr(uploaded_files, ocr_config: dict, state: AppState):
    """Run OCR on uploaded files"""
    engine_name = ocr_config["engine_name"]
    detector_name = ocr_config["detector_name"]
    recognizer_name = ocr_config["recognizer_name"]
    use_batch = ocr_config["use_batch"]
    debug_mode = ocr_config["debug_mode"]
    device = ocr_config["device"]

    try:
        # Create engine
        if engine_name:
            engine_display_name = engine_name
            with st.spinner(f"Loading {engine_name} engine..."):
                engine = OCREngineFactory.create(
                    engine=engine_name,
                    batch_size=config.OCR_BATCH_SIZE,
                    device=device,
                    debug_mode=debug_mode
                )
        else:
            engine_display_name = f"{detector_name}+{recognizer_name}"
            with st.spinner(f"Loading {detector_name} detector + {recognizer_name} recognizer..."):
                engine = OCREngineFactory.create(
                    detector=detector_name,
                    recognizer=recognizer_name,
                    batch_size=config.OCR_BATCH_SIZE,
                    device=device,
                    debug_mode=debug_mode
                )

        # Collect all images from uploaded files
        all_images = []
        image_sources = []  # Track which file each image came from

        # Convert files to images
        with st.spinner("Loading files..."):
            for file in uploaded_files:
                if "pdf" in file.type:
                    # Save PDF temporarily and convert
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    # Convert PDF to images
                    pdf_images = preprocessing.pdf_to_images(tmp_path, dpi=config.PDF_DPI)
                    all_images.extend(pdf_images)

                    # Track sources
                    for j in range(len(pdf_images)):
                        image_sources.append(f"{file.name}:page_{j+1}")

                else:
                    # Single image
                    image = Image.open(file)
                    all_images.append(image)
                    image_sources.append(file.name)

        st.info(f"Total images to process: {len(all_images)}")

        # Initialize image cache and store original images
        state.image_cache = ImageCache()
        state.image_cache.add_images(all_images)
        st.session_state.current_image_idx = 0  # Reset to first image

        # Process images based on pipeline choice
        if use_batch:
            # Batch pipeline: Process all at once
            with st.spinner(f"Batch processing {len(all_images)} images with {engine_display_name}..."):
                progress_bar = st.progress(0)
                state.ocr_results = engine.recognize_batch(all_images)

                # Set sources
                for result, source in zip(state.ocr_results, image_sources):
                    result.source = source

                progress_bar.progress(100)

            st.success(f"Batch OCR complete! Processed {len(state.ocr_results)} images")

        else:
            # Streaming pipeline: Process one at a time
            state.ocr_results = []
            progress_bar = st.progress(0)

            for i, (image, source) in enumerate(zip(all_images, image_sources)):
                with st.spinner(f"Processing {i+1}/{len(all_images)}: {source}"):
                    result = engine.recognize(image)
                    result.source = source
                    state.ocr_results.append(result)

                progress_bar.progress((i + 1) / len(all_images))

            st.success(f"Streaming OCR complete! Processed {len(state.ocr_results)} images")

    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        st.code(traceback.format_exc())
        state.ocr_results = None


def render_navigation(num_results: int):
    """Render image navigation controls"""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("First", disabled=st.session_state.current_image_idx == 0):
            st.session_state.current_image_idx = 0
            st.rerun()

    with col2:
        if st.button("Prev", disabled=st.session_state.current_image_idx == 0):
            st.session_state.current_image_idx -= 1
            st.rerun()

    with col3:
        st.markdown(
            f"<div style='text-align: center; padding-top: 5px;'><strong>Image {st.session_state.current_image_idx + 1} of {num_results}</strong></div>",
            unsafe_allow_html=True
        )

    with col4:
        if st.button("Next", disabled=st.session_state.current_image_idx >= num_results - 1):
            st.session_state.current_image_idx += 1
            st.rerun()

    with col5:
        if st.button("Last", disabled=st.session_state.current_image_idx >= num_results - 1):
            st.session_state.current_image_idx = num_results - 1
            st.rerun()


def render_result_display(state: AppState):
    """Render OCR results display"""
    if not state.ocr_results:
        return

    st.divider()
    st.subheader("OCR Results")

    # Summary
    total_words = sum(len(r.words) for r in state.ocr_results)
    total_time = sum(r.processing_time for r in state.ocr_results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", len(state.ocr_results))
    with col2:
        st.metric("Total Words", total_words)
    with col3:
        st.metric("Total Time", f"{total_time:.2f}s")

    st.divider()

    # Navigation controls
    render_navigation(len(state.ocr_results))

    st.divider()

    # Display current image
    idx = st.session_state.current_image_idx
    result = state.ocr_results[idx]
    current_image = state.image_cache.get_image(idx)

    if current_image is None:
        st.error("Error loading image from cache")
        return

    # Result header
    st.markdown(f"### {result.source or f'Image {idx+1}'}")

    # Toggle between original and annotated
    show_original = st.checkbox("Show Original Image (without annotations)", key=f"show_original_{idx}")

    # Debug mode toggle (only if detector regions are available)
    show_detector_regions = False
    if result.detector_regions is not None and not show_original:
        show_detector_regions = st.checkbox(
            "Show Detector Regions (Debug)",
            value=False,
            key=f"show_detector_{idx}",
            help="Toggle between detector regions (red dashed boxes) and recognized words (green boxes)"
        )

    if show_original:
        # Show original image
        st.image(current_image, use_column_width=True)

        # Display confidence stats
        stats = result.confidence_stats
        if stats.available:
            st.info(f"Confidence Statistics - Mean: {stats.mean:.1%}, Std: {stats.std:.3f}")
    else:
        # Show interactive annotated image with OCR viewer component
        if show_detector_regions and result.detector_regions is not None:
            # Show detector regions (debug mode)
            if len(result.detector_regions) == 0:
                st.warning("Debug Mode: No text regions detected by the detector")
                st.image(current_image, use_column_width=True)
            else:
                st.info(f"Debug Mode: Showing {len(result.detector_regions)} detector regions")
                ocr_viewer(current_image, result, show_detector_regions=True, key=f"ocr_viewer_debug_{idx}")
        elif result.words:
            # Show recognized words (normal mode)
            ocr_viewer(current_image, result, key=f"ocr_viewer_{idx}")
        else:
            st.info("No text detected in this image")
            st.image(current_image, use_column_width=True)

    # Show debug info if available
    if result.detector_regions is not None and not show_original:
        if len(result.detector_regions) == 0:
            st.caption("Debug: Detector found 0 text regions")
        else:
            st.caption(f"Debug: Detector found {len(result.detector_regions)} text regions")

    st.divider()

    # Extracted text
    text = " ".join(word.text for word in result.words)
    st.text_area(
        "Extracted Text",
        text,
        height=150,
        key=f"ocr_text_{idx}"
    )

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Words", len(result.words))
    with col2:
        stats = result.confidence_stats
        if stats.available:
            st.metric("Confidence (Mean)", f"{stats.mean:.1%}")
        else:
            st.metric("Confidence", "N/A")
    with col3:
        stats = result.confidence_stats
        if stats.available:
            st.metric("Confidence (Std)", f"{stats.std:.3f}")
        else:
            st.metric("Std Dev", "N/A")
    with col4:
        st.metric("Time", f"{result.processing_time:.2f}s")

    # Word details (expandable)
    with st.expander("View Word Details"):
        if result.words:
            for j, word in enumerate(result.words):
                conf_str = f"{word.confidence:.1%}" if word.confidence != DEFAULT_CONFIDENCE else "N/A"
                bbox_str = f"[{word.bbox.left},{word.bbox.top},{word.bbox.width},{word.bbox.height}]"
                st.text(f"{j+1}. {word.text} (conf: {conf_str}, bbox: {bbox_str})")
        else:
            st.write("No words detected")

    st.divider()

    # Placeholders for future features
    st.subheader("Step 2: Transliteration (Optional)")
    st.info("Transliteration service coming soon!")

    st.subheader("Step 3: Translation")
    st.info("Translation service coming soon!")


def render_ocr_page():
    """Main OCR page render function"""
    state = st.session_state.state

    # Sidebar configuration
    ocr_config = render_ocr_sidebar()

    # File uploader
    uploaded_files = render_file_uploader()

    # OCR Button
    if uploaded_files and st.button("Run OCR", type="primary"):
        process_ocr(uploaded_files, ocr_config, state)

    # Display results
    render_result_display(state)
