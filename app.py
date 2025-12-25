"""
Metafrasis - Ancient Greek OCR, Transliteration, and Translation
"""
from dataclasses import dataclass
from typing import List, Optional
import tempfile
import traceback

import streamlit as st
from PIL import Image

from services.ocr import (
    OCREngineFactory,
    OCRResult,
    preprocessing,
    DEFAULT_CONFIDENCE,
    ImageCache,
    ocr_viewer,
)
import config


@dataclass
class AppState:
    """Application state that persists across Streamlit reruns"""
    ocr_results: List[OCRResult] = None
    image_cache: Optional[ImageCache] = None
    transliterated_text: str = ""
    translated_text: str = ""


# Initialize state
if "state" not in st.session_state:
    st.session_state.state = AppState()

# Initialize navigation state
if "current_image_idx" not in st.session_state:
    st.session_state.current_image_idx = 0

state = st.session_state.state

# ============================================================================
# UI
# ============================================================================
st.title("🏛️ Metafrasis")
st.write("Ancient Greek OCR, Transliteration, and Translation")

# Sidebar - OCR Settings
st.sidebar.header("OCR Settings")
available_engines = OCREngineFactory.available_engines()

if not available_engines:
    st.error("No OCR engines available! Check your installation.")
    st.stop()

engine_name = st.sidebar.selectbox(
    "Select OCR Engine",
    available_engines,
    index=0 if config.DEFAULT_OCR_ENGINE not in available_engines
          else available_engines.index(config.DEFAULT_OCR_ENGINE)
)

# Pipeline selection
pipeline = st.sidebar.radio(
    "Processing Pipeline",
    ["Streaming (Sequential)", "Batch (Parallel)"],
    help="Streaming: Process one at a time. Batch: Process all together (faster for multiple files)"
)
use_batch = "Batch" in pipeline

# Show device info
device = config.get_device()
st.sidebar.info(f"🖥️ Device: {device.upper()}")

# File uploader - support multiple images and PDFs
uploaded_files = st.file_uploader(
    "Upload image(s) or PDF(s)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded")

    # Show previews
    with st.expander("View Uploaded Files"):
        for i, file in enumerate(uploaded_files):
            if "image" in file.type:
                st.image(file, caption=f"{i+1}. {file.name}", width=200)
            else:
                st.text(f"{i+1}. 📄 {file.name} (PDF)")

    # OCR Button
    if st.button("🔍 Run OCR", type="primary"):
        try:
            # Create engine
            with st.spinner(f"Loading {engine_name} engine..."):
                engine = OCREngineFactory.create(
                    engine_name,
                    batch_size=config.OCR_BATCH_SIZE,
                    device=device
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
                with st.spinner(f"Batch processing {len(all_images)} images with {engine_name}..."):
                    progress_bar = st.progress(0)
                    state.ocr_results = engine.recognize_batch(all_images)

                    # Set sources
                    for result, source in zip(state.ocr_results, image_sources):
                        result.source = source

                    progress_bar.progress(100)

                st.success(f"✅ Batch OCR complete! Processed {len(state.ocr_results)} images")

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

                st.success(f"✅ Streaming OCR complete! Processed {len(state.ocr_results)} images")

        except Exception as e:
            st.error(f"❌ OCR failed: {str(e)}")
            st.code(traceback.format_exc())
            state.ocr_results = None

# Display OCR results
if state.ocr_results:
    st.divider()
    st.subheader("📝 OCR Results")

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
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮️ First", disabled=st.session_state.current_image_idx == 0):
            st.session_state.current_image_idx = 0
            st.rerun()

    with col2:
        if st.button("◀️ Prev", disabled=st.session_state.current_image_idx == 0):
            st.session_state.current_image_idx -= 1
            st.rerun()

    with col3:
        st.markdown(
            f"<div style='text-align: center; padding-top: 5px;'><strong>Image {st.session_state.current_image_idx + 1} of {len(state.ocr_results)}</strong></div>",
            unsafe_allow_html=True
        )

    with col4:
        if st.button("Next ▶️", disabled=st.session_state.current_image_idx >= len(state.ocr_results) - 1):
            st.session_state.current_image_idx += 1
            st.rerun()

    with col5:
        if st.button("Last ⏭️", disabled=st.session_state.current_image_idx >= len(state.ocr_results) - 1):
            st.session_state.current_image_idx = len(state.ocr_results) - 1
            st.rerun()

    st.divider()

    # Display current image
    idx = st.session_state.current_image_idx
    result = state.ocr_results[idx]
    current_image = state.image_cache.get_image(idx)

    if current_image is None:
        st.error("Error loading image from cache")
    else:
        # Result header
        st.markdown(f"### {result.source or f'Image {idx+1}'}")

        # Toggle between original and annotated
        show_original = st.checkbox("Show Original Image (without annotations)", key=f"show_original_{idx}")

        if show_original:
            # Show original image
            st.image(current_image, use_column_width=True)

            # Display confidence stats
            stats = result.confidence_stats
            if stats.available:
                st.info(f"📊 Confidence Statistics - Mean: {stats.mean:.1%}, Std: {stats.std:.3f}")
        else:
            # Show interactive annotated image with OCR viewer component
            if result.words:
                ocr_viewer(current_image, result, key=f"ocr_viewer_{idx}")
            else:
                st.info("No text detected in this image")
                st.image(current_image, use_column_width=True)

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

    # ========================================================================
    # STEP 2: Transliteration (Placeholder)
    # ========================================================================
    st.subheader("Step 2: Transliteration (Optional)")
    st.info("🚧 Transliteration service coming soon!")

    # ========================================================================
    # STEP 3: Translation (Placeholder)
    # ========================================================================
    st.subheader("Step 3: Translation")
    st.info("🚧 Translation service coming soon!")
