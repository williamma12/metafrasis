"""
Annotation Page - Create training data for OCR models

Features:
- Dataset management (create, load, save)
- Image upload and navigation
- Interactive canvas for drawing regions
- Text input for region labels
- Auto-detection using existing OCR detectors
- Auto-save as you annotate
- Export as zip for download
"""
import streamlit as st
from PIL import Image
from pathlib import Path
from datetime import datetime
import shutil

from app.backend.state import AnnotationState
from app.services.annotation import (
    AnnotationStorage,
    AnnotationDataset,
    AnnotatedImage,
    Region,
    Point,
    AnnotationExporter,
    annotation_canvas,
    parse_canvas_result,
)


def get_annotation_state() -> AnnotationState:
    """Get annotation state from session state"""
    return st.session_state.annotation_state


def trigger_autosave(storage: AnnotationStorage, state: AnnotationState):
    """Save dataset immediately if there are unsaved changes"""
    if not state.dataset or not state.unsaved_changes:
        return

    # Update timestamp and save synchronously (fast for JSON files)
    state.dataset.updated_at = datetime.now()
    storage.save(state.dataset)
    state.unsaved_changes = False


def render_dataset_sidebar(storage: AnnotationStorage, state: AnnotationState):
    """Render dataset management controls in sidebar"""
    st.sidebar.header("Dataset")

    # List existing datasets
    datasets = storage.list_datasets()

    # Create new dataset
    with st.sidebar.expander("Create New Dataset", expanded=not datasets):
        new_name = st.text_input("Dataset Name", key="new_dataset_name")
        if st.button("Create", disabled=not new_name):
            if storage.exists(new_name):
                st.error(f"Dataset '{new_name}' already exists")
            else:
                dataset = AnnotationDataset(name=new_name)
                storage.save(dataset)
                state.current_dataset = new_name
                state.dataset = dataset
                state.current_image_idx = 0
                st.rerun()

    # Load existing dataset
    if datasets:
        selected = st.sidebar.selectbox(
            "Load Dataset",
            [""] + datasets,
            index=0 if not state.current_dataset else datasets.index(state.current_dataset) + 1 if state.current_dataset in datasets else 0,
            key="select_dataset"
        )

        if selected and selected != state.current_dataset:
            state.current_dataset = selected
            state.dataset = storage.load(selected)
            state.current_image_idx = 0
            state.selected_region_id = None
            state.unsaved_changes = False
            st.rerun()

    # Show dataset stats
    if state.dataset:
        st.sidebar.divider()
        st.sidebar.markdown(f"**{state.dataset.name}**")
        st.sidebar.caption(f"Images: {len(state.dataset.images)}")
        st.sidebar.caption(f"Regions: {state.dataset.total_regions}")
        st.sidebar.caption(f"Labeled: {state.dataset.labeled_regions}")

        # Download section
        st.sidebar.divider()
        exporter = AnnotationExporter()
        zip_bytes = exporter.export_dataset_bytes(state.dataset, storage)
        st.sidebar.download_button(
            label="Download Dataset",
            data=zip_bytes,
            file_name=f"{state.dataset.name}.zip",
            mime="application/zip",
            use_container_width=True,
        )


def render_image_upload(storage: AnnotationStorage, state: AnnotationState):
    """Render image upload section"""
    if not state.dataset:
        return

    st.subheader("Add Images")

    uploaded_files = st.file_uploader(
        "Upload images to annotate",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="annotation_upload"
    )

    if uploaded_files and st.button("Add to Dataset"):
        import tempfile

        for file in uploaded_files:
            image = Image.open(file)
            width, height = image.size

            # Save to temp file, then copy via storage
            with tempfile.NamedTemporaryFile(suffix=Path(file.name).suffix, delete=False) as tmp:
                image.save(tmp.name)
                relative_path = storage.copy_image(Path(tmp.name), state.dataset.name)
                Path(tmp.name).unlink()  # Clean up temp file

            # Add to dataset
            annotated_image = AnnotatedImage(
                image_path=relative_path,
                width=width,
                height=height,
            )
            state.dataset.add_image(annotated_image)

        state.unsaved_changes = True
        trigger_autosave(storage, state)
        st.success(f"Added {len(uploaded_files)} image(s)")
        st.rerun()


def render_image_navigation(storage: AnnotationStorage, state: AnnotationState):
    """Render image navigation controls"""
    if not state.dataset or not state.dataset.images:
        return

    num_images = len(state.dataset.images)
    current_image = state.dataset.images[state.current_image_idx]

    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 2, 1, 1, 1])

    with col1:
        if st.button("First", disabled=state.current_image_idx == 0, key="ann_first"):
            state.current_image_idx = 0
            state.selected_region_id = None
            st.rerun()

    with col2:
        if st.button("Prev", disabled=state.current_image_idx == 0, key="ann_prev"):
            state.current_image_idx -= 1
            state.selected_region_id = None
            st.rerun()

    with col3:
        st.markdown(
            f"<div style='text-align: center; padding-top: 5px;'><strong>Image {state.current_image_idx + 1} of {num_images}</strong></div>",
            unsafe_allow_html=True
        )

    with col4:
        if st.button("Next", disabled=state.current_image_idx >= num_images - 1, key="ann_next"):
            state.current_image_idx += 1
            state.selected_region_id = None
            st.rerun()

    with col5:
        if st.button("Last", disabled=state.current_image_idx >= num_images - 1, key="ann_last"):
            state.current_image_idx = num_images - 1
            state.selected_region_id = None
            st.rerun()

    with col6:
        if st.button("Delete", type="secondary", key="ann_delete_image"):
            # Delete current image
            storage.delete_image(state.dataset.name, current_image.id, delete_file=True)
            # Reload dataset
            state.dataset = storage.load(state.dataset.name)
            # Adjust index if needed
            if state.current_image_idx >= len(state.dataset.images):
                state.current_image_idx = max(0, len(state.dataset.images) - 1)
            state.selected_region_id = None
            st.rerun()


def render_drawing_toolbar(state: AnnotationState):
    """Render drawing mode toolbar"""
    st.markdown("### Drawing Tools")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "Select",
            type="primary" if state.drawing_mode == "select" else "secondary",
            use_container_width=True,
            key="mode_select"
        ):
            state.drawing_mode = "select"
            st.rerun()

    with col2:
        if st.button(
            "Rectangle",
            type="primary" if state.drawing_mode == "rectangle" else "secondary",
            use_container_width=True,
            key="mode_rect"
        ):
            state.drawing_mode = "rectangle"
            st.rerun()

    with col3:
        if st.button(
            "Polygon",
            type="primary" if state.drawing_mode == "polygon" else "secondary",
            use_container_width=True,
            key="mode_poly"
        ):
            state.drawing_mode = "polygon"
            st.rerun()

    with col4:
        auto_detect = st.toggle("Auto-detect", value=state.auto_detect_enabled, key="auto_detect_toggle")
        if auto_detect != state.auto_detect_enabled:
            state.auto_detect_enabled = auto_detect


def render_region_sidebar(storage: AnnotationStorage, state: AnnotationState, current_image: AnnotatedImage):
    """Render region list and editor in sidebar"""
    st.sidebar.divider()
    st.sidebar.header("Regions")

    if not current_image.regions:
        st.sidebar.info("No regions yet. Draw on the image to add regions.")
        return

    # Region list
    for region in current_image.regions:
        is_selected = region.id == state.selected_region_id
        label = region.text[:20] + "..." if region.text and len(region.text) > 20 else region.text or f"[{region.type}]"
        prefix = "auto " if region.auto_detected else ""

        if st.sidebar.button(
            f"{'> ' if is_selected else ''}{prefix}{label}",
            key=f"region_{region.id}",
            use_container_width=True,
        ):
            state.selected_region_id = region.id
            st.rerun()

    # Region editor
    if state.selected_region_id:
        region = current_image.get_region(state.selected_region_id)
        if region:
            st.sidebar.divider()
            st.sidebar.subheader("Edit Region")

            # Text input
            new_text = st.sidebar.text_area(
                "Text",
                value=region.text or "",
                key=f"text_{region.id}",
                height=100,
            )
            # Normalize for comparison: both as empty string or actual text
            current_text = region.text or ""
            if new_text != current_text:
                region.text = new_text if new_text else None
                region.verified = True
                state.unsaved_changes = True
                trigger_autosave(storage, state)

            # Verified checkbox
            verified = st.sidebar.checkbox("Verified", value=region.verified, key=f"verified_{region.id}")
            if verified != region.verified:
                region.verified = verified
                state.unsaved_changes = True
                trigger_autosave(storage, state)

            # Delete button
            if st.sidebar.button("Delete Region", type="secondary", key=f"delete_{region.id}"):
                current_image.remove_region(region.id)
                state.selected_region_id = None
                state.unsaved_changes = True
                trigger_autosave(storage, state)
                st.rerun()


def render_annotation_canvas(storage: AnnotationStorage, state: AnnotationState, current_image: AnnotatedImage):
    """Render the annotation canvas"""
    # Load image - try new structure first, then fall back to old
    image_path = storage.get_image_path_for_dataset(state.dataset.name, current_image.image_path)
    if not image_path.exists():
        # Fall back to old structure
        image_path = storage.get_image_path(current_image.image_path)
    if not image_path.exists():
        st.error(f"Image not found: {current_image.image_path}")
        return

    image = Image.open(image_path)

    # Render canvas
    result = annotation_canvas(
        image=image,
        regions=current_image.regions,
        selected_region_id=state.selected_region_id,
        drawing_mode=state.drawing_mode,
        key=f"canvas_{current_image.id}_{state.drawing_mode}"
    )

    # Handle canvas result
    if result:
        new_regions, selected_id, action = parse_canvas_result(result)
        action_timestamp = result.get("actionTimestamp")

        # Skip if we already processed this action (prevents infinite loop)
        if action_timestamp and action_timestamp == state.last_action_timestamp:
            return

        if action == "add":
            # Replace all regions with updated list
            current_image.regions = new_regions
            state.selected_region_id = selected_id
            state.unsaved_changes = True
            state.last_action_timestamp = action_timestamp
            trigger_autosave(storage, state)
            st.rerun()
        elif action == "delete":
            current_image.regions = new_regions
            state.selected_region_id = None
            state.unsaved_changes = True
            state.last_action_timestamp = action_timestamp
            trigger_autosave(storage, state)
            st.rerun()
        elif action == "select":
            if selected_id != state.selected_region_id:
                state.selected_region_id = selected_id
                state.last_action_timestamp = action_timestamp
                st.rerun()


def run_auto_detection(storage: AnnotationStorage, state: AnnotationState, current_image: AnnotatedImage):
    """Run auto-detection on current image"""
    st.info("Running auto-detection...")

    try:
        from app.services.ocr.factory import OCREngineFactory

        # Load image - try new structure first, then fall back to old
        image_path = storage.get_image_path_for_dataset(state.dataset.name, current_image.image_path)
        if not image_path.exists():
            image_path = storage.get_image_path(current_image.image_path)
        image = Image.open(image_path)

        # Use CRAFT detector
        engine = OCREngineFactory.create(detector="craft", recognizer="ppocr")

        # Get detector to find regions
        if hasattr(engine, 'detector'):
            detector = engine.detector
            if not detector.is_loaded:
                detector.load_model()

            regions = detector.detect(image)

            # Convert to annotation regions
            for text_region in regions:
                bbox = text_region.bbox
                points = [
                    Point(x=bbox.left, y=bbox.top),
                    Point(x=bbox.left + bbox.width, y=bbox.top),
                    Point(x=bbox.left + bbox.width, y=bbox.top + bbox.height),
                    Point(x=bbox.left, y=bbox.top + bbox.height),
                ]
                region = Region(
                    type="rectangle",
                    points=points,
                    auto_detected=True,
                    verified=False,
                )
                current_image.add_region(region)

            state.unsaved_changes = True
            trigger_autosave(storage, state)
            st.success(f"Detected {len(regions)} regions")
            st.rerun()

    except Exception as e:
        st.error(f"Auto-detection failed: {e}")


def render_annotation_page():
    """Main annotation page render function"""
    state = get_annotation_state()
    storage = AnnotationStorage()

    # Sidebar: Dataset management
    render_dataset_sidebar(storage, state)

    # Main content
    if not state.dataset:
        st.info("Create or load a dataset to start annotating.")
        render_image_upload(storage, state)
        return

    # Image upload
    with st.expander("Add Images", expanded=len(state.dataset.images) == 0):
        render_image_upload(storage, state)

    if not state.dataset.images:
        st.info("Upload images to start annotating.")
        return

    st.divider()

    # Navigation
    render_image_navigation(storage, state)

    st.divider()

    # Drawing toolbar
    render_drawing_toolbar(state)

    # Auto-detect button
    if state.auto_detect_enabled:
        current_image = state.dataset.images[state.current_image_idx]
        if st.button("Run Auto-Detection", key="run_autodetect"):
            run_auto_detection(storage, state, current_image)

    st.divider()

    # Canvas and region sidebar
    current_image = state.dataset.images[state.current_image_idx]

    # Region list in sidebar
    render_region_sidebar(storage, state, current_image)

    # Canvas
    render_annotation_canvas(storage, state, current_image)
