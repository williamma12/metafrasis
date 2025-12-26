"""
Application state management for Metafrasis

Contains dataclasses for session state that persists across Streamlit reruns.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from services.ocr import OCRResult, ImageCache


@dataclass
class AppState:
    """Application state for OCR tab that persists across Streamlit reruns"""
    ocr_results: List[OCRResult] = None
    image_cache: Optional[ImageCache] = None
    transliterated_text: str = ""
    translated_text: str = ""


@dataclass
class AnnotationState:
    """Application state for annotation tab"""
    current_dataset: Optional[str] = None
    current_image_idx: int = 0
    selected_region_id: Optional[str] = None
    drawing_mode: str = "rectangle"  # rectangle, polygon, select
    auto_detect_enabled: bool = False
    unsaved_changes: bool = False
    # Store loaded dataset in memory
    dataset: Optional["AnnotationDataset"] = None  # Forward reference


def init_session_state():
    """Initialize session state if not already done"""
    import streamlit as st

    if "state" not in st.session_state:
        st.session_state.state = AppState()

    if "annotation_state" not in st.session_state:
        st.session_state.annotation_state = AnnotationState()

    if "current_image_idx" not in st.session_state:
        st.session_state.current_image_idx = 0
