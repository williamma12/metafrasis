"""
Interactive OCR viewer component for displaying images with bounding boxes
"""
import os
import streamlit.components.v1 as components
from typing import List, Optional
from PIL import Image
import base64
from io import BytesIO

from .base import OCRResult, Word, DEFAULT_CONFIDENCE
import config

# Declare the custom component
# Mode is controlled by VIEWER_RELEASE environment variable (defaults to development mode)
# Development mode (default): Connects to Vite dev server at http://localhost:5173
# Production mode (VIEWER_RELEASE=true): Loads pre-built component from build directory
_RELEASE = config.VIEWER_RELEASE_MODE

if not _RELEASE:
    _ocr_viewer = components.declare_component(
        "ocr_viewer",
        url="http://localhost:5173",  # Vite dev server
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "../../frontend/ocr_viewer/build")
    _ocr_viewer = components.declare_component(
        "ocr_viewer",
        path=build_dir
    )


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded data URL string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def ocr_viewer(
    image: Image.Image,
    result: OCRResult,
    show_detector_regions: bool = False,
    key: Optional[str] = None,
) -> Optional[dict]:
    """
    Display interactive OCR result viewer with clickable bounding boxes

    Args:
        image: PIL Image object to display
        result: OCRResult object with words and bounding boxes
        show_detector_regions: If True, show detector regions instead of recognized words
        key: Streamlit component key

    Returns:
        Component value containing visibility state for each word/region
    """
    # Convert image to base64 data URL
    image_url = image_to_base64(image)

    # Prepare detector regions data if requested
    detector_regions_data = []
    if show_detector_regions and result.detector_regions:
        detector_regions_data = [
            {
                "bbox": {
                    "left": region.bbox.left,
                    "top": region.bbox.top,
                    "width": region.bbox.width,
                    "height": region.bbox.height,
                },
                "confidence": region.confidence if region.confidence != DEFAULT_CONFIDENCE else -1,
                "index": idx,  # For labeling regions as #1, #2, etc.
            }
            for idx, region in enumerate(result.detector_regions)
        ]

    # Prepare words data for the component (only if not showing detector regions)
    words_data = []
    if not show_detector_regions:
        words_data = [
            {
                "text": word.text,
                "bbox": {
                    "left": word.bbox.left,
                    "top": word.bbox.top,
                    "width": word.bbox.width,
                    "height": word.bbox.height,
                },
                "confidence": word.confidence if word.confidence != DEFAULT_CONFIDENCE else -1,
            }
            for word in result.words
        ]

    # Default: all boxes visible
    num_items = len(detector_regions_data) if show_detector_regions else len(result.words)
    default_visibility = [True] * num_items

    # Call the component
    component_value = _ocr_viewer(
        imageUrl=image_url,
        words=words_data,
        detectorRegions=detector_regions_data,
        defaultVisibility=default_visibility,
        key=key,
        default={"visibility": default_visibility},
    )

    return component_value
