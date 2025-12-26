"""
Annotation Canvas - Streamlit component for drawing regions on images

Provides an interactive canvas for creating and editing annotation regions.
"""
import os
import streamlit.components.v1 as components
from typing import Optional, List, Dict, Any
from PIL import Image
import base64
from io import BytesIO

from .models import Region, Point
import config

# Declare the custom component
_RELEASE = config.ANNOTATION_CANVAS_RELEASE_MODE

if not _RELEASE:
    _annotation_canvas = components.declare_component(
        "annotation_canvas",
        url="http://localhost:5174",  # Vite dev server
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "../../frontend/annotation_canvas/build")
    _annotation_canvas = components.declare_component(
        "annotation_canvas",
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


def region_to_dict(region: Region) -> Dict[str, Any]:
    """Convert Region to dict format for JS component"""
    return {
        "id": region.id,
        "type": region.type,
        "points": [{"x": p.x, "y": p.y} for p in region.points],
        "text": region.text,
        "auto_detected": region.auto_detected,
        "verified": region.verified,
    }


def dict_to_region(data: Dict[str, Any]) -> Region:
    """Convert dict from JS component to Region"""
    return Region(
        id=data["id"],
        type=data["type"],
        points=[Point(x=p["x"], y=p["y"]) for p in data["points"]],
        text=data.get("text"),
        auto_detected=data.get("auto_detected", False),
        verified=data.get("verified", False),
    )


def annotation_canvas(
    image: Image.Image,
    regions: List[Region],
    selected_region_id: Optional[str] = None,
    drawing_mode: str = "select",
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Display interactive annotation canvas for drawing regions on images

    Args:
        image: PIL Image object to display
        regions: List of existing Region objects
        selected_region_id: ID of currently selected region (or None)
        drawing_mode: Drawing mode - 'rectangle', 'polygon', or 'select'
        key: Streamlit component key

    Returns:
        Dict with updated state:
        - regions: List of region dicts
        - selectedRegionId: ID of selected region or None
        - action: What action was performed ('add', 'delete', 'update', 'select', None)
    """
    # Convert image to base64 data URL
    image_url = image_to_base64(image)

    # Convert regions to dict format
    regions_data = [region_to_dict(r) for r in regions]

    # Call the component
    component_value = _annotation_canvas(
        imageUrl=image_url,
        regions=regions_data,
        selectedRegionId=selected_region_id,
        drawingMode=drawing_mode,
        key=key,
        default={
            "regions": regions_data,
            "selectedRegionId": selected_region_id,
            "action": None,
        },
    )

    return component_value


def parse_canvas_result(result: Optional[Dict[str, Any]]) -> tuple:
    """
    Parse the result from annotation_canvas component

    Args:
        result: Raw result dict from component

    Returns:
        Tuple of (regions: List[Region], selected_id: str | None, action: str | None)
    """
    if result is None:
        return [], None, None

    regions = [dict_to_region(r) for r in result.get("regions", [])]
    selected_id = result.get("selectedRegionId")
    action = result.get("action")

    return regions, selected_id, action
