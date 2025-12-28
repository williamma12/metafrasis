"""
Annotation Service

Provides data models, storage, and UI components for creating OCR training datasets.

Usage:
    from services.annotation import AnnotationStorage, AnnotationDataset, Region, Point

    # Create a new dataset
    dataset = AnnotationDataset(name="my_dataset")

    # Add an image with regions
    from services.annotation.models import AnnotatedImage
    image = AnnotatedImage(image_path="images/page1.jpg", width=1024, height=768)
    image.add_region(Region.from_bbox(100, 200, 300, 50, text="Hello World"))
    dataset.add_image(image)

    # Save to disk
    storage = AnnotationStorage()
    storage.save(dataset)

    # Load from disk
    loaded = storage.load("my_dataset")

    # Delete an image from dataset (also removes file from disk)
    storage.delete_image("my_dataset", image_id="img_abc123")

    # Export as zip archive
    from services.annotation import AnnotationExporter
    exporter = AnnotationExporter()
    zip_path = exporter.export_dataset(dataset, storage)  # Saves to data/exports/
    zip_bytes = exporter.export_dataset_bytes(dataset, storage)  # For browser download

    # Export to directory (images copied to exports/<dataset>/images/)
    export_dir = exporter.export_to_directory(dataset, storage)

    # Use annotation canvas (in Streamlit app)
    from services.annotation import annotation_canvas
    result = annotation_canvas(pil_image, regions, drawing_mode="rectangle")
"""
from .models import (
    Point,
    Region,
    AnnotatedImage,
    AnnotationDataset,
)
from .storage import AnnotationStorage
from .exporter import AnnotationExporter

# Lazy imports for Streamlit components (avoid loading Streamlit in non-UI contexts)
_canvas_module = None


def __getattr__(name):
    """Lazy load Streamlit canvas components to avoid import warnings in training."""
    global _canvas_module
    if name in ("annotation_canvas", "parse_canvas_result"):
        if _canvas_module is None:
            from . import canvas as _canvas_module
        return getattr(_canvas_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Point",
    "Region",
    "AnnotatedImage",
    "AnnotationDataset",
    "AnnotationStorage",
    "AnnotationExporter",
    "annotation_canvas",
    "parse_canvas_result",
]
