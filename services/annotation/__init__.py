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

    # Export as zip archive
    from services.annotation import AnnotationExporter
    exporter = AnnotationExporter()
    zip_path = exporter.export_dataset(dataset, storage)  # Saves to data/datasets/
    zip_bytes = exporter.export_dataset_bytes(dataset, storage)  # For browser download

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
from .canvas import annotation_canvas, parse_canvas_result
from .exporter import AnnotationExporter

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
