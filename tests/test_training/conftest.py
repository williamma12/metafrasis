"""Fixtures for training tests."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from services.annotation.models import (
    AnnotationDataset,
    AnnotatedImage,
    Region,
    Point,
)
from services.annotation.storage import AnnotationStorage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_points():
    """Sample rectangle points."""
    return [
        Point(x=10, y=10),
        Point(x=110, y=10),
        Point(x=110, y=40),
        Point(x=10, y=40),
    ]


@pytest.fixture
def sample_region(sample_points):
    """Sample annotated region."""
    return Region(
        id="test_region_1",
        type="rectangle",
        points=sample_points,
        text="Hello",
        verified=True,
    )


@pytest.fixture
def sample_polygon_region():
    """Sample polygon region."""
    return Region(
        id="test_region_2",
        type="polygon",
        points=[
            Point(x=50, y=50),
            Point(x=150, y=50),
            Point(x=160, y=80),
            Point(x=140, y=100),
            Point(x=60, y=100),
            Point(x=40, y=80),
        ],
        text="World",
        verified=True,
    )


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    # Create a simple test image with some text-like features
    img = Image.new("RGB", (200, 100), color="white")
    pixels = img.load()

    # Draw some dark rectangles to simulate text
    for x in range(20, 100):
        for y in range(20, 40):
            pixels[x, y] = (0, 0, 0)

    for x in range(60, 180):
        for y in range(60, 80):
            pixels[x, y] = (50, 50, 50)

    image_path = temp_dir / "test_image.png"
    img.save(image_path)
    return image_path, img


@pytest.fixture
def sample_annotated_image(sample_region, sample_polygon_region):
    """Sample annotated image."""
    return AnnotatedImage(
        id="img_test1",
        image_path="images/test_image.png",
        width=200,
        height=100,
        regions=[sample_region, sample_polygon_region],
    )


@pytest.fixture
def sample_dataset(sample_annotated_image):
    """Sample annotation dataset."""
    dataset = AnnotationDataset(
        name="test_dataset",
        version="1.0",
    )
    dataset.add_image(sample_annotated_image)
    return dataset


@pytest.fixture
def sample_storage(temp_dir, sample_image):
    """Create storage with sample image."""
    storage = AnnotationStorage(base_path=temp_dir)

    # Create dataset directory and copy image
    dataset_dir = temp_dir / "test_dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path, img = sample_image
    img.save(images_dir / "test_image.png")

    return storage


@pytest.fixture
def numpy_polygon():
    """Sample polygon as numpy array."""
    return np.array([
        [10, 10],
        [110, 10],
        [110, 40],
        [10, 40],
    ], dtype=np.float32)


@pytest.fixture
def sample_heatmap():
    """Sample heatmap for testing."""
    heatmap = np.zeros((100, 200), dtype=np.float32)
    # Add some "text regions"
    heatmap[20:40, 20:100] = 0.8
    heatmap[60:80, 60:180] = 0.9
    return heatmap


@pytest.fixture
def sample_predictions():
    """Sample OCR predictions for testing metrics."""
    return [
        "hello world",
        "this is a test",
        "ancient greek",
        "αβγδ",
    ]


@pytest.fixture
def sample_references():
    """Sample OCR references for testing metrics."""
    return [
        "hello world",
        "this is a tset",  # typo
        "ancient greek text",  # missing word
        "αβγδε",  # missing character
    ]
