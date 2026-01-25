"""
Shared pytest fixtures for annotation tests
"""
import pytest
from pathlib import Path
from datetime import datetime
from PIL import Image

from app.services.annotation import (
    Point,
    Region,
    AnnotatedImage,
    AnnotationDataset,
    AnnotationStorage,
    AnnotationExporter,
)


@pytest.fixture
def sample_point():
    """Create a sample Point"""
    return Point(x=10.5, y=20.5)


@pytest.fixture
def sample_points():
    """Create sample points for a rectangle"""
    return [
        Point(x=0, y=0),
        Point(x=100, y=0),
        Point(x=100, y=50),
        Point(x=0, y=50),
    ]


@pytest.fixture
def sample_polygon_points():
    """Create sample points for a polygon"""
    return [
        Point(x=50, y=0),
        Point(x=100, y=25),
        Point(x=100, y=75),
        Point(x=50, y=100),
        Point(x=0, y=75),
        Point(x=0, y=25),
    ]


@pytest.fixture
def sample_region(sample_points):
    """Create a sample rectangle Region"""
    return Region(
        id="test123",
        type="rectangle",
        points=sample_points,
        text="Hello World",
        auto_detected=False,
        verified=True,
    )


@pytest.fixture
def sample_region_polygon(sample_polygon_points):
    """Create a sample polygon Region"""
    return Region(
        id="poly456",
        type="polygon",
        points=sample_polygon_points,
        text="Polygon Text",
        auto_detected=True,
        verified=False,
    )


@pytest.fixture
def sample_region_no_text(sample_points):
    """Create a Region without text"""
    return Region(
        id="notext789",
        type="rectangle",
        points=sample_points,
        text=None,
        auto_detected=True,
        verified=False,
    )


@pytest.fixture
def sample_image():
    """Create a simple test PIL image"""
    return Image.new('RGB', (200, 100), color='white')


@pytest.fixture
def sample_annotated_image(sample_region, sample_region_no_text):
    """Create a sample AnnotatedImage with regions"""
    img = AnnotatedImage(
        id="img_abc123",
        image_path="images/test.png",  # New structure: relative to dataset dir
        width=200,
        height=100,
    )
    img.add_region(sample_region)
    img.add_region(sample_region_no_text)
    return img


@pytest.fixture
def sample_annotated_image_empty():
    """Create an AnnotatedImage with no regions"""
    return AnnotatedImage(
        id="img_empty",
        image_path="images/empty.png",  # New structure: relative to dataset dir
        width=300,
        height=200,
    )


@pytest.fixture
def sample_dataset(sample_annotated_image, sample_annotated_image_empty):
    """Create a sample AnnotationDataset"""
    dataset = AnnotationDataset(
        name="test_dataset",
        version="1.0",
    )
    dataset.add_image(sample_annotated_image)
    dataset.add_image(sample_annotated_image_empty)
    return dataset


@pytest.fixture
def sample_dataset_empty():
    """Create an empty AnnotationDataset"""
    return AnnotationDataset(
        name="empty_dataset",
        version="1.0",
    )


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory"""
    storage_dir = tmp_path / "annotations"
    storage_dir.mkdir(exist_ok=True)
    return storage_dir


@pytest.fixture
def temp_storage(temp_storage_dir):
    """Create an AnnotationStorage with temporary directory"""
    return AnnotationStorage(base_path=temp_storage_dir)


@pytest.fixture
def temp_export_dir(tmp_path):
    """Create a temporary export directory"""
    export_dir = tmp_path / "exports"
    export_dir.mkdir(exist_ok=True)
    return export_dir


@pytest.fixture
def temp_exporter(temp_export_dir):
    """Create an AnnotationExporter with temporary directory"""
    return AnnotationExporter(output_dir=temp_export_dir)


@pytest.fixture
def storage_with_images(temp_storage, sample_dataset, sample_image):
    """Create storage with actual image files"""
    # Create dataset images directory (new structure)
    images_dir = temp_storage._dataset_images_dir(sample_dataset.name)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save test images
    for img in sample_dataset.images:
        # New structure: images are relative to dataset dir
        image_path = temp_storage.get_image_path_for_dataset(sample_dataset.name, img.image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        sample_image.save(image_path)

    # Save the dataset
    temp_storage.save(sample_dataset)

    return temp_storage, sample_dataset


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Automatically cleanup temporary files after each test"""
    yield
    # Cleanup happens automatically with tmp_path fixture
