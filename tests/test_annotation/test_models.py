"""
Tests for annotation data models
"""
import pytest
import json
from datetime import datetime

from services.annotation import (
    Point,
    Region,
    AnnotatedImage,
    AnnotationDataset,
)


class TestPoint:
    """Tests for Point dataclass"""

    def test_point_creation(self):
        """Test Point can be created with valid values"""
        point = Point(x=10.5, y=20.5)
        assert point.x == 10.5
        assert point.y == 20.5

    def test_point_to_dict(self):
        """Test Point serialization to dict"""
        point = Point(x=15.0, y=25.0)
        data = point.to_dict()

        assert data == {"x": 15.0, "y": 25.0}

    def test_point_from_dict(self):
        """Test Point deserialization from dict"""
        data = {"x": 30.0, "y": 40.0}
        point = Point.from_dict(data)

        assert point.x == 30.0
        assert point.y == 40.0

    def test_point_roundtrip(self):
        """Test Point serialization roundtrip"""
        original = Point(x=123.456, y=789.012)
        data = original.to_dict()
        restored = Point.from_dict(data)

        assert restored.x == original.x
        assert restored.y == original.y


class TestRegion:
    """Tests for Region dataclass"""

    def test_region_creation_default(self):
        """Test Region creation with defaults"""
        region = Region()

        assert region.id is not None
        assert len(region.id) == 8
        assert region.type == "rectangle"
        assert region.points == []
        assert region.text is None
        assert region.auto_detected is False
        assert region.verified is False
        assert isinstance(region.created_at, datetime)

    def test_region_creation_with_values(self, sample_points):
        """Test Region creation with explicit values"""
        region = Region(
            id="custom_id",
            type="polygon",
            points=sample_points,
            text="Test text",
            auto_detected=True,
            verified=True,
        )

        assert region.id == "custom_id"
        assert region.type == "polygon"
        assert len(region.points) == 4
        assert region.text == "Test text"
        assert region.auto_detected is True
        assert region.verified is True

    def test_region_from_bbox(self):
        """Test Region creation from bounding box"""
        region = Region.from_bbox(10, 20, 100, 50, text="bbox text")

        assert region.type == "rectangle"
        assert len(region.points) == 4
        assert region.text == "bbox text"

        # Check points form correct rectangle
        assert region.points[0].x == 10
        assert region.points[0].y == 20
        assert region.points[1].x == 110
        assert region.points[1].y == 20
        assert region.points[2].x == 110
        assert region.points[2].y == 70
        assert region.points[3].x == 10
        assert region.points[3].y == 70

    def test_region_get_bbox(self, sample_region):
        """Test Region bounding box calculation"""
        x, y, w, h = sample_region.get_bbox()

        assert x == 0
        assert y == 0
        assert w == 100
        assert h == 50

    def test_region_get_bbox_empty(self):
        """Test Region bounding box with no points"""
        region = Region(points=[])
        x, y, w, h = region.get_bbox()

        assert (x, y, w, h) == (0, 0, 0, 0)

    def test_region_get_bbox_polygon(self, sample_region_polygon):
        """Test Region bounding box for polygon"""
        x, y, w, h = sample_region_polygon.get_bbox()

        assert x == 0
        assert y == 0
        assert w == 100
        assert h == 100

    def test_region_to_dict(self, sample_region):
        """Test Region serialization to dict"""
        data = sample_region.to_dict()

        assert data["id"] == "test123"
        assert data["type"] == "rectangle"
        assert len(data["points"]) == 4
        assert data["text"] == "Hello World"
        assert data["auto_detected"] is False
        assert data["verified"] is True
        assert "created_at" in data

    def test_region_from_dict(self, sample_region):
        """Test Region deserialization from dict"""
        data = sample_region.to_dict()
        restored = Region.from_dict(data)

        assert restored.id == sample_region.id
        assert restored.type == sample_region.type
        assert len(restored.points) == len(sample_region.points)
        assert restored.text == sample_region.text
        assert restored.auto_detected == sample_region.auto_detected
        assert restored.verified == sample_region.verified

    def test_region_roundtrip(self, sample_region):
        """Test Region serialization roundtrip"""
        data = sample_region.to_dict()
        restored = Region.from_dict(data)

        assert restored.id == sample_region.id
        assert restored.type == sample_region.type
        assert restored.text == sample_region.text


class TestAnnotatedImage:
    """Tests for AnnotatedImage dataclass"""

    def test_annotated_image_creation_default(self):
        """Test AnnotatedImage creation with defaults"""
        img = AnnotatedImage()

        assert img.id.startswith("img_")
        assert img.image_path == ""
        assert img.width == 0
        assert img.height == 0
        assert img.regions == []

    def test_annotated_image_creation_with_values(self):
        """Test AnnotatedImage creation with explicit values"""
        img = AnnotatedImage(
            id="custom_img",
            image_path="path/to/image.png",
            width=800,
            height=600,
        )

        assert img.id == "custom_img"
        assert img.image_path == "path/to/image.png"
        assert img.width == 800
        assert img.height == 600
        assert img.regions == []

    def test_annotated_image_add_region(self, sample_region):
        """Test adding a region to an image"""
        img = AnnotatedImage(image_path="test.png", width=200, height=100)
        img.add_region(sample_region)

        assert len(img.regions) == 1
        assert img.regions[0].id == sample_region.id

    def test_annotated_image_remove_region(self, sample_annotated_image):
        """Test removing a region from an image"""
        initial_count = len(sample_annotated_image.regions)
        region_id = sample_annotated_image.regions[0].id

        result = sample_annotated_image.remove_region(region_id)

        assert result is True
        assert len(sample_annotated_image.regions) == initial_count - 1

    def test_annotated_image_remove_region_not_found(self, sample_annotated_image):
        """Test removing a non-existent region"""
        result = sample_annotated_image.remove_region("nonexistent_id")

        assert result is False

    def test_annotated_image_get_region(self, sample_annotated_image):
        """Test getting a region by ID"""
        region_id = sample_annotated_image.regions[0].id
        region = sample_annotated_image.get_region(region_id)

        assert region is not None
        assert region.id == region_id

    def test_annotated_image_get_region_not_found(self, sample_annotated_image):
        """Test getting a non-existent region"""
        region = sample_annotated_image.get_region("nonexistent_id")

        assert region is None

    def test_annotated_image_to_dict(self, sample_annotated_image):
        """Test AnnotatedImage serialization to dict"""
        data = sample_annotated_image.to_dict()

        assert data["id"] == sample_annotated_image.id
        assert data["image_path"] == sample_annotated_image.image_path
        assert data["width"] == sample_annotated_image.width
        assert data["height"] == sample_annotated_image.height
        assert len(data["regions"]) == len(sample_annotated_image.regions)

    def test_annotated_image_from_dict(self, sample_annotated_image):
        """Test AnnotatedImage deserialization from dict"""
        data = sample_annotated_image.to_dict()
        restored = AnnotatedImage.from_dict(data)

        assert restored.id == sample_annotated_image.id
        assert restored.image_path == sample_annotated_image.image_path
        assert restored.width == sample_annotated_image.width
        assert restored.height == sample_annotated_image.height
        assert len(restored.regions) == len(sample_annotated_image.regions)


class TestAnnotationDataset:
    """Tests for AnnotationDataset dataclass"""

    def test_dataset_creation_default(self):
        """Test AnnotationDataset creation with defaults"""
        dataset = AnnotationDataset()

        assert dataset.name == "untitled"
        assert dataset.version == "1.0"
        assert isinstance(dataset.created_at, datetime)
        assert isinstance(dataset.updated_at, datetime)
        assert dataset.images == []

    def test_dataset_creation_with_values(self):
        """Test AnnotationDataset creation with explicit values"""
        dataset = AnnotationDataset(
            name="my_dataset",
            version="2.0",
        )

        assert dataset.name == "my_dataset"
        assert dataset.version == "2.0"
        assert dataset.images == []

    def test_dataset_add_image(self, sample_annotated_image):
        """Test adding an image to a dataset"""
        dataset = AnnotationDataset(name="test")
        initial_updated_at = dataset.updated_at

        dataset.add_image(sample_annotated_image)

        assert len(dataset.images) == 1
        assert dataset.images[0].id == sample_annotated_image.id
        assert dataset.updated_at >= initial_updated_at

    def test_dataset_remove_image(self, sample_dataset):
        """Test removing an image from a dataset"""
        initial_count = len(sample_dataset.images)
        image_id = sample_dataset.images[0].id

        result = sample_dataset.remove_image(image_id)

        assert result is True
        assert len(sample_dataset.images) == initial_count - 1

    def test_dataset_remove_image_not_found(self, sample_dataset):
        """Test removing a non-existent image"""
        result = sample_dataset.remove_image("nonexistent_id")

        assert result is False

    def test_dataset_get_image(self, sample_dataset):
        """Test getting an image by ID"""
        image_id = sample_dataset.images[0].id
        image = sample_dataset.get_image(image_id)

        assert image is not None
        assert image.id == image_id

    def test_dataset_get_image_by_path(self, sample_dataset):
        """Test getting an image by path"""
        image_path = sample_dataset.images[0].image_path
        image = sample_dataset.get_image_by_path(image_path)

        assert image is not None
        assert image.image_path == image_path

    def test_dataset_total_regions(self, sample_dataset):
        """Test total regions count"""
        # sample_dataset has 2 images: one with 2 regions, one with 0
        assert sample_dataset.total_regions == 2

    def test_dataset_labeled_regions(self, sample_dataset):
        """Test labeled regions count"""
        # Only one region has text in sample_dataset
        assert sample_dataset.labeled_regions == 1

    def test_dataset_verified_regions(self, sample_dataset):
        """Test verified regions count"""
        # Only one region is verified in sample_dataset
        assert sample_dataset.verified_regions == 1

    def test_dataset_to_json(self, sample_dataset):
        """Test AnnotationDataset JSON serialization"""
        json_str = sample_dataset.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["name"] == sample_dataset.name
        assert data["version"] == sample_dataset.version
        assert len(data["images"]) == len(sample_dataset.images)

    def test_dataset_from_json(self, sample_dataset):
        """Test AnnotationDataset JSON deserialization"""
        json_str = sample_dataset.to_json()
        restored = AnnotationDataset.from_json(json_str)

        assert restored.name == sample_dataset.name
        assert restored.version == sample_dataset.version
        assert len(restored.images) == len(sample_dataset.images)

    def test_dataset_roundtrip(self, sample_dataset):
        """Test AnnotationDataset serialization roundtrip"""
        json_str = sample_dataset.to_json()
        restored = AnnotationDataset.from_json(json_str)

        assert restored.name == sample_dataset.name
        assert restored.total_regions == sample_dataset.total_regions
        assert restored.labeled_regions == sample_dataset.labeled_regions

    def test_dataset_to_dict(self, sample_dataset):
        """Test AnnotationDataset to_dict method"""
        data = sample_dataset.to_dict()

        assert data["name"] == sample_dataset.name
        assert data["version"] == sample_dataset.version
        assert "created_at" in data
        assert "updated_at" in data
        assert len(data["images"]) == len(sample_dataset.images)

    def test_dataset_from_dict(self, sample_dataset):
        """Test AnnotationDataset from_dict method"""
        data = sample_dataset.to_dict()
        restored = AnnotationDataset.from_dict(data)

        assert restored.name == sample_dataset.name
        assert restored.version == sample_dataset.version
