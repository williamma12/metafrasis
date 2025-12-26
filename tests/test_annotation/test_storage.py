"""
Tests for AnnotationStorage service
"""
import pytest
from pathlib import Path

from services.annotation import (
    AnnotationStorage,
    AnnotationDataset,
    AnnotatedImage,
    Region,
    Point,
)


class TestAnnotationStorageInit:
    """Tests for AnnotationStorage initialization"""

    def test_storage_creates_directories(self, tmp_path):
        """Test storage creates base and images directories"""
        base_path = tmp_path / "new_storage"
        storage = AnnotationStorage(base_path=base_path)

        assert base_path.exists()
        assert (base_path / "images").exists()

    def test_storage_default_path(self):
        """Test storage uses default path when none provided"""
        storage = AnnotationStorage()

        assert storage.base_path == Path("data/annotations")


class TestAnnotationStorageSaveLoad:
    """Tests for saving and loading datasets"""

    def test_save_dataset(self, temp_storage, sample_dataset):
        """Test saving a dataset"""
        path = temp_storage.save(sample_dataset)

        assert path.exists()
        assert path.suffix == ".json"
        assert path.stem == sample_dataset.name

    def test_load_dataset(self, temp_storage, sample_dataset):
        """Test loading a dataset"""
        temp_storage.save(sample_dataset)
        loaded = temp_storage.load(sample_dataset.name)

        assert loaded is not None
        assert loaded.name == sample_dataset.name
        assert len(loaded.images) == len(sample_dataset.images)

    def test_load_dataset_not_found(self, temp_storage):
        """Test loading a non-existent dataset"""
        loaded = temp_storage.load("nonexistent")

        assert loaded is None

    def test_save_updates_timestamp(self, temp_storage, sample_dataset):
        """Test that saving updates the updated_at timestamp"""
        original_updated = sample_dataset.updated_at

        import time
        time.sleep(0.01)  # Small delay to ensure different timestamp

        temp_storage.save(sample_dataset)

        assert sample_dataset.updated_at > original_updated

    def test_save_load_roundtrip(self, temp_storage, sample_dataset):
        """Test full save/load roundtrip preserves data"""
        temp_storage.save(sample_dataset)
        loaded = temp_storage.load(sample_dataset.name)

        assert loaded.name == sample_dataset.name
        assert loaded.version == sample_dataset.version
        assert len(loaded.images) == len(sample_dataset.images)
        assert loaded.total_regions == sample_dataset.total_regions

    def test_save_overwrites_existing(self, temp_storage, sample_dataset):
        """Test saving overwrites existing dataset"""
        temp_storage.save(sample_dataset)

        # Modify and save again
        sample_dataset.version = "2.0"
        temp_storage.save(sample_dataset)

        loaded = temp_storage.load(sample_dataset.name)
        assert loaded.version == "2.0"


class TestAnnotationStorageExists:
    """Tests for checking dataset existence"""

    def test_exists_true(self, temp_storage, sample_dataset):
        """Test exists returns True for existing dataset"""
        temp_storage.save(sample_dataset)

        assert temp_storage.exists(sample_dataset.name) is True

    def test_exists_false(self, temp_storage):
        """Test exists returns False for non-existent dataset"""
        assert temp_storage.exists("nonexistent") is False


class TestAnnotationStorageDelete:
    """Tests for deleting datasets"""

    def test_delete_dataset(self, temp_storage, sample_dataset):
        """Test deleting a dataset"""
        temp_storage.save(sample_dataset)
        result = temp_storage.delete(sample_dataset.name)

        assert result is True
        assert temp_storage.exists(sample_dataset.name) is False

    def test_delete_dataset_not_found(self, temp_storage):
        """Test deleting a non-existent dataset"""
        result = temp_storage.delete("nonexistent")

        assert result is False


class TestAnnotationStorageList:
    """Tests for listing datasets"""

    def test_list_datasets_empty(self, temp_storage):
        """Test listing datasets when none exist"""
        datasets = temp_storage.list_datasets()

        assert datasets == []

    def test_list_datasets(self, temp_storage):
        """Test listing multiple datasets"""
        # Create several datasets
        for name in ["alpha", "beta", "gamma"]:
            dataset = AnnotationDataset(name=name)
            temp_storage.save(dataset)

        datasets = temp_storage.list_datasets()

        assert len(datasets) == 3
        assert "alpha" in datasets
        assert "beta" in datasets
        assert "gamma" in datasets

    def test_list_datasets_sorted(self, temp_storage):
        """Test listed datasets are sorted"""
        for name in ["zebra", "apple", "mango"]:
            dataset = AnnotationDataset(name=name)
            temp_storage.save(dataset)

        datasets = temp_storage.list_datasets()

        assert datasets == ["apple", "mango", "zebra"]


class TestAnnotationStorageImages:
    """Tests for image copying and path resolution"""

    def test_copy_image(self, temp_storage, sample_image, tmp_path):
        """Test copying an image to storage"""
        # Save sample image to temp location
        source_path = tmp_path / "source_image.png"
        sample_image.save(source_path)

        # Copy to storage
        relative_path = temp_storage.copy_image(source_path, "test_dataset")

        assert relative_path == "images/test_dataset/source_image.png"
        assert (temp_storage.base_path / relative_path).exists()

    def test_copy_image_collision(self, temp_storage, sample_image, tmp_path):
        """Test copying images with same name"""
        source_path = tmp_path / "image.png"
        sample_image.save(source_path)

        # Copy twice
        path1 = temp_storage.copy_image(source_path, "test_dataset")
        path2 = temp_storage.copy_image(source_path, "test_dataset")

        assert path1 != path2
        assert "image.png" in path1
        assert "image_1.png" in path2

    def test_get_image_path(self, temp_storage):
        """Test getting absolute path from relative path"""
        relative_path = "images/test_dataset/image.png"
        absolute_path = temp_storage.get_image_path(relative_path)

        assert absolute_path == temp_storage.base_path / relative_path


class TestAnnotationStorageStats:
    """Tests for getting dataset statistics"""

    def test_get_dataset_stats(self, temp_storage, sample_dataset):
        """Test getting stats for a dataset"""
        temp_storage.save(sample_dataset)
        stats = temp_storage.get_dataset_stats(sample_dataset.name)

        assert stats is not None
        assert stats["name"] == sample_dataset.name
        assert stats["version"] == sample_dataset.version
        assert stats["num_images"] == len(sample_dataset.images)
        assert stats["total_regions"] == sample_dataset.total_regions
        assert stats["labeled_regions"] == sample_dataset.labeled_regions
        assert stats["verified_regions"] == sample_dataset.verified_regions
        assert "created_at" in stats
        assert "updated_at" in stats

    def test_get_dataset_stats_not_found(self, temp_storage):
        """Test getting stats for non-existent dataset"""
        stats = temp_storage.get_dataset_stats("nonexistent")

        assert stats is None
