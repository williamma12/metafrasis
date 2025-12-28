"""
Tests for AnnotationExporter service
"""
import pytest
import json
import zipfile
from pathlib import Path

from services.annotation import (
    AnnotationExporter,
    AnnotationStorage,
    AnnotationDataset,
    AnnotatedImage,
    Region,
    Point,
)


class TestAnnotationExporterInit:
    """Tests for AnnotationExporter initialization"""

    def test_exporter_default_path(self):
        """Test exporter uses default path when none provided"""
        exporter = AnnotationExporter()

        assert exporter.output_dir == Path("data/exports")

    def test_exporter_custom_path(self, tmp_path):
        """Test exporter uses custom path when provided"""
        custom_path = tmp_path / "custom_exports"
        exporter = AnnotationExporter(output_dir=custom_path)

        assert exporter.output_dir == custom_path


class TestAnnotationExporterExportDataset:
    """Tests for export_dataset method"""

    def test_export_dataset_creates_zip(self, storage_with_images, temp_exporter):
        """Test export creates a zip file"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"
        assert zip_path.stem == dataset.name

    def test_export_dataset_creates_output_dir(self, storage_with_images, tmp_path):
        """Test export creates output directory if it doesn't exist"""
        storage, dataset = storage_with_images
        new_dir = tmp_path / "new_exports"
        exporter = AnnotationExporter(output_dir=new_dir)

        zip_path = exporter.export_dataset(dataset, storage)

        assert new_dir.exists()
        assert zip_path.exists()

    def test_export_dataset_contains_json(self, storage_with_images, temp_exporter):
        """Test exported zip contains dataset.json"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "dataset.json" in zf.namelist()

            # Verify JSON content
            json_content = zf.read("dataset.json").decode("utf-8")
            data = json.loads(json_content)
            assert data["name"] == dataset.name
            assert len(data["images"]) == len(dataset.images)

    def test_export_dataset_contains_images(self, storage_with_images, temp_exporter):
        """Test exported zip contains images"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            image_files = [n for n in zf.namelist() if n.startswith("images/")]
            assert len(image_files) == len(dataset.images)

    def test_export_dataset_contains_metadata(self, storage_with_images, temp_exporter):
        """Test exported zip contains metadata.json"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "metadata.json" in zf.namelist()

            # Verify metadata content
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
            assert metadata["dataset_name"] == dataset.name
            assert metadata["num_images"] == len(dataset.images)
            assert metadata["total_regions"] == dataset.total_regions
            assert "exported_at" in metadata

    def test_export_dataset_custom_name(self, storage_with_images, temp_exporter):
        """Test export with custom output name"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(
            dataset, storage, output_name="custom_name"
        )

        assert zip_path.stem == "custom_name"


class TestAnnotationExporterExportBytes:
    """Tests for export_dataset_bytes method"""

    def test_export_bytes_returns_bytes(self, storage_with_images, temp_exporter):
        """Test export_dataset_bytes returns bytes"""
        storage, dataset = storage_with_images

        result = temp_exporter.export_dataset_bytes(dataset, storage)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_export_bytes_valid_zip(self, storage_with_images, temp_exporter):
        """Test exported bytes is a valid zip file"""
        storage, dataset = storage_with_images

        result = temp_exporter.export_dataset_bytes(dataset, storage)

        # Write to temp file and verify
        import io
        buffer = io.BytesIO(result)

        with zipfile.ZipFile(buffer, "r") as zf:
            assert "dataset.json" in zf.namelist()
            assert "metadata.json" in zf.namelist()

    def test_export_bytes_contains_same_as_file(self, storage_with_images, temp_exporter):
        """Test bytes export contains same content as file export"""
        storage, dataset = storage_with_images

        # Export to file
        zip_path = temp_exporter.export_dataset(dataset, storage)

        # Export to bytes
        zip_bytes = temp_exporter.export_dataset_bytes(dataset, storage)

        # Compare contents
        import io
        with zipfile.ZipFile(zip_path, "r") as zf_file:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf_bytes:
                file_names = set(zf_file.namelist())
                bytes_names = set(zf_bytes.namelist())

                assert file_names == bytes_names


class TestAnnotationExporterEmptyDataset:
    """Tests for exporting empty datasets"""

    def test_export_empty_dataset(self, temp_storage, temp_exporter):
        """Test exporting a dataset with no images"""
        dataset = AnnotationDataset(name="empty")
        temp_storage.save(dataset)

        zip_path = temp_exporter.export_dataset(dataset, temp_storage)

        assert zip_path.exists()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "dataset.json" in zf.namelist()
            assert "metadata.json" in zf.namelist()

            # No images
            image_files = [n for n in zf.namelist() if n.startswith("images/")]
            assert len(image_files) == 0

    def test_export_dataset_missing_images(self, temp_storage, temp_exporter):
        """Test exporting dataset when some images are missing from disk"""
        dataset = AnnotationDataset(name="missing_images")
        dataset.add_image(AnnotatedImage(
            image_path="images/missing/nonexistent.png",
            width=100,
            height=100,
        ))
        temp_storage.save(dataset)

        # Should not raise, just skip missing images
        zip_path = temp_exporter.export_dataset(dataset, temp_storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Dataset JSON should still be there
            assert "dataset.json" in zf.namelist()

            # No images since the file doesn't exist
            image_files = [n for n in zf.namelist() if n.startswith("images/")]
            assert len(image_files) == 0


class TestAnnotationExporterZipContents:
    """Tests for detailed zip contents verification"""

    def test_zip_is_compressed(self, storage_with_images, temp_exporter):
        """Test that zip uses deflate compression"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_DEFLATED

    def test_dataset_json_matches_original(self, storage_with_images, temp_exporter):
        """Test that exported dataset.json matches original dataset"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            json_content = zf.read("dataset.json").decode("utf-8")
            exported = AnnotationDataset.from_json(json_content)

            assert exported.name == dataset.name
            assert exported.version == dataset.version
            assert len(exported.images) == len(dataset.images)
            assert exported.total_regions == dataset.total_regions
            assert exported.labeled_regions == dataset.labeled_regions

    def test_metadata_has_correct_counts(self, storage_with_images, temp_exporter):
        """Test that metadata has correct counts"""
        storage, dataset = storage_with_images

        zip_path = temp_exporter.export_dataset(dataset, storage)

        with zipfile.ZipFile(zip_path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))

            assert metadata["num_images"] == len(dataset.images)
            assert metadata["total_regions"] == dataset.total_regions
            assert metadata["labeled_regions"] == dataset.labeled_regions
            assert metadata["version"] == dataset.version


class TestAnnotationExporterExportToDirectory:
    """Tests for export_to_directory method"""

    def test_export_to_directory_creates_dir(self, storage_with_images, temp_exporter):
        """Test export creates a directory structure"""
        storage, dataset = storage_with_images

        export_dir = temp_exporter.export_to_directory(dataset, storage)

        assert export_dir.exists()
        assert export_dir.is_dir()
        assert (export_dir / "dataset.json").exists()
        assert (export_dir / "metadata.json").exists()
        assert (export_dir / "images").is_dir()

    def test_export_to_directory_contains_images(self, storage_with_images, temp_exporter):
        """Test exported directory contains all images"""
        storage, dataset = storage_with_images

        export_dir = temp_exporter.export_to_directory(dataset, storage)

        images_dir = export_dir / "images"
        image_files = list(images_dir.glob("*"))

        assert len(image_files) == len(dataset.images)

    def test_export_to_directory_updates_paths(self, storage_with_images, temp_exporter):
        """Test that image paths in dataset.json are updated to exported structure"""
        storage, dataset = storage_with_images

        export_dir = temp_exporter.export_to_directory(dataset, storage)

        # Load exported dataset
        with open(export_dir / "dataset.json") as f:
            exported = AnnotationDataset.from_json(f.read())

        # All paths should start with images/
        for image in exported.images:
            assert image.image_path.startswith("images/")

    def test_export_to_directory_custom_name(self, storage_with_images, temp_exporter):
        """Test export with custom output name"""
        storage, dataset = storage_with_images

        export_dir = temp_exporter.export_to_directory(
            dataset, storage, output_name="custom_export"
        )

        assert export_dir.name == "custom_export"
        assert export_dir.exists()

    def test_export_to_directory_overwrites_existing(self, storage_with_images, temp_exporter):
        """Test that exporting to same directory overwrites it"""
        storage, dataset = storage_with_images

        # Export twice
        export_dir1 = temp_exporter.export_to_directory(dataset, storage)
        export_dir2 = temp_exporter.export_to_directory(dataset, storage)

        assert export_dir1 == export_dir2
        assert export_dir2.exists()

    def test_export_to_directory_metadata(self, storage_with_images, temp_exporter):
        """Test that metadata.json is created correctly"""
        storage, dataset = storage_with_images

        export_dir = temp_exporter.export_to_directory(dataset, storage)

        with open(export_dir / "metadata.json") as f:
            metadata = json.load(f)

        assert metadata["dataset_name"] == dataset.name
        assert metadata["num_images"] == len(dataset.images)
        assert "exported_at" in metadata
