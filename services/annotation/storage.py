"""
Annotation Storage Service

Handles saving and loading annotation datasets to/from JSON files.

Directory structure:
    data/annotations/
        <dataset_name>/
            dataset.json
            images/
                image1.jpg
                image2.jpg
"""
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .models import AnnotationDataset


class AnnotationStorage:
    """
    Storage service for annotation datasets

    Each dataset is stored in its own directory:
        <base_path>/<dataset_name>/
            dataset.json    - Dataset metadata and annotations
            images/         - Image files

    This structure matches the zip export format for easy training.
    """

    def __init__(self, base_path: Path = None):
        """
        Initialize storage service

        Args:
            base_path: Base directory for annotations (default: data/annotations)
        """
        if base_path is None:
            base_path = Path("data/annotations")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, name: str) -> Path:
        """Get the directory path for a dataset"""
        return self.base_path / name

    def _dataset_json_path(self, name: str) -> Path:
        """Get the JSON file path for a dataset"""
        return self._dataset_dir(name) / "dataset.json"

    def _dataset_images_dir(self, name: str) -> Path:
        """Get the images directory for a dataset"""
        return self._dataset_dir(name) / "images"

    def save(self, dataset: AnnotationDataset) -> Path:
        """
        Save dataset to JSON file

        Args:
            dataset: Dataset to save

        Returns:
            Path to the saved file
        """
        # Update timestamp
        dataset.updated_at = datetime.now()

        # Create dataset directory
        dataset_dir = self._dataset_dir(dataset.name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        file_path = self._dataset_json_path(dataset.name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(dataset.to_json(indent=2))

        return file_path

    def load(self, name: str) -> Optional[AnnotationDataset]:
        """
        Load dataset from JSON file

        Args:
            name: Dataset name

        Returns:
            Loaded dataset or None if not found
        """
        # Try new structure first
        file_path = self._dataset_json_path(name)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return AnnotationDataset.from_json(f.read())

        # Fall back to old structure (for migration)
        old_path = self.base_path / f"{name}.json"
        if old_path.exists():
            with open(old_path, "r", encoding="utf-8") as f:
                return AnnotationDataset.from_json(f.read())

        return None

    def exists(self, name: str) -> bool:
        """Check if a dataset exists"""
        return (
            self._dataset_json_path(name).exists()
            or (self.base_path / f"{name}.json").exists()  # Old structure
        )

    def delete(self, name: str) -> bool:
        """
        Delete a dataset and all its files

        Args:
            name: Dataset name

        Returns:
            True if deleted, False if not found
        """
        # Try new structure
        dataset_dir = self._dataset_dir(name)
        if dataset_dir.exists() and dataset_dir.is_dir():
            shutil.rmtree(dataset_dir)
            return True

        # Try old structure
        old_path = self.base_path / f"{name}.json"
        if old_path.exists():
            old_path.unlink()
            # Also delete old images directory if it exists
            old_images = self.base_path / "images" / name
            if old_images.exists():
                shutil.rmtree(old_images)
            return True

        return False

    def list_datasets(self) -> List[str]:
        """
        List available datasets

        Returns:
            List of dataset names
        """
        datasets = set()

        # New structure: directories with dataset.json
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "dataset.json").exists():
                datasets.add(item.name)

        # Old structure: <name>.json files at root
        for file_path in self.base_path.glob("*.json"):
            datasets.add(file_path.stem)

        return sorted(datasets)

    def get_image_path(self, relative_path: str) -> Path:
        """
        Get absolute path for a relative image path

        Args:
            relative_path: Relative path stored in dataset (e.g., "images/photo.jpg")

        Returns:
            Absolute path to the image
        """
        # For new structure, the path is relative to the dataset directory
        # For old structure, path was "images/<dataset>/<filename>"
        # We need to handle both cases

        # Check if it's in the new per-dataset structure
        # New paths look like: "images/filename.jpg" (relative to dataset dir)
        # Old paths look like: "images/<dataset>/filename.jpg" (relative to base)

        # First, try interpreting as old-style path (relative to base_path)
        old_style = self.base_path / relative_path
        if old_style.exists():
            return old_style

        # For new structure, we need the dataset name to construct the path
        # The caller should use get_image_path_for_dataset() instead
        # But for compatibility, return the base_path version
        return self.base_path / relative_path

    def get_image_path_for_dataset(self, dataset_name: str, relative_path: str) -> Path:
        """
        Get absolute path for an image in a specific dataset

        Args:
            dataset_name: Name of the dataset
            relative_path: Relative path stored in dataset (e.g., "images/photo.jpg")

        Returns:
            Absolute path to the image
        """
        return self._dataset_dir(dataset_name) / relative_path

    def copy_image(self, source_path: Path, dataset_name: str) -> str:
        """
        Copy an image to the dataset's image directory

        Args:
            source_path: Path to source image
            dataset_name: Name of the dataset

        Returns:
            Relative path to the copied image (for storing in dataset)
        """
        source_path = Path(source_path)

        # Create dataset images directory
        images_dir = self._dataset_images_dir(dataset_name)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy image with unique name if needed
        dest_name = source_path.name
        dest_path = images_dir / dest_name

        # Handle name collisions
        counter = 1
        while dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            dest_name = f"{stem}_{counter}{suffix}"
            dest_path = images_dir / dest_name
            counter += 1

        shutil.copy2(source_path, dest_path)

        # Return relative path (relative to dataset directory)
        return f"images/{dest_name}"

    def get_dataset_stats(self, name: str) -> Optional[dict]:
        """
        Get statistics for a dataset

        Args:
            name: Dataset name

        Returns:
            Dict with stats or None if dataset not found
        """
        dataset = self.load(name)
        if dataset is None:
            return None

        return {
            "name": dataset.name,
            "version": dataset.version,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat(),
            "num_images": len(dataset.images),
            "total_regions": dataset.total_regions,
            "labeled_regions": dataset.labeled_regions,
            "verified_regions": dataset.verified_regions,
        }

    def delete_image(
        self,
        dataset_name: str,
        image_id: str,
        delete_file: bool = True,
    ) -> bool:
        """
        Delete an image from a dataset and optionally remove the image file.

        Args:
            dataset_name: Name of the dataset
            image_id: ID of the image to delete
            delete_file: Whether to also delete the image file from disk

        Returns:
            True if image was found and deleted, False otherwise
        """
        dataset = self.load(dataset_name)
        if dataset is None:
            return False

        # Find the image to get its path before removal
        image = dataset.get_image(image_id)
        if image is None:
            return False

        image_path = image.image_path

        # Remove from dataset
        if not dataset.remove_image(image_id):
            return False

        # Save updated dataset
        self.save(dataset)

        # Delete image file if requested
        if delete_file and image_path:
            # Try new structure first
            full_path = self.get_image_path_for_dataset(dataset_name, image_path)
            if not full_path.exists():
                # Fall back to old structure
                full_path = self.base_path / image_path

            if full_path.exists():
                full_path.unlink()

        return True

    def migrate_dataset(self, name: str) -> bool:
        """
        Migrate a dataset from old structure to new per-dataset directory structure.

        Old structure:
            data/annotations/<name>.json
            data/annotations/images/<name>/<files>

        New structure:
            data/annotations/<name>/dataset.json
            data/annotations/<name>/images/<files>

        Args:
            name: Dataset name to migrate

        Returns:
            True if migrated, False if already new structure or not found
        """
        old_json = self.base_path / f"{name}.json"
        new_json = self._dataset_json_path(name)

        # Already migrated or doesn't exist
        if new_json.exists() or not old_json.exists():
            return False

        # Load dataset
        with open(old_json, "r", encoding="utf-8") as f:
            dataset = AnnotationDataset.from_json(f.read())

        # Create new directory structure
        dataset_dir = self._dataset_dir(name)
        new_images_dir = self._dataset_images_dir(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        new_images_dir.mkdir(parents=True, exist_ok=True)

        # Move images and update paths
        old_images_dir = self.base_path / "images" / name
        if old_images_dir.exists():
            for image_file in old_images_dir.iterdir():
                if image_file.is_file():
                    shutil.move(str(image_file), str(new_images_dir / image_file.name))

            # Remove old images directory if empty
            if not any(old_images_dir.iterdir()):
                old_images_dir.rmdir()

        # Update image paths in dataset
        for annotated_image in dataset.images:
            old_path = annotated_image.image_path
            # Old path: "images/<dataset>/<filename>" -> New path: "images/<filename>"
            if old_path.startswith(f"images/{name}/"):
                filename = old_path[len(f"images/{name}/"):]
                annotated_image.image_path = f"images/{filename}"

        # Save to new location
        with open(new_json, "w", encoding="utf-8") as f:
            f.write(dataset.to_json(indent=2))

        # Remove old JSON file
        old_json.unlink()

        # Clean up old images parent directory if empty
        old_images_parent = self.base_path / "images"
        if old_images_parent.exists() and not any(old_images_parent.iterdir()):
            old_images_parent.rmdir()

        return True

    def migrate_all(self) -> List[str]:
        """
        Migrate all datasets from old structure to new structure.

        Returns:
            List of migrated dataset names
        """
        migrated = []
        for json_file in self.base_path.glob("*.json"):
            name = json_file.stem
            if self.migrate_dataset(name):
                migrated.append(name)
        return migrated
