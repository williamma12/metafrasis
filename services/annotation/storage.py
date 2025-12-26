"""
Annotation Storage Service

Handles saving and loading annotation datasets to/from JSON files.
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

    Handles:
    - Saving datasets to JSON files
    - Loading datasets from JSON files
    - Listing available datasets
    - Copying images to dataset directory
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

        # Create images subdirectory
        self.images_path = self.base_path / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)

    def _dataset_path(self, name: str) -> Path:
        """Get the file path for a dataset"""
        return self.base_path / f"{name}.json"

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

        # Write to file
        file_path = self._dataset_path(dataset.name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(dataset.to_json(indent=2))

        return file_path

    def load(self, name: str) -> Optional[AnnotationDataset]:
        """
        Load dataset from JSON file

        Args:
            name: Dataset name (without .json extension)

        Returns:
            Loaded dataset or None if not found
        """
        file_path = self._dataset_path(name)
        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return AnnotationDataset.from_json(f.read())

    def exists(self, name: str) -> bool:
        """Check if a dataset exists"""
        return self._dataset_path(name).exists()

    def delete(self, name: str) -> bool:
        """
        Delete a dataset

        Args:
            name: Dataset name

        Returns:
            True if deleted, False if not found
        """
        file_path = self._dataset_path(name)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_datasets(self) -> List[str]:
        """
        List available datasets

        Returns:
            List of dataset names (without .json extension)
        """
        datasets = []
        for file_path in self.base_path.glob("*.json"):
            datasets.append(file_path.stem)
        return sorted(datasets)

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

        # Create dataset-specific image directory
        dataset_images = self.images_path / dataset_name
        dataset_images.mkdir(parents=True, exist_ok=True)

        # Copy image with unique name if needed
        dest_name = source_path.name
        dest_path = dataset_images / dest_name

        # Handle name collisions
        counter = 1
        while dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            dest_name = f"{stem}_{counter}{suffix}"
            dest_path = dataset_images / dest_name
            counter += 1

        shutil.copy2(source_path, dest_path)

        # Return relative path
        return str(dest_path.relative_to(self.base_path))

    def get_image_path(self, relative_path: str) -> Path:
        """
        Get absolute path for a relative image path

        Args:
            relative_path: Relative path stored in dataset

        Returns:
            Absolute path to the image
        """
        return self.base_path / relative_path

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
