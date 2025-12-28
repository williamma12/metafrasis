"""
Annotation Exporter

Export annotated datasets as:
- Zip archive containing dataset.json and images/
- Directory structure with dataset.json and images/
"""
import json
import shutil
import zipfile
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

from .models import AnnotationDataset
from .storage import AnnotationStorage


class AnnotationExporter:
    """
    Export annotation datasets as downloadable zip archives or directory exports
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize exporter

        Args:
            output_dir: Output directory for exports (default: data/exports)
        """
        if output_dir is None:
            output_dir = Path("data/exports")
        self.output_dir = Path(output_dir)

    def export_dataset(
        self,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
        output_name: Optional[str] = None,
    ) -> Path:
        """
        Export dataset as a zip archive to disk

        Creates a zip file containing:
        - dataset.json: Full dataset with annotations
        - images/: All source images

        Args:
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images
            output_name: Output filename (default: dataset name)

        Returns:
            Path to exported zip file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_name = output_name or dataset.name
        zip_path = self.output_dir / f"{output_name}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            self._write_dataset_to_zip(zf, dataset, storage)

        return zip_path

    def export_dataset_bytes(
        self,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
    ) -> bytes:
        """
        Export dataset as a zip archive in memory (for browser download)

        Creates a zip file containing:
        - dataset.json: Full dataset with annotations
        - images/: All source images

        Args:
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images

        Returns:
            Bytes of the zip file
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            self._write_dataset_to_zip(zf, dataset, storage)

        buffer.seek(0)
        return buffer.getvalue()

    def _write_dataset_to_zip(
        self,
        zf: zipfile.ZipFile,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
    ) -> None:
        """
        Write dataset contents to a zip file

        Args:
            zf: Open ZipFile to write to
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images
        """
        # Write dataset JSON
        dataset_json = dataset.to_json(indent=2)
        zf.writestr("dataset.json", dataset_json)

        # Write all images
        for image in dataset.images:
            # Try new structure first, then fall back to old
            image_path = storage.get_image_path_for_dataset(dataset.name, image.image_path)
            if not image_path.exists():
                image_path = storage.get_image_path(image.image_path)
            if image_path.exists():
                # Store in images/ subdirectory within zip
                arcname = f"images/{image_path.name}"
                zf.write(image_path, arcname)

        # Write export metadata
        metadata = {
            "dataset_name": dataset.name,
            "version": dataset.version,
            "num_images": len(dataset.images),
            "total_regions": dataset.total_regions,
            "labeled_regions": dataset.labeled_regions,
            "exported_at": datetime.now().isoformat(),
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    def export_to_directory(
        self,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
        output_name: Optional[str] = None,
    ) -> Path:
        """
        Export dataset to a directory structure.

        Creates a directory containing:
        - dataset.json: Full dataset with annotations
        - images/: All source images copied to this directory

        Args:
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images
            output_name: Output directory name (default: dataset name)

        Returns:
            Path to exported directory
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_name = output_name or dataset.name
        export_dir = self.output_dir / output_name

        # Clean existing export directory
        if export_dir.exists():
            shutil.rmtree(export_dir)

        export_dir.mkdir(parents=True, exist_ok=True)

        # Create images subdirectory
        images_dir = export_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Copy images and update paths in dataset copy
        exported_images = []
        for image in dataset.images:
            # Try new structure first, then fall back to old
            source_path = storage.get_image_path_for_dataset(dataset.name, image.image_path)
            if not source_path.exists():
                source_path = storage.get_image_path(image.image_path)
            if source_path.exists():
                # Copy image to export directory
                dest_path = images_dir / source_path.name
                shutil.copy2(source_path, dest_path)

                # Update image path in exported dataset
                from .models import AnnotatedImage
                exported_image = AnnotatedImage(
                    id=image.id,
                    image_path=f"images/{source_path.name}",
                    width=image.width,
                    height=image.height,
                    regions=image.regions,
                )
                exported_images.append(exported_image)

        # Create exported dataset with updated paths
        exported_dataset = AnnotationDataset(
            name=dataset.name,
            version=dataset.version,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            images=exported_images,
        )

        # Write dataset JSON
        dataset_path = export_dir / "dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(exported_dataset.to_json(indent=2))

        # Write export metadata
        metadata = {
            "dataset_name": dataset.name,
            "version": dataset.version,
            "num_images": len(exported_images),
            "total_regions": dataset.total_regions,
            "labeled_regions": dataset.labeled_regions,
            "exported_at": datetime.now().isoformat(),
        }
        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return export_dir
