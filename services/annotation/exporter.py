"""
Annotation Exporter

Export annotated datasets as a zip archive containing:
- dataset.json: The full dataset with all annotations
- images/: All source images
"""
import json
import zipfile
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

from .models import AnnotationDataset
from .storage import AnnotationStorage


class AnnotationExporter:
    """
    Export annotation datasets as downloadable zip archives
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize exporter

        Args:
            output_dir: Output directory for exports (default: data/datasets)
        """
        if output_dir is None:
            output_dir = Path("data/datasets")
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
