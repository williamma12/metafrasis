"""
Annotation Exporter

Export annotated datasets to formats suitable for training OCR models:
- Detector training: COCO-style JSON with bounding boxes
- Recognizer training: Cropped images with text labels
"""
import json
import csv
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from PIL import Image
import numpy as np

from .models import AnnotationDataset, AnnotatedImage, Region, Point
from .storage import AnnotationStorage


class AnnotationExporter:
    """
    Export annotation datasets for OCR model training

    Supports:
    - Detector training: Bounding box annotations in COCO format
    - Recognizer training: Cropped region images with text labels
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

    def export_for_detector(
        self,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
        format: str = "coco",
        output_name: Optional[str] = None,
    ) -> Path:
        """
        Export bounding boxes for detector training

        Args:
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images
            format: Export format ('coco' for COCO JSON)
            output_name: Output filename (default: dataset name)

        Returns:
            Path to exported file
        """
        if format != "coco":
            raise ValueError(f"Unsupported format: {format}. Only 'coco' is supported.")

        output_name = output_name or dataset.name
        detector_dir = self.output_dir / "detector" / output_name
        detector_dir.mkdir(parents=True, exist_ok=True)

        # Build COCO format
        coco = {
            "info": {
                "description": f"OCR Detection Dataset: {dataset.name}",
                "version": dataset.version,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": [
                {"id": 1, "name": "text", "supercategory": "text"}
            ],
            "images": [],
            "annotations": [],
        }

        annotation_id = 1

        for img_idx, image in enumerate(dataset.images):
            # Add image entry
            coco["images"].append({
                "id": img_idx + 1,
                "file_name": image.image_path,
                "width": image.width,
                "height": image.height,
            })

            # Add annotations for each region
            for region in image.regions:
                x, y, w, h = region.get_bbox()

                # Build segmentation for polygons
                if region.type == "polygon" and region.points:
                    segmentation = [[coord for p in region.points for coord in (p.x, p.y)]]
                else:
                    segmentation = []

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx + 1,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                })
                annotation_id += 1

        # Write COCO JSON
        output_path = detector_dir / "annotations.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        # Write summary
        summary = {
            "dataset": dataset.name,
            "format": "coco",
            "num_images": len(dataset.images),
            "num_annotations": annotation_id - 1,
            "exported_at": datetime.now().isoformat(),
        }
        summary_path = detector_dir / "export_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return output_path

    def export_for_recognizer(
        self,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
        output_name: Optional[str] = None,
        labeled_only: bool = True,
        min_size: int = 5,
    ) -> Path:
        """
        Export cropped images and text labels for recognizer training

        Creates:
        - {region_id}.png - Cropped region image
        - {region_id}.txt - Text label
        - labels.csv - Master list of all exports

        Args:
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images
            output_name: Output directory name (default: dataset name)
            labeled_only: Only export regions with text labels
            min_size: Minimum region size in pixels

        Returns:
            Path to output directory
        """
        output_name = output_name or dataset.name
        recog_dir = self.output_dir / "recognizer" / output_name
        images_dir = recog_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        exports = []

        for image in dataset.images:
            # Load source image
            image_path = storage.get_image_path(image.image_path)
            if not image_path.exists():
                continue

            source_img = Image.open(image_path)

            for region in image.regions:
                # Skip unlabeled regions if required
                if labeled_only and not region.text:
                    continue

                # Get bounding box
                x, y, w, h = region.get_bbox()

                # Skip tiny regions
                if w < min_size or h < min_size:
                    continue

                # Crop region
                try:
                    crop = self._crop_region(source_img, region)
                except Exception as e:
                    print(f"Failed to crop region {region.id}: {e}")
                    continue

                # Save cropped image
                crop_filename = f"{region.id}.png"
                crop_path = images_dir / crop_filename
                crop.save(crop_path)

                # Save text label
                if region.text:
                    label_path = images_dir / f"{region.id}.txt"
                    label_path.write_text(region.text, encoding="utf-8")

                exports.append({
                    "region_id": region.id,
                    "source_image": image.image_path,
                    "crop_path": str(crop_path.relative_to(recog_dir)),
                    "text": region.text or "",
                    "type": region.type,
                    "auto_detected": region.auto_detected,
                    "verified": region.verified,
                    "width": crop.width,
                    "height": crop.height,
                })

        # Write labels CSV
        csv_path = recog_dir / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if exports:
                writer = csv.DictWriter(f, fieldnames=exports[0].keys())
                writer.writeheader()
                writer.writerows(exports)

        # Write summary
        summary = {
            "dataset": dataset.name,
            "num_exports": len(exports),
            "labeled_only": labeled_only,
            "exported_at": datetime.now().isoformat(),
        }
        summary_path = recog_dir / "export_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return recog_dir

    def _crop_region(self, image: Image.Image, region: Region) -> Image.Image:
        """
        Crop a region from an image

        Handles both rectangles and polygons.
        For polygons, crops the bounding box and masks the outside.

        Args:
            image: Source PIL Image
            region: Region to crop

        Returns:
            Cropped PIL Image
        """
        x, y, w, h = region.get_bbox()

        # Ensure bounds are within image
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(int(w), image.width - x)
        h = min(int(h), image.height - y)

        # Simple crop for rectangles
        if region.type == "rectangle":
            return image.crop((x, y, x + w, y + h))

        # For polygons, use mask
        crop = image.crop((x, y, x + w, y + h))

        # Create polygon mask
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)

        # Translate polygon points to crop coordinates
        polygon_points = [(p.x - x, p.y - y) for p in region.points]
        draw.polygon(polygon_points, fill=255)

        # Apply mask (set outside to white)
        result = Image.new("RGB", (w, h), (255, 255, 255))
        result.paste(crop, mask=mask)

        return result

    def export_both(
        self,
        dataset: AnnotationDataset,
        storage: AnnotationStorage,
        output_name: Optional[str] = None,
    ) -> dict:
        """
        Export for both detector and recognizer training

        Args:
            dataset: AnnotationDataset to export
            storage: AnnotationStorage for accessing images
            output_name: Output name (default: dataset name)

        Returns:
            Dict with paths to both exports
        """
        detector_path = self.export_for_detector(dataset, storage, output_name=output_name)
        recognizer_path = self.export_for_recognizer(dataset, storage, output_name=output_name)

        return {
            "detector": detector_path,
            "recognizer": recognizer_path,
        }
