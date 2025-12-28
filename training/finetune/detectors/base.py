"""
Base trainer class for text detection models.

Provides shared functionality for detection model training including
dataset loading, loss computation, and metric evaluation.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from training.evaluate.detectors import PrecisionRecallF1Metric

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from services.annotation.models import AnnotationDataset, Region
from services.annotation.storage import AnnotationStorage
from training.evaluate.detectors import PrecisionRecallF1Metric
from ..base import BaseTrainer


class DetectorDataset(Dataset):
    """
    Base dataset class for detector training.

    Supports two data formats:
    1. Annotation format: Load directly from AnnotationDataset JSON files
    2. Exported format: Load pre-generated target maps from disk

    For annotation format, set `annotation_dataset` parameter.
    For exported format, the data_dir should contain images/ and targets/ subdirs.
    """

    def __init__(
        self,
        data_dir: Path,
        img_size: int = 640,
        augment: bool = False,
        annotation_dataset: Optional[AnnotationDataset] = None,
        storage: Optional[AnnotationStorage] = None,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment

        # Annotation-based loading
        self.annotation_dataset = annotation_dataset
        self.storage = storage
        self.use_annotations = annotation_dataset is not None

        if self.use_annotations:
            # Load from annotation dataset
            self.samples = list(range(len(annotation_dataset.images)))
        else:
            # Load from exported format
            self.images_dir = self.data_dir / "images"
            self.targets_dir = self.data_dir / "targets"

            self.samples = sorted(self.images_dir.glob("*.png"))
            if not self.samples:
                self.samples = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def load_targets(self, sample_name: str) -> Dict[str, np.ndarray]:
        """Load target maps for a sample from exported format. Override in subclass."""
        pass

    @abstractmethod
    def generate_targets(
        self, image: Image.Image, regions: List[Region]
    ) -> Dict[str, np.ndarray]:
        """Generate target maps from regions. Override in subclass."""
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.use_annotations:
            return self._getitem_annotation(idx)
        else:
            return self._getitem_exported(idx)

    def _getitem_annotation(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Load item from annotation dataset."""
        annotated_image = self.annotation_dataset.images[idx]

        # Load image
        image_path = self.storage.get_image_path(annotated_image.image_path)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        # Scale region points to match resized image
        scale_x = self.img_size / original_size[0]
        scale_y = self.img_size / original_size[1]

        from services.annotation.models import Point
        scaled_regions = []
        for region in annotated_image.regions:
            scaled_points = [
                Point(x=p.x * scale_x, y=p.y * scale_y)
                for p in region.points
            ]
            scaled_region = Region(
                id=region.id,
                type=region.type,
                points=scaled_points,
                text=region.text,
                auto_detected=region.auto_detected,
                verified=region.verified,
            )
            scaled_regions.append(scaled_region)

        # Generate target maps
        targets = self.generate_targets(image, scaled_regions)

        # Preprocess image
        image_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).float()

        targets = {k: torch.tensor(v).float() for k, v in targets.items()}
        return image_tensor, targets

    def _getitem_exported(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Load item from exported format."""
        image_path = self.samples[idx]
        sample_name = image_path.stem

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1).float()

        # Load targets
        targets = self.load_targets(sample_name)
        targets = {k: torch.tensor(v).float() for k, v in targets.items()}

        return image, targets


class DetectorTrainer(BaseTrainer):
    """
    Base class for detection model trainers (CRAFT, DB).

    Provides shared functionality for:
    - Dataset loading with target maps
    - Training step with gradient clipping
    - Validation with detection metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector trainer.

        Args:
            config: Training configuration dictionary
        """
        super().__init__(config)
        self.eval_metric = PrecisionRecallF1Metric(
            iou_threshold=config.get("iou_threshold", 0.5)
        )

    @property
    def dataset_class(self) -> type:
        """Return the dataset class to use. Override in subclass."""
        return DetectorDataset

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """Return dataset-specific kwargs for detectors."""
        return {
            "img_size": self.config.get("img_size", 640),
        }

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Tuple of (images, targets)

        Returns:
            Dictionary of loss values
        """
        images, targets = batch
        images = images.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}

        # Forward pass
        outputs = self.model(images)

        # Compute loss
        loss, loss_dict = self.compute_loss(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        return {"loss": loss.item(), **loss_dict}

    @abstractmethod
    def compute_loss(
        self, outputs: Any, targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute detection loss.

        Args:
            outputs: Model outputs
            targets: Target maps

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        pass

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            dataloader: Validation dataloader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        loss_sums: Dict[str, float] = {}

        with torch.no_grad():
            for batch in dataloader:
                images, targets = batch
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                outputs = self.model(images)
                loss, loss_dict = self.compute_loss(outputs, targets)

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_sums[k] = loss_sums.get(k, 0.0) + v

        n = len(dataloader)
        avg_losses = {f"val_{k}": v / n for k, v in loss_sums.items()}
        avg_losses["val_loss"] = total_loss / n

        return avg_losses

    def outputs_to_polygons(
        self, outputs: Any, threshold: float = 0.5
    ) -> List[np.ndarray]:
        """
        Post-process model outputs to polygon predictions.

        Override this method to enable detection metrics computation.
        Returns empty list by default (metrics will be skipped).

        Args:
            outputs: Model outputs (e.g., heatmaps)
            threshold: Detection threshold

        Returns:
            List of predicted polygons as [N, 2] arrays
        """
        return []

    def compute_detection_metrics(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute detection metrics (precision, recall, F1).

        Args:
            predictions: List of predicted polygons
            targets: List of target polygons

        Returns:
            Dictionary of metric values
        """
        if not predictions or not targets:
            return {}

        result = self.eval_metric.compute(predictions, targets)
        return {
            "precision": result.details["precision"],
            "recall": result.details["recall"],
            "f1": result.details["f1"],
        }
