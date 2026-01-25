"""
DB (Differentiable Binarization) detector trainer.

DBNet uses ResNet backbone with FPN and produces:
- Probability maps
- Threshold maps
- Binary maps (via differentiable binarization)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from ml.models import DBNet
from app.services.annotation.models import Region
from ml.training.data.base import (
    region_to_polygon,
    get_region_mask,
    compute_shrunk_polygon,
    compute_distance_map,
)
from .base import DetectorTrainer, DetectorDataset


class DBDataset(DetectorDataset):
    """Dataset for DB training with probability and threshold maps."""

    def __init__(self, *args, shrink_ratio: float = 0.4, thresh_min: float = 0.3, thresh_max: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def load_targets(self, sample_name: str) -> Dict[str, np.ndarray]:
        """Load probability, threshold, and shrink mask from exported format."""
        prob_map = np.load(
            self.targets_dir / "probability_map" / f"{sample_name}.npy"
        )
        thresh_map = np.load(
            self.targets_dir / "threshold_map" / f"{sample_name}.npy"
        )
        shrink_mask = np.load(
            self.targets_dir / "shrink_mask" / f"{sample_name}.npy"
        )

        # Return at full resolution (model outputs at same resolution as input)
        return {
            "prob_map": prob_map[np.newaxis, ...].astype(np.float32),
            "thresh_map": thresh_map[np.newaxis, ...].astype(np.float32),
            "shrink_mask": shrink_mask[np.newaxis, ...].astype(np.float32),
        }

    def generate_targets(
        self, image: Image.Image, regions: List[Region]
    ) -> Dict[str, np.ndarray]:
        """Generate DB-style probability, threshold, and shrink mask from annotation regions."""
        width, height = image.size

        # Generate probability map (shrunk polygons)
        prob_map = np.zeros((height, width), dtype=np.float32)
        for region in regions:
            polygon = region_to_polygon(region)
            shrunk = compute_shrunk_polygon(polygon, self.shrink_ratio)
            shrunk_int = shrunk.astype(np.int32)
            cv2.fillPoly(prob_map, [shrunk_int], 1.0)

        # Generate threshold map (distance-based)
        thresh_map = self._generate_threshold_map((width, height), regions)

        # Generate shrink mask (original polygons)
        shrink_mask = np.zeros((height, width), dtype=np.float32)
        for region in regions:
            polygon = region_to_polygon(region)
            polygon_int = polygon.astype(np.int32)
            cv2.fillPoly(shrink_mask, [polygon_int], 1.0)

        # Return at full resolution (model outputs at same resolution as input)
        return {
            "prob_map": prob_map[np.newaxis, ...].astype(np.float32),
            "thresh_map": thresh_map[np.newaxis, ...].astype(np.float32),
            "shrink_mask": shrink_mask[np.newaxis, ...].astype(np.float32),
        }

    def _generate_threshold_map(
        self, size: Tuple[int, int], regions: List[Region]
    ) -> np.ndarray:
        """Generate threshold map based on distance from text boundaries."""
        width, height = size
        thresh_map = np.zeros((height, width), dtype=np.float32)

        for region in regions:
            # Create mask for this region
            mask = get_region_mask(region, size)

            # Compute distance from boundary
            dist_inside = cv2.distanceTransform(
                (mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5
            )
            dist_outside = cv2.distanceTransform(
                (mask <= 0.5).astype(np.uint8), cv2.DIST_L2, 5
            )

            # Combine: positive inside, negative outside
            dist = dist_inside - dist_outside

            # Normalize to threshold range
            if dist.max() > dist.min():
                dist_norm = (dist - dist.min()) / (dist.max() - dist.min())
                local_thresh = self.thresh_min + dist_norm * (self.thresh_max - self.thresh_min)
            else:
                local_thresh = np.full_like(dist, self.thresh_min)

            # Only update where this region exists
            region_mask = mask > 0.5
            thresh_map[region_mask] = np.maximum(thresh_map[region_mask], local_thresh[region_mask])

        return thresh_map


class DBLoss(nn.Module):
    """
    Loss function for DB training.

    Combines:
    - BCE loss on probability map
    - L1 loss on threshold map
    - Dice loss on binary map
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        l1_weight: float = 10.0,
        dice_weight: float = 1.0,
    ):
        """
        Initialize DB loss.

        Args:
            bce_weight: Weight for BCE loss on probability map
            l1_weight: Weight for L1 loss on threshold map
            dice_weight: Weight for Dice loss on binary map
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.l1_weight = l1_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        prob_pred: torch.Tensor,
        thresh_pred: torch.Tensor,
        binary_pred: torch.Tensor,
        prob_target: torch.Tensor,
        thresh_target: torch.Tensor,
        shrink_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DB loss.

        Args:
            prob_pred: Predicted probability map
            thresh_pred: Predicted threshold map
            binary_pred: Predicted binary map
            prob_target: Target probability map
            thresh_target: Target threshold map
            shrink_mask: Shrink mask for BCE loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # BCE loss on probability map (using shrink mask)
        bce_loss = F.binary_cross_entropy(
            prob_pred * shrink_mask,
            prob_target * shrink_mask,
            reduction="sum",
        ) / (shrink_mask.sum() + 1e-6)

        # L1 loss on threshold map (only where text exists)
        text_mask = prob_target > 0.5
        if text_mask.sum() > 0:
            l1_loss = F.l1_loss(
                thresh_pred[text_mask], thresh_target[text_mask]
            )
        else:
            l1_loss = torch.tensor(0.0, device=prob_pred.device)

        # Dice loss on binary map
        dice_loss = self._dice_loss(binary_pred, prob_target)

        total_loss = (
            self.bce_weight * bce_loss
            + self.l1_weight * l1_loss
            + self.dice_weight * dice_loss
        )

        return total_loss, {
            "bce_loss": bce_loss.item(),
            "l1_loss": l1_loss.item(),
            "dice_loss": dice_loss.item(),
        }

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss for binary segmentation."""
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice


class DBTrainer(DetectorTrainer):
    """Trainer for DBNet text detector."""

    @property
    def name(self) -> str:
        return "db"

    @property
    def dataset_class(self) -> type:
        return DBDataset

    def create_model(self) -> nn.Module:
        """Create DBNet model."""
        return DBNet(pretrained=self.config.get("pretrained_backbone", True))

    def create_criterion(self) -> nn.Module:
        """Create DB loss."""
        return DBLoss(
            bce_weight=self.config.get("bce_weight", 1.0),
            l1_weight=self.config.get("l1_weight", 10.0),
            dice_weight=self.config.get("dice_weight", 1.0),
        )

    def compute_loss(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DB loss.

        Args:
            outputs: Tuple of (prob_pred, thresh_pred, binary_pred)
            targets: Dict with prob_map, thresh_map, shrink_mask

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        prob_pred, thresh_pred, binary_pred = outputs
        prob_target = targets["prob_map"]
        thresh_target = targets["thresh_map"]
        shrink_mask = targets["shrink_mask"]

        return self.criterion(
            prob_pred,
            thresh_pred,
            binary_pred,
            prob_target,
            thresh_target,
            shrink_mask,
        )


if __name__ == "__main__":
    DBTrainer.main()
