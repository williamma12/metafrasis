"""
CRAFT (Character Region Awareness For Text) detector trainer.

CRAFT uses a VGG16-BN backbone with U-Net style upsampling to produce:
- Region score maps (character centers)
- Affinity score maps (character adjacency)
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

from ml.models import CRAFT
from app.services.annotation.models import Region
from ml.training.data.base import region_to_polygon, compute_polygon_centroid, generate_gaussian_heatmap
from .base import DetectorTrainer, DetectorDataset


class CRAFTDataset(DetectorDataset):
    """Dataset for CRAFT training with region and affinity maps."""

    def __init__(self, *args, gaussian_sigma: float = 10.0, affinity_max_distance: float = 50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_sigma = gaussian_sigma
        self.affinity_max_distance = affinity_max_distance

    def load_targets(self, sample_name: str) -> Dict[str, np.ndarray]:
        """Load region and affinity target maps from exported format."""
        region_map = np.load(
            self.targets_dir / "region_map" / f"{sample_name}.npy"
        )
        affinity_map = np.load(
            self.targets_dir / "affinity_map" / f"{sample_name}.npy"
        )

        # Return at full resolution (model outputs at same resolution as input)
        return {
            "region_map": region_map[np.newaxis, ...],  # Add channel dim
            "affinity_map": affinity_map[np.newaxis, ...],
        }

    def generate_targets(
        self, image: Image.Image, regions: List[Region]
    ) -> Dict[str, np.ndarray]:
        """Generate CRAFT-style region and affinity maps from annotation regions."""
        width, height = image.size

        # Generate at full resolution (model outputs at same resolution as input)
        region_map = self._generate_region_map((width, height), regions)
        affinity_map = self._generate_affinity_map((width, height), regions)

        return {
            "region_map": region_map[np.newaxis, ...].astype(np.float32),
            "affinity_map": affinity_map[np.newaxis, ...].astype(np.float32),
        }

    def _generate_region_map(self, size: Tuple[int, int], regions: List[Region]) -> np.ndarray:
        """Generate region score map with Gaussian heatmaps for each text region."""
        width, height = size
        region_map = np.zeros((height, width), dtype=np.float32)

        for region in regions:
            polygon = region_to_polygon(region)
            mask = np.zeros((height, width), dtype=np.float32)
            polygon_int = polygon.astype(np.int32)
            cv2.fillPoly(mask, [polygon_int], 1.0)

            # Adaptive sigma based on region size
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            sigma = max(2.0, min(x_max - x_min, y_max - y_min) / 4)

            kernel_size = int(sigma * 6) | 1
            if kernel_size > 1:
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)

            if mask.max() > 0:
                mask = mask / mask.max()

            region_map = np.maximum(region_map, mask)

        return region_map

    def _generate_affinity_map(self, size: Tuple[int, int], regions: List[Region]) -> np.ndarray:
        """Generate affinity score map between adjacent regions."""
        width, height = size
        affinity_map = np.zeros((height, width), dtype=np.float32)

        if len(regions) < 2:
            return affinity_map

        # Find adjacent pairs
        pairs = self._find_adjacent_pairs(regions)

        for region1, region2 in pairs:
            poly1 = region_to_polygon(region1)
            poly2 = region_to_polygon(region2)
            cx1, cy1 = compute_polygon_centroid(poly1)
            cx2, cy2 = compute_polygon_centroid(poly2)

            mid_x = (cx1 + cx2) / 2
            mid_y = (cy1 + cy2) / 2
            distance = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
            sigma = min(self.gaussian_sigma, distance / 4)

            if sigma > 1:
                heatmap = generate_gaussian_heatmap((mid_x, mid_y), size, sigma=sigma)
                affinity_map = np.maximum(affinity_map, heatmap)

        return affinity_map

    def _find_adjacent_pairs(self, regions: List[Region]) -> List[Tuple[Region, Region]]:
        """Find pairs of adjacent regions for affinity links."""
        pairs = []
        for i, region1 in enumerate(regions):
            poly1 = region_to_polygon(region1)
            cx1, cy1 = compute_polygon_centroid(poly1)
            bbox1 = (poly1[:, 0].min(), poly1[:, 1].min(), poly1[:, 0].max(), poly1[:, 1].max())
            h1 = bbox1[3] - bbox1[1]

            for j, region2 in enumerate(regions[i + 1:], i + 1):
                poly2 = region_to_polygon(region2)
                cx2, cy2 = compute_polygon_centroid(poly2)
                bbox2 = (poly2[:, 0].min(), poly2[:, 1].min(), poly2[:, 0].max(), poly2[:, 1].max())
                h2 = bbox2[3] - bbox2[1]

                h_distance = min(abs(bbox1[2] - bbox2[0]), abs(bbox2[2] - bbox1[0]))
                v_distance = abs(cy1 - cy2)
                avg_height = (h1 + h2) / 2

                if h_distance < self.affinity_max_distance and v_distance < avg_height * 0.5:
                    pairs.append((region1, region2))

        return pairs


class CRAFTLoss(nn.Module):
    """
    Loss function for CRAFT training.

    Uses MSE loss with online hard example mining (OHEM)
    to balance positive and negative samples.
    """

    def __init__(self, neg_ratio: float = 3.0):
        """
        Initialize CRAFT loss.

        Args:
            neg_ratio: Ratio of negative to positive samples for OHEM
        """
        super().__init__()
        self.neg_ratio = neg_ratio

    def forward(
        self,
        region_pred: torch.Tensor,
        affinity_pred: torch.Tensor,
        region_target: torch.Tensor,
        affinity_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CRAFT loss.

        Args:
            region_pred: Predicted region map [B, 1, H, W]
            affinity_pred: Predicted affinity map [B, 1, H, W]
            region_target: Target region map [B, 1, H, W]
            affinity_target: Target affinity map [B, 1, H, W]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Create positive/negative masks
        pos_mask = (region_target > 0.5) | (affinity_target > 0.5)
        neg_mask = ~pos_mask

        # Count samples
        n_pos = pos_mask.sum().item()
        n_neg = min(neg_mask.sum().item(), int(n_pos * self.neg_ratio))

        if n_pos == 0:
            # No positive samples, use all samples
            region_loss = F.mse_loss(region_pred, region_target)
            affinity_loss = F.mse_loss(affinity_pred, affinity_target)
        else:
            # Compute per-pixel losses
            region_loss_map = F.mse_loss(
                region_pred, region_target, reduction="none"
            )
            affinity_loss_map = F.mse_loss(
                affinity_pred, affinity_target, reduction="none"
            )

            # Positive loss
            pos_region_loss = (region_loss_map * pos_mask.float()).sum() / n_pos
            pos_affinity_loss = (affinity_loss_map * pos_mask.float()).sum() / n_pos

            if n_neg > 0:
                # Hard negative mining
                neg_loss = (region_loss_map + affinity_loss_map) * neg_mask.float()
                neg_loss_flat = neg_loss.view(-1)
                _, hard_neg_idx = neg_loss_flat.topk(n_neg)
                neg_region_loss = region_loss_map.view(-1)[hard_neg_idx].mean()
                neg_affinity_loss = affinity_loss_map.view(-1)[hard_neg_idx].mean()

                region_loss = pos_region_loss + neg_region_loss
                affinity_loss = pos_affinity_loss + neg_affinity_loss
            else:
                region_loss = pos_region_loss
                affinity_loss = pos_affinity_loss

        total_loss = region_loss + affinity_loss

        return total_loss, {
            "region_loss": region_loss.item(),
            "affinity_loss": affinity_loss.item(),
        }


class CRAFTTrainer(DetectorTrainer):
    """Trainer for CRAFT text detector."""

    @property
    def name(self) -> str:
        return "craft"

    @property
    def dataset_class(self) -> type:
        return CRAFTDataset

    def create_model(self) -> nn.Module:
        """Create CRAFT model."""
        return CRAFT(pretrained=self.config.get("pretrained_backbone", True))

    def create_criterion(self) -> nn.Module:
        """Create CRAFT loss with OHEM."""
        return CRAFTLoss(neg_ratio=self.config.get("neg_ratio", 3.0))

    def compute_loss(
        self, outputs: Tuple[torch.Tensor, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CRAFT loss.

        Args:
            outputs: Tuple of (region_pred, affinity_pred)
            targets: Dict with region_map and affinity_map

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        region_pred, affinity_pred = outputs
        region_target = targets["region_map"]
        affinity_target = targets["affinity_map"]

        return self.criterion(
            region_pred, affinity_pred, region_target, affinity_target
        )


if __name__ == "__main__":
    CRAFTTrainer.main()
