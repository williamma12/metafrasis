"""
CRAFT (Character Region Awareness For Text) detector trainer.

CRAFT uses a VGG16-BN backbone with U-Net style upsampling to produce:
- Region score maps (character centers)
- Affinity score maps (character adjacency)
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from models import CRAFT
from .base import DetectorTrainer, DetectorDataset


class CRAFTDataset(DetectorDataset):
    """Dataset for CRAFT training with region and affinity maps."""

    def load_targets(self, sample_name: str) -> Dict[str, np.ndarray]:
        """Load region and affinity target maps."""
        target_size = self.img_size // 2  # CRAFT outputs at 1/2 resolution

        region_map = np.load(
            self.targets_dir / "region_map" / f"{sample_name}.npy"
        )
        affinity_map = np.load(
            self.targets_dir / "affinity_map" / f"{sample_name}.npy"
        )

        # Resize targets to match model output size
        from PIL import Image

        region_map = np.array(
            Image.fromarray(region_map).resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )
        )
        affinity_map = np.array(
            Image.fromarray(affinity_map).resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )
        )

        return {
            "region_map": region_map[np.newaxis, ...],  # Add channel dim
            "affinity_map": affinity_map[np.newaxis, ...],
        }


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
