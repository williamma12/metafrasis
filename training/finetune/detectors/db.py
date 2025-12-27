"""
DB (Differentiable Binarization) detector trainer.

DBNet uses ResNet backbone with FPN and produces:
- Probability maps
- Threshold maps
- Binary maps (via differentiable binarization)
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from models import DBNet
from .base import DetectorTrainer, DetectorDataset


class DBDataset(DetectorDataset):
    """Dataset for DB training with probability and threshold maps."""

    def load_targets(self, sample_name: str) -> Dict[str, np.ndarray]:
        """Load probability, threshold, and shrink mask."""
        target_size = self.img_size // 4  # DBNet outputs at 1/4 resolution
        from PIL import Image

        prob_map = np.load(
            self.targets_dir / "probability_map" / f"{sample_name}.npy"
        )
        thresh_map = np.load(
            self.targets_dir / "threshold_map" / f"{sample_name}.npy"
        )
        shrink_mask = np.load(
            self.targets_dir / "shrink_mask" / f"{sample_name}.npy"
        )

        # Resize targets
        prob_map = np.array(
            Image.fromarray(prob_map).resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )
        )
        thresh_map = np.array(
            Image.fromarray(thresh_map).resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )
        )
        shrink_mask = np.array(
            Image.fromarray(shrink_mask.astype(np.uint8)).resize(
                (target_size, target_size), Image.Resampling.NEAREST
            )
        )

        return {
            "prob_map": prob_map[np.newaxis, ...].astype(np.float32),
            "thresh_map": thresh_map[np.newaxis, ...].astype(np.float32),
            "shrink_mask": shrink_mask[np.newaxis, ...].astype(np.float32),
        }


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
