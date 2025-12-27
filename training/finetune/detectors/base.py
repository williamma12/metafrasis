"""
Base trainer class for text detection models.

Provides shared functionality for detection model training including
dataset loading, loss computation, and metric evaluation.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from ..base import BaseTrainer


class DetectorDataset(Dataset):
    """
    Base dataset class for detector training.

    Loads images and pre-generated target maps from disk.
    """

    def __init__(
        self,
        data_dir: Path,
        img_size: int = 640,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment

        self.images_dir = self.data_dir / "images"
        self.targets_dir = self.data_dir / "targets"

        self.samples = sorted(self.images_dir.glob("*.png"))
        if not self.samples:
            self.samples = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def load_targets(self, sample_name: str) -> Dict[str, np.ndarray]:
        """Load target maps for a sample. Override in subclass."""
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

    @property
    def dataset_class(self) -> type:
        """Return the dataset class to use. Override in subclass."""
        return DetectorDataset

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        data_dir = Path(self.config["data_dir"])
        img_size = self.config.get("img_size", 640)
        batch_size = self.config.get("batch_size", 16)
        num_workers = self.config.get("num_workers", 4)

        train_dataset = self.dataset_class(
            data_dir / "train",
            img_size=img_size,
            augment=True,
        )

        val_dataset = self.dataset_class(
            data_dir / "val",
            img_size=img_size,
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")

        return train_loader, val_loader

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
