"""
Base trainer class for all model fine-tuning.

Provides shared training loop infrastructure with abstract methods
for model-specific implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import (
    EarlyStopping,
    LRScheduler,
    MetricTracker,
    get_device,
    count_parameters,
    save_checkpoint,
    setup_logging,
)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Provides the main training loop and common utilities.
    Subclasses must implement model creation, data loading,
    and training/validation steps.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = get_device()
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.output_dir, self.name)
        self.metrics = MetricTracker()

        # Will be set during training
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.criterion: Optional[nn.Module] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Trainer name for logging."""
        pass

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the model."""
        pass

    @abstractmethod
    def create_criterion(self) -> nn.Module:
        """Create and return the loss function."""
        pass

    @abstractmethod
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create and return train and validation dataloaders."""
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: A batch from the dataloader

        Returns:
            Dictionary of loss values
        """
        pass

    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run validation on the given dataloader.

        Args:
            dataloader: Validation dataloader

        Returns:
            Dictionary of validation metrics
        """
        pass

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create optimizer. Override for custom optimizer configs.

        Args:
            model: The model to optimize

        Returns:
            Configured optimizer
        """
        return torch.optim.Adam(
            model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler:
        """
        Create learning rate scheduler. Override for custom scheduling.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Configured scheduler
        """
        return LRScheduler(
            optimizer,
            scheduler_type=self.config.get("scheduler", "cosine"),
            warmup_epochs=self.config.get("warmup_epochs", 5),
            total_epochs=self.config.get("epochs", 100),
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss_dict = self.train_step(batch)
            total_loss += loss_dict.get("loss", 0.0)
            num_batches += 1
            self.metrics.update(loss_dict)

        return total_loss / max(num_batches, 1)

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training results including final metrics
        """
        self.logger.info(f"Starting training with device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Setup
        self.model = self.create_model().to(self.device)
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        early_stopping = EarlyStopping(
            patience=self.config.get("patience", 10),
            mode="min",
        )

        # Log model info
        params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {params}")

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        self.logger.info(f"Train batches: {len(train_loader)}")
        self.logger.info(f"Val batches: {len(val_loader)}")

        # Training loop
        best_loss = float("inf")
        epochs = self.config.get("epochs", 100)

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics.get("loss", val_metrics.get("val_loss", train_loss))

            # Update scheduler
            self.scheduler.step()

            # Log progress
            lr = self.scheduler.get_lr()[0]
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"lr={lr:.6f}"
            )

            # Update metrics
            self.metrics.update({
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            })
            self.metrics.end_epoch()

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.output_dir / "best_model.pt",
                    scheduler=self.scheduler,
                )
                self.logger.info(f"Saved best model with loss={val_loss:.4f}")

            # Early stopping
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save final model and metrics
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            val_loss,
            self.output_dir / "final_model.pt",
            scheduler=self.scheduler,
        )
        self.metrics.save(self.output_dir / "metrics.json")

        self.logger.info("Training complete!")
        self.logger.info(f"Best validation loss: {best_loss:.4f}")

        return {
            "best_loss": best_loss,
            "final_loss": val_loss,
            "epochs_trained": epoch + 1,
            "output_dir": str(self.output_dir),
        }
