"""
CRNN (Convolutional Recurrent Neural Network) recognizer trainer.

CRNN architecture:
- CNN backbone for feature extraction
- Bidirectional LSTM for sequence modeling
- CTC loss for alignment-free training
"""

from typing import Any, Dict

import torch.nn as nn

from models import CRNN
from .base import CTCRecognizerTrainer


class CRNNTrainer(CTCRecognizerTrainer):
    """Trainer for CRNN text recognizer."""

    @property
    def name(self) -> str:
        return "crnn"

    def create_model(self) -> nn.Module:
        """Create CRNN model."""
        # Ensure dataloaders are created first to get vocabulary size
        if self.char_to_idx is None:
            # This will be called from train() after create_dataloaders
            raise RuntimeError(
                "create_dataloaders must be called before create_model"
            )

        return CRNN(
            img_height=self.config.get("img_height", 32),
            num_channels=1,  # Grayscale input
            num_classes=len(self.char_to_idx),
            hidden_size=self.config.get("hidden_size", 256),
        )

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Overrides to ensure dataloaders are created before model.
        """
        self.logger.info(f"Starting training with device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Create dataloaders first to get vocabulary
        train_loader, val_loader = self.create_dataloaders()

        # Now create model with correct num_classes
        from ..utils import count_parameters, save_checkpoint, EarlyStopping

        self.model = self.create_model().to(self.device)
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        early_stopping = EarlyStopping(
            patience=self.config.get("patience", 10),
            mode="min",
        )

        params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {params}")
        self.logger.info(f"Train batches: {len(train_loader)}")
        self.logger.info(f"Val batches: {len(val_loader)}")

        # Training loop
        best_loss = float("inf")
        epochs = self.config.get("epochs", 100)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics.get("val_loss", train_loss)

            self.scheduler.step()

            lr = self.scheduler.get_lr()[0]
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"accuracy={val_metrics.get('accuracy', 0):.4f}, "
                f"cer={val_metrics.get('cer', 0):.4f}, "
                f"lr={lr:.6f}"
            )

            self.metrics.update({
                "train_loss": train_loss,
                **val_metrics,
            })
            self.metrics.end_epoch()

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

            if early_stopping(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

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
