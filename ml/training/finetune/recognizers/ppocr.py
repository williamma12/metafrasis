"""
PP-OCR recognizer trainer.

PP-OCR uses a lightweight MobileNetV3 backbone with BiLSTM and CTC loss.
Optimized for efficient training on limited hardware.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models import PPOCRModel
from .base import CTCRecognizerTrainer, RecognizerDataset, collate_fn


class PPOCRTrainer(CTCRecognizerTrainer):
    """
    Trainer for PP-OCR text recognizer.

    Adds gradient accumulation for training with larger effective batch sizes
    on limited memory.
    """

    @property
    def name(self) -> str:
        return "ppocr"

    def create_model(self) -> nn.Module:
        """Create PP-OCR model."""
        if self.char_to_idx is None:
            raise RuntimeError(
                "create_dataloaders must be called before create_model"
            )

        return PPOCRModel(
            in_channels=1,  # Grayscale input
            num_classes=len(self.char_to_idx),
            backbone_scale=self.config.get("backbone_scale", 0.5),
            hidden_size=self.config.get("hidden_size", 256),
        )

    def create_dataloaders(self) -> tuple:
        """Create dataloaders with PP-OCR specific settings."""
        from pathlib import Path

        data_dir = Path(self.config["data_dir"])
        img_height = self.config.get("img_height", 32)
        img_width = self.config.get("img_width", 320)  # Wider for PP-OCR
        batch_size = self.config.get("batch_size", 64)
        num_workers = self.config.get("num_workers", 4)

        train_dataset = RecognizerDataset(
            data_dir / "train",
            img_height=img_height,
            img_width=img_width,
        )

        self.char_to_idx = train_dataset.char_to_idx
        self.idx_to_char = train_dataset.idx_to_char

        val_dataset = RecognizerDataset(
            data_dir / "val",
            img_height=img_height,
            img_width=img_width,
            char_to_idx=self.char_to_idx,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        self.logger.info(f"Vocabulary size: {len(self.char_to_idx)}")

        return train_loader, val_loader

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute a single training step with gradient accumulation.

        PP-OCR model outputs [B, T, C], need to permute to [T, B, C].
        """
        images, targets, target_lengths = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        # Forward pass - PP-OCR outputs [B, T, C]
        output = self.model(images)
        output = output.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]
        output = torch.log_softmax(output, dim=2)

        # CTC loss
        input_lengths = torch.full(
            (images.size(0),),
            output.size(0),
            dtype=torch.long,
            device=self.device,
        )
        loss = self.criterion(output, targets, input_lengths, target_lengths)

        # Backward pass
        accumulation_steps = self.config.get("accumulation_steps", 1)
        loss = loss / accumulation_steps

        loss.backward()

        # Step optimizer every accumulation_steps
        if not hasattr(self, "_accumulation_count"):
            self._accumulation_count = 0
        self._accumulation_count += 1

        if self._accumulation_count >= accumulation_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulation_count = 0

        return {"loss": loss.item() * accumulation_steps}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation with PP-OCR output handling."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images, targets, target_lengths = batch
                images = images.to(self.device)
                targets_dev = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward pass - PP-OCR outputs [B, T, C]
                output = self.model(images)
                output = output.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]
                output_log = torch.log_softmax(output, dim=2)

                # CTC loss
                input_lengths = torch.full(
                    (images.size(0),),
                    output.size(0),
                    dtype=torch.long,
                    device=self.device,
                )
                loss = self.criterion(output_log, targets_dev, input_lengths, target_lengths)
                total_loss += loss.item()

                # Decode predictions
                predictions = self.decode_predictions(output)
                all_predictions.extend(predictions)

                # Decode targets
                for i in range(len(predictions)):
                    target_text = "".join([
                        self.idx_to_char.get(idx.item(), "")
                        for idx in targets[i, : target_lengths[i]]
                    ])
                    all_targets.append(target_text)

        n = len(dataloader)
        metrics = self.compute_recognition_metrics(all_predictions, all_targets)
        metrics["val_loss"] = total_loss / n

        return metrics

    def train(self) -> Dict[str, Any]:
        """Main training loop with gradient accumulation."""
        self.logger.info(f"Starting training with device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Create dataloaders first
        train_loader, val_loader = self.create_dataloaders()

        from ..utils import count_parameters, save_checkpoint, EarlyStopping

        self.model = self.create_model().to(self.device)
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        early_stopping = EarlyStopping(
            patience=self.config.get("patience", 15),
            mode="min",
        )

        params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {params}")
        self.logger.info(f"Train batches: {len(train_loader)}")
        self.logger.info(f"Val batches: {len(val_loader)}")
        self.logger.info(
            f"Accumulation steps: {self.config.get('accumulation_steps', 1)}"
        )

        best_loss = float("inf")
        epochs = self.config.get("epochs", 100)

        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Reset at start of epoch
            self._accumulation_count = 0

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
                f"lr={lr:.6f}"
            )

            self.metrics.update({"train_loss": train_loss, **val_metrics})
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

        return {
            "best_loss": best_loss,
            "final_loss": val_loss,
            "epochs_trained": epoch + 1,
            "output_dir": str(self.output_dir),
        }


if __name__ == "__main__":
    PPOCRTrainer.main()
