"""
trOCR recognizer trainer with LoRA adapters.

trOCR is a transformer-based OCR model that uses:
- Vision Transformer (ViT) encoder
- GPT-2 style decoder
- LoRA for efficient fine-tuning
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from .base import TransformerRecognizerTrainer


class TrOCRDataset(Dataset):
    """
    Dataset for trOCR training.

    Uses HuggingFace processor for image preprocessing and tokenization.
    """

    def __init__(
        self,
        data_dir: Path,
        processor: Any,
        max_length: int = 64,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length

        # Load labels
        labels_file = self.data_dir / "labels.txt"
        self.samples = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    filename, text = line.split("\t", 1)
                    self.samples.append((filename, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename, text = self.samples[idx]

        # Load image
        image_path = self.data_dir / "images" / filename
        image = Image.open(image_path).convert("RGB")

        # Process image and text
        encoding = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Remove batch dimension
        return {
            "pixel_values": encoding.pixel_values.squeeze(0),
            "labels": encoding.input_ids.squeeze(0),
        }


def trocr_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for trOCR batches."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


class TrOCRTrainer(TransformerRecognizerTrainer):
    """
    Trainer for trOCR with LoRA adapters.

    Uses HuggingFace transformers and PEFT for efficient fine-tuning.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.processor = None
        self.tokenizer = None

    @property
    def name(self) -> str:
        return "trocr"

    def create_model(self) -> nn.Module:
        """Create trOCR model with LoRA adapters."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "trOCR training requires: pip install transformers peft"
            )

        model_name = self.config.get(
            "model_name", "microsoft/trocr-base-handwritten"
        )

        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get(
                "target_modules", ["q_proj", "v_proj"]
            ),
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model = get_peft_model(model, lora_config)

        self.logger.info("LoRA configuration:")
        self.logger.info(f"  r={lora_config.r}")
        self.logger.info(f"  alpha={lora_config.lora_alpha}")
        self.logger.info(f"  dropout={lora_config.lora_dropout}")
        self.logger.info(f"  target_modules={lora_config.target_modules}")

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    def create_criterion(self) -> nn.Module:
        """
        trOCR uses built-in loss computation.

        The loss is computed inside model.forward() when labels are provided.
        """
        return None  # Not used directly

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create dataloaders for trOCR."""
        if self.processor is None:
            # Create model first to get processor
            _ = self.create_model()

        data_dir = Path(self.config["data_dir"])
        max_length = self.config.get("max_length", 64)
        batch_size = self.config.get("batch_size", 8)
        num_workers = self.config.get("num_workers", 4)

        train_dataset = TrOCRDataset(
            data_dir / "train",
            processor=self.processor,
            max_length=max_length,
        )

        val_dataset = TrOCRDataset(
            data_dir / "val",
            processor=self.processor,
            max_length=max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=trocr_collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=trocr_collate_fn,
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with labels - loss computed internally
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def decode_predictions(self, output: torch.Tensor) -> List[str]:
        """
        Decode transformer output using generate.

        Args:
            output: Not used - we call model.generate() instead

        Returns:
            List of decoded strings
        """
        # This is called with model output, but for trOCR we use generate()
        # The actual decoding happens in validate()
        return []

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation with generation."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Compute loss
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                total_loss += outputs.loss.item()

                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.config.get("max_length", 64),
                    num_beams=self.config.get("num_beams", 4),
                )

                # Decode predictions
                predictions = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                all_predictions.extend(predictions)

                # Decode targets
                targets = self.processor.batch_decode(
                    labels, skip_special_tokens=True
                )
                all_targets.extend(targets)

        n = len(dataloader)
        metrics = self.compute_recognition_metrics(all_predictions, all_targets)
        metrics["val_loss"] = total_loss / n

        return metrics

    def train(self) -> Dict[str, Any]:
        """Main training loop for trOCR."""
        self.logger.info(f"Starting training with device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")

        from ..utils import count_parameters, save_checkpoint, EarlyStopping

        # Create model first (sets up processor)
        self.model = self.create_model().to(self.device)

        # Then create dataloaders
        train_loader, val_loader = self.create_dataloaders()

        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        early_stopping = EarlyStopping(
            patience=self.config.get("patience", 10),
            mode="min",
        )

        params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {params}")

        best_loss = float("inf")
        epochs = self.config.get("epochs", 50)

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
                f"cer={val_metrics.get('cer', 0):.4f}, "
                f"lr={lr:.6f}"
            )

            self.metrics.update({"train_loss": train_loss, **val_metrics})
            self.metrics.end_epoch()

            if val_loss < best_loss:
                best_loss = val_loss
                # Save LoRA adapter
                self.model.save_pretrained(self.output_dir / "best_adapter")
                self.logger.info(f"Saved best adapter with loss={val_loss:.4f}")

            if early_stopping(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save final adapter
        self.model.save_pretrained(self.output_dir / "final_adapter")
        self.metrics.save(self.output_dir / "metrics.json")

        self.logger.info("Training complete!")

        return {
            "best_loss": best_loss,
            "final_loss": val_loss,
            "epochs_trained": epoch + 1,
            "output_dir": str(self.output_dir),
        }
