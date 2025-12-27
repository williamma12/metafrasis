"""
Base trainer classes for text recognition models.

Provides shared functionality for recognition model training including
dataset loading, CTC/transformer decoding, and recognition metrics.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from ..base import BaseTrainer


class RecognizerDataset(Dataset):
    """
    Dataset for text recognition training.

    Loads images and labels from the standard recognizer format:
    - images/ directory containing cropped text line images
    - labels.txt with filename<tab>text format
    """

    def __init__(
        self,
        data_dir: Path,
        img_height: int = 32,
        img_width: int = 128,
        char_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width

        # Load labels
        labels_file = self.data_dir / "labels.txt"
        self.samples = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    filename, text = line.split("\t", 1)
                    self.samples.append((filename, text))

        # Build or use provided character mapping
        if char_to_idx is None:
            self.char_to_idx = self._build_vocab()
        else:
            self.char_to_idx = char_to_idx

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def _build_vocab(self) -> Dict[str, int]:
        """Build character vocabulary from labels."""
        chars = set()
        for _, text in self.samples:
            chars.update(text)

        # Sort for deterministic ordering, reserve 0 for CTC blank
        char_to_idx = {"<blank>": 0}
        for i, char in enumerate(sorted(chars), 1):
            char_to_idx[char] = i

        return char_to_idx

    def encode_text(self, text: str) -> List[int]:
        """Convert text to sequence of indices."""
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode_text(self, indices: List[int]) -> str:
        """Convert indices back to text."""
        chars = []
        for idx in indices:
            if idx == 0:  # Skip blank
                continue
            char = self.idx_to_char.get(idx, "")
            chars.append(char)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        filename, text = self.samples[idx]

        # Load and preprocess image
        image_path = self.data_dir / "images" / filename
        image = Image.open(image_path).convert("L")  # Grayscale

        # Resize to fixed size
        image = image.resize(
            (self.img_width, self.img_height), Image.Resampling.BILINEAR
        )

        # Convert to tensor and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.tensor(image).unsqueeze(0)  # Add channel dim

        # Encode text
        target = torch.tensor(self.encode_text(text), dtype=torch.long)
        target_length = len(target)

        return image, target, target_length


def collate_fn(batch):
    """Collate function for variable-length targets."""
    images, targets, target_lengths = zip(*batch)

    # Stack images
    images = torch.stack(images)

    # Pad targets to same length
    max_len = max(target_lengths)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, target in enumerate(targets):
        padded_targets[i, : len(target)] = target

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, padded_targets, target_lengths


class RecognizerTrainer(BaseTrainer):
    """
    Base class for recognition model trainers (CRNN, PP-OCR, trOCR).

    Provides shared functionality for:
    - Dataset loading with character vocabulary
    - Training step with sequence output handling
    - Validation with recognition metrics (CER, WER, accuracy)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.char_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_char: Optional[Dict[int, str]] = None

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        data_dir = Path(self.config["data_dir"])
        img_height = self.config.get("img_height", 32)
        img_width = self.config.get("img_width", 128)
        batch_size = self.config.get("batch_size", 32)
        num_workers = self.config.get("num_workers", 4)

        train_dataset = RecognizerDataset(
            data_dir / "train",
            img_height=img_height,
            img_width=img_width,
        )

        # Store vocabulary
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

    @property
    def num_classes(self) -> int:
        """Number of output classes (vocabulary size)."""
        if self.char_to_idx is None:
            raise RuntimeError("Dataloaders must be created before accessing num_classes")
        return len(self.char_to_idx)

    @abstractmethod
    def decode_predictions(self, output: torch.Tensor) -> List[str]:
        """
        Decode model output to text.

        Args:
            output: Model output tensor

        Returns:
            List of decoded strings
        """
        pass

    def compute_recognition_metrics(
        self, predictions: List[str], targets: List[str]
    ) -> Dict[str, float]:
        """
        Compute recognition metrics.

        Args:
            predictions: Predicted texts
            targets: Target texts

        Returns:
            Dictionary with accuracy, CER, WER
        """
        correct = sum(p == t for p, t in zip(predictions, targets))
        accuracy = correct / len(targets) if targets else 0.0

        # Character error rate
        total_chars = sum(len(t) for t in targets)
        char_errors = sum(
            self._levenshtein(p, t) for p, t in zip(predictions, targets)
        )
        cer = char_errors / total_chars if total_chars > 0 else 0.0

        return {
            "accuracy": accuracy,
            "cer": cer,
        }

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]


class CTCRecognizerTrainer(RecognizerTrainer):
    """
    Base class for CTC-based recognizers (CRNN, PP-OCR).

    Provides:
    - CTC loss function
    - CTC greedy decoding
    - Training step with CTC loss computation
    """

    def create_criterion(self) -> nn.Module:
        """Create CTC loss function."""
        return nn.CTCLoss(blank=0, zero_infinity=True)

    def decode_predictions(self, output: torch.Tensor) -> List[str]:
        """
        Decode CTC output using greedy decoding.

        Args:
            output: Model output [T, B, C] or [B, T, C]

        Returns:
            List of decoded strings
        """
        # Ensure output is [T, B, C]
        if output.dim() == 3 and output.size(0) != output.size(1):
            if output.size(1) > output.size(0):
                output = output.permute(1, 0, 2)

        # Get best path
        _, indices = output.max(2)  # [T, B]
        indices = indices.t()  # [B, T]

        decoded = []
        for seq in indices:
            # Remove duplicates and blanks
            chars = []
            prev = -1
            for idx in seq.tolist():
                if idx != prev and idx != 0:  # 0 is blank
                    char = self.idx_to_char.get(idx, "")
                    chars.append(char)
                prev = idx
            decoded.append("".join(chars))

        return decoded

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute a single CTC training step."""
        images, targets, target_lengths = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        # Forward pass
        output = self.model(images)

        # Ensure output is [T, B, C] for CTC loss
        if output.dim() == 3:
            if output.size(1) > output.size(0):
                output = output.permute(1, 0, 2)

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
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation with CTC decoding."""
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

                # Forward pass
                output = self.model(images)

                # Ensure output is [T, B, C]
                if output.dim() == 3 and output.size(1) > output.size(0):
                    output = output.permute(1, 0, 2)

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


class TransformerRecognizerTrainer(RecognizerTrainer):
    """
    Base class for transformer-based recognizers (trOCR).

    Provides:
    - Cross-entropy loss function
    - Transformer generate decoding
    - Training step for encoder-decoder models
    """

    def create_criterion(self) -> nn.Module:
        """Create cross-entropy loss for transformer."""
        return nn.CrossEntropyLoss(ignore_index=-100)

    @abstractmethod
    def decode_predictions(self, output: torch.Tensor) -> List[str]:
        """
        Decode transformer output.

        Override this to use model.generate() with proper tokenizer.
        """
        pass
