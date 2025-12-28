"""
Base trainer class for all model fine-tuning.

Provides shared training loop infrastructure with abstract methods
for model-specific implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from services.annotation.models import AnnotationDataset
from services.annotation.storage import AnnotationStorage
from .utils import (
    EarlyStopping,
    LRScheduler,
    MetricTracker,
    get_device,
    count_parameters,
    save_checkpoint,
    setup_logging,
    prepare_data_dir,
    cleanup_temp_dir,
)


class ZipExportStorage:
    """
    Minimal storage interface for zip-exported datasets.

    Works with the directory structure created by AnnotationExporter:
    - dataset.json (with image paths like "images/filename.png")
    - images/
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

    def get_image_path(self, relative_path: str) -> Path:
        """Get absolute path for a relative image path."""
        return self.base_path / relative_path

    def load(self, name: str) -> Optional[AnnotationDataset]:
        """Load dataset from JSON file."""
        if name == "dataset":
            json_path = self.base_path / "dataset.json"
        else:
            json_path = self.base_path / f"{name}.json"

        if not json_path.exists():
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            return AnnotationDataset.from_json(f.read())


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
        self.device = config.get("device") or get_device()
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

    @property
    @abstractmethod
    def dataset_class(self) -> type:
        """Return the dataset class to use for training."""
        pass

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """
        Return dataset-specific keyword arguments.
        Override in subclasses for custom config (e.g., img_size, img_height).
        """
        return {}

    def get_collate_fn(self) -> Optional[Callable]:
        """
        Return collate function for the dataloader.
        Override in subclasses if needed.
        """
        return None

    def create_val_dataset(
        self,
        data_dir: Path,
        val_data: Optional[AnnotationDataset],
        storage: Any,
        train_dataset: Dataset,
        dataset_kwargs: Dict[str, Any],
    ) -> Dataset:
        """
        Create validation dataset.

        Override in subclasses for custom behavior (e.g., shared vocabulary
        between train and val datasets in recognizers).

        Args:
            data_dir: Path to data directory
            val_data: Validation annotation dataset (None for pre-exported format)
            storage: Storage instance
            train_dataset: Already-created training dataset
            dataset_kwargs: Dataset keyword arguments

        Returns:
            Validation dataset
        """
        if val_data is not None:
            return self.dataset_class(
                data_dir,
                annotation_dataset=val_data,
                storage=storage,
                augment=False,
                **dataset_kwargs,
            )
        else:
            return self.dataset_class(
                data_dir / "val",
                augment=False,
                **dataset_kwargs,
            )

    def post_create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> None:
        """
        Called after datasets are created.

        Override for post-processing like storing vocabulary mappings.
        """
        pass

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.

        Supports three data formats:
        1. Zip file: Exported dataset zip from AnnotationExporter
           - Contains: dataset.json, images/, metadata.json
        2. Annotation format: data_dir points to annotations directory with JSON files
           - Expects: data_dir/<dataset_name>.json or train.json/val.json
        3. Exported format: data_dir contains train/ and val/ subdirectories
           - Expects: data_dir/train/images/

        Config options:
        - split_ratio: float (default 0.9)

        Uses PyTorch's random_split for train/val splitting.
        """
        raw_data_dir = Path(self.config["data_dir"])
        batch_size = self.config.get("batch_size", 16)
        num_workers = self.config.get("num_workers", 4)
        split_ratio = self.config.get("split_ratio", 0.9)

        # Handle zip files - extract if needed
        data_dir, is_temp = prepare_data_dir(raw_data_dir)
        self._temp_data_dir = data_dir if is_temp else None

        # Get dataset-specific kwargs
        dataset_kwargs = self.get_dataset_kwargs()

        # Try to load annotation data and storage
        full_data, storage = self._load_annotation_dataset(data_dir)

        if full_data is not None:
            # Annotation-based format - create full dataset then split with PyTorch
            full_dataset = self.dataset_class(
                data_dir,
                annotation_dataset=full_data,
                storage=storage,
                **dataset_kwargs,
            )

            # Use PyTorch random_split
            n_total = len(full_dataset)
            n_train = int(n_total * split_ratio)
            n_val = n_total - n_train

            train_dataset, val_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

            self.logger.info(f"Split dataset: {n_train} train, {n_val} val (using random_split)")

        else:
            # Pre-exported format with train/val subdirs
            train_dataset = self.dataset_class(
                data_dir / "train",
                **dataset_kwargs,
            )
            val_dataset = self.create_val_dataset(
                data_dir, None, storage, train_dataset, dataset_kwargs
            )

        # Post-processing hook (for recognizers to store vocabulary)
        # For random_split, both subsets share the same base dataset
        base_dataset = getattr(train_dataset, 'dataset', train_dataset)
        self.post_create_dataloaders(base_dataset, val_dataset)

        # Create dataloaders
        collate_fn = self.get_collate_fn()

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

        return train_loader, val_loader

    def _load_annotation_dataset(
        self,
        data_dir: Path,
    ) -> Tuple[Optional[AnnotationDataset], Optional[Any]]:
        """
        Load annotation dataset from various formats.

        Args:
            data_dir: Path to data directory

        Returns:
            Tuple of (dataset, storage) or (None, None) for pre-exported format
        """
        # Check for zip export format (has dataset.json at root)
        if (data_dir / "dataset.json").exists() and (data_dir / "images").exists():
            self.logger.info("Loading from zip export format")
            storage = ZipExportStorage(data_dir)
            full_dataset = storage.load("dataset")

            if full_dataset is None:
                raise ValueError(f"Failed to load dataset.json from {data_dir}")

            return full_dataset, storage

        # Check for pre-exported format with train/val subdirs
        if (data_dir / "train").exists() and (data_dir / "train" / "images").exists():
            self.logger.info("Loading from exported format")
            return None, None

        # Check for pre-exported recognizer format
        if (data_dir / "train").exists() and (data_dir / "train" / "labels.txt").exists():
            self.logger.info("Loading from exported format")
            return None, None

        # Annotation format - look for JSON files
        self.logger.info("Loading from annotation format")
        storage = AnnotationStorage(data_dir)

        # Find any JSON file
        json_files = [f for f in data_dir.glob("*.json") if f.stem != "metadata"]
        if not json_files:
            raise ValueError(f"No annotation files found in {data_dir}")

        dataset_name = json_files[0].stem
        full_dataset = storage.load(dataset_name)
        if full_dataset is None:
            raise ValueError(f"Failed to load dataset: {dataset_name}")

        return full_dataset, storage

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

        # Optional ONNX export
        if self.config.get("export_onnx", False):
            self._export_onnx()

        return {
            "best_loss": best_loss,
            "final_loss": val_loss,
            "epochs_trained": epoch + 1,
            "output_dir": str(self.output_dir),
        }

    def _export_onnx(self):
        """Export model to ONNX format if configured."""
        try:
            from training.export import export_to_onnx

            onnx_path = self.output_dir / f"{self.name}.onnx"
            input_shape = self._get_onnx_input_shape()

            export_to_onnx(
                self.model,
                onnx_path,
                input_shape=input_shape,
            )
            self.logger.info(f"Exported ONNX model to {onnx_path}")
        except Exception as e:
            self.logger.warning(f"Failed to export ONNX: {e}")

    def _get_onnx_input_shape(self) -> tuple:
        """
        Get input shape for ONNX export. Override in subclass.

        Returns:
            Tuple of (batch, channels, height, width)
        """
        img_size = self.config.get("img_size", 640)
        return (1, 3, img_size, img_size)

    @classmethod
    def main(cls):
        """
        CLI entry point for running training.

        Usage: python -m training.finetune.detectors.craft <config.yaml>
        """
        import argparse

        parser = argparse.ArgumentParser(
            description=f"Train {cls.__name__}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "config",
            type=str,
            help="Path to YAML config file",
        )
        args = parser.parse_args()

        # Load config and train
        from .utils import load_config

        config = load_config(args.config)
        trainer = cls(config)
        results = trainer.train()

        print(f"\nTraining complete!")
        print(f"  Best loss: {results['best_loss']:.4f}")
        print(f"  Epochs: {results['epochs_trained']}")
        print(f"  Output: {results['output_dir']}")

        return results
