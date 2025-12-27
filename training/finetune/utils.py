"""
Shared utilities for model fine-tuning.

Provides common functionality for training loops:
- Configuration loading
- Checkpoint management
- Logging setup
- Learning rate scheduling
- Early stopping
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import json
import logging
import yaml
import torch
from datetime import datetime


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    output_path: Path,
    scheduler: Optional[Any] = None,
    extra_state: Optional[Dict] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        loss: Current loss value
        output_path: Path to save the checkpoint
        scheduler: Optional learning rate scheduler
        extra_state: Optional additional state to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra_state is not None:
        checkpoint.update(extra_state)

    torch.save(checkpoint, output_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the checkpoint to

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "timestamp": checkpoint.get("timestamp"),
    }


def setup_logging(
    output_dir: Path,
    name: str = "training",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up logging for training.

    Creates both console and file handlers.

    Args:
        output_dir: Directory for log files
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


@dataclass
class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Tracks validation loss and stops training if no improvement
    for a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better) or 'max' for metrics (higher is better)
    """
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    counter: int = field(default=0, init=False)
    best_value: float = field(default=None, init=False)
    should_stop: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.should_stop = False
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")


class LRScheduler:
    """
    Learning rate scheduler wrapper with warmup support.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ('step', 'cosine', 'linear')
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        **kwargs: Additional scheduler arguments
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 0,
        total_epochs: int = 100,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Create main scheduler
        if scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=kwargs.get("eta_min", 1e-6),
            )
        elif scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=kwargs.get("end_factor", 0.01),
                total_iters=total_epochs - warmup_epochs,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def step(self):
        """Take a scheduler step."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = base_lr * warmup_factor
        else:
            self.scheduler.step()

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            "current_epoch": self.current_epoch,
            "scheduler_state": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state["current_epoch"]
        self.scheduler.load_state_dict(state["scheduler_state"])


class MetricTracker:
    """
    Track and aggregate training metrics.

    Provides running averages and history for logging.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.running: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]):
        """Add new metric values."""
        for name, value in metrics.items():
            if name not in self.running:
                self.running[name] = []
            self.running[name].append(value)

    def get_average(self, name: str) -> float:
        """Get running average for a metric."""
        if name not in self.running or not self.running[name]:
            return 0.0
        return sum(self.running[name]) / len(self.running[name])

    def get_all_averages(self) -> Dict[str, float]:
        """Get running averages for all metrics."""
        return {name: self.get_average(name) for name in self.running}

    def end_epoch(self):
        """End current epoch, save averages to history."""
        for name, values in self.running.items():
            if name not in self.history:
                self.history[name] = []
            if values:
                self.history[name].append(sum(values) / len(values))

        # Clear running values
        self.running = {}

    def get_history(self, name: str) -> List[float]:
        """Get history for a metric."""
        return self.history.get(name, [])

    def save(self, path: Path):
        """Save metrics history to JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
