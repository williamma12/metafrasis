"""Tests for finetune base classes."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from training.finetune.base import BaseTrainer


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class MockDataset:
    """Mock dataset for testing."""
    pass


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""

    @property
    def name(self) -> str:
        return "test_trainer"

    @property
    def dataset_class(self) -> type:
        return MockDataset

    def create_model(self) -> nn.Module:
        return MockModel()

    def create_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def create_dataloaders(self):
        # Return mock dataloaders
        return MagicMock(), MagicMock()

    def train_step(self, batch):
        return {"loss": 0.1}

    def validate(self, dataloader):
        return {"val_loss": 0.05, "accuracy": 0.9}


class TestBaseTrainer:
    """Tests for BaseTrainer abstract base class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cannot_instantiate_directly(self, temp_dir):
        """Test that BaseTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTrainer({"output_dir": str(temp_dir)})

    def test_concrete_implementation(self, temp_dir):
        """Test creating a concrete implementation."""
        config = {"output_dir": str(temp_dir)}
        trainer = ConcreteTrainer(config)
        assert trainer.name == "test_trainer"
        assert trainer.config == config

    def test_output_dir_created(self, temp_dir):
        """Test that output directory is created."""
        output_dir = temp_dir / "training_output"
        config = {"output_dir": str(output_dir)}
        trainer = ConcreteTrainer(config)
        assert output_dir.exists()

    def test_device_selection(self, temp_dir):
        """Test device selection."""
        config = {"output_dir": str(temp_dir)}
        trainer = ConcreteTrainer(config)
        # Should be either 'cpu', 'cuda', or 'mps'
        assert trainer.device.type in ["cpu", "cuda", "mps"]

    def test_create_optimizer_default(self, temp_dir):
        """Test default optimizer creation."""
        config = {"output_dir": str(temp_dir)}
        trainer = ConcreteTrainer(config)
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_custom_lr(self, temp_dir):
        """Test optimizer with custom learning rate."""
        config = {
            "output_dir": str(temp_dir),
            "learning_rate": 0.001
        }
        trainer = ConcreteTrainer(config)
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)
        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_create_scheduler_default(self, temp_dir):
        """Test default scheduler creation."""
        config = {"output_dir": str(temp_dir)}
        trainer = ConcreteTrainer(config)
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)
        scheduler = trainer.create_scheduler(optimizer)
        assert scheduler is not None

    def test_train_epoch(self, temp_dir):
        """Test train_epoch method."""
        config = {"output_dir": str(temp_dir)}
        trainer = ConcreteTrainer(config)

        # Create mock dataloader
        mock_loader = [
            {"input": torch.randn(4, 10), "target": torch.randn(4, 2)},
            {"input": torch.randn(4, 10), "target": torch.randn(4, 2)},
        ]

        trainer.model = trainer.create_model()
        trainer.criterion = trainer.create_criterion()
        trainer.optimizer = trainer.create_optimizer(trainer.model)

        loss = trainer.train_epoch(mock_loader)
        assert isinstance(loss, float)
        assert loss >= 0


class TestMetricTracker:
    """Tests for MetricTracker utility."""

    def test_metric_tracker_update(self):
        """Test updating metric tracker."""
        from training.finetune.utils import MetricTracker

        tracker = MetricTracker()
        tracker.update({"loss": 0.5, "accuracy": 0.8})
        tracker.end_epoch()

        assert "loss" in tracker.history
        assert "accuracy" in tracker.history
        assert tracker.history["loss"][0] == 0.5
        assert tracker.history["accuracy"][0] == 0.8

    def test_metric_tracker_multiple_epochs(self):
        """Test tracking across multiple epochs."""
        from training.finetune.utils import MetricTracker

        tracker = MetricTracker()

        for i in range(3):
            tracker.update({"loss": 0.5 - i * 0.1})
            tracker.end_epoch()

        assert len(tracker.history["loss"]) == 3
        assert tracker.history["loss"][0] == 0.5
        assert tracker.history["loss"][2] == pytest.approx(0.3, abs=0.001)

    def test_metric_tracker_save(self, temp_dir):
        """Test saving metrics to file."""
        from training.finetune.utils import MetricTracker

        temp_dir = Path(tempfile.mkdtemp())
        tracker = MetricTracker()
        tracker.update({"loss": 0.5})
        tracker.end_epoch()

        save_path = temp_dir / "metrics.json"
        tracker.save(save_path)

        assert save_path.exists()


class TestEarlyStopping:
    """Tests for EarlyStopping utility."""

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after no improvement."""
        from training.finetune.utils import EarlyStopping

        early_stopping = EarlyStopping(patience=3, mode="min")

        # 4 epochs without improvement
        assert not early_stopping(0.5)
        assert not early_stopping(0.6)  # worse
        assert not early_stopping(0.7)  # worse
        assert early_stopping(0.8)  # triggers

    def test_early_stopping_improvement_resets(self):
        """Test that improvement resets counter."""
        from training.finetune.utils import EarlyStopping

        early_stopping = EarlyStopping(patience=2, mode="min")

        assert not early_stopping(0.5)
        assert not early_stopping(0.6)  # worse
        assert not early_stopping(0.4)  # better - resets
        assert not early_stopping(0.5)  # worse
        assert early_stopping(0.6)  # triggers

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        from training.finetune.utils import EarlyStopping

        early_stopping = EarlyStopping(patience=2, mode="max")

        assert not early_stopping(0.5)
        assert not early_stopping(0.4)  # worse in max mode
        assert early_stopping(0.3)  # triggers


class TestLRScheduler:
    """Tests for LRScheduler utility."""

    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        from training.finetune.utils import LRScheduler

        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        scheduler = LRScheduler(
            optimizer,
            scheduler_type="cosine",
            warmup_epochs=0,
            total_epochs=10
        )

        initial_lr = scheduler.get_lr()[0]
        assert initial_lr == 0.01

        # After stepping
        for _ in range(5):
            scheduler.step()

        mid_lr = scheduler.get_lr()[0]
        assert mid_lr < initial_lr

    def test_step_scheduler(self):
        """Test step scheduler."""
        from training.finetune.utils import LRScheduler

        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        scheduler = LRScheduler(
            optimizer,
            scheduler_type="step",
            warmup_epochs=0,
            total_epochs=100,
            step_size=10,
            gamma=0.1
        )

        initial_lr = scheduler.get_lr()[0]

        # Step 10 times
        for _ in range(10):
            scheduler.step()

        lr_after_step = scheduler.get_lr()[0]
        assert abs(lr_after_step - initial_lr * 0.1) < 1e-6


class TestCheckpointing:
    """Tests for save/load checkpoint utilities."""

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        from training.finetune.utils import save_checkpoint

        temp_dir = Path(tempfile.mkdtemp())
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())

        path = temp_dir / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch=5, loss=0.1, output_path=path)

        assert path.exists()
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["epoch"] == 5
        assert checkpoint["loss"] == 0.1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        from training.finetune.utils import save_checkpoint, load_checkpoint

        temp_dir = Path(tempfile.mkdtemp())
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())

        path = temp_dir / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch=5, loss=0.1, output_path=path)

        # Create new model and optimizer
        new_model = MockModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        checkpoint = load_checkpoint(path, new_model, new_optimizer)
        assert checkpoint["epoch"] == 5
