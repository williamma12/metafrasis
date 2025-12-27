"""Tests for concrete trainer implementations."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class TestCRNNTrainer:
    """Tests for CRNN trainer."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_import(self):
        """Test that CRNNTrainer can be imported."""
        from training.finetune import CRNNTrainer
        assert CRNNTrainer is not None

    def test_name_property(self, temp_dir):
        """Test trainer name property."""
        from training.finetune import CRNNTrainer

        config = {"output_dir": str(temp_dir), "data_dir": str(temp_dir)}
        trainer = CRNNTrainer(config)
        assert trainer.name == "crnn"

    def test_inherits_from_ctc_trainer(self):
        """Test that CRNNTrainer inherits from CTCRecognizerTrainer."""
        from training.finetune import CRNNTrainer, CTCRecognizerTrainer

        assert issubclass(CRNNTrainer, CTCRecognizerTrainer)


class TestPPOCRTrainer:
    """Tests for PP-OCR trainer."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_import(self):
        """Test that PPOCRTrainer can be imported."""
        from training.finetune import PPOCRTrainer
        assert PPOCRTrainer is not None

    def test_name_property(self, temp_dir):
        """Test trainer name property."""
        from training.finetune import PPOCRTrainer

        config = {"output_dir": str(temp_dir), "data_dir": str(temp_dir)}
        trainer = PPOCRTrainer(config)
        assert trainer.name == "ppocr"

    def test_inherits_from_ctc_trainer(self):
        """Test that PPOCRTrainer inherits from CTCRecognizerTrainer."""
        from training.finetune import PPOCRTrainer, CTCRecognizerTrainer

        assert issubclass(PPOCRTrainer, CTCRecognizerTrainer)

    def test_has_gradient_accumulation(self, temp_dir):
        """Test that PPOCRTrainer has accumulation step tracking."""
        from training.finetune import PPOCRTrainer

        config = {
            "output_dir": str(temp_dir),
            "data_dir": str(temp_dir),
            "accumulation_steps": 4
        }
        trainer = PPOCRTrainer(config)
        assert trainer.config.get("accumulation_steps") == 4


class TestTrOCRTrainer:
    """Tests for TrOCR trainer."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_import(self):
        """Test that TrOCRTrainer can be imported."""
        from training.finetune import TrOCRTrainer
        assert TrOCRTrainer is not None

    def test_name_property(self, temp_dir):
        """Test trainer name property."""
        from training.finetune import TrOCRTrainer

        config = {"output_dir": str(temp_dir), "data_dir": str(temp_dir)}
        trainer = TrOCRTrainer(config)
        assert trainer.name == "trocr"

    def test_inherits_from_transformer_trainer(self):
        """Test that TrOCRTrainer inherits from TransformerRecognizerTrainer."""
        from training.finetune import TrOCRTrainer, TransformerRecognizerTrainer

        assert issubclass(TrOCRTrainer, TransformerRecognizerTrainer)

    def test_lora_config_defaults(self, temp_dir):
        """Test that LoRA config has defaults."""
        from training.finetune import TrOCRTrainer

        config = {"output_dir": str(temp_dir), "data_dir": str(temp_dir)}
        trainer = TrOCRTrainer(config)

        # These should have defaults
        assert trainer.config.get("lora_r", 16) == 16
        assert trainer.config.get("lora_alpha", 32) == 32


class TestCRAFTTrainer:
    """Tests for CRAFT detector trainer."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_import(self):
        """Test that CRAFTTrainer can be imported."""
        from training.finetune import CRAFTTrainer
        assert CRAFTTrainer is not None

    def test_name_property(self, temp_dir):
        """Test trainer name property."""
        from training.finetune import CRAFTTrainer

        config = {"output_dir": str(temp_dir), "data_dir": str(temp_dir)}
        trainer = CRAFTTrainer(config)
        assert trainer.name == "craft"

    def test_inherits_from_detector_trainer(self):
        """Test that CRAFTTrainer inherits from DetectorTrainer."""
        from training.finetune import CRAFTTrainer, DetectorTrainer

        assert issubclass(CRAFTTrainer, DetectorTrainer)


class TestDBTrainer:
    """Tests for DB detector trainer."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_import(self):
        """Test that DBTrainer can be imported."""
        from training.finetune import DBTrainer
        assert DBTrainer is not None

    def test_name_property(self, temp_dir):
        """Test trainer name property."""
        from training.finetune import DBTrainer

        config = {"output_dir": str(temp_dir), "data_dir": str(temp_dir)}
        trainer = DBTrainer(config)
        assert trainer.name == "db"

    def test_inherits_from_detector_trainer(self):
        """Test that DBTrainer inherits from DetectorTrainer."""
        from training.finetune import DBTrainer, DetectorTrainer

        assert issubclass(DBTrainer, DetectorTrainer)


class TestCRAFTLoss:
    """Tests for CRAFT loss function."""

    def test_import(self):
        """Test that CRAFTLoss can be imported."""
        from training.finetune.detectors import CRAFTLoss
        assert CRAFTLoss is not None

    def test_loss_computation(self):
        """Test loss computation with mock inputs."""
        from training.finetune.detectors import CRAFTLoss

        loss_fn = CRAFTLoss(neg_ratio=3.0)

        # Mock inputs
        batch_size = 2
        height, width = 64, 64

        region_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
        affinity_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
        region_target = torch.zeros(batch_size, 1, height, width)
        affinity_target = torch.zeros(batch_size, 1, height, width)

        # Add some positive samples
        region_target[:, :, 20:40, 20:40] = 1.0
        affinity_target[:, :, 20:40, 30:35] = 1.0

        loss, loss_dict = loss_fn(
            region_pred, affinity_pred, region_target, affinity_target
        )

        assert isinstance(loss, torch.Tensor)
        assert "region_loss" in loss_dict
        assert "affinity_loss" in loss_dict


class TestDBLoss:
    """Tests for DB loss function."""

    def test_import(self):
        """Test that DBLoss can be imported."""
        from training.finetune.detectors import DBLoss
        assert DBLoss is not None

    def test_loss_computation(self):
        """Test loss computation with mock inputs."""
        from training.finetune.detectors import DBLoss

        loss_fn = DBLoss(bce_weight=1.0, l1_weight=10.0, dice_weight=1.0)

        batch_size = 2
        height, width = 64, 64

        prob_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
        thresh_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
        binary_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
        prob_target = torch.zeros(batch_size, 1, height, width)
        thresh_target = torch.zeros(batch_size, 1, height, width)
        shrink_mask = torch.ones(batch_size, 1, height, width)

        # Add some positive samples
        prob_target[:, :, 20:40, 20:40] = 1.0
        thresh_target[:, :, 20:40, 20:40] = 0.5

        loss, loss_dict = loss_fn(
            prob_pred, thresh_pred, binary_pred,
            prob_target, thresh_target, shrink_mask
        )

        assert isinstance(loss, torch.Tensor)
        assert "bce_loss" in loss_dict
        assert "l1_loss" in loss_dict
        assert "dice_loss" in loss_dict


class TestRecognizerDataset:
    """Tests for RecognizerDataset."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_import(self):
        """Test that RecognizerDataset can be imported."""
        from training.finetune.recognizers import RecognizerDataset
        assert RecognizerDataset is not None

    def test_dataset_creation(self, temp_dir):
        """Test creating a dataset with sample data."""
        from training.finetune.recognizers import RecognizerDataset
        from PIL import Image

        # Create sample data
        images_dir = temp_dir / "images"
        images_dir.mkdir()

        # Create sample image
        img = Image.new("RGB", (100, 32), color="white")
        img.save(images_dir / "test.png")

        # Create labels file
        with open(temp_dir / "labels.txt", "w") as f:
            f.write("test.png\thello world\n")

        # Create dataset
        dataset = RecognizerDataset(
            temp_dir,
            img_height=32,
            img_width=128
        )

        assert len(dataset) == 1
        assert "h" in dataset.char_to_idx


class TestDetectorDataset:
    """Tests for DetectorDataset."""

    def test_import(self):
        """Test that DetectorDataset can be imported."""
        from training.finetune.detectors import DetectorDataset
        assert DetectorDataset is not None
