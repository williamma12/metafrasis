"""
Integration tests for training pipeline.

Tests end-to-end training on small sample datasets.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from services.annotation.models import (
    AnnotationDataset,
    AnnotatedImage,
    Region,
    Point,
)
from services.annotation.storage import AnnotationStorage


class TestDetectorTrainingPipeline:
    """End-to-end tests for detector training."""

    @pytest.fixture
    def training_dataset(self, tmp_path):
        """Create a minimal training dataset with images and annotations."""
        # Create storage
        storage = AnnotationStorage(base_path=tmp_path)

        # Create dataset
        dataset = AnnotationDataset(name="integration_test", version="1.0")

        # Create test images with annotations
        for i in range(3):
            # Create image with some content
            img = Image.new("RGB", (256, 256), color="white")
            pixels = img.load()
            # Draw a dark rectangle to simulate text
            for x in range(50 + i * 20, 150 + i * 20):
                for y in range(50, 100):
                    pixels[x, y] = (20, 20, 20)

            # Save image
            dataset_dir = tmp_path / "integration_test"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            img_path = images_dir / f"test_{i}.png"
            img.save(img_path)

            # Create annotation
            region = Region(
                id=f"region_{i}",
                type="rectangle",
                points=[
                    Point(x=50 + i * 20, y=50),
                    Point(x=150 + i * 20, y=50),
                    Point(x=150 + i * 20, y=100),
                    Point(x=50 + i * 20, y=100),
                ],
                text=f"Text {i}",
                verified=True,
            )

            annotated_image = AnnotatedImage(
                id=f"img_{i}",
                image_path=f"images/test_{i}.png",
                width=256,
                height=256,
                regions=[region],
            )
            dataset.add_image(annotated_image)

        # Save dataset
        storage.save(dataset)

        return tmp_path / "integration_test", storage

    @pytest.fixture
    def training_config(self, training_dataset, tmp_path):
        """Create minimal training config."""
        data_dir, _ = training_dataset
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(exist_ok=True)

        return {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "warmup_epochs": 0,
            "patience": 10,
            "split_ratio": 0.67,  # 2 train, 1 val
            "num_workers": 0,  # Avoid multiprocessing issues in tests
            "img_size": 256,
        }

    @pytest.mark.slow
    def test_db_trainer_runs(self, training_config):
        """Test that DB trainer can run on minimal dataset."""
        from training.finetune.detectors.db import DBTrainer

        trainer = DBTrainer(training_config)
        results = trainer.train()

        assert results is not None
        assert "best_loss" in results
        assert "epochs_trained" in results
        assert results["epochs_trained"] == training_config["epochs"]

        # Check output files exist
        output_dir = Path(training_config["output_dir"])
        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "final_model.pt").exists()
        assert (output_dir / "metrics.json").exists()

    @pytest.mark.slow
    def test_craft_trainer_runs(self, training_config):
        """Test that CRAFT trainer can run on minimal dataset."""
        from training.finetune.detectors.craft import CRAFTTrainer

        trainer = CRAFTTrainer(training_config)
        results = trainer.train()

        assert results is not None
        assert "best_loss" in results
        assert results["epochs_trained"] == training_config["epochs"]


class TestRecognizerTrainingPipeline:
    """End-to-end tests for recognizer training."""

    @pytest.fixture
    def recognizer_dataset(self, tmp_path):
        """Create a minimal recognizer training dataset."""
        storage = AnnotationStorage(base_path=tmp_path)

        dataset = AnnotationDataset(name="recognizer_test", version="1.0")

        # Create test images with text regions
        for i in range(4):
            # Create narrow image like a text line
            img = Image.new("RGB", (200, 32), color="white")
            pixels = img.load()
            # Draw some dark pixels to simulate text
            for x in range(20, 180):
                for y in range(8, 24):
                    if (x + y) % 3 == 0:
                        pixels[x, y] = (30, 30, 30)

            # Save image
            dataset_dir = tmp_path / "recognizer_test"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            img.save(images_dir / f"line_{i}.png")

            # Create annotation with text
            region = Region(
                id=f"region_{i}",
                type="rectangle",
                points=[
                    Point(x=0, y=0),
                    Point(x=200, y=0),
                    Point(x=200, y=32),
                    Point(x=0, y=32),
                ],
                text=f"test{i}",
                verified=True,
            )

            annotated_image = AnnotatedImage(
                id=f"line_{i}",
                image_path=f"images/line_{i}.png",
                width=200,
                height=32,
                regions=[region],
            )
            dataset.add_image(annotated_image)

        storage.save(dataset)
        return tmp_path / "recognizer_test", storage

    @pytest.fixture
    def recognizer_config(self, recognizer_dataset, tmp_path):
        """Create minimal recognizer training config."""
        data_dir, _ = recognizer_dataset
        output_dir = tmp_path / "recognizer_outputs"
        output_dir.mkdir(exist_ok=True)

        return {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "warmup_epochs": 0,
            "patience": 10,
            "split_ratio": 0.75,  # 3 train, 1 val
            "num_workers": 0,
            "img_height": 32,
            "img_width": 200,
            "device": "cpu",  # CTC loss not supported on MPS
        }

    @pytest.mark.slow
    def test_crnn_trainer_runs(self, recognizer_config):
        """Test that CRNN trainer can run on minimal dataset."""
        from training.finetune.recognizers.crnn import CRNNTrainer

        trainer = CRNNTrainer(recognizer_config)
        results = trainer.train()

        assert results is not None
        assert "best_loss" in results
        assert results["epochs_trained"] == recognizer_config["epochs"]


class TestDatasetLoading:
    """Tests for dataset loading functionality."""

    @pytest.fixture
    def annotation_dataset(self, tmp_path):
        """Create dataset in annotation format."""
        storage = AnnotationStorage(base_path=tmp_path)

        dataset = AnnotationDataset(name="test_ds", version="1.0")

        # Create one image
        img = Image.new("RGB", (100, 100), color="white")
        dataset_dir = tmp_path / "test_ds"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        img.save(images_dir / "img.png")

        region = Region(
            id="r1",
            type="rectangle",
            points=[
                Point(x=10, y=10),
                Point(x=90, y=10),
                Point(x=90, y=90),
                Point(x=10, y=90),
            ],
            text="test",
            verified=True,
        )

        annotated_image = AnnotatedImage(
            id="img1",
            image_path="images/img.png",
            width=100,
            height=100,
            regions=[region],
        )
        dataset.add_image(annotated_image)
        storage.save(dataset)

        return tmp_path / "test_ds"

    def test_detector_dataset_loads_annotation_format(self, annotation_dataset, tmp_path):
        """Test that detector dataset can load from annotation format."""
        from training.finetune.detectors.db import DBDataset
        from training.finetune.base import ZipExportStorage

        storage = ZipExportStorage(annotation_dataset)
        data = storage.load("dataset")

        assert data is not None
        assert len(data.images) == 1

        # Create dataset
        dataset = DBDataset(
            annotation_dataset,
            annotation_dataset=data,
            storage=storage,
            img_size=256,
        )

        assert len(dataset) == 1

        # Get item - returns (image_tensor, targets_dict)
        image, targets = dataset[0]
        assert image is not None
        assert isinstance(targets, dict)

    def test_recognizer_dataset_loads_annotation_format(self, annotation_dataset, tmp_path):
        """Test that recognizer dataset can load from annotation format."""
        from training.finetune.recognizers.base import RecognizerDataset
        from training.finetune.base import ZipExportStorage

        storage = ZipExportStorage(annotation_dataset)
        data = storage.load("dataset")

        assert data is not None

        # Create dataset
        dataset = RecognizerDataset(
            annotation_dataset,
            annotation_dataset=data,
            storage=storage,
            img_height=32,
            img_width=200,
        )

        assert len(dataset) == 1

        # Get item - returns (image_tensor, label_tensor, label_length)
        sample = dataset[0]
        assert sample is not None
