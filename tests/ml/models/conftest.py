"""
Shared pytest fixtures for ML model tests.
"""
import pytest
import torch


@pytest.fixture
def sample_image_tensor():
    """Standard RGB image tensor for testing."""
    return torch.randn(2, 3, 256, 256)


@pytest.fixture
def sample_grayscale_tensor():
    """Grayscale image tensor for CRNN testing."""
    return torch.randn(2, 1, 32, 200)


@pytest.fixture
def sample_sequence_tensor():
    """Sequence tensor for recognition testing."""
    return torch.randn(2, 50, 256)


@pytest.fixture
def device():
    """Returns available device (cpu/cuda/mps)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@pytest.fixture
def sample_feature_maps():
    """Multi-scale feature maps for FPN testing."""
    return {
        'f2': torch.randn(2, 64, 160, 160),
        'f3': torch.randn(2, 128, 80, 80),
        'f4': torch.randn(2, 256, 40, 40),
        'f5': torch.randn(2, 512, 20, 20),
    }


@pytest.fixture
def sample_vgg_features():
    """VGG-style multi-scale features for CRAFT testing."""
    return {
        'pool1': torch.randn(2, 64, 128, 128),
        'pool2': torch.randn(2, 128, 64, 64),
        'pool3': torch.randn(2, 256, 32, 32),
        'pool4': torch.randn(2, 512, 16, 16),
        'pool5': torch.randn(2, 512, 8, 8),
    }


@pytest.fixture
def sample_ctc_logits():
    """CTC logits for decoding tests [B, T, num_classes]."""
    # Batch=2, Time=10, Classes=4 (a, b, c, blank)
    return torch.randn(2, 10, 4)


@pytest.fixture
def sample_db_features():
    """Feature map for DBHead testing."""
    return torch.randn(2, 1024, 160, 160)
