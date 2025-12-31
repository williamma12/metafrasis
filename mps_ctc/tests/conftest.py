"""
Pytest configuration and shared fixtures for mps_ctc tests.

This module provides:
1. Skip markers for MPS and native extension availability
2. Shared fixtures for creating test data
3. Test configuration and custom markers
"""

import pytest
import torch
from typing import Tuple


# =============================================================================
# AVAILABILITY CHECKS
# =============================================================================


def has_mps() -> bool:
    """Check if MPS backend is available."""
    return torch.backends.mps.is_available()


def has_native_extension() -> bool:
    """Check if the native MPS CTC extension is compiled and available."""
    try:
        from mps_ctc import _C
        return hasattr(_C, 'ctc_loss_forward') and hasattr(_C, 'ctc_loss_backward')
    except ImportError:
        return False


# =============================================================================
# PYTEST MARKERS
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_mps: mark test to run only when MPS is available"
    )
    config.addinivalue_line(
        "markers", "requires_native: mark test to run only when native extension is compiled"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Apply skip markers based on environment."""
    skip_mps = pytest.mark.skip(reason="MPS not available")
    skip_native = pytest.mark.skip(reason="Native extension not compiled")

    for item in items:
        if "requires_mps" in item.keywords and not has_mps():
            item.add_marker(skip_mps)
        if "requires_native" in item.keywords and not has_native_extension():
            item.add_marker(skip_native)


# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def small_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create small test data for quick tests and debugging.

    Returns:
        Tuple of (log_probs, targets, input_lengths, target_lengths)
        - log_probs: [10, 2, 5]
        - targets: [2, 3]
        - input_lengths: [2]
        - target_lengths: [2]
    """
    T, B, C, S = 10, 2, 5, 3
    torch.manual_seed(42)

    log_probs = torch.randn(T, B, C).log_softmax(2)
    targets = torch.randint(1, C, (B, S))
    input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.full((B,), S, dtype=torch.long)

    return log_probs, targets, input_lengths, target_lengths


@pytest.fixture
def medium_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create medium-sized test data for standard tests.

    Returns:
        Tuple of (log_probs, targets, input_lengths, target_lengths)
        - log_probs: [50, 4, 20]
        - targets: [4, 10]
        - input_lengths: [4]
        - target_lengths: [4]
    """
    T, B, C, S = 50, 4, 20, 10
    torch.manual_seed(42)

    log_probs = torch.randn(T, B, C).log_softmax(2)
    targets = torch.randint(1, C, (B, S))
    input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.randint(5, S + 1, (B,), dtype=torch.long)

    return log_probs, targets, input_lengths, target_lengths


@pytest.fixture
def large_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create large test data for stress tests.

    Returns:
        Tuple of (log_probs, targets, input_lengths, target_lengths)
        - log_probs: [200, 16, 100]
        - targets: [16, 50]
        - input_lengths: [16]
        - target_lengths: [16]
    """
    T, B, C, S = 200, 16, 100, 50
    torch.manual_seed(42)

    log_probs = torch.randn(T, B, C).log_softmax(2)
    targets = torch.randint(1, C, (B, S))
    input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.randint(20, S + 1, (B,), dtype=torch.long)

    return log_probs, targets, input_lengths, target_lengths


@pytest.fixture
def variable_length_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data with variable sequence lengths.

    Returns:
        Tuple of (log_probs, targets, input_lengths, target_lengths)
        with varying input and target lengths per batch element.
    """
    T, B, C, S = 100, 4, 30, 20
    torch.manual_seed(42)

    log_probs = torch.randn(T, B, C).log_softmax(2)
    targets = torch.randint(1, C, (B, S))
    input_lengths = torch.tensor([100, 80, 60, 90], dtype=torch.long)
    target_lengths = torch.tensor([20, 15, 10, 18], dtype=torch.long)

    return log_probs, targets, input_lengths, target_lengths


@pytest.fixture
def repeated_char_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data with repeated characters in targets.

    This tests the CTC handling of repeated characters which requires
    blank tokens between them.

    Returns:
        Tuple of (log_probs, targets, input_lengths, target_lengths)
    """
    T, B, C = 30, 2, 10
    torch.manual_seed(42)

    log_probs = torch.randn(T, B, C).log_softmax(2)
    # Targets with repeated chars: [1, 1, 2, 2, 3]
    targets = torch.tensor([[1, 1, 2, 2, 3], [2, 2, 2, 1, 1]], dtype=torch.long)
    input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.full((B,), 5, dtype=torch.long)

    return log_probs, targets, input_lengths, target_lengths


@pytest.fixture
def mps_small_data(small_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Small test data on MPS device.

    Skips if MPS is not available.
    """
    if not has_mps():
        pytest.skip("MPS not available")

    log_probs, targets, input_lengths, target_lengths = small_data
    return (
        log_probs.to('mps'),
        targets.to('mps'),
        input_lengths.to('mps'),
        target_lengths.to('mps'),
    )


@pytest.fixture
def mps_medium_data(medium_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Medium test data on MPS device.

    Skips if MPS is not available.
    """
    if not has_mps():
        pytest.skip("MPS not available")

    log_probs, targets, input_lengths, target_lengths = medium_data
    return (
        log_probs.to('mps'),
        targets.to('mps'),
        input_lengths.to('mps'),
        target_lengths.to('mps'),
    )


@pytest.fixture
def native_format_data(
    mps_medium_data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test data formatted for native extension (contiguous, correct dtypes).

    The native extension requires:
    - log_probs: float32, contiguous
    - targets: int32, contiguous
    - input_lengths: int32, contiguous
    - target_lengths: int32, contiguous
    """
    log_probs, targets, input_lengths, target_lengths = mps_medium_data
    return (
        log_probs.contiguous().float(),
        targets.contiguous().int(),
        input_lengths.contiguous().int(),
        target_lengths.contiguous().int(),
    )


# =============================================================================
# CPU REFERENCE FIXTURE
# =============================================================================


@pytest.fixture
def cpu_ctc_loss():
    """
    Fixture providing a function to compute CPU reference CTC loss.

    Returns:
        Function that computes CTC loss using PyTorch's CPU implementation.
    """
    import torch.nn.functional as F

    def compute_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='none'):
        """Compute CTC loss on CPU."""
        # Ensure everything is on CPU
        log_probs = log_probs.cpu()
        input_lengths = input_lengths.cpu()
        target_lengths = target_lengths.cpu()

        # Convert 2D targets to 1D concatenated format
        if targets.dim() == 2:
            targets_list = [
                targets.cpu()[b, :target_lengths[b].item()]
                for b in range(targets.size(0))
            ]
            targets_1d = torch.cat(targets_list)
        else:
            targets_1d = targets.cpu()

        return F.ctc_loss(
            log_probs,
            targets_1d,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=False
        )

    return compute_loss


# =============================================================================
# HELPER FIXTURES
# =============================================================================


@pytest.fixture
def seed():
    """Fixture to set random seed for reproducibility."""
    def set_seed(value: int = 42):
        torch.manual_seed(value)
        if has_mps():
            # MPS doesn't have a separate seed function yet
            pass
    return set_seed


@pytest.fixture
def tolerance():
    """Fixture providing tolerance values for numerical comparisons."""
    return {
        'rtol': 1e-3,
        'atol': 1e-3,
        'rtol_strict': 1e-4,
        'atol_strict': 1e-4,
    }
