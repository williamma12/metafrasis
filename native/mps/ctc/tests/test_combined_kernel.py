"""
Tests for the combined CTC forward-backward kernel (ctc_forward_backward_combined).

This kernel combines forward pass, backward pass, and gradient computation into
a single kernel launch with state-level parallelism using SIMD reduction.

Run with: pytest mps_ctc/tests/test_combined_kernel.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# FIXTURES AND HELPERS
# =============================================================================


def has_mps() -> bool:
    """Check if MPS backend is available."""
    return torch.backends.mps.is_available()


def has_native_extension() -> bool:
    """Check if the native MPS CTC extension is available."""
    try:
        from native.mps.ctc import _C
        return (
            hasattr(_C, 'ctc_loss_forward') and
            hasattr(_C, 'ctc_loss_backward') and
            hasattr(_C, 'ctc_loss_combined')
        )
    except ImportError:
        return False


def has_combined_kernel() -> bool:
    """Check if the combined kernel is available."""
    try:
        from native.mps.ctc import _C
        return hasattr(_C, 'ctc_loss_combined')
    except ImportError:
        return False


# Skip markers
requires_mps = pytest.mark.skipif(not has_mps(), reason="MPS not available")
requires_native = pytest.mark.skipif(
    not has_native_extension(),
    reason="Native MPS CTC extension not compiled"
)
requires_combined = pytest.mark.skipif(
    not (has_mps() and has_combined_kernel()),
    reason="Requires MPS and combined kernel"
)


def create_ctc_data(
    T: int = 50,
    B: int = 4,
    C: int = 20,
    S: int = 10,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test data for CTC loss."""
    torch.manual_seed(seed)

    log_probs = torch.randn(T, B, C, device=device).log_softmax(2)
    targets = torch.randint(1, C, (B, S), device=device)  # Avoid blank (0)
    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    target_lengths = torch.randint(1, S + 1, (B,), dtype=torch.long, device=device)

    return log_probs, targets, input_lengths, target_lengths


def cpu_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = 'none'
) -> torch.Tensor:
    """Compute CTC loss using PyTorch's CPU reference implementation."""
    if targets.dim() == 2:
        targets_list = [targets[b, :target_lengths[b].item()].cpu() for b in range(targets.size(0))]
        targets_1d = torch.cat(targets_list)
    else:
        targets_1d = targets.cpu()

    return F.ctc_loss(
        log_probs.cpu(),
        targets_1d,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=blank,
        reduction=reduction,
        zero_infinity=False
    )


# =============================================================================
# COMBINED KERNEL TESTS
# =============================================================================


class TestCombinedKernelBasic:
    """Basic tests for the combined forward-backward kernel."""

    @requires_combined
    def test_combined_kernel_available(self):
        """Test that combined kernel is available."""
        from native.mps.ctc import _C
        assert hasattr(_C, 'ctc_loss_combined')
        assert callable(_C.ctc_loss_combined)

    @requires_combined
    def test_combined_basic(self):
        """Test basic combined kernel execution."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(2, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        assert loss.shape == (2,)
        assert grad.shape == log_probs.shape
        assert not torch.isnan(loss).any()
        assert not torch.isnan(grad).any()

    @requires_combined
    def test_combined_output_devices(self):
        """Test that outputs are on MPS device."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(2, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        assert loss.device.type == 'mps'
        assert grad.device.type == 'mps'


class TestCombinedVsSeparateKernels:
    """Tests comparing combined kernel against separate forward/backward kernels."""

    @requires_combined
    def test_loss_matches_forward_kernel(self):
        """Test that combined loss matches separate forward kernel."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=50, B=4, C=20, S=10, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(4, device='mps', dtype=torch.float32)

        # Separate forward kernel
        loss_separate, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        # Combined kernel
        loss_combined, _ = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        torch.testing.assert_close(
            loss_combined, loss_separate,
            rtol=1e-3, atol=1e-3,
            msg="Combined loss should match separate forward kernel"
        )

    @requires_combined
    def test_gradient_matches_backward_kernel(self):
        """Test that combined gradient matches separate backward kernel."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=50, B=4, C=20, S=10, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(4, device='mps', dtype=torch.float32)

        # Separate forward + backward
        loss_fwd, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        grad_separate = _C.ctc_loss_backward(
            grad_output, log_probs, alpha, targets,
            input_lengths, target_lengths, blank=0
        )

        # Combined kernel
        _, grad_combined = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        torch.testing.assert_close(
            grad_combined, grad_separate,
            rtol=1e-3, atol=1e-3,
            msg="Combined gradient should match separate backward kernel"
        )

    @requires_combined
    @pytest.mark.parametrize("T,B,C,S", [
        (10, 1, 5, 3),
        (20, 2, 10, 5),
        (50, 8, 20, 10),
        (100, 16, 50, 20),
    ])
    def test_combined_matches_separate_various_sizes(self, T, B, C, S):
        """Test combined vs separate kernels with various sizes."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=T, B=B, C=C, S=S, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        # Separate kernels
        loss_fwd, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        grad_separate = _C.ctc_loss_backward(
            grad_output, log_probs, alpha, targets,
            input_lengths, target_lengths, blank=0
        )

        # Combined kernel
        loss_combined, grad_combined = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        torch.testing.assert_close(loss_combined, loss_fwd, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(grad_combined, grad_separate, rtol=1e-3, atol=1e-3)


class TestCombinedVsCPUReference:
    """Tests comparing combined kernel against PyTorch CPU reference."""

    @requires_combined
    def test_loss_matches_cpu(self):
        """Test that combined loss matches CPU reference."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=50, B=4, C=20, S=10, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(4, device='mps', dtype=torch.float32)

        # Combined kernel
        loss_combined, _ = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        # CPU reference
        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(
            loss_combined.cpu(), cpu_loss,
            rtol=1e-3, atol=1e-3,
            msg="Combined loss should match CPU reference"
        )

    @requires_combined
    def test_gradient_matches_cpu(self):
        """Test that combined gradient matches CPU reference."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        # CPU gradient
        log_probs_cpu = log_probs.cpu().clone().requires_grad_(True)
        targets_cpu = targets.cpu()
        input_lengths_cpu = input_lengths.cpu()
        target_lengths_cpu = target_lengths.cpu()

        targets_1d = torch.cat([
            targets_cpu[b, :target_lengths_cpu[b].item()]
            for b in range(targets_cpu.size(0))
        ])

        cpu_loss = F.ctc_loss(
            log_probs_cpu, targets_1d, input_lengths_cpu, target_lengths_cpu,
            blank=0, reduction='sum'
        )
        cpu_loss.backward()
        cpu_grad = log_probs_cpu.grad.clone()

        # Combined kernel
        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(4, device='mps', dtype=torch.float32)

        _, grad_combined = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        torch.testing.assert_close(
            grad_combined.cpu(), cpu_grad,
            rtol=1e-3, atol=1e-3,
            msg="Combined gradient should match CPU reference"
        )


class TestCombinedVariableLengths:
    """Tests for combined kernel with variable sequence lengths."""

    @requires_combined
    def test_variable_input_lengths(self):
        """Test with variable input lengths per batch element."""
        from native.mps.ctc import _C

        T, B, C, S = 100, 4, 30, 20
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.tensor([100, 80, 60, 90], device='mps', dtype=torch.int)
        target_lengths = torch.tensor([20, 15, 10, 18], device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        # Combined kernel
        loss_combined, grad_combined = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        # CPU reference
        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(
            loss_combined.cpu(), cpu_loss,
            rtol=1e-3, atol=1e-3
        )

    @requires_combined
    def test_gradients_zero_beyond_input_length(self):
        """Test that gradients are zero beyond each sequence's input_length."""
        from native.mps.ctc import _C

        T, B, C, S = 50, 2, 10, 5
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.tensor([30, 40], device='mps', dtype=torch.int)
        target_lengths = torch.tensor([5, 5], device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        _, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        grad_cpu = grad.cpu()

        # Check that gradients beyond input_lengths are zero
        assert torch.allclose(grad_cpu[30:, 0, :], torch.zeros(T - 30, C))
        assert torch.allclose(grad_cpu[40:, 1, :], torch.zeros(T - 40, C))


class TestCombinedEdgeCases:
    """Edge case tests for combined kernel."""

    @requires_combined
    def test_single_batch(self):
        """Test with batch size of 1."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=1, C=10, S=5, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(1, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        assert loss.shape == (1,)
        assert not torch.isnan(loss).any()
        assert not torch.isnan(grad).any()

    @requires_combined
    def test_single_char_target(self):
        """Test with single character targets."""
        from native.mps.ctc import _C

        T, B, C = 20, 2, 10
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, 1), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.ones(B, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)

    @requires_combined
    def test_repeated_characters(self):
        """Test with repeated characters in target."""
        from native.mps.ctc import _C

        T, B, C = 30, 2, 10
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.tensor(
            [[1, 1, 2, 2, 3], [2, 2, 2, 1, 1]],
            device='mps', dtype=torch.int
        )
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), 5, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)

    @requires_combined
    def test_min_viable_input_length(self):
        """Test with minimum viable input length for target."""
        from native.mps.ctc import _C

        B, C, S = 2, 10, 3
        T = 2 * S + 1  # Minimum for repeated chars

        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        assert not torch.isnan(loss).any()
        assert not torch.isnan(grad).any()

    @requires_combined
    def test_non_zero_blank(self):
        """Test with non-zero blank index."""
        from native.mps.ctc import _C

        T, B, C, S = 20, 2, 10, 5
        blank = 5

        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(0, blank, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=blank
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)


class TestCombinedNumericalStability:
    """Numerical stability tests for combined kernel."""

    @requires_combined
    def test_long_sequence(self):
        """Test numerical stability with long sequences."""
        from native.mps.ctc import _C

        T, B, C, S = 200, 2, 50, 30
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        assert not torch.isnan(loss).any(), "Loss should not be NaN for long sequences"
        assert not torch.isinf(loss).any(), "Loss should not be Inf for long sequences"
        assert not torch.isnan(grad).any(), "Gradient should not be NaN"

    @requires_combined
    def test_extreme_log_probs(self):
        """Test with extreme (very confident) log probabilities."""
        from native.mps.ctc import _C

        T, B, C, S = 30, 2, 10, 5
        torch.manual_seed(42)

        logits = torch.randn(T, B, C, device='mps') * 10
        log_probs = logits.log_softmax(2).contiguous().float()

        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        assert not torch.isnan(loss).any()
        assert not torch.isnan(grad).any()

    @requires_combined
    def test_uniform_log_probs(self):
        """Test with uniform (uncertain) log probabilities."""
        from native.mps.ctc import _C

        T, B, C, S = 30, 2, 10, 5
        torch.manual_seed(42)

        logits = torch.zeros(T, B, C, device='mps')
        log_probs = logits.log_softmax(2).contiguous().float()

        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)
        grad_output = torch.ones(B, device='mps', dtype=torch.float32)

        loss, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)


class TestCombinedGradientScaling:
    """Tests for gradient scaling with grad_output."""

    @requires_combined
    def test_grad_output_scaling(self):
        """Test that grad_output properly scales gradients."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        # Gradient with ones
        grad_output_1 = torch.ones(4, device='mps', dtype=torch.float32)
        _, grad_1 = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output_1, blank=0
        )

        # Gradient with twos (should be 2x grad_1)
        grad_output_2 = torch.full((4,), 2.0, device='mps', dtype=torch.float32)
        _, grad_2 = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output_2, blank=0
        )

        torch.testing.assert_close(
            grad_2, grad_1 * 2,
            rtol=1e-4, atol=1e-4,
            msg="Gradient should scale linearly with grad_output"
        )

    @requires_combined
    def test_per_sample_grad_output(self):
        """Test with different grad_output per sample."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        # Different scale per batch element
        grad_output = torch.tensor([1.0, 2.0, 0.5, 3.0], device='mps', dtype=torch.float32)

        _, grad = _C.ctc_loss_combined(
            log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
        )

        # Check gradient is not all zeros and has expected structure
        assert not torch.isnan(grad).any()
        assert grad.abs().sum() > 0


class TestCombinedConsistency:
    """Tests for consistency across multiple calls."""

    @requires_combined
    def test_deterministic(self):
        """Test that combined kernel produces deterministic results."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(4, device='mps', dtype=torch.float32)

        # Run multiple times
        results = []
        for _ in range(3):
            loss, grad = _C.ctc_loss_combined(
                log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
            )
            torch.mps.synchronize()
            results.append((loss.clone(), grad.clone()))

        # All results should be identical
        for i in range(1, len(results)):
            torch.testing.assert_close(results[0][0], results[i][0])
            torch.testing.assert_close(results[0][1], results[i][1])

    @requires_combined
    def test_synchronization(self):
        """Test that operations properly synchronize."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        grad_output = torch.ones(4, device='mps', dtype=torch.float32)

        # Run multiple times quickly
        for _ in range(5):
            loss, grad = _C.ctc_loss_combined(
                log_probs, targets, input_lengths, target_lengths, grad_output, blank=0
            )
            torch.mps.synchronize()

        assert not torch.isnan(loss).any()
        assert not torch.isnan(grad).any()
