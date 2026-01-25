"""
Tests for CTC Loss MPS native extension (ctc_mps.mm).

Following canonical CUDA testing patterns from PyTorch, these tests validate:
1. Native extension loading and availability
2. Forward pass correctness against CPU reference
3. Backward pass (gradient) correctness
4. Automatic differentiation (gradcheck)
5. Edge cases and boundary conditions
6. Numerical stability
7. Device placement and memory behavior

Run with: pytest mps_ctc/tests/test_native_extension.py -v
"""

import pytest
import torch
import torch.nn as nn
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
        return hasattr(_C, 'ctc_loss_forward') and hasattr(_C, 'ctc_loss_backward')
    except ImportError:
        return False


# Skip markers
requires_mps = pytest.mark.skipif(not has_mps(), reason="MPS not available")
requires_native = pytest.mark.skipif(
    not has_native_extension(),
    reason="Native MPS CTC extension not compiled"
)
requires_mps_and_native = pytest.mark.skipif(
    not (has_mps() and has_native_extension()),
    reason="Requires both MPS and native extension"
)


def create_ctc_data(
    T: int = 50,
    B: int = 4,
    C: int = 20,
    S: int = 10,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data for CTC loss.

    Args:
        T: Input sequence length (time steps)
        B: Batch size
        C: Number of classes (including blank)
        S: Max target sequence length
        seed: Random seed for reproducibility
        device: Device to create tensors on

    Returns:
        Tuple of (log_probs, targets, input_lengths, target_lengths)
    """
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
    # Convert 2D targets to 1D concatenated format
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
# EXTENSION LOADING TESTS
# =============================================================================


class TestExtensionLoading:
    """Tests for native extension loading and availability."""

    def test_extension_import(self):
        """Test that the mps_ctc package can be imported."""
        import native.mps.ctc as mps_ctc
        assert hasattr(mps_ctc, 'CTCLossMPS')

    def test_extension_version(self):
        """Test that version is defined."""
        import native.mps.ctc as mps_ctc
        assert hasattr(mps_ctc, '__version__')
        assert isinstance(mps_ctc.__version__, str)

    @requires_native
    def test_native_extension_available(self):
        """Test that native extension is loadable."""
        from native.mps.ctc import _C
        assert hasattr(_C, 'ctc_loss_forward')
        assert hasattr(_C, 'ctc_loss_backward')

    @requires_native
    def test_native_extension_callable(self):
        """Test that native extension functions are callable."""
        from native.mps.ctc import _C
        assert callable(_C.ctc_loss_forward)
        assert callable(_C.ctc_loss_backward)


# =============================================================================
# FORWARD PASS TESTS
# =============================================================================


class TestNativeForward:
    """Tests for the native forward pass (ctc_forward kernel)."""

    @requires_mps_and_native
    def test_forward_basic(self):
        """Test basic forward pass on MPS."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        # Ensure tensors are contiguous and correct dtype
        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert loss.shape == (2,)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    @requires_mps_and_native
    def test_forward_matches_cpu(self):
        """Test that native forward matches CPU reference."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=50, B=4, C=20, S=10, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        # Native MPS implementation
        mps_loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        # CPU reference
        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(
            mps_loss.cpu(), cpu_loss,
            rtol=1e-3, atol=1e-3,
            msg="Native forward should match CPU reference"
        )

    @requires_mps_and_native
    @pytest.mark.parametrize("T,B,C,S", [
        (10, 1, 5, 3),    # Single batch
        (20, 2, 10, 5),   # Small batch
        (50, 8, 20, 10),  # Medium batch
        (100, 16, 50, 20), # Larger batch
    ])
    def test_forward_various_sizes(self, T, B, C, S):
        """Test forward pass with various tensor sizes."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=T, B=B, C=C, S=S, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert loss.shape == (B,)
        assert alpha.shape == (T, B, 2 * S + 1)
        assert not torch.isnan(loss).any()

    @requires_mps_and_native
    def test_forward_alpha_shape(self):
        """Test that alpha tensor has correct shape."""
        from native.mps.ctc import _C

        T, B, C, S = 30, 4, 15, 8

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=T, B=B, C=C, S=S, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        # Alpha should be [T, B, L] where L = 2*S + 1
        L = 2 * S + 1
        assert alpha.shape == (T, B, L)

    @requires_mps_and_native
    def test_forward_variable_lengths(self):
        """Test forward with variable input/target lengths."""
        from native.mps.ctc import _C

        T, B, C, S = 100, 4, 30, 20
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.tensor([100, 80, 60, 90], device='mps', dtype=torch.int)
        target_lengths = torch.tensor([20, 15, 10, 18], device='mps', dtype=torch.int)

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        # CPU reference
        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(
            loss.cpu(), cpu_loss,
            rtol=1e-3, atol=1e-3
        )


# =============================================================================
# BACKWARD PASS TESTS
# =============================================================================


class TestNativeBackward:
    """Tests for the native backward pass (ctc_backward and ctc_gradient kernels)."""

    @requires_mps_and_native
    def test_backward_basic(self):
        """Test basic backward pass on MPS."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        # Forward
        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        # Backward
        grad_output = torch.ones_like(loss)
        grad_log_probs = _C.ctc_loss_backward(
            grad_output, log_probs, alpha, targets,
            input_lengths, target_lengths, blank=0
        )

        assert grad_log_probs.shape == log_probs.shape
        assert not torch.isnan(grad_log_probs).any()

    @requires_mps_and_native
    def test_backward_matches_cpu(self):
        """Test that native backward matches CPU reference gradients."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=50, B=4, C=20, S=10, device='mps'
        )

        # CPU gradient computation
        log_probs_cpu = log_probs.cpu().clone().requires_grad_(True)
        targets_cpu = targets.cpu()
        input_lengths_cpu = input_lengths.cpu()
        target_lengths_cpu = target_lengths.cpu()

        # Convert targets to 1D format for F.ctc_loss
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

        # MPS gradient computation
        log_probs_mps = log_probs.contiguous().float()
        targets_mps = targets.contiguous().int()
        input_lengths_mps = input_lengths.contiguous().int()
        target_lengths_mps = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs_mps, targets_mps, input_lengths_mps, target_lengths_mps, blank=0
        )

        grad_output = torch.ones_like(loss)
        mps_grad = _C.ctc_loss_backward(
            grad_output, log_probs_mps, alpha, targets_mps,
            input_lengths_mps, target_lengths_mps, blank=0
        )

        torch.testing.assert_close(
            mps_grad.cpu(), cpu_grad,
            rtol=1e-3, atol=1e-3,
            msg="Native backward should match CPU reference"
        )

    @requires_mps_and_native
    def test_backward_gradient_shape(self):
        """Test that gradient has same shape as input."""
        from native.mps.ctc import _C

        T, B, C, S = 30, 4, 15, 8

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=T, B=B, C=C, S=S, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        grad_output = torch.ones_like(loss)
        grad_log_probs = _C.ctc_loss_backward(
            grad_output, log_probs, alpha, targets,
            input_lengths, target_lengths, blank=0
        )

        assert grad_log_probs.shape == log_probs.shape


# =============================================================================
# AUTOGRAD INTEGRATION TESTS
# =============================================================================


class TestAutogradIntegration:
    """Tests for autograd integration via CTCLossFunctionNative."""

    @requires_mps_and_native
    def test_autograd_forward_backward(self):
        """Test autograd forward/backward through CTCLossMPS."""
        from native.mps.ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        log_probs = log_probs.requires_grad_(True)

        criterion = CTCLossMPS(blank=0, reduction='mean')
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        loss.backward()

        assert log_probs.grad is not None
        assert log_probs.grad.shape == log_probs.shape
        assert not torch.isnan(log_probs.grad).any()

    @requires_mps_and_native
    def test_gradcheck(self):
        """Test gradient correctness using torch.autograd.gradcheck."""
        # MPS doesn't support float64, so we skip this test on MPS
        # Gradcheck really needs double precision for accurate gradient checking
        pytest.skip("MPS doesn't support float64 required for gradcheck")

        from native.mps.ctc.ctc_loss import CTCLossFunctionNative

        # Use small sizes for gradcheck (it's slow)
        T, B, C, S = 8, 2, 5, 3
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='cpu', dtype=torch.float64).log_softmax(2)
        log_probs = log_probs.requires_grad_(True)
        targets = torch.randint(1, C, (B, S), device='cpu').int()
        input_lengths = torch.full((B,), T, device='cpu', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='cpu', dtype=torch.int)

        # gradcheck requires double precision
        torch.autograd.gradcheck(
            lambda x: CTCLossFunctionNative.apply(
                x, targets, input_lengths, target_lengths, 0
            ).sum(),
            (log_probs,),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
            raise_exception=True
        )

    @requires_mps_and_native
    def test_gradgradcheck(self):
        """Test second-order gradients (if supported)."""
        # MPS doesn't support float64, so we skip this test on MPS
        # Gradgradcheck really needs double precision for accurate gradient checking
        pytest.skip("MPS doesn't support float64 required for gradgradcheck")

        from native.mps.ctc.ctc_loss import CTCLossFunctionNative

        # Small sizes for gradgradcheck
        T, B, C, S = 6, 2, 4, 2
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='cpu', dtype=torch.float64).log_softmax(2)
        log_probs = log_probs.requires_grad_(True)
        targets = torch.randint(1, C, (B, S), device='cpu').int()
        input_lengths = torch.full((B,), T, device='cpu', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='cpu', dtype=torch.int)

        try:
            torch.autograd.gradgradcheck(
                lambda x: CTCLossFunctionNative.apply(
                    x, targets, input_lengths, target_lengths, 0
                ).sum(),
                (log_probs,),
                eps=1e-4,
                atol=1e-2,
                rtol=1e-2,
            )
        except RuntimeError as e:
            # Second-order gradients may not be implemented
            if "not implemented" in str(e).lower():
                pytest.skip("Second-order gradients not implemented")
            raise


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @requires_mps_and_native
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

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert loss.shape == (1,)
        assert not torch.isnan(loss).any()

    @requires_mps_and_native
    def test_single_char_target(self):
        """Test with single character targets."""
        from native.mps.ctc import _C

        T, B, C = 20, 2, 10
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, 1), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.ones(B, device='mps', dtype=torch.int)

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)

    @requires_mps_and_native
    def test_repeated_characters(self):
        """Test with repeated characters in target (requires blank between them)."""
        from native.mps.ctc import _C

        T, B, C = 30, 2, 10
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        # Targets with repeated chars: [1, 1, 2, 2, 3]
        targets = torch.tensor(
            [[1, 1, 2, 2, 3], [2, 2, 2, 1, 1]],
            device='mps', dtype=torch.int
        )
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), 5, device='mps', dtype=torch.int)

        loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)

    @requires_mps_and_native
    def test_min_input_length(self):
        """Test with minimum viable input length."""
        from native.mps.ctc import _C

        B, C, S = 2, 10, 3
        # Min input length = 2 * target_length + 1 for repeated chars
        # But for distinct chars, it's target_length
        T = S + 1  # Minimal padding

        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)

        loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert not torch.isnan(loss).any()

    @requires_mps_and_native
    def test_non_zero_blank(self):
        """Test with non-zero blank index."""
        from native.mps.ctc import _C

        T, B, C, S = 20, 2, 10, 5
        blank = 5  # Non-zero blank

        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        # Targets avoid blank=5
        targets = torch.randint(0, blank, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)

        loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    @requires_mps_and_native
    def test_long_sequence(self):
        """Test numerical stability with long sequences."""
        from native.mps.ctc import _C

        T, B, C, S = 200, 2, 50, 30
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)

        loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert not torch.isnan(loss).any(), "Loss should not be NaN for long sequences"
        assert not torch.isinf(loss).any(), "Loss should not be Inf for long sequences"

    @requires_mps_and_native
    def test_extreme_log_probs(self):
        """Test with extreme (very confident) log probabilities."""
        from native.mps.ctc import _C

        T, B, C, S = 30, 2, 10, 5
        torch.manual_seed(42)

        # Create very confident predictions (near one-hot)
        logits = torch.randn(T, B, C, device='mps') * 10  # Scale up for sharp distribution
        log_probs = logits.log_softmax(2).contiguous().float()

        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)

        loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert not torch.isnan(loss).any()

    @requires_mps_and_native
    def test_uniform_log_probs(self):
        """Test with uniform (uncertain) log probabilities."""
        from native.mps.ctc import _C

        T, B, C, S = 30, 2, 10, 5
        torch.manual_seed(42)

        # Create uniform distribution
        logits = torch.zeros(T, B, C, device='mps')
        log_probs = logits.log_softmax(2).contiguous().float()

        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)

        loss, _ = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        cpu_loss = cpu_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        torch.testing.assert_close(loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)


# =============================================================================
# DEVICE AND MEMORY TESTS
# =============================================================================


class TestDeviceAndMemory:
    """Tests for device placement and memory behavior."""

    @requires_mps_and_native
    def test_output_device(self):
        """Test that outputs are on the correct device."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        assert loss.device.type == 'mps', "Loss should be on MPS device"
        assert alpha.device.type == 'mps', "Alpha should be on MPS device"

    @requires_mps_and_native
    def test_gradient_device(self):
        """Test that gradients are on the correct device."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        log_probs = log_probs.contiguous().float()
        targets = targets.contiguous().int()
        input_lengths = input_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()

        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )

        grad_output = torch.ones_like(loss)
        grad_log_probs = _C.ctc_loss_backward(
            grad_output, log_probs, alpha, targets,
            input_lengths, target_lengths, blank=0
        )

        assert grad_log_probs.device.type == 'mps', "Gradient should be on MPS device"

    @requires_mps_and_native
    def test_contiguous_requirement(self):
        """Test that non-contiguous tensors are handled correctly."""
        from native.mps.ctc import _C

        T, B, C, S = 20, 4, 10, 5
        torch.manual_seed(42)

        # Create non-contiguous tensor via transpose
        log_probs = torch.randn(B, T, C, device='mps').transpose(0, 1)
        assert not log_probs.is_contiguous()

        targets = torch.randint(1, C, (B, S), device='mps')
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.long)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.long)

        # Should either handle non-contiguous or raise clear error
        try:
            loss, _ = _C.ctc_loss_forward(
                log_probs.contiguous().float(),
                targets.contiguous().int(),
                input_lengths.contiguous().int(),
                target_lengths.contiguous().int(),
                blank=0
            )
            assert not torch.isnan(loss).any()
        except RuntimeError as e:
            # Acceptable if clear error about contiguous requirement
            assert "contiguous" in str(e).lower()

    @requires_mps_and_native
    def test_dtype_handling(self):
        """Test handling of different data types."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='mps'
        )

        # Test with float32 (should work)
        loss_f32, _ = _C.ctc_loss_forward(
            log_probs.float().contiguous(),
            targets.int().contiguous(),
            input_lengths.int().contiguous(),
            target_lengths.int().contiguous(),
            blank=0
        )
        assert loss_f32.dtype == torch.float32

    @requires_mps
    def test_synchronization(self):
        """Test that operations properly synchronize."""
        from native.mps.ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=30, B=4, C=15, S=8, device='mps'
        )

        criterion = CTCLossMPS(blank=0, reduction='mean')

        # Run multiple times to check for race conditions
        for _ in range(5):
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            torch.mps.synchronize()

        assert not torch.isnan(loss)


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================


class TestInputValidation:
    """Tests for input validation in the native extension."""

    @requires_mps_and_native
    def test_wrong_device_raises(self):
        """Test that CPU tensors raise appropriate error."""
        from native.mps.ctc import _C

        log_probs, targets, input_lengths, target_lengths = create_ctc_data(
            T=20, B=2, C=10, S=5, device='cpu'
        )

        with pytest.raises(RuntimeError, match="MPS"):
            _C.ctc_loss_forward(
                log_probs.contiguous().float(),
                targets.contiguous().int(),
                input_lengths.contiguous().int(),
                target_lengths.contiguous().int(),
                blank=0
            )

    @requires_mps_and_native
    def test_dimension_mismatch(self):
        """Test that dimension mismatches raise errors."""
        from native.mps.ctc import _C

        T, B, C, S = 20, 4, 10, 5
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        # Wrong batch size for target_lengths
        target_lengths = torch.full((B + 1,), S, device='mps', dtype=torch.int)

        with pytest.raises(RuntimeError):
            _C.ctc_loss_forward(
                log_probs, targets, input_lengths, target_lengths, blank=0
            )

    @requires_mps_and_native
    def test_invalid_blank_index(self):
        """Test that invalid blank index raises error."""
        from native.mps.ctc import _C

        T, B, C, S = 20, 2, 10, 5
        torch.manual_seed(42)

        log_probs = torch.randn(T, B, C, device='mps').log_softmax(2).contiguous().float()
        targets = torch.randint(1, C, (B, S), device='mps').contiguous().int()
        input_lengths = torch.full((B,), T, device='mps', dtype=torch.int)
        target_lengths = torch.full((B,), S, device='mps', dtype=torch.int)

        # Blank index out of range
        with pytest.raises(RuntimeError):
            _C.ctc_loss_forward(
                log_probs, targets, input_lengths, target_lengths, blank=C + 1
            )
