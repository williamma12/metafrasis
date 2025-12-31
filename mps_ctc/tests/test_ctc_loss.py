"""
Tests for CTC Loss MPS implementation.

Validates forward and backward passes against PyTorch's CPU reference.
"""

import pytest
import torch
import torch.nn as nn


class TestCTCLossMPS:
    """Test suite for CTCLossMPS."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        T, B, C = 50, 4, 20  # time, batch, classes
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2)
        targets = torch.randint(1, C, (B, 10))  # Avoid blank (0)
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.randint(5, 10, (B,), dtype=torch.long)
        return log_probs, targets, input_lengths, target_lengths

    @pytest.fixture
    def small_data(self):
        """Create small test data for debugging."""
        T, B, C = 10, 2, 5
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2)
        targets = torch.randint(1, C, (B, 3))
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), 3, dtype=torch.long)
        return log_probs, targets, input_lengths, target_lengths

    def test_forward_matches_cpu_small(self, small_data):
        """Test forward pass on small data matches CPU reference."""
        from mps_ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = small_data

        # CPU reference
        cpu_loss = nn.CTCLoss(blank=0, reduction='none')(
            log_probs, targets, input_lengths, target_lengths
        )

        # MPS implementation (runs pure Python version on CPU for now)
        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        # Force pure implementation by testing on CPU first
        mps_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(mps_loss, cpu_loss, rtol=1e-4, atol=1e-4)

    def test_forward_matches_cpu(self, sample_data):
        """Test forward pass matches CPU reference."""
        from mps_ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = sample_data

        # CPU reference
        cpu_loss = nn.CTCLoss(blank=0, reduction='none')(
            log_probs, targets, input_lengths, target_lengths
        )

        # Our implementation (pure Python, on CPU)
        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        our_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(our_loss, cpu_loss, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_forward_on_mps_device(self, sample_data):
        """Test forward pass works on MPS device."""
        from mps_ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = sample_data

        # Move to MPS
        log_probs_mps = log_probs.to('mps')
        targets_mps = targets.to('mps')
        input_lengths_mps = input_lengths.to('mps')
        target_lengths_mps = target_lengths.to('mps')

        # CPU reference
        cpu_loss = nn.CTCLoss(blank=0, reduction='none')(
            log_probs, targets, input_lengths, target_lengths
        )

        # MPS implementation
        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        mps_loss = mps_criterion(
            log_probs_mps, targets_mps, input_lengths_mps, target_lengths_mps
        )

        torch.testing.assert_close(mps_loss.cpu(), cpu_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_backward_on_mps_device(self, sample_data):
        """Test backward pass works on MPS device and produces correct gradients."""
        from mps_ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = sample_data

        # CPU gradient
        log_probs_cpu = log_probs.clone().requires_grad_(True)
        cpu_loss = nn.CTCLoss(blank=0)(
            log_probs_cpu, targets, input_lengths, target_lengths
        )
        cpu_loss.backward()

        # MPS gradient
        log_probs_mps = log_probs.to('mps').requires_grad_(True)
        targets_mps = targets.to('mps')
        input_lengths_mps = input_lengths.to('mps')
        target_lengths_mps = target_lengths.to('mps')

        mps_criterion = CTCLossMPS(blank=0)
        mps_loss = mps_criterion(
            log_probs_mps, targets_mps, input_lengths_mps, target_lengths_mps
        )
        mps_loss.backward()

        torch.testing.assert_close(
            log_probs_mps.grad.cpu(), log_probs_cpu.grad,
            rtol=1e-3, atol=1e-3
        )

    def test_reduction_mean(self, sample_data):
        """Test mean reduction."""
        from mps_ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = sample_data

        cpu_loss = nn.CTCLoss(blank=0, reduction='mean')(
            log_probs, targets, input_lengths, target_lengths
        )

        mps_criterion = CTCLossMPS(blank=0, reduction='mean')
        our_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(our_loss, cpu_loss, rtol=1e-4, atol=1e-4)

    def test_reduction_sum(self, sample_data):
        """Test sum reduction."""
        from mps_ctc import CTCLossMPS

        log_probs, targets, input_lengths, target_lengths = sample_data

        cpu_loss = nn.CTCLoss(blank=0, reduction='sum')(
            log_probs, targets, input_lengths, target_lengths
        )

        mps_criterion = CTCLossMPS(blank=0, reduction='sum')
        our_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(our_loss, cpu_loss, rtol=1e-4, atol=1e-4)

    def test_variable_length_inputs(self):
        """Test with variable-length inputs and targets."""
        from mps_ctc import CTCLossMPS

        T, B, C = 100, 4, 30
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2)
        targets = torch.randint(1, C, (B, 20))
        input_lengths = torch.tensor([100, 80, 60, 90], dtype=torch.long)
        target_lengths = torch.tensor([20, 15, 10, 18], dtype=torch.long)

        cpu_loss = nn.CTCLoss(blank=0, reduction='none')(
            log_probs, targets, input_lengths, target_lengths
        )

        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        our_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(our_loss, cpu_loss, rtol=1e-4, atol=1e-4)

    def test_numerical_stability_long_sequence(self):
        """Test numerical stability with long sequences."""
        from mps_ctc import CTCLossMPS

        T, B, C = 200, 2, 50
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2)
        targets = torch.randint(1, C, (B, 30))
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), 30, dtype=torch.long)

        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        assert not torch.isnan(loss).any(), "Loss should not contain NaN"
        assert not torch.isinf(loss).any(), "Loss should not contain Inf"

    def test_single_char_target(self):
        """Test with single character targets."""
        from mps_ctc import CTCLossMPS

        T, B, C = 20, 2, 10
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2)
        targets = torch.randint(1, C, (B, 1))
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), 1, dtype=torch.long)

        cpu_loss = nn.CTCLoss(blank=0, reduction='none')(
            log_probs, targets, input_lengths, target_lengths
        )

        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        our_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(our_loss, cpu_loss, rtol=1e-4, atol=1e-4)

    def test_repeated_characters(self):
        """Test with repeated characters in target."""
        from mps_ctc import CTCLossMPS

        T, B, C = 30, 2, 10
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2)
        # Create targets with repeated chars: [1, 1, 2, 2, 3]
        targets = torch.tensor([[1, 1, 2, 2, 3], [2, 2, 2, 1, 1]], dtype=torch.long)
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), 5, dtype=torch.long)

        cpu_loss = nn.CTCLoss(blank=0, reduction='none')(
            log_probs, targets, input_lengths, target_lengths
        )

        mps_criterion = CTCLossMPS(blank=0, reduction='none')
        our_loss = mps_criterion(
            log_probs, targets, input_lengths, target_lengths
        )

        torch.testing.assert_close(our_loss, cpu_loss, rtol=1e-4, atol=1e-4)


class TestCTCLossMPSPerformance:
    """Performance tests for CTCLossMPS."""

    @pytest.mark.skip(reason="Pure Python implementation is too slow; enable after Metal implementation")
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    @pytest.mark.slow
    def test_large_batch_performance(self):
        """Test performance with large batch."""
        from mps_ctc import CTCLossMPS
        import time

        T, B, C = 100, 32, 100
        torch.manual_seed(42)
        log_probs = torch.randn(T, B, C).log_softmax(2).to('mps')
        targets = torch.randint(1, C, (B, 20)).to('mps')
        input_lengths = torch.full((B,), T, dtype=torch.long).to('mps')
        target_lengths = torch.full((B,), 20, dtype=torch.long).to('mps')

        mps_criterion = CTCLossMPS(blank=0)

        # Warmup
        for _ in range(3):
            log_probs_warmup = log_probs.clone().requires_grad_(True)
            loss = mps_criterion(log_probs_warmup, targets, input_lengths, target_lengths)
            loss.backward()

        # Timed runs
        torch.mps.synchronize()
        start = time.time()
        n_runs = 10
        for _ in range(n_runs):
            log_probs_grad = log_probs.clone().requires_grad_(True)
            loss = mps_criterion(log_probs_grad, targets, input_lengths, target_lengths)
            loss.backward()
        torch.mps.synchronize()
        elapsed = time.time() - start

        avg_time = elapsed / n_runs * 1000  # ms
        print(f"\nAverage time per forward+backward: {avg_time:.2f} ms")

        # Should complete in reasonable time
        assert avg_time < 1000, f"Too slow: {avg_time:.2f} ms"
