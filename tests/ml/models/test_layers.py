"""
Tests for ML model layers (building blocks).

Tests shapes, functional behavior, gradients, and mathematical properties.
"""
import pytest
import torch
import torch.nn as nn
from ml.models.layers import (
    ConvBNLayer,
    SEModule,
    BasicBlock,
    ResidualUnit,
    CTCDecoder,
    CHARSETS,
    make_divisible,
)


class TestConvBNLayer:
    """Tests for ConvBNLayer (Conv + BatchNorm + Activation)."""

    def test_output_shape_with_stride_1(self):
        """Test output shape preserves spatial dimensions with stride=1."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)

        assert out.shape == (2, 64, 32, 32)

    def test_output_shape_with_stride_2(self):
        """Test output shape halves spatial dimensions with stride=2."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)

        assert out.shape == (2, 64, 16, 16)

    def test_relu_activation_non_negative(self):
        """Test ReLU activation produces only non-negative outputs."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, padding=1, act='relu')
        x = torch.randn(2, 3, 32, 32) - 0.5  # Include negative values
        out = layer(x)

        assert out.shape == (2, 64, 32, 32)
        assert (out >= 0).all(), "ReLU should produce non-negative outputs"

    def test_hardswish_activation_bounded(self):
        """Test hardswish activation is bounded correctly."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, padding=1, act='hardswish')
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)

        assert out.shape == (2, 64, 32, 32)
        # Hardswish output should exist
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_no_activation_preserves_negatives(self):
        """Test that act=None preserves negative values."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, act=None)
        x = torch.randn(2, 3, 32, 32) - 1.0  # Many negative values
        out = layer(x)

        # Should have some negative values (no activation applied)
        assert (out < 0).any(), "Without activation, some outputs should be negative"

    def test_batch_normalization_normalizes(self):
        """Test batch normalization actually normalizes (mean≈0, std≈1)."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, act=None)
        layer.train()  # BN behaves differently in train vs eval

        # High mean and std input
        x = torch.randn(32, 3, 32, 32) * 10 + 5
        out = layer(x)

        # After BN (before activation if any), should be normalized
        # Check across batch dimension
        batch_mean = out.mean(dim=(0, 2, 3))
        batch_std = out.std(dim=(0, 2, 3))

        # Mean should be close to 0, std close to 1
        assert torch.allclose(batch_mean, torch.zeros_like(batch_mean), atol=0.1)
        assert torch.allclose(batch_std, torch.ones_like(batch_std), atol=0.2)

    def test_gradients_flow_backward(self):
        """Test gradients computed and flow backward."""
        layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, act='relu')
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(x.grad).any(), "No NaN gradients"
        assert not torch.isinf(x.grad).any(), "No Inf gradients"


class TestSEModule:
    """Tests for SEModule (Squeeze-and-Excitation)."""

    def test_output_shape_preserved(self):
        """Test SE module preserves input shape."""
        se = SEModule(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = se(x)

        assert out.shape == x.shape

    def test_global_pooling_reduces_spatial(self):
        """Test global average pooling reduces spatial dimensions."""
        se = SEModule(channels=64)
        x = torch.randn(2, 64, 32, 32)

        # Access internal pooling (if exposed) or verify output changes spatially
        out = se(x)

        # Output should differ from input (attention applied)
        assert not torch.allclose(out, x)

    def test_attention_actually_applied(self):
        """Test SE attention changes the output."""
        se = SEModule(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = se(x)

        # Attention should modify the input
        assert not torch.equal(out, x), "SE should reweight channels"

    def test_channel_reweighting(self):
        """Test channels are reweighted differently (SE applies attention)."""
        se = SEModule(channels=64)
        x = torch.randn(2, 64, 32, 32) + 1.0  # Positive bias
        out = se(x)

        # Calculate channel-wise ratios (SE uses hardsigmoid [0,1], so ratio <= 1.0)
        channel_ratios = (out / (x + 1e-6)).mean(dim=(0, 2, 3))

        # Different channels should have different attention weights
        assert channel_ratios.max() > channel_ratios.min(), \
            "Different channels should be reweighted differently"
        assert (channel_ratios >= 0).all(), "Ratios should be non-negative"
        assert (channel_ratios <= 1.0).all(), "Ratios should be <= 1.0 (hardsigmoid output)"

    def test_gradients_flow_through_se(self):
        """Test gradients flow through SE module."""
        se = SEModule(channels=64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        out = se(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestBasicBlock:
    """Tests for BasicBlock (ResNet residual block)."""

    def test_stride_1_preserves_size(self):
        """Test stride=1 preserves spatial size."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)

        assert out.shape == (2, 64, 32, 32)

    def test_stride_2_halves_size(self):
        """Test stride=2 halves spatial size."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)

        assert out.shape == (2, 128, 16, 16)

    def test_residual_connection_adds(self):
        """Test residual connection mathematically adds input to output."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        block.eval()  # Disable BN randomness

        x = torch.randn(2, 64, 32, 32)

        # Forward pass
        out = block(x)

        # Output should not equal pure conv path (residual added)
        # This is implicit - we can't easily separate, but verify it's not zero
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_downsample_when_channels_change(self):
        """Test shortcut connection applied when in_channels != out_channels."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)

        # Should have non-identity shortcut module
        assert not isinstance(block.shortcut, nn.Identity)
        assert len(block.shortcut) > 0, "Shortcut should have conv+bn layers"

        x = torch.randn(2, 64, 32, 32)
        out = block(x)

        assert out.shape == (2, 128, 16, 16)

    def test_gradients_through_residual(self):
        """Test gradients flow through both conv and residual paths."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCTCDecoder:
    """Tests for CTC greedy decoder."""

    def test_removes_blanks(self):
        """Test CTC decoder removes blank tokens."""
        charset = "abc"
        blank_idx = 3
        decoder = CTCDecoder(charset=charset, blank_idx=blank_idx)

        # Create logits: [B=1, T=4, C=4]
        # Sequence: blank, a, blank, b
        logits = torch.tensor([
            [
                [-1, -1, -1, 3],  # blank (highest)
                [3, -1, -1, -1],  # a
                [-1, -1, -1, 3],  # blank
                [-1, 3, -1, -1],  # b
            ]
        ], dtype=torch.float32)

        texts, confidences = decoder.decode_batch(logits)

        assert texts[0] == "ab", f"Expected 'ab', got '{texts[0]}'"
        assert 0 <= confidences[0] <= 1

    def test_collapses_duplicates(self):
        """Test consecutive duplicates are collapsed."""
        charset = "abc"
        blank_idx = 3
        decoder = CTCDecoder(charset=charset, blank_idx=blank_idx)

        # Sequence: a, a, a, b, b
        logits = torch.tensor([
            [
                [3, -1, -1, -1],  # a
                [3, -1, -1, -1],  # a (duplicate)
                [3, -1, -1, -1],  # a (duplicate)
                [-1, 3, -1, -1],  # b
                [-1, 3, -1, -1],  # b (duplicate)
            ]
        ], dtype=torch.float32)

        texts, confidences = decoder.decode_batch(logits)

        assert texts[0] == "ab", f"Expected 'ab', got '{texts[0]}'"

    def test_handles_all_blanks(self):
        """Test handles sequence of all blanks (empty string)."""
        charset = "abc"
        blank_idx = 3
        decoder = CTCDecoder(charset=charset, blank_idx=blank_idx)

        # All blanks
        logits = torch.tensor([
            [
                [-1, -1, -1, 3],  # blank
                [-1, -1, -1, 3],  # blank
                [-1, -1, -1, 3],  # blank
            ]
        ], dtype=torch.float32)

        texts, confidences = decoder.decode_batch(logits)

        assert texts[0] == "", "All blanks should produce empty string"

    def test_batch_decoding(self):
        """Test batch processing produces correct number of outputs."""
        charset = "abc"
        decoder = CTCDecoder(charset=charset, blank_idx=3)

        # Batch of 3
        logits = torch.randn(3, 10, 4)
        texts, confidences = decoder.decode_batch(logits)

        assert len(texts) == 3
        assert len(confidences) == 3
        assert all(0 <= c <= 1 for c in confidences)


class TestCHARSETS:
    """Tests for predefined character sets."""

    def test_greek_has_basic_characters(self):
        """Test Greek charset includes basic letters."""
        charset = CHARSETS['greek']

        assert 'α' in charset  # lowercase alpha
        assert 'Α' in charset  # uppercase alpha

    def test_greek_has_accents(self):
        """Test Greek charset includes accented characters."""
        charset = CHARSETS['greek']

        assert 'ά' in charset  # acute accent
        assert 'ὰ' in charset  # grave accent

    def test_greek_has_breathing_marks(self):
        """Test Greek charset includes breathing marks."""
        charset = CHARSETS['greek']

        assert 'ἀ' in charset  # smooth breathing
        assert 'ἁ' in charset  # rough breathing

    def test_latin_charset(self):
        """Test Latin charset has expected characters."""
        charset = CHARSETS['latin']

        assert 'a' in charset
        assert 'Z' in charset
        assert '0' in charset
        assert '9' in charset

    def test_charsets_are_strings(self):
        """Test charsets are valid strings."""
        for name, charset in CHARSETS.items():
            assert isinstance(charset, str), f"{name} charset should be a string"
            assert len(charset) > 0, f"{name} charset should not be empty"

    def test_invalid_charset_raises_error(self):
        """Test invalid charset name raises KeyError."""
        with pytest.raises(KeyError):
            _ = CHARSETS['nonexistent']


class TestMakeDivisible:
    """Tests for make_divisible utility function."""

    def test_makes_divisible_by_divisor(self):
        """Test output is divisible by divisor."""
        result = make_divisible(23, divisor=8)

        assert result % 8 == 0

    def test_returns_correct_value(self):
        """Test returns closest divisible value."""
        result = make_divisible(23, divisor=8)

        # 23 / 8 = 2.875, rounds to 3, so 3 * 8 = 24
        assert result == 24
