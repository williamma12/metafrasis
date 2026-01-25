"""
Tests for neck networks (feature aggregation).

Tests multi-scale fusion, sequence encoding, and LSTM processing.
"""
import pytest
import torch
from ml.models.necks import FPN, BidirectionalLSTM, SequenceEncoder


class TestFPN:
    """Tests for Feature Pyramid Network."""

    def test_takes_four_features_outputs_concatenated(self):
        """Test FPN takes 4 feature maps and outputs concatenated pyramid."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        f2 = torch.randn(2, 64, 160, 160)
        f3 = torch.randn(2, 128, 80, 80)
        f4 = torch.randn(2, 256, 40, 40)
        f5 = torch.randn(2, 512, 20, 20)

        out = fpn((f2, f3, f4, f5))

        assert isinstance(out, torch.Tensor)

    def test_output_channels_correct(self):
        """Test output channels = out_channels * 4."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        f2 = torch.randn(2, 64, 160, 160)
        f3 = torch.randn(2, 128, 80, 80)
        f4 = torch.randn(2, 256, 40, 40)
        f5 = torch.randn(2, 512, 20, 20)

        out = fpn((f2, f3, f4, f5))

        assert out.shape[1] == 256 * 4, f"Expected {256 * 4} channels, got {out.shape[1]}"

    def test_all_pyramid_levels_upsampled_to_same_size(self):
        """Test all pyramid levels upsampled to same spatial resolution."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        f2 = torch.randn(2, 64, 160, 160)
        f3 = torch.randn(2, 128, 80, 80)
        f4 = torch.randn(2, 256, 40, 40)
        f5 = torch.randn(2, 512, 20, 20)

        out = fpn((f2, f3, f4, f5))

        # Output should have same spatial size as f2 (highest resolution)
        assert out.shape[2:] == (160, 160), \
            f"Expected spatial size (160, 160), got {out.shape[2:]}"

    def test_top_down_pathway_upsamples_correctly(self):
        """Test top-down pathway upsamples correctly (bilinear)."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        f2 = torch.randn(2, 64, 160, 160)
        f3 = torch.randn(2, 128, 80, 80)
        f4 = torch.randn(2, 256, 40, 40)
        f5 = torch.randn(2, 512, 20, 20)

        out = fpn((f2, f3, f4, f5))

        # Output should be non-zero (upsampling happened)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_lateral_connections_reduce_channels(self):
        """Test lateral connections reduce channels correctly."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        # Verify lateral conv layers exist
        assert fpn.lateral1 is not None
        assert fpn.lateral2 is not None
        assert fpn.lateral3 is not None
        assert fpn.lateral4 is not None

    def test_smoothing_convolutions_applied(self):
        """Test smoothing convolutions applied after fusion."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        # Verify smooth conv layers exist
        assert fpn.smooth1 is not None
        assert fpn.smooth2 is not None
        assert fpn.smooth3 is not None
        assert fpn.smooth4 is not None

    def test_works_with_different_input_resolutions(self):
        """Test FPN works with different input resolutions."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        # Test with 320x320 input (f2 would be 80x80)
        f2 = torch.randn(1, 64, 80, 80)
        f3 = torch.randn(1, 128, 40, 40)
        f4 = torch.randn(1, 256, 20, 20)
        f5 = torch.randn(1, 512, 10, 10)

        out = fpn((f2, f3, f4, f5))

        assert out.shape[2:] == (80, 80)
        assert out.shape[1] == 256 * 4

    def test_gradients_flow_through_all_pyramid_levels(self):
        """Test gradients flow through all pyramid levels."""
        fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)

        f2 = torch.randn(2, 64, 160, 160, requires_grad=True)
        f3 = torch.randn(2, 128, 80, 80, requires_grad=True)
        f4 = torch.randn(2, 256, 40, 40, requires_grad=True)
        f5 = torch.randn(2, 512, 20, 20, requires_grad=True)

        out = fpn((f2, f3, f4, f5))
        loss = out.sum()
        loss.backward()

        # Gradients should flow to all inputs
        assert f2.grad is not None
        assert f3.grad is not None
        assert f4.grad is not None
        assert f5.grad is not None

        # No NaN or Inf
        assert not torch.isnan(f2.grad).any()
        assert not torch.isnan(f3.grad).any()


class TestBidirectionalLSTM:
    """Tests for Bidirectional LSTM with projection."""

    def test_forward_backward_concatenated(self):
        """Test forward and backward states are concatenated."""
        lstm = BidirectionalLSTM(input_size=256, hidden_size=128, output_size=256)
        x = torch.randn(2, 50, 256)  # [B, T, C]

        out = lstm(x)

        # Output should have shape [B, T, output_size]
        assert out.shape == (2, 50, 256)

    def test_linear_projection_applied(self):
        """Test linear projection reduces dimension correctly."""
        lstm = BidirectionalLSTM(input_size=256, hidden_size=128, output_size=64)
        x = torch.randn(2, 50, 256)

        out = lstm(x)

        # After projection, output size should be 64
        assert out.shape == (2, 50, 64)

    def test_batch_first_true(self):
        """Test batch_first=True handles [B, T, C] input correctly."""
        lstm = BidirectionalLSTM(
            input_size=256,
            hidden_size=128,
            output_size=256,
            batch_first=True
        )
        x = torch.randn(2, 50, 256)

        out = lstm(x)

        assert out.shape == (2, 50, 256)

    def test_batch_first_false(self):
        """Test batch_first=False handles [T, B, C] input correctly."""
        lstm = BidirectionalLSTM(
            input_size=256,
            hidden_size=128,
            output_size=256,
            batch_first=False
        )
        x = torch.randn(50, 2, 256)  # [T, B, C]

        out = lstm(x)

        assert out.shape == (50, 2, 256)

    def test_different_sequence_lengths_work(self):
        """Test different sequence lengths work."""
        lstm = BidirectionalLSTM(input_size=256, hidden_size=128, output_size=256)

        x1 = torch.randn(2, 30, 256)
        out1 = lstm(x1)
        assert out1.shape == (2, 30, 256)

        x2 = torch.randn(2, 100, 256)
        out2 = lstm(x2)
        assert out2.shape == (2, 100, 256)

    def test_gradients_flow_backward_through_time(self):
        """Test gradients flow backward through time."""
        lstm = BidirectionalLSTM(input_size=256, hidden_size=128, output_size=256)
        x = torch.randn(2, 50, 256, requires_grad=True)

        out = lstm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestSequenceEncoder:
    """Tests for stacked BiLSTM sequence encoder."""

    def test_reshapes_input_correctly(self):
        """Test reshapes [B, C, 1, W] → [B, W, C] correctly."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)
        x = torch.randn(2, 512, 1, 100)  # [B, C, 1, W]

        out = encoder(x)

        # Output should be [B, W, hidden_size]
        assert out.shape == (2, 100, 256)

    def test_two_bilstm_layers_stacked(self):
        """Test two BiLSTM layers are stacked."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)

        # Verify both LSTM layers exist
        assert encoder.lstm1 is not None
        assert encoder.lstm2 is not None

    def test_output_shape_correct(self):
        """Test output shape is [B, W, hidden_size]."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)
        x = torch.randn(2, 512, 1, 100)

        out = encoder(x)

        assert out.shape == (2, 100, 256), \
            f"Expected shape (2, 100, 256), got {out.shape}"

    def test_second_lstm_receives_output_from_first(self):
        """Test second LSTM receives output from first."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)
        x = torch.randn(2, 512, 1, 100)

        out = encoder(x)

        # Output should be non-zero (both LSTMs processed)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_handles_variable_sequence_lengths(self):
        """Test handles variable sequence lengths."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)

        x1 = torch.randn(2, 512, 1, 50)
        out1 = encoder(x1)
        assert out1.shape == (2, 50, 256)

        x2 = torch.randn(2, 512, 1, 200)
        out2 = encoder(x2)
        assert out2.shape == (2, 200, 256)

    def test_gradients_flow_through_both_lstm_layers(self):
        """Test gradients flow through both LSTM layers."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)
        x = torch.randn(2, 512, 1, 100, requires_grad=True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_out_channels_attribute_correct(self):
        """Test out_channels attribute matches hidden_size."""
        encoder = SequenceEncoder(in_channels=512, hidden_size=256)

        assert encoder.out_channels == 256
