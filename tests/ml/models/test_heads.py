"""
Tests for task-specific head networks.

Tests CTC head for recognition and DB head for detection.
"""
import pytest
import torch
from ml.models.heads import CTCHead, DBHead


class TestCTCHead:
    """Tests for CTC prediction head."""

    def test_linear_projection_correct_classes(self):
        """Test linear projection produces correct number of classes."""
        head = CTCHead(in_channels=256, num_classes=100)
        x = torch.randn(2, 50, 256)  # [B, T, C]

        out = head(x)

        assert out.shape == (2, 50, 100), \
            f"Expected shape (2, 50, 100), got {out.shape}"

    def test_output_shape_correct(self):
        """Test output shape is [B, T, num_classes]."""
        head = CTCHead(in_channels=256, num_classes=100)
        x = torch.randn(2, 50, 256)

        out = head(x)

        assert out.shape[0] == 2  # Batch
        assert out.shape[1] == 50  # Time/sequence length
        assert out.shape[2] == 100  # Classes

    def test_no_activation_applied(self):
        """Test no activation applied (logits output for CTC loss)."""
        head = CTCHead(in_channels=256, num_classes=100)
        x = torch.randn(2, 50, 256)

        out = head(x)

        # Logits can be negative (no sigmoid/softmax)
        assert (out < 0).any(), "Should output logits (some negative values)"
        assert (out > 0).any(), "Should output logits (some positive values)"

    def test_works_with_different_sequence_lengths(self):
        """Test works with different sequence lengths."""
        head = CTCHead(in_channels=256, num_classes=100)

        x1 = torch.randn(2, 30, 256)
        out1 = head(x1)
        assert out1.shape == (2, 30, 100)

        x2 = torch.randn(2, 100, 256)
        out2 = head(x2)
        assert out2.shape == (2, 100, 100)

    def test_handles_batch_size_one(self):
        """Test handles batch size = 1."""
        head = CTCHead(in_channels=256, num_classes=100)
        x = torch.randn(1, 50, 256)

        out = head(x)

        assert out.shape == (1, 50, 100)

    def test_gradients_flow_through_linear(self):
        """Test gradients flow through linear layer."""
        head = CTCHead(in_channels=256, num_classes=100)
        x = torch.randn(2, 50, 256, requires_grad=True)

        out = head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_out_channels_attribute_correct(self):
        """Test out_channels attribute is num_classes."""
        head = CTCHead(in_channels=256, num_classes=100)

        assert head.out_channels == 100


class TestDBHead:
    """Tests for Differentiable Binarization head."""

    def test_returns_tuple_of_three_tensors(self):
        """Test returns tuple of 3 tensors (prob, thresh, binary)."""
        head = DBHead(in_channels=1024)
        x = torch.randn(2, 1024, 160, 160)

        result = head(x)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_maps_same_spatial_size(self):
        """Test all maps have same spatial size."""
        head = DBHead(in_channels=1024)
        x = torch.randn(2, 1024, 160, 160)

        prob_map, thresh_map, binary_map = head(x)

        # All should have same shape
        assert prob_map.shape == thresh_map.shape
        assert prob_map.shape == binary_map.shape

    def test_prob_map_in_range_zero_one(self):
        """Test prob map in [0, 1] after sigmoid."""
        head = DBHead(in_channels=1024)
        x = torch.randn(2, 1024, 160, 160)

        prob_map, thresh_map, binary_map = head(x)

        assert (prob_map >= 0).all(), "Prob map should be >= 0"
        assert (prob_map <= 1).all(), "Prob map should be <= 1"

    def test_thresh_map_in_range_zero_one(self):
        """Test thresh map in [0, 1] after sigmoid."""
        head = DBHead(in_channels=1024)
        x = torch.randn(2, 1024, 160, 160)

        prob_map, thresh_map, binary_map = head(x)

        assert (thresh_map >= 0).all(), "Thresh map should be >= 0"
        assert (thresh_map <= 1).all(), "Thresh map should be <= 1"

    def test_binary_map_computed_using_diff_binarization(self):
        """Test binary map computed using differentiable binarization formula."""
        head = DBHead(in_channels=1024, k=50)
        x = torch.randn(2, 1024, 160, 160)

        prob_map, thresh_map, binary_map = head(x)

        # Binary map should be different from prob map (formula applied)
        assert not torch.allclose(binary_map, prob_map), \
            "Binary map should differ from prob map (formula applied)"

    def test_binary_map_formula_correct(self):
        """Test binary map = sigmoid(k * (prob - thresh))."""
        head = DBHead(in_channels=1024, k=50)
        x = torch.randn(2, 1024, 160, 160)

        prob_map, thresh_map, binary_map = head(x)

        # Manually compute expected binary map
        expected = torch.sigmoid(head.k * (prob_map - thresh_map))

        # Should match (within numerical precision)
        assert torch.allclose(binary_map, expected, atol=1e-6), \
            "Binary map should match sigmoid(k * (prob - thresh))"

    def test_amplification_factor_k_applied(self):
        """Test amplification factor k=50 is applied."""
        head = DBHead(in_channels=1024, k=50)

        assert head.k == 50

    def test_upsampling_increases_resolution(self):
        """Test upsampling increases resolution correctly."""
        head = DBHead(in_channels=1024)
        x = torch.randn(2, 1024, 160, 160)

        prob_map, thresh_map, binary_map = head(x)

        # Upsampled by 4x (two stride-2 transposed convs)
        assert prob_map.shape == (2, 1, 640, 640), \
            f"Expected (2, 1, 640, 640), got {prob_map.shape}"

    def test_works_with_different_input_sizes(self):
        """Test works with different input sizes."""
        head = DBHead(in_channels=1024)

        x1 = torch.randn(1, 1024, 80, 80)
        prob1, thresh1, binary1 = head(x1)
        assert prob1.shape == (1, 1, 320, 320)

        x2 = torch.randn(1, 1024, 120, 120)
        prob2, thresh2, binary2 = head(x2)
        assert prob2.shape == (1, 1, 480, 480)

    def test_gradients_flow_through_both_branches(self):
        """Test gradients flow through probability and threshold branches."""
        head = DBHead(in_channels=1024)
        x = torch.randn(2, 1024, 160, 160, requires_grad=True)

        prob_map, thresh_map, binary_map = head(x)

        # Compute loss from all three outputs
        loss = prob_map.sum() + thresh_map.sum() + binary_map.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_threshold_branch_learns_separately(self):
        """Test threshold branch has separate parameters from prob branch."""
        head = DBHead(in_channels=1024)

        # Verify both branches exist
        assert head.prob_conv is not None
        assert head.thresh_conv is not None

        # Verify they are different modules
        assert head.prob_conv is not head.thresh_conv
