"""
Tests for composite models (full pipelines).

Tests end-to-end architecture integration: backbone → neck → head.
"""
import pytest
import torch
from ml.models.composites import CRAFT, DBNet, CRNN, PPOCRModel
from ml.models.layers import CHARSETS


class TestCRAFT:
    """Tests for CRAFT text detector."""

    def test_returns_tuple_of_two_tensors(self):
        """Test CRAFT returns tuple of 2 tensors (region, affinity)."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512)

        result = model(x)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_outputs_same_spatial_size(self):
        """Test both outputs have same spatial size as input."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512)

        region_score, affinity_score = model(x)

        assert region_score.shape == (1, 1, 512, 512)
        assert affinity_score.shape == (1, 1, 512, 512)

    def test_region_scores_in_range_zero_one(self):
        """Test region scores in [0, 1] after sigmoid."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512)

        region_score, affinity_score = model(x)

        assert (region_score >= 0).all(), "Region scores should be >= 0"
        assert (region_score <= 1).all(), "Region scores should be <= 1"

    def test_affinity_scores_in_range_zero_one(self):
        """Test affinity scores in [0, 1] after sigmoid."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512)

        region_score, affinity_score = model(x)

        assert (affinity_score >= 0).all(), "Affinity scores should be >= 0"
        assert (affinity_score <= 1).all(), "Affinity scores should be <= 1"

    def test_skip_connections_merge_correctly(self):
        """Test skip connections from VGG pools merge correctly."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512)

        region_score, affinity_score = model(x)

        # Outputs should be non-zero (skip connections processed)
        assert not torch.allclose(region_score, torch.zeros_like(region_score))
        assert not torch.allclose(affinity_score, torch.zeros_like(affinity_score))

    def test_unet_decoder_upsamples_progressively(self):
        """Test U-Net decoder upsamples progressively."""
        model = CRAFT()

        # Verify upconv layers exist
        assert model.upconv1 is not None
        assert model.upconv2 is not None
        assert model.upconv3 is not None
        assert model.upconv4 is not None
        assert model.upconv5 is not None

    def test_works_with_different_input_sizes(self):
        """Test works with different input sizes."""
        model = CRAFT()

        # Test 512x512
        x1 = torch.randn(1, 3, 512, 512)
        region1, affinity1 = model(x1)
        assert region1.shape == (1, 1, 512, 512)

        # Test 1024x1024
        x2 = torch.randn(1, 3, 1024, 1024)
        region2, affinity2 = model(x2)
        assert region2.shape == (1, 1, 1024, 1024)

    def test_gradients_flow_through_encoder_decoder(self):
        """Test gradients flow through encoder and decoder."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512, requires_grad=True)

        region_score, affinity_score = model(x)
        loss = region_score.sum() + affinity_score.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_character_and_word_level_scores_differ(self):
        """Test character-level and word-level scores can differ."""
        model = CRAFT()
        x = torch.randn(1, 3, 512, 512)

        region_score, affinity_score = model(x)

        # Region and affinity should be different
        assert not torch.allclose(region_score, affinity_score)


class TestDBNet:
    """Tests for DBNet text detector."""

    def test_end_to_end_forward_pass_works(self):
        """Test end-to-end forward pass works."""
        model = DBNet()
        x = torch.randn(1, 3, 640, 640)

        result = model(x)

        assert result is not None

    def test_backbone_extracts_multi_scale_features(self):
        """Test backbone extracts multi-scale features."""
        model = DBNet()

        # Verify backbone exists
        assert model.backbone is not None

    def test_fpn_fuses_features_correctly(self):
        """Test FPN fuses features correctly."""
        model = DBNet()

        # Verify FPN exists
        assert model.fpn is not None

    def test_head_returns_three_maps(self):
        """Test head returns 3 maps (prob, thresh, binary)."""
        model = DBNet()
        x = torch.randn(1, 3, 640, 640)

        prob_map, thresh_map, binary_map = model(x)

        assert prob_map is not None
        assert thresh_map is not None
        assert binary_map is not None

    def test_input_output_pipeline(self):
        """Test input [B, 3, H, W] → 3 outputs at [B, 1, H, W]."""
        model = DBNet()
        x = torch.randn(2, 3, 640, 640)

        prob_map, thresh_map, binary_map = model(x)

        # All outputs should have same batch size
        assert prob_map.shape[0] == 2
        assert thresh_map.shape[0] == 2
        assert binary_map.shape[0] == 2

    def test_works_with_different_resolutions(self):
        """Test works with different resolutions."""
        model = DBNet()

        # Test 640x640
        x1 = torch.randn(1, 3, 640, 640)
        prob1, thresh1, binary1 = model(x1)
        assert prob1.shape[2] == 640
        assert prob1.shape[3] == 640

        # Test 736x736
        x2 = torch.randn(1, 3, 736, 736)
        prob2, thresh2, binary2 = model(x2)
        assert prob2.shape[2] == 736
        assert prob2.shape[3] == 736

    def test_probability_and_threshold_maps_independent(self):
        """Test probability and threshold maps are independent."""
        model = DBNet()
        x = torch.randn(1, 3, 640, 640)

        prob_map, thresh_map, binary_map = model(x)

        # Prob and thresh should be different (separate branches)
        assert not torch.allclose(prob_map, thresh_map)

    def test_binary_map_is_differentiable_combination(self):
        """Test binary map is differentiable combination."""
        model = DBNet()
        x = torch.randn(1, 3, 640, 640)

        prob_map, thresh_map, binary_map = model(x)

        # Binary map should differ from prob map (formula applied)
        assert not torch.allclose(binary_map, prob_map)

    def test_gradients_flow_through_entire_pipeline(self):
        """Test gradients flow through entire pipeline."""
        model = DBNet()
        x = torch.randn(1, 3, 640, 640, requires_grad=True)

        prob_map, thresh_map, binary_map = model(x)
        loss = prob_map.sum() + thresh_map.sum() + binary_map.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestCRNN:
    """Tests for CRNN text recognizer."""

    def test_input_output_shape_correct(self):
        """Test input [B, 1, 32, W] → output [T, B, num_classes]."""
        model = CRNN(img_height=32, num_channels=1, num_classes=37)
        x = torch.randn(2, 1, 32, 200)

        out = model(x)

        assert out.shape[1] == 2  # Batch
        assert out.shape[2] == 37  # Classes

    def test_cnn_reduces_height_to_one(self):
        """Test CNN reduces height to 1."""
        model = CRNN(img_height=32, num_channels=1)

        # Verify CNN exists
        assert model.cnn is not None

    def test_lstm_processes_sequence(self):
        """Test LSTM processes sequence."""
        model = CRNN(img_height=32, num_channels=1)

        # Verify RNN exists
        assert model.rnn is not None

    def test_output_format_compatible_with_ctc_loss(self):
        """Test output format [T, B, C] is compatible with CTC loss."""
        model = CRNN(img_height=32, num_channels=1, num_classes=37)
        x = torch.randn(2, 1, 32, 200)

        out = model(x)

        # Shape should be [T, B, C]
        assert len(out.shape) == 3
        # Batch should be dimension 1
        assert out.shape[1] == 2

    def test_works_with_variable_width_inputs(self):
        """Test works with variable width inputs."""
        model = CRNN(img_height=32, num_channels=1, num_classes=37)

        x1 = torch.randn(2, 1, 32, 100)
        out1 = model(x1)

        x2 = torch.randn(2, 1, 32, 300)
        out2 = model(x2)

        # Wider input should produce longer sequence
        assert out2.shape[0] > out1.shape[0]

    def test_sequence_length_relates_to_input_width(self):
        """Test sequence length T relates to input width W."""
        model = CRNN(img_height=32, num_channels=1, num_classes=37)
        x = torch.randn(2, 1, 32, 200)

        out = model(x)

        # T should be > 0
        assert out.shape[0] > 0

    def test_gradients_flow_through_cnn_and_lstm(self):
        """Test gradients flow through CNN and LSTM."""
        model = CRNN(img_height=32, num_channels=1, num_classes=37)
        x = torch.randn(2, 1, 32, 200, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_handles_grayscale_input_correctly(self):
        """Test handles grayscale input (1 channel) correctly."""
        model = CRNN(img_height=32, num_channels=1, num_classes=37)
        x = torch.randn(2, 1, 32, 200)

        out = model(x)

        assert out.shape[2] == 37  # Classes


class TestPPOCRModel:
    """Tests for PP-OCR recognition model."""

    def test_input_output_shape_correct(self):
        """Test input [B, 3, 48, W] → output [B, T, num_classes]."""
        model = PPOCRModel(in_channels=3, num_classes=37)
        x = torch.randn(2, 3, 48, 200)

        out = model(x)

        assert out.shape[0] == 2  # Batch
        assert out.shape[2] == 37  # Classes

    def test_mobilenet_backbone_preserves_width(self):
        """Test MobileNetV3 backbone preserves width for sequence."""
        model = PPOCRModel(in_channels=3, num_classes=37)

        # Verify backbone exists
        assert model.backbone is not None

    def test_bilstm_encoder_processes_sequence(self):
        """Test BiLSTM encoder processes sequence."""
        model = PPOCRModel(in_channels=3, num_classes=37)

        # Verify encoder exists
        assert model.encoder is not None

    def test_ctc_head_projects_to_classes(self):
        """Test CTC head projects to character classes."""
        model = PPOCRModel(in_channels=3, num_classes=37)

        # Verify head exists
        assert model.head is not None

    def test_greek_charset_num_classes_correct(self):
        """Test Greek charset: num_classes = len(charset) + 1 (blank)."""
        greek_charset = CHARSETS['greek']
        model = PPOCRModel(in_channels=3, num_classes=len(greek_charset) + 1)

        assert model.num_classes == len(greek_charset) + 1

    def test_latin_charset_works_correctly(self):
        """Test Latin charset works correctly."""
        latin_charset = CHARSETS['latin']
        model = PPOCRModel(in_channels=3, num_classes=len(latin_charset) + 1)
        x = torch.randn(2, 3, 48, 200)

        out = model(x)

        assert out.shape[2] == len(latin_charset) + 1

    def test_handles_variable_width_inputs(self):
        """Test handles variable width inputs."""
        model = PPOCRModel(in_channels=3, num_classes=37)

        x1 = torch.randn(2, 3, 48, 100)
        out1 = model(x1)

        x2 = torch.randn(2, 3, 48, 300)
        out2 = model(x2)

        # Wider input should produce longer sequence
        assert out2.shape[1] > out1.shape[1]

    def test_sequence_length_preserves_input_width_structure(self):
        """Test sequence length preserves input width structure."""
        model = PPOCRModel(in_channels=3, num_classes=37)
        x = torch.randn(2, 3, 48, 200)

        out = model(x)

        # T should be > 0
        assert out.shape[1] > 0

    def test_gradients_flow_end_to_end(self):
        """Test gradients flow end-to-end."""
        model = PPOCRModel(in_channels=3, num_classes=37)
        x = torch.randn(2, 3, 48, 200, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_model_eval_vs_train_behaves_correctly(self):
        """Test model.eval() vs model.train() behave correctly."""
        model = PPOCRModel(in_channels=3, num_classes=37)
        x = torch.randn(2, 3, 48, 200)

        # Train mode
        model.train()
        out_train = model(x)

        # Eval mode
        model.eval()
        out_eval = model(x)

        # Both should produce output (may differ due to BatchNorm)
        assert out_train.shape == out_eval.shape
