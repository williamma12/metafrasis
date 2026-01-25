"""
Tests for backbone networks (feature extractors).

Tests output shapes, channel counts, spatial scaling, and gradient flow.
"""
import pytest
import torch
from ml.models.backbones import VGG16BN, ResNetBackbone, MobileNetV3Backbone, CRNNCNN


class TestVGG16BN:
    """Tests for VGG16 with Batch Normalization backbone."""

    def test_returns_dict_with_five_feature_maps(self):
        """Test VGG16BN returns dict with exactly 5 feature maps."""
        model = VGG16BN()
        x = torch.randn(2, 3, 256, 256)
        features = model(x)

        assert isinstance(features, dict)
        assert len(features) == 5

    def test_feature_map_keys(self):
        """Test feature map keys are pool1 through pool5."""
        model = VGG16BN()
        x = torch.randn(2, 3, 256, 256)
        features = model(x)

        expected_keys = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
        assert set(features.keys()) == set(expected_keys)

    def test_spatial_dimensions_decrease_correctly(self):
        """Test spatial dimensions decrease by factor of 2 at each pool."""
        model = VGG16BN()
        x = torch.randn(2, 3, 256, 256)
        features = model(x)

        # pool1: H/2 = 128
        assert features['pool1'].shape == (2, 64, 128, 128)

        # pool2: H/4 = 64
        assert features['pool2'].shape == (2, 128, 64, 64)

        # pool3: H/8 = 32
        assert features['pool3'].shape == (2, 256, 32, 32)

        # pool4: H/16 = 16
        assert features['pool4'].shape == (2, 512, 16, 16)

        # pool5: H/32 = 8
        assert features['pool5'].shape == (2, 512, 8, 8)

    def test_channel_counts_increase_deeper(self):
        """Test channel counts increase deeper in network."""
        model = VGG16BN()
        x = torch.randn(2, 3, 256, 256)
        features = model(x)

        assert features['pool1'].shape[1] == 64
        assert features['pool2'].shape[1] == 128
        assert features['pool3'].shape[1] == 256
        assert features['pool4'].shape[1] == 512
        assert features['pool5'].shape[1] == 512

    def test_all_outputs_non_zero(self):
        """Test all feature maps are non-zero (forward pass executed)."""
        model = VGG16BN()
        x = torch.randn(2, 3, 256, 256)
        features = model(x)

        for key, feat in features.items():
            assert not torch.allclose(feat, torch.zeros_like(feat)), \
                f"{key} should not be all zeros"

    def test_works_with_different_input_sizes(self):
        """Test VGG16BN works with different input sizes."""
        model = VGG16BN()

        # Test 320x320
        x1 = torch.randn(1, 3, 320, 320)
        features1 = model(x1)
        assert features1['pool1'].shape == (1, 64, 160, 160)
        assert features1['pool5'].shape == (1, 512, 10, 10)

        # Test 512x512
        x2 = torch.randn(1, 3, 512, 512)
        features2 = model(x2)
        assert features2['pool1'].shape == (1, 64, 256, 256)
        assert features2['pool5'].shape == (1, 512, 16, 16)

    def test_gradients_flow_backward(self):
        """Test gradients flow back through all pools."""
        model = VGG16BN()
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        features = model(x)

        # Compute loss from all features
        loss = sum(feat.sum() for feat in features.values())
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_batch_size_one_works(self):
        """Test VGG16BN works with batch size = 1."""
        model = VGG16BN()
        x = torch.randn(1, 3, 256, 256)
        features = model(x)

        assert features['pool1'].shape[0] == 1
        assert features['pool5'].shape[0] == 1


class TestResNetBackbone:
    """Tests for ResNet-18 backbone."""

    def test_returns_tuple_of_four_feature_maps(self):
        """Test ResNetBackbone returns tuple of exactly 4 feature maps."""
        model = ResNetBackbone()
        x = torch.randn(2, 3, 640, 640)
        features = model(x)

        assert isinstance(features, tuple)
        assert len(features) == 4

    def test_channel_counts_correct(self):
        """Test channel counts are [64, 128, 256, 512]."""
        model = ResNetBackbone()
        x = torch.randn(2, 3, 640, 640)
        f2, f3, f4, f5 = model(x)

        assert f2.shape[1] == 64
        assert f3.shape[1] == 128
        assert f4.shape[1] == 256
        assert f5.shape[1] == 512

    def test_spatial_scales_correct(self):
        """Test spatial scales are [1/4, 1/8, 1/16, 1/32]."""
        model = ResNetBackbone()
        x = torch.randn(2, 3, 640, 640)
        f2, f3, f4, f5 = model(x)

        # f2: 1/4 resolution
        assert f2.shape[2:] == (160, 160)

        # f3: 1/8 resolution
        assert f3.shape[2:] == (80, 80)

        # f4: 1/16 resolution
        assert f4.shape[2:] == (40, 40)

        # f5: 1/32 resolution
        assert f5.shape[2:] == (20, 20)

    def test_residual_connections_work(self):
        """Test residual connections work (output contains residual signal)."""
        model = ResNetBackbone()
        x = torch.randn(2, 3, 640, 640)
        f2, f3, f4, f5 = model(x)

        # All features should be non-zero (residuals add signal)
        assert not torch.allclose(f2, torch.zeros_like(f2))
        assert not torch.allclose(f3, torch.zeros_like(f3))
        assert not torch.allclose(f4, torch.zeros_like(f4))
        assert not torch.allclose(f5, torch.zeros_like(f5))

    def test_gradients_propagate_through_residuals(self):
        """Test gradients flow through residual connections."""
        model = ResNetBackbone()
        x = torch.randn(2, 3, 640, 640, requires_grad=True)
        f2, f3, f4, f5 = model(x)

        loss = f2.sum() + f3.sum() + f4.sum() + f5.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_works_with_variable_input_sizes(self):
        """Test ResNetBackbone works with different input sizes."""
        model = ResNetBackbone()

        # Test 512x512
        x1 = torch.randn(1, 3, 512, 512)
        f2, f3, f4, f5 = model(x1)
        assert f2.shape == (1, 64, 128, 128)
        assert f5.shape == (1, 512, 16, 16)

        # Test 1024x1024
        x2 = torch.randn(1, 3, 1024, 1024)
        f2, f3, f4, f5 = model(x2)
        assert f2.shape == (1, 64, 256, 256)
        assert f5.shape == (1, 512, 32, 32)

    def test_batch_size_one_works(self):
        """Test ResNetBackbone works with batch size = 1."""
        model = ResNetBackbone()
        x = torch.randn(1, 3, 640, 640)
        f2, f3, f4, f5 = model(x)

        assert f2.shape[0] == 1
        assert f3.shape[0] == 1
        assert f4.shape[0] == 1
        assert f5.shape[0] == 1


class TestMobileNetV3Backbone:
    """Tests for MobileNetV3-small backbone."""

    def test_output_height_is_one(self):
        """Test output height = 1 (asymmetric pooling verified)."""
        model = MobileNetV3Backbone(in_channels=3, scale=0.5)
        x = torch.randn(2, 3, 48, 200)
        out = model(x)

        assert out.shape[2] == 1, "Height should be reduced to 1"

    def test_width_preserved_for_sequence(self):
        """Test width is preserved (or minimally changed) for sequence modeling."""
        model = MobileNetV3Backbone(in_channels=3, scale=0.5)
        x = torch.randn(2, 3, 48, 200)
        out = model(x)

        # Width should be related to input width
        # After stride-2 convolutions, width is reduced but not to 1
        assert out.shape[3] > 1, "Width should be preserved for sequence"

    def test_scale_parameter_changes_channels(self):
        """Test scale parameter actually changes channel count."""
        model_small = MobileNetV3Backbone(in_channels=3, scale=0.5)
        model_large = MobileNetV3Backbone(in_channels=3, scale=1.0)

        x = torch.randn(2, 3, 48, 200)

        out_small = model_small(x)
        out_large = model_large(x)

        # Larger scale should have more channels
        assert out_large.shape[1] > out_small.shape[1], \
            f"Scale=1.0 ({out_large.shape[1]} ch) should have more channels than scale=0.5 ({out_small.shape[1]} ch)"

    def test_inverted_residuals_work(self):
        """Test inverted residuals work correctly."""
        model = MobileNetV3Backbone(in_channels=3, scale=0.5)
        x = torch.randn(2, 3, 48, 200)
        out = model(x)

        # Output should be non-zero (inverted residuals processed)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_works_with_different_widths(self):
        """Test MobileNetV3 handles variable-length sequences."""
        model = MobileNetV3Backbone(in_channels=3, scale=0.5)

        # Test different widths
        x1 = torch.randn(1, 3, 48, 100)
        out1 = model(x1)
        assert out1.shape[2] == 1

        x2 = torch.randn(1, 3, 48, 300)
        out2 = model(x2)
        assert out2.shape[2] == 1

        # Wider input should produce wider output
        assert out2.shape[3] > out1.shape[3]

    def test_gradients_flow_through_bottlenecks(self):
        """Test gradients flow through inverted bottleneck blocks."""
        model = MobileNetV3Backbone(in_channels=3, scale=0.5)
        x = torch.randn(2, 3, 48, 200, requires_grad=True)
        out = model(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestCRNNCNN:
    """Tests for CRNN CNN backbone."""

    def test_output_shape_correct(self):
        """Test input [B, 1, 32, W] → output [B, 512, 1, W']."""
        model = CRNNCNN(num_channels=1, img_height=32)
        x = torch.randn(2, 1, 32, 200)
        out = model(x)

        assert out.shape[0] == 2  # Batch
        assert out.shape[1] == 512  # Channels
        assert out.shape[2] == 1  # Height reduced to 1

    def test_height_progressively_reduced(self):
        """Test height is progressively reduced: 32→16→8→4→2→1."""
        # This is tested implicitly by the final output shape
        model = CRNNCNN(num_channels=1, img_height=32)
        x = torch.randn(2, 1, 32, 200)
        out = model(x)

        assert out.shape[2] == 1, "Height should be reduced to 1"

    def test_width_preserved_or_minimally_changed(self):
        """Test width is preserved or minimally changed for sequence."""
        model = CRNNCNN(num_channels=1, img_height=32)
        x = torch.randn(2, 1, 32, 200)
        out = model(x)

        # Width should be related to input width (some reduction is expected)
        # Based on the architecture, width is reduced by pooling
        assert out.shape[3] > 10, "Width should be preserved for sequence modeling"

    def test_handles_variable_widths(self):
        """Test CRNNCNN handles variable input widths."""
        model = CRNNCNN(num_channels=1, img_height=32)

        x1 = torch.randn(1, 1, 32, 100)
        out1 = model(x1)

        x2 = torch.randn(1, 1, 32, 300)
        out2 = model(x2)

        # Wider input should produce wider output
        assert out2.shape[3] > out1.shape[3]

    def test_grayscale_input_processed(self):
        """Test grayscale input (1 channel) is processed correctly."""
        model = CRNNCNN(num_channels=1, img_height=32)
        x = torch.randn(2, 1, 32, 200)
        out = model(x)

        assert out.shape == (2, 512, 1, out.shape[3])

    def test_gradients_flow_through_all_layers(self):
        """Test gradients flow through all conv layers."""
        model = CRNNCNN(num_channels=1, img_height=32)
        x = torch.randn(2, 1, 32, 200, requires_grad=True)
        out = model(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_out_channels_attribute_correct(self):
        """Test out_channels attribute is 512."""
        model = CRNNCNN(num_channels=1, img_height=32)

        assert model.out_channels == 512

    def test_works_with_rgb_input(self):
        """Test CRNNCNN can work with RGB input (3 channels)."""
        model = CRNNCNN(num_channels=3, img_height=32)
        x = torch.randn(2, 3, 32, 200)
        out = model(x)

        assert out.shape == (2, 512, 1, out.shape[3])
