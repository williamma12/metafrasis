"""
MobileNetV3 backbone for text recognition

Used in PP-OCR recognizer.
Modified for OCR with asymmetric pooling to preserve sequence length.
"""
import torch
import torch.nn as nn

from ..layers import ConvBNLayer, ResidualUnit, make_divisible


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-small backbone for text recognition

    Modified for OCR:
    - Asymmetric pooling to preserve width (sequence length)
    - Output height = 1 for sequence modeling
    - Configurable width multiplier (scale)

    Output: [B, out_channels, 1, W'] where W' depends on input width
    """

    def __init__(self, in_channels: int = 3, scale: float = 0.5):
        """
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            scale: Width multiplier (default: 0.5 for efficiency)
        """
        super().__init__()
        self.scale = scale

        # Initial conv: stride 2 reduces height
        self.conv1 = ConvBNLayer(
            in_channels,
            make_divisible(16 * scale),
            3,
            stride=2,
            padding=1,
            act="hardswish"
        )

        # MobileNetV3 blocks
        # Config: [in_ch, mid_ch, out_ch, kernel, stride, use_se, act]
        block_configs = [
            [16, 16, 16, 3, 2, True, "relu"],       # H/4
            [16, 72, 24, 3, 1, False, "relu"],
            [24, 88, 24, 3, 1, False, "relu"],
            [24, 96, 40, 5, 2, True, "hardswish"],  # H/8
            [40, 240, 40, 5, 1, True, "hardswish"],
            [40, 240, 40, 5, 1, True, "hardswish"],
            [40, 120, 48, 5, 1, True, "hardswish"],
            [48, 144, 48, 5, 1, True, "hardswish"],
            [48, 288, 96, 5, 2, True, "hardswish"],  # H/16
            [96, 576, 96, 5, 1, True, "hardswish"],
            [96, 576, 96, 5, 1, True, "hardswish"],
        ]

        self.blocks = nn.ModuleList()
        in_ch = make_divisible(16 * scale)

        for cfg in block_configs:
            base_in, mid_ch, out_ch, kernel, stride, use_se, act = cfg
            mid_ch = make_divisible(mid_ch * scale)
            out_ch = make_divisible(out_ch * scale)

            block = ResidualUnit(
                in_channels=in_ch,
                mid_channels=mid_ch,
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride,
                use_se=use_se,
                act=act
            )
            self.blocks.append(block)
            in_ch = out_ch

        # Final conv
        self.conv2 = ConvBNLayer(
            in_ch,
            make_divisible(576 * scale),
            1,
            act="hardswish"
        )

        # Adaptive pool to height=1
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        self.out_channels = make_divisible(576 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature map [B, out_channels, 1, W']
        """
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
