"""
ResNet backbone for text detection

Used in DB (Differentiable Binarization) detector.
Returns multi-scale feature maps for FPN.
"""
import torch
import torch.nn as nn
from typing import Tuple

from ..layers import BasicBlock


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone for feature extraction

    Returns multi-scale features for FPN fusion.

    Output scales:
        f2: 1/4 resolution, 64 channels
        f3: 1/8 resolution, 128 channels
        f4: 1/16 resolution, 256 channels
        f5: 1/32 resolution, 512 channels
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks (simplified ResNet-18)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int
    ) -> nn.Sequential:
        """Create a ResNet layer with basic blocks"""
        layers = []

        # First block may downsample
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Multi-scale features: (f2, f3, f4, f5)
        """
        # Stem
        x = self.conv1(x)      # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)    # [B, 64, H/4, W/4]

        # ResNet layers
        f2 = self.layer1(x)    # [B, 64, H/4, W/4]
        f3 = self.layer2(f2)   # [B, 128, H/8, W/8]
        f4 = self.layer3(f3)   # [B, 256, H/16, W/16]
        f5 = self.layer4(f4)   # [B, 512, H/32, W/32]

        return f2, f3, f4, f5
