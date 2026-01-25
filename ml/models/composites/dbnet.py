"""
DBNet (Differentiable Binarization Network) model

Text detection model using differentiable binarization for fast
and accurate scene text detection.

Based on: "Real-time Scene Text Detection with Differentiable Binarization"
(AAAI 2020) https://arxiv.org/abs/1911.08947
"""
import torch
import torch.nn as nn
from typing import Tuple

from ..backbones import ResNetBackbone
from ..necks import FPN
from ..heads import DBHead


class DBNet(nn.Module):
    """
    DB (Differentiable Binarization) Network

    Architecture:
        - Backbone: ResNet-18 for multi-scale feature extraction
        - Neck: FPN for feature fusion
        - Head: Dual-branch for probability and threshold maps

    The model outputs three maps:
    1. Probability map: likelihood of text at each pixel
    2. Threshold map: adaptive threshold per pixel
    3. Binary map: differentiable binarization result

    Input: [B, 3, H, W] RGB image
    Output: (prob_map, thresh_map, binary_map), each [B, 1, H, W]
    """

    def __init__(self, pretrained: bool = False, k: int = 50):
        """
        Args:
            pretrained: Whether to use pretrained backbone weights
            k: Amplification factor for differentiable binarization
        """
        super().__init__()

        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)
        self.head = DBHead(in_channels=256 * 4, k=k)  # 4 scales concatenated

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Tuple of:
                prob_map: Probability map [B, 1, H, W]
                thresh_map: Threshold map [B, 1, H, W]
                binary_map: Binary map [B, 1, H, W]
        """
        # Extract multi-scale features
        features = self.backbone(x)

        # Fuse features with FPN
        fused = self.fpn(features)

        # Detection heads
        prob_map, thresh_map, binary_map = self.head(fused)

        return prob_map, thresh_map, binary_map
