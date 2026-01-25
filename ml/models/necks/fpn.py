"""
Feature Pyramid Network (FPN) for multi-scale feature fusion

Used in DB (Differentiable Binarization) detector.
Fuses multi-scale backbone features using top-down pathway.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion

    Top-down pathway with lateral connections that fuses features
    from different scales into a single feature map.

    Architecture:
        Input: Multi-scale features [f2, f3, f4, f5]
        - Lateral connections reduce channels
        - Top-down upsampling + element-wise addition
        - Smoothing convolutions after fusion
        - Final concatenation of all scales

    Args:
        in_channels: List of input channel counts [64, 128, 256, 512]
        out_channels: Output channels per scale (default: 256)
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()

        # Lateral connections (reduce channels)
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)

        # Smooth layers (after upsampling)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.out_channels = out_channels * 4  # 4 scales concatenated

    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Forward pass with top-down feature fusion

        Args:
            features: (f2, f3, f4, f5) multi-scale features from backbone
                f2: [B, C1, H/4, W/4]
                f3: [B, C2, H/8, W/8]
                f4: [B, C3, H/16, W/16]
                f5: [B, C4, H/32, W/32]

        Returns:
            Fused feature map [B, out_channels*4, H/4, W/4]
        """
        f2, f3, f4, f5 = features

        # Top-down pathway with lateral connections
        p5 = self.lateral4(f5)
        p4 = self.lateral3(f4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral2(f3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral1(f2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        # Smooth after fusion
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)

        # Upsample all to same size as p2 (highest resolution)
        p3 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=p2.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate all scales
        fused = torch.cat([p2, p3, p4, p5], dim=1)

        return fused
