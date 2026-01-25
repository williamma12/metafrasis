"""
CRAFT (Character Region Awareness For Text detection) model

U-Net style architecture with VGG16-BN backbone for text detection.
Outputs region and affinity score maps for character-level detection.

Based on: "Character Region Awareness for Text Detection" (CVPR 2019)
Reference: https://github.com/clovaai/CRAFT-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..backbones import VGG16BN


class CRAFT(nn.Module):
    """
    CRAFT: Character Region Awareness For Text detection

    Architecture:
        - Backbone: VGG16-BN for multi-scale feature extraction
        - Decoder: U-Net style decoder with skip connections
        - Output: 2-channel maps (region score + affinity score)

    The region score indicates the probability of a pixel being part
    of a character center. The affinity score indicates the probability
    of a pixel being part of the space between adjacent characters.

    Input: [B, 3, H, W] RGB image
    Output: (region_score, affinity_score), each [B, 1, H, W]
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Backbone
        self.basenet = VGG16BN(pretrained=pretrained)

        # U-Net decoder with skip connections
        # Upsampling from 1/32 to 1/16
        self.upconv1 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Merge with pool4 (1/16) and upsample to 1/8
        self.upconv2 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Merge with pool3 (1/8) and upsample to 1/4
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Merge with pool2 (1/4) and upsample to 1/2
        self.upconv4 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Merge with pool1 (1/2) and upsample to 1/1
        self.upconv5 = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final convolution to produce 2 output channels
        # Channel 0: Region score (character center probability)
        # Channel 1: Affinity score (character link probability)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1)  # 2 channels: region + affinity
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Tuple of:
                region_score: Character region map [B, 1, H, W]
                affinity_score: Character link map [B, 1, H, W]
        """
        # Get feature maps from backbone
        features = self.basenet(x)

        # Decoder with skip connections
        # 1/32 -> 1/16
        h = self.upconv1(features['pool5'])
        h = F.interpolate(
            h, size=features['pool4'].shape[2:],
            mode='bilinear', align_corners=False
        )
        h = torch.cat([h, features['pool4']], dim=1)

        # 1/16 -> 1/8
        h = self.upconv2(h)
        h = F.interpolate(
            h, size=features['pool3'].shape[2:],
            mode='bilinear', align_corners=False
        )
        h = torch.cat([h, features['pool3']], dim=1)

        # 1/8 -> 1/4
        h = self.upconv3(h)
        h = F.interpolate(
            h, size=features['pool2'].shape[2:],
            mode='bilinear', align_corners=False
        )
        h = torch.cat([h, features['pool2']], dim=1)

        # 1/4 -> 1/2
        h = self.upconv4(h)
        h = F.interpolate(
            h, size=features['pool1'].shape[2:],
            mode='bilinear', align_corners=False
        )
        h = torch.cat([h, features['pool1']], dim=1)

        # 1/2 -> 1/1
        h = self.upconv5(h)
        h = F.interpolate(
            h, size=(x.size(2), x.size(3)),
            mode='bilinear', align_corners=False
        )

        # Final classification
        h = self.conv_cls(h)

        # Split into region and affinity scores
        region_score = h[:, 0:1, :, :]
        affinity_score = h[:, 1:2, :, :]

        # Apply sigmoid to get probabilities
        region_score = torch.sigmoid(region_score)
        affinity_score = torch.sigmoid(affinity_score)

        return region_score, affinity_score
