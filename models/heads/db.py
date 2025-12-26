"""
DB Head for text detection

Differentiable Binarization detection head.
Outputs probability map, threshold map, and binary map for text detection.

Based on: "Real-time Scene Text Detection with Differentiable Binarization"
(AAAI 2020) https://arxiv.org/abs/1911.08947
"""
import torch
import torch.nn as nn


class DBHead(nn.Module):
    """
    DB (Differentiable Binarization) detection head

    Produces three outputs for text detection:
    1. Probability map: Likelihood of each pixel being text
    2. Threshold map: Adaptive threshold per pixel
    3. Binary map: Differentiable binarization result

    Architecture:
        Input: [B, C, H, W] fused features from FPN
        -> Two parallel branches:
           - Probability branch: Conv -> BN -> ReLU -> Upsample -> Conv -> Sigmoid
           - Threshold branch: Conv -> BN -> ReLU -> Upsample -> Conv -> Sigmoid
        -> Differentiable binarization combines both

    Args:
        in_channels: Input feature channels (default: 1024 for 4-scale FPN)
        k: Amplification factor for differentiable binarization (default: 50)
    """

    def __init__(self, in_channels: int = 1024, k: int = 50):
        super().__init__()

        self.k = k  # Amplification factor for differentiable binarization

        # Probability map head
        self.prob_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # Threshold map head
        self.thresh_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Fused features [B, C, H, W]

        Returns:
            Tuple of:
                prob_map: Probability map [B, 1, H*4, W*4]
                thresh_map: Threshold map [B, 1, H*4, W*4]
                binary_map: Differentiable binary map [B, 1, H*4, W*4]
        """
        prob_map = self.prob_conv(x)
        thresh_map = self.thresh_conv(x)

        # Differentiable binarization
        binary_map = self._diff_binarize(prob_map, thresh_map)

        return prob_map, thresh_map, binary_map

    def _diff_binarize(
        self,
        prob_map: torch.Tensor,
        thresh_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable binarization using approximate step function

        DB = 1 / (1 + exp(-k * (P - T)))

        Where:
            P = probability map
            T = threshold map
            k = amplification factor

        Args:
            prob_map: Probability map [B, 1, H, W]
            thresh_map: Threshold map [B, 1, H, W]

        Returns:
            Binary map [B, 1, H, W]
        """
        return torch.reciprocal(1 + torch.exp(-self.k * (prob_map - thresh_map)))
