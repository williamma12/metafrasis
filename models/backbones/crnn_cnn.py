"""
CRNN CNN backbone for text recognition

Based on the CRNN paper (TPAMI 2016).
VGG-style CNN that reduces height to 1 while preserving width.
"""
import torch
import torch.nn as nn


class CRNNCNN(nn.Module):
    """
    CNN backbone from CRNN paper

    VGG-style architecture that:
    - Reduces height from 32 to 1
    - Preserves width for sequence modeling
    - Outputs 512-channel feature map

    Input: [B, num_channels, 32, W]
    Output: [B, 512, 1, W']
    """

    def __init__(self, num_channels: int = 1, img_height: int = 32):
        """
        Args:
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
            img_height: Height of input images (must be divisible by 16)
        """
        super().__init__()

        assert img_height % 16 == 0, 'img_height must be divisible by 16'

        # CNN backbone
        # Input: [B, num_channels, 32, W]
        self.cnn = nn.Sequential(
            # Conv 1
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [B, 64, 16, W/2]

            # Conv 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [B, 128, 8, W/4]

            # Conv 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Conv 4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # [B, 256, 4, W/4+1]

            # Conv 5
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Conv 6
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # [B, 512, 2, W/4+1]

            # Conv 7
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)  # [B, 512, 1, W/4]
        )

        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature map [B, 512, 1, W']
        """
        return self.cnn(x)
