"""
VGG16 with Batch Normalization backbone

Used as the encoder in CRAFT text detector.
Returns multi-scale feature maps for skip connections.
"""
import torch
import torch.nn as nn
from typing import Dict


class VGG16BN(nn.Module):
    """
    VGG16 with Batch Normalization backbone for feature extraction

    Returns intermediate feature maps at 5 scales for U-Net style decoders.

    Output scales:
        pool1: 1/2 resolution, 64 channels
        pool2: 1/4 resolution, 128 channels
        pool3: 1/8 resolution, 256 channels
        pool4: 1/16 resolution, 512 channels
        pool5: 1/32 resolution, 512 channels
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        # Conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        # Conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)

        # Conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        # Conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning intermediate feature maps for skip connections

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary of feature maps at different scales
        """
        # Conv1
        h = self.relu(self.bn1_1(self.conv1_1(x)))
        h = self.relu(self.bn1_2(self.conv1_2(h)))
        h = self.pool(h)
        pool1 = h

        # Conv2
        h = self.relu(self.bn2_1(self.conv2_1(h)))
        h = self.relu(self.bn2_2(self.conv2_2(h)))
        h = self.pool(h)
        pool2 = h

        # Conv3
        h = self.relu(self.bn3_1(self.conv3_1(h)))
        h = self.relu(self.bn3_2(self.conv3_2(h)))
        h = self.relu(self.bn3_3(self.conv3_3(h)))
        h = self.pool(h)
        pool3 = h

        # Conv4
        h = self.relu(self.bn4_1(self.conv4_1(h)))
        h = self.relu(self.bn4_2(self.conv4_2(h)))
        h = self.relu(self.bn4_3(self.conv4_3(h)))
        h = self.pool(h)
        pool4 = h

        # Conv5
        h = self.relu(self.bn5_1(self.conv5_1(h)))
        h = self.relu(self.bn5_2(self.conv5_2(h)))
        h = self.relu(self.bn5_3(self.conv5_3(h)))
        h = self.pool(h)
        pool5 = h

        return {
            'pool1': pool1,  # 1/2
            'pool2': pool2,  # 1/4
            'pool3': pool3,  # 1/8
            'pool4': pool4,  # 1/16
            'pool5': pool5   # 1/32
        }
