"""
PP-OCR Recognition model (PyTorch implementation)

PaddleOCR-style recognition model using MobileNetV3 backbone,
BiLSTM sequence encoder, and CTC head.

Reference: https://github.com/PaddlePaddle/PaddleOCR
"""
import torch
import torch.nn as nn

from ..backbones import MobileNetV3Backbone
from ..necks import SequenceEncoder
from ..heads import CTCHead


class PPOCRModel(nn.Module):
    """
    PP-OCR v3/v4 recognition model in PyTorch

    Architecture:
        - Backbone: MobileNetV3-small (efficient feature extraction)
        - Neck: Stacked BiLSTM (sequence modeling)
        - Head: CTC linear layer (classification)

    Optimized for mobile/edge deployment with width multiplier.
    Uses asymmetric pooling to preserve sequence length for CTC.

    Input: [B, C, H, W] where H=32 or 48
    Output: [B, T, num_classes] logits for CTC decoding
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 37,
        backbone_scale: float = 0.5,
        hidden_size: int = 256
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (charset size + blank)
            backbone_scale: MobileNet width multiplier (default: 0.5)
            hidden_size: BiLSTM hidden dimension (default: 256)
        """
        super().__init__()

        self.backbone = MobileNetV3Backbone(in_channels, scale=backbone_scale)
        self.encoder = SequenceEncoder(self.backbone.out_channels, hidden_size)
        self.head = CTCHead(self.encoder.out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Image tensor [B, C, H, W]

        Returns:
            CTC logits [B, T, num_classes] where T is sequence length
        """
        # Backbone: [B, C, H, W] -> [B, C', 1, W']
        x = self.backbone(x)

        # Encoder: [B, C', 1, W'] -> [B, W', hidden]
        x = self.encoder(x)

        # Head: [B, W', hidden] -> [B, W', num_classes]
        x = self.head(x)

        return x
