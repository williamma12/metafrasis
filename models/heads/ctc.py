"""
CTC Head for text recognition

Connectionist Temporal Classification prediction head.
Used as the output layer in CTC-based recognition models.
"""
import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """
    CTC prediction head

    Linear projection from sequence features to character classes.
    Outputs logits for CTC loss computation and greedy decoding.

    Architecture:
        Input: [B, T, in_channels] sequence features
        -> Linear projection
        Output: [B, T, num_classes] logits

    Args:
        in_channels: Input feature dimension (from sequence encoder)
        num_classes: Number of output classes (charset size + 1 for blank)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.out_channels = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [B, T, C] sequence features from encoder

        Returns:
            [B, T, num_classes] logits for CTC decoding
        """
        return self.fc(x)
