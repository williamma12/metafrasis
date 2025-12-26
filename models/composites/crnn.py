"""
CRNN (Convolutional Recurrent Neural Network) model

Text recognition model combining CNN feature extraction with
bidirectional LSTM sequence modeling and CTC decoding.

Based on: "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition" (TPAMI 2016)
"""
import torch
import torch.nn as nn

from ..backbones import CRNNCNN
from ..necks import BidirectionalLSTM


class CRNN(nn.Module):
    """
    CRNN architecture for text recognition

    Architecture:
        - CNN: VGG-style feature extraction
        - RNN: Stacked bidirectional LSTM for sequence modeling
        - CTC: Output logits for CTC decoding

    The CNN backbone reduces the input height to 1 while preserving
    width for sequence modeling. The BiLSTM processes the sequence
    and projects to the output vocabulary.

    Input: [B, C, H, W] where H=32 (grayscale or RGB)
    Output: [T, B, num_classes] logits for CTC decoding
    """

    def __init__(
        self,
        img_height: int = 32,
        num_channels: int = 1,
        num_classes: int = 37,
        hidden_size: int = 256
    ):
        """
        Args:
            img_height: Height of input images (must be divisible by 16)
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes (charset size + blank token)
            hidden_size: Number of hidden units in LSTM
        """
        super().__init__()

        self.num_classes = num_classes

        # CNN backbone
        self.cnn = CRNNCNN(num_channels=num_channels, img_height=img_height)

        # RNN - two stacked BiLSTMs (using batch_first=False for compatibility)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(
                self.cnn.out_channels,
                hidden_size,
                hidden_size,
                batch_first=False
            ),
            BidirectionalLSTM(
                hidden_size,
                hidden_size,
                num_classes,
                batch_first=False
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Image tensor [B, C, H, W]

        Returns:
            CTC output [T, B, num_classes] where T is sequence length
        """
        # CNN feature extraction
        conv = self.cnn(x)  # [B, 512, 1, W/4]

        # Prepare for RNN: [B, C, 1, W] -> [W, B, C]
        b, c, h, w = conv.size()
        assert h == 1, f"CNN output height must be 1, got {h}"
        conv = conv.squeeze(2)  # [B, 512, W]
        conv = conv.permute(2, 0, 1)  # [W, B, 512]

        # RNN sequence modeling
        output = self.rnn(conv)  # [T, B, num_classes]

        return output
