"""
Sequence encoding layers for text recognition

Contains LSTM-based sequence modeling components:
- BidirectionalLSTM: Single BiLSTM layer with projection
- SequenceEncoder: Stacked BiLSTM for recognition models
"""
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM with linear projection

    Used for sequence modeling in text recognition (CRNN, PP-OCR, etc.)
    Processes sequences in both directions and projects to output dimension.

    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden dimension
        output_size: Output projection dimension
        batch_first: If True, input shape is [B, T, C], else [T, B, C]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_first: bool = True
    ):
        super().__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=batch_first
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor
               - If batch_first: [B, T, input_size]
               - Else: [T, B, input_size]

        Returns:
            Output tensor with same batch format, last dim = output_size
        """
        self.lstm.flatten_parameters()
        recurrent, _ = self.lstm(x)

        if self.batch_first:
            # [B, T, 2*hidden] -> [B, T, output]
            output = self.linear(recurrent)
        else:
            # [T, B, 2*hidden] -> [T, B, output]
            T, B, H = recurrent.size()
            output = self.linear(recurrent.reshape(T * B, H))
            output = output.reshape(T, B, -1)

        return output


class SequenceEncoder(nn.Module):
    """
    Stacked BiLSTM sequence encoder

    Two-layer bidirectional LSTM for sequence modeling.
    Used as the "neck" in recognition models between backbone and CTC head.

    Architecture:
        Input: [B, C, 1, W] feature map from backbone
        -> Reshape to [B, W, C] sequence
        -> BiLSTM layer 1
        -> BiLSTM layer 2
        Output: [B, W, hidden_size] sequence features

    Args:
        in_channels: Input feature dimension (backbone output channels)
        hidden_size: LSTM hidden dimension (also output dimension)
    """

    def __init__(self, in_channels: int, hidden_size: int = 256):
        super().__init__()
        self.lstm1 = BidirectionalLSTM(in_channels, hidden_size, hidden_size)
        self.lstm2 = BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        self.out_channels = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [B, C, 1, W] feature map (height must be 1)

        Returns:
            [B, W, hidden_size] sequence features
        """
        b, c, h, w = x.size()
        assert h == 1, f"Input height must be 1, got {h}"

        # Reshape: [B, C, 1, W] -> [B, W, C]
        x = x.squeeze(2).permute(0, 2, 1)

        x = self.lstm1(x)
        x = self.lstm2(x)

        return x
