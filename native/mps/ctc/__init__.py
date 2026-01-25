"""
MPS-accelerated CTC Loss for PyTorch.

Provides a drop-in replacement for nn.CTCLoss that runs on Apple Silicon GPUs
using Metal compute shaders.

Usage:
    from mps_ctc import CTCLossMPS

    criterion = CTCLossMPS(blank=0, reduction='mean')
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
"""

from .ctc_loss import CTCLossMPS

__all__ = ["CTCLossMPS"]
__version__ = "0.1.0"
