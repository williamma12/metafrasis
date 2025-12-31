"""
CTC Loss implementation with MPS support.

Provides CTCLossMPS - a drop-in replacement for nn.CTCLoss that works on Apple Silicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional

# Try to import compiled extension
HAS_NATIVE_EXTENSION = False
try:
    from . import _C
    # Verify the extension has the required functions
    if hasattr(_C, 'ctc_loss_forward') and hasattr(_C, 'ctc_loss_backward'):
        HAS_NATIVE_EXTENSION = True
except ImportError:
    pass


def log_sum_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-sum-exp for two tensors."""
    max_val = torch.maximum(a, b)
    # Handle -inf cases
    result = max_val + torch.log1p(torch.exp(-torch.abs(a - b)))
    # Where both are -inf, result should be -inf
    both_inf = (a == float('-inf')) & (b == float('-inf'))
    result = torch.where(both_inf, a, result)
    return result


def log_sum_exp_3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-sum-exp for three tensors."""
    return log_sum_exp(log_sum_exp(a, b), c)


class CTCLossFunctionPure(Function):
    """
    Pure PyTorch implementation of CTC loss forward-backward algorithm.

    This is a reference implementation that works on any device including MPS.
    It's slower than the native Metal implementation but useful for:
    - Testing correctness
    - Fallback when native extension isn't available
    """

    @staticmethod
    def forward(
        ctx,
        log_probs: torch.Tensor,      # [T, B, C]
        targets: torch.Tensor,         # [B, S] or [sum(target_lengths)]
        input_lengths: torch.Tensor,   # [B]
        target_lengths: torch.Tensor,  # [B]
        blank: int = 0,
        zero_infinity: bool = False,
    ) -> torch.Tensor:
        """
        Compute CTC loss using forward-backward algorithm.

        Args:
            log_probs: Log probabilities [T, B, C] where T=time, B=batch, C=classes
            targets: Target sequences [B, S] where S=max target length
            input_lengths: Length of each input sequence [B]
            target_lengths: Length of each target sequence [B]
            blank: Index of blank label (default 0)
            zero_infinity: Replace infinite losses with zero

        Returns:
            Loss tensor [B] (unreduced)
        """
        T, B, C = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype

        # Handle 1D targets (concatenated format)
        if targets.dim() == 1:
            # Split into per-batch targets
            targets_list = torch.split(targets, target_lengths.tolist())
            max_target_len = target_lengths.max().item()
            targets_2d = torch.zeros(B, max_target_len, dtype=targets.dtype, device=device)
            for b, t in enumerate(targets_list):
                targets_2d[b, :len(t)] = t
            targets = targets_2d

        S = targets.shape[1]  # Max target length

        # Expanded sequence length: blank between each char + blanks at ends
        # e.g., "abc" -> [blank, a, blank, b, blank, c, blank] = 2*3+1 = 7
        max_L = 2 * S + 1

        # Initialize alpha (forward variables) with -inf
        # alpha[t, b, s] = log probability of reaching state s at time t
        alpha = torch.full((T, B, max_L), float('-inf'), dtype=dtype, device=device)

        # Compute losses for each batch element
        losses = torch.zeros(B, dtype=dtype, device=device)

        for b in range(B):
            T_b = input_lengths[b].item()
            S_b = target_lengths[b].item()
            L_b = 2 * S_b + 1  # Expanded sequence length for this batch

            if T_b == 0 or S_b == 0:
                losses[b] = 0.0 if zero_infinity else float('inf')
                continue

            # Build expanded label sequence: [blank, c1, blank, c2, blank, ...]
            expanded_labels = torch.zeros(L_b, dtype=torch.long, device=device)
            for s in range(L_b):
                if s % 2 == 0:
                    expanded_labels[s] = blank
                else:
                    expanded_labels[s] = targets[b, s // 2]

            # Initialize alpha at t=0
            # Can only start at state 0 (blank) or state 1 (first char)
            alpha[0, b, 0] = log_probs[0, b, expanded_labels[0]]
            if L_b > 1:
                alpha[0, b, 1] = log_probs[0, b, expanded_labels[1]]

            # Forward recursion
            for t in range(1, T_b):
                for s in range(L_b):
                    label = expanded_labels[s]
                    log_prob = log_probs[t, b, label]

                    # Transition from same state
                    result = alpha[t-1, b, s]

                    # Transition from previous state
                    if s > 0:
                        result = log_sum_exp(result, alpha[t-1, b, s-1])

                    # Skip transition (for non-blank, non-repeated chars)
                    if s > 1:
                        prev_label = expanded_labels[s-2]
                        # Can skip if current is not blank and not same as s-2
                        if label != blank and label != prev_label:
                            result = log_sum_exp(result, alpha[t-1, b, s-2])

                    alpha[t, b, s] = result + log_prob

            # Final loss: log-sum-exp of final states
            # Can end at last blank (L_b-1) or last char (L_b-2)
            final_alpha = alpha[T_b-1, b, L_b-1]
            if L_b > 1:
                final_alpha = log_sum_exp(final_alpha, alpha[T_b-1, b, L_b-2])

            losses[b] = -final_alpha

        # Handle infinities
        if zero_infinity:
            losses = torch.where(torch.isinf(losses), torch.zeros_like(losses), losses)

        # Save for backward
        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, alpha)
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity

        return losses

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Compute gradients using backward algorithm.

        Uses a hybrid approach for correctness: delegates to PyTorch's CPU
        implementation for the gradient computation.
        """
        log_probs, targets, input_lengths, target_lengths, alpha = ctx.saved_tensors
        blank = ctx.blank
        zero_infinity = ctx.zero_infinity

        # Move to CPU for gradient computation using PyTorch's implementation
        log_probs_cpu = log_probs.detach().cpu()
        log_probs_cpu.requires_grad_(True)
        input_lengths_cpu = input_lengths.cpu()
        target_lengths_cpu = target_lengths.cpu()
        grad_output_cpu = grad_output.cpu()

        # Targets need to be 1D concatenated format for ctc_loss
        # Convert from 2D [B, S] to 1D concatenated format
        if targets.dim() == 2:
            B = targets.size(0)
            targets_list = [targets[b, :target_lengths_cpu[b].item()].cpu() for b in range(B)]
            targets_cpu = torch.cat(targets_list)
        else:
            targets_cpu = targets.cpu()

        # Enable gradient computation (disabled by default inside backward())
        with torch.enable_grad():
            # Compute loss on CPU with no reduction
            cpu_loss = F.ctc_loss(
                log_probs_cpu, targets_cpu, input_lengths_cpu, target_lengths_cpu,
                blank=blank, reduction='none', zero_infinity=zero_infinity,
            )

            # Compute gradient using autograd.grad
            # Scale each loss by its corresponding grad_output
            weighted_loss = (cpu_loss * grad_output_cpu).sum()
            grad_log_probs = torch.autograd.grad(
                weighted_loss, log_probs_cpu, create_graph=False
            )[0]

        # Move gradient back to original device
        grad_log_probs = grad_log_probs.to(log_probs.device)

        return grad_log_probs, None, None, None, None, None


class CTCLossFunctionNative(Function):
    """
    Native Metal implementation of CTC loss using compiled extension.
    """

    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank):
        loss, alpha = _C.ctc_loss_forward(
            log_probs, targets, input_lengths, target_lengths, blank
        )
        ctx.save_for_backward(log_probs, alpha, targets, input_lengths, target_lengths)
        ctx.blank = blank
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, alpha, targets, input_lengths, target_lengths = ctx.saved_tensors
        grad_log_probs = _C.ctc_loss_backward(
            grad_output, log_probs, alpha, targets,
            input_lengths, target_lengths, ctx.blank
        )
        return grad_log_probs, None, None, None, None


class CTCLossMPS(nn.Module):
    """
    Drop-in replacement for nn.CTCLoss with MPS (Apple Silicon GPU) support.

    This module provides CTC loss computation that works on MPS devices,
    where PyTorch's native nn.CTCLoss is not supported.

    Args:
        blank: Index of the blank label (default: 0)
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum' (default: 'mean')
        zero_infinity: Whether to zero infinite losses (default: False)

    Example:
        >>> criterion = CTCLossMPS(blank=0, reduction='mean')
        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).to('mps')
        >>> targets = torch.randint(1, 20, (16, 30)).to('mps')
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long).to('mps')
        >>> target_lengths = torch.randint(10, 30, (16,)).to('mps')
        >>> loss = criterion(log_probs, targets, input_lengths, target_lengths)
    """

    def __init__(
        self,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False,
    ):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

        # Check for native extension
        self._use_native = HAS_NATIVE_EXTENSION

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            log_probs: Log probabilities of shape [T, B, C] where:
                      T = input sequence length
                      B = batch size
                      C = number of classes (including blank)
            targets: Target sequences [B, S] or [sum(target_lengths)]
            input_lengths: Lengths of input sequences [B]
            target_lengths: Lengths of target sequences [B]

        Returns:
            Loss tensor (scalar if reduction='mean'/'sum', [B] if 'none')
        """
        # For non-MPS devices, use PyTorch's implementation
        if log_probs.device.type != 'mps':
            return F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=self.blank, reduction=self.reduction,
                zero_infinity=self.zero_infinity,
            )

        # Use appropriate implementation for MPS
        if self._use_native:
            loss = CTCLossFunctionNative.apply(
                log_probs, targets, input_lengths, target_lengths, self.blank
            )
        else:
            loss = CTCLossFunctionPure.apply(
                log_probs, targets, input_lengths, target_lengths,
                self.blank, self.zero_infinity,
            )

        # Apply reduction (PyTorch CTCLoss normalizes by target_lengths for 'mean')
        if self.reduction == 'mean':
            # Divide each loss by its target length, then take batch mean
            return (loss / target_lengths.float()).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f'blank={self.blank}, reduction={self.reduction}, zero_infinity={self.zero_infinity}'
