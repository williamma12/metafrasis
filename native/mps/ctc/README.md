# MPS-Accelerated CTC Loss for PyTorch

A drop-in replacement for `torch.nn.CTCLoss` that runs on Apple Silicon GPUs using Metal compute shaders.

## Why This Exists

PyTorch's built-in `nn.CTCLoss` doesn't support MPS (Metal Performance Shaders) backend. This forces training to fall back to CPU, which is 5-15x slower on Apple Silicon.

This package provides a native Metal implementation for CTC loss, enabling full GPU acceleration for OCR and speech recognition training on Mac.

## Installation

```bash
# From the mps_ctc directory
pip install -e .
```

## Usage

```python
from mps_ctc import CTCLossMPS

# Drop-in replacement for nn.CTCLoss
criterion = CTCLossMPS(blank=0, reduction='mean')

# Works exactly like nn.CTCLoss
log_probs = model(images).log_softmax(2)  # [T, B, C]
loss = criterion(log_probs, targets, input_lengths, target_lengths)
loss.backward()
```

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Pure Python Forward | ✅ Working | Reference implementation, works on any device |
| Pure Python Backward | ✅ Working | Delegates to CPU PyTorch for gradients |
| Metal Forward Kernel | 📝 Skeleton | Needs implementation |
| Metal Backward Kernel | 📝 Skeleton | Needs implementation |
| Metal Gradient Kernel | 📝 Skeleton | Needs implementation |

The current implementation uses pure Python with CPU fallback for gradients. This works correctly but isn't as fast as a native Metal implementation.

## Project Structure

```
mps_ctc/
├── __init__.py              # Package exports
├── ctc_loss.py              # Python CTCLossMPS class
├── setup.py                 # Build configuration
├── csrc/
│   ├── ctc_kernels.metal    # Metal compute shaders (skeleton)
│   └── ctc_mps.mm           # Objective-C++ PyTorch bindings (skeleton)
└── tests/
    └── test_ctc_loss.py     # Test suite
```

## Implementing the Metal Kernels

### Step 1: Understand the CTC Algorithm

CTC uses dynamic programming (forward-backward algorithm):

1. **Expand targets**: Insert blank symbols between characters
   - "cat" → [ε, c, ε, a, ε, t, ε] (7 states)

2. **Forward pass**: Compute α[t,s] = P(emit first s symbols | first t frames)

3. **Backward pass**: Compute β[t,s] = P(emit remaining symbols | frames t to T)

4. **Loss**: -log(α[T-1, L-1] + α[T-1, L-2])

5. **Gradient**: Derived from α, β, and log probabilities

### Step 2: Implement Log-Sum-Exp

All computations must be in log space to prevent underflow:

```metal
inline float log_sum_exp(float a, float b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    float max_val = max(a, b);
    return max_val + log1p(exp(-abs(a - b)));
}
```

### Step 3: Implement Forward Kernel

See `csrc/ctc_kernels.metal` for the full algorithm. Key points:

- Each thread handles one batch element
- Initialize α[0, 0] and α[0, 1] for first timestep
- Iterate through timesteps, computing transitions
- Handle skip transitions for non-blank, non-repeated characters

### Step 4: Implement Backward Kernel

Similar to forward but:
- Iterate time backwards (T-1 to 0)
- Transitions go to s, s+1, s+2 instead of from s, s-1, s-2

### Step 5: Implement Gradient Kernel

```metal
// For each (t, b, c):
grad[t,b,c] = sum over s where label[s] == c of:
              exp(alpha[t,s] + beta[t,s] - total_log_prob)
              - exp(log_probs[t,b,c])
```

### Step 6: Build and Test

```bash
# Compile Metal shaders
cd csrc
xcrun -sdk macosx metal -c ctc_kernels.metal -o ctc_kernels.air
xcrun -sdk macosx metallib ctc_kernels.air -o ctc_kernels.metallib
cd ..

# Build extension
pip install -e .

# Run tests
pytest tests/ -v
```

## Memory Layout

All tensors use row-major layout:

| Tensor | Shape | Index Formula |
|--------|-------|---------------|
| log_probs | [T, B, C] | `t * (B * C) + b * C + c` |
| alpha/beta | [T, B, L] | `t * (B * L) + b * L + s` |
| targets | [B, S] | `b * S + s_idx` |
| grad | [T, B, C] | `t * (B * C) + b * C + c` |

Where:
- T = max input sequence length
- B = batch size
- C = number of classes (vocabulary size)
- S = max target sequence length
- L = 2*S + 1 (expanded sequence length)

## Debugging Tips

1. **Test forward first**: Compare loss values against CPU reference
2. **Check numerical stability**: Long sequences can cause -inf propagation
3. **Verify indexing**: Off-by-one errors are common in DP algorithms
4. **Use small examples**: Start with T=10, B=2, C=5, S=3

## References

- [CTC Original Paper (Graves 2006)](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Distill.pub CTC Explanation](https://distill.pub/2017/ctc/)
- [PyTorch CTCLoss Source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossCTC.cpp)
- [Metal Compute Documentation](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
