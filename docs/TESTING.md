# Testing Guide

This document describes how to run tests in the Metafrasis project.

## Quick Start

```bash
# Show all available test commands
make help

# Run all tests (Python + Frontend)
make test-all

# Run just ML tests
make test-ml

# Run just frontend tests
make test-frontend
```

## Test Organization

### Python Tests (pytest)

**ML Model Tests** (`tests/ml/models/`) - **148 tests**
- `test_layers.py` (29 tests) - ConvBNLayer, SEModule, BasicBlock, CTCDecoder, charsets
- `test_backbones.py` (29 tests) - VGG16BN, ResNet, MobileNetV3, CRNNCNN
- `test_necks.py` (23 tests) - FPN, BiLSTM, SequenceEncoder
- `test_heads.py` (18 tests) - CTCHead, DBHead
- `test_composites.py` (40 tests) - CRAFT, DBNet, CRNN, PPOCRModel
- `test_registry.py` (15 tests) - Model registry validation

**Backend Tests** (`tests/app/`) - **52 tests**
- Page tests, service tests, component tests

### Frontend Tests (vitest)

**OCR Viewer** (`app/frontend/ocr_viewer/tests/`) - **16 tests**
- Rendering, interactivity, layout, edge cases

**Annotation Canvas** (`app/frontend/annotation_canvas/tests/`) - **30 tests**
- Rectangle mode, polygon mode, select mode, keyboard shortcuts

## Makefile Targets

### Running Tests

```bash
# All tests
make test-all          # Run everything (Python + Frontend)
make test              # Alias for test-all

# Python tests
make test-python       # All Python tests (ML + Backend)
make test-ml           # Just ML model tests
make test-backend      # Just backend/service tests

# Frontend tests
make test-frontend              # All frontend tests
make test-ocr-viewer           # Just OCR Viewer
make test-annotation-canvas    # Just Annotation Canvas

# Quick tests (skip slow tests)
make test-quick        # Run with -m "not slow"
```

### Coverage Reports

```bash
# Generate coverage reports
make test-coverage-ml       # ML tests with coverage
make test-coverage-backend  # Backend tests with coverage

# View HTML coverage report
open htmlcov/index.html
```

### Other Commands

```bash
# Install dependencies
make install           # Install Python + npm dependencies

# Code quality
make lint              # Run ruff linter
make lint-fix          # Auto-fix linting issues

# Cleanup
make clean             # Remove test artifacts, caches, coverage reports

# Test summary
make test-summary      # Show test counts by category
```

## Running Tests Manually

### Python Tests (pytest)

```bash
# All ML tests
uv run pytest tests/ml/models/ -v

# Specific test file
uv run pytest tests/ml/models/test_layers.py -v

# Specific test function
uv run pytest tests/ml/models/test_layers.py::TestConvBNLayer::test_output_shape_with_stride_1

# With coverage
uv run pytest tests/ml/models/ --cov=ml.models --cov-report=html

# Skip slow tests
uv run pytest tests/ -m "not slow"

# Run tests matching pattern
uv run pytest -k "CTC"
```

### Frontend Tests (vitest)

```bash
# OCR Viewer
cd app/frontend/ocr_viewer
npm test              # Interactive mode
npm test -- --run     # Run once and exit

# Annotation Canvas
cd app/frontend/annotation_canvas
npm test -- --run

# With UI
npm run test:ui

# With coverage
npm run test:coverage
```

## Test Markers

Python tests use pytest markers to categorize tests:

- `@pytest.mark.slow` - Long-running tests (can skip with `-m "not slow"`)
- `@pytest.mark.requires_craft` - Requires CRAFT model weights
- `@pytest.mark.requires_crnn` - Requires CRNN model weights
- `@pytest.mark.requires_tesseract` - Requires Tesseract installation
- `@pytest.mark.requires_mps` - Requires Apple Metal Performance Shaders
- `@pytest.mark.requires_native` - Requires native extension compilation

## Continuous Integration

The test suite is designed to run in CI environments:

```bash
# CI-friendly command (exits with error code on failure)
make test-all

# Or run components separately
make test-python && make test-frontend
```

## Test Coverage Summary

| Category | Tests | Status |
|----------|-------|--------|
| ML Models | 148 | ✅ 100% passing |
| Backend | 52 | ✅ 100% passing |
| OCR Viewer | 16 | ✅ 100% passing |
| Annotation Canvas | 30 | ⚠️ 33% passing |
| **Total** | **246** | **~88% passing** |

## Writing New Tests

### Python Tests

```python
# tests/ml/models/test_example.py
import pytest
import torch
from ml.models.example import ExampleModel

class TestExampleModel:
    """Tests for ExampleModel."""

    def test_output_shape(self):
        """Test output shape is correct."""
        model = ExampleModel(in_channels=3, out_channels=64)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)

        assert out.shape == (2, 64, 32, 32)

    def test_gradients_flow(self):
        """Test gradients flow backward."""
        model = ExampleModel(in_channels=3, out_channels=64)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

### Frontend Tests

```typescript
// tests/Example.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import Example from '../src/Example'

describe('Example', () => {
  it('renders correctly', async () => {
    render(<Example message="Hello" />)

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument()
    })
  })
})
```

## Troubleshooting

### Tests fail with "Image not loading"

Frontend tests mock the Image object to trigger onload. If you see loading issues:
- Check that `tests/setup.ts` includes the Image mock
- Ensure tests use `await waitFor()` to wait for async rendering

### Tests fail with "Module not found"

```bash
# Reinstall dependencies
make install

# Or manually
uv sync
cd app/frontend/ocr_viewer && npm install
cd app/frontend/annotation_canvas && npm install
```

### Coverage report not generated

```bash
# Install coverage dependencies
uv pip install pytest-cov

# Run with coverage
make test-coverage-ml
```

### Slow tests timing out

```bash
# Skip slow tests
make test-quick

# Or increase timeout
uv run pytest --timeout=300
```

## Best Practices

1. **Always write assertions** - Never use comments in place of `assert` or `expect()`
2. **Test shapes AND behavior** - Don't just test tensor shapes, test functional properties
3. **Test gradient flow** - Ensure backpropagation works
4. **Use fixtures** - Share common test data via pytest fixtures or test utils
5. **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`
6. **Mock external dependencies** - Don't download models or make API calls in tests
7. **Use descriptive names** - Test names should clearly describe what they verify

## Reference

- [pytest documentation](https://docs.pytest.org/)
- [Vitest documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
