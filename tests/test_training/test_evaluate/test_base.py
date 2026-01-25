"""Tests for evaluate base classes."""

import pytest
from ml.training.evaluate.base import Metric, MetricResult


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_create_simple(self):
        """Test creating a simple metric result."""
        result = MetricResult(name="accuracy", value=0.95)
        assert result.name == "accuracy"
        assert result.value == 0.95
        assert result.details is None

    def test_create_with_details(self):
        """Test creating a metric result with details."""
        result = MetricResult(
            name="precision",
            value=0.85,
            details={"tp": 85, "fp": 15}
        )
        assert result.name == "precision"
        assert result.value == 0.85
        assert result.details == {"tp": 85, "fp": 15}

    def test_repr(self):
        """Test string representation."""
        result = MetricResult(name="cer", value=0.05)
        assert "cer" in str(result)
        assert "0.05" in str(result)


class TestMetricABC:
    """Tests for Metric abstract base class."""

    def test_cannot_instantiate(self):
        """Test that Metric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Metric()

    def test_concrete_implementation(self):
        """Test creating a concrete implementation."""
        from ml.training.evaluate import CERMetric

        metric = CERMetric()
        assert metric.name == "cer"
        assert hasattr(metric, "compute")
        assert hasattr(metric, "update")
        assert hasattr(metric, "aggregate")
        assert hasattr(metric, "reset")
