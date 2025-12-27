"""
Base metric class for model evaluation.

Provides abstract interface for all evaluation metrics with support
for batch processing and aggregation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricResult:
    """
    Container for metric computation results.

    Attributes:
        name: Name of the metric
        value: Primary metric value
        details: Optional detailed breakdown
    """
    name: str
    value: float
    details: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


class Metric(ABC):
    """
    Abstract base class for all evaluation metrics.

    Supports both single-shot computation and batch accumulation.

    Usage:
        # Single computation
        result = metric.compute(predictions, targets)

        # Batch processing
        for batch_pred, batch_target in dataloader:
            metric.update(batch_pred, batch_target)
        result = metric.aggregate()
        metric.reset()
    """

    def __init__(self):
        """Initialize metric state."""
        self._predictions: List[Any] = []
        self._targets: List[Any] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass

    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> MetricResult:
        """
        Compute metric from predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            MetricResult with computed value
        """
        pass

    def update(self, predictions: Any, targets: Any) -> None:
        """
        Update running statistics for batch processing.

        Args:
            predictions: Batch of predictions
            targets: Batch of targets
        """
        self._predictions.append(predictions)
        self._targets.append(targets)

    def aggregate(self) -> MetricResult:
        """
        Compute final metric from accumulated state.

        Returns:
            MetricResult from all accumulated batches
        """
        if not self._predictions:
            return MetricResult(name=self.name, value=0.0)

        # Flatten accumulated data
        all_predictions = self._flatten(self._predictions)
        all_targets = self._flatten(self._targets)

        return self.compute(all_predictions, all_targets)

    def reset(self) -> None:
        """Reset accumulated state."""
        self._predictions = []
        self._targets = []

    def _flatten(self, data: List[Any]) -> Any:
        """
        Flatten accumulated batches.

        Override this method for custom data structures.
        """
        if not data:
            return []

        # Handle list of lists
        if isinstance(data[0], list):
            return [item for batch in data for item in batch]

        # Return as-is for other types
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
