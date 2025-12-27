"""
Precision, Recall, and F1 metrics for detection.
"""

from typing import Any, Dict, List

import numpy as np

from ..base import MetricResult
from .base import DetectorMetric


class PrecisionRecallF1Metric(DetectorMetric):
    """
    Compute Precision, Recall, and F1 score for detection.

    - Precision: TP / (TP + FP) - fraction of predictions that are correct
    - Recall: TP / (TP + FN) - fraction of targets that are detected
    - F1: 2 * (Precision * Recall) / (Precision + Recall)
    """

    @property
    def name(self) -> str:
        return "precision_recall_f1"

    def compute(
        self, predictions: List[np.ndarray], targets: List[np.ndarray]
    ) -> MetricResult:
        """
        Compute precision, recall, and F1.

        Args:
            predictions: List of predicted polygons
            targets: List of target polygons

        Returns:
            MetricResult with F1 as primary value, precision/recall in details
        """
        matches, unmatched_preds, unmatched_targets = self.match_predictions(
            predictions, targets
        )

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_targets)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=f1,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "iou_threshold": self.iou_threshold,
            },
        )


class PrecisionMetric(DetectorMetric):
    """Compute precision only."""

    @property
    def name(self) -> str:
        return "precision"

    def compute(
        self, predictions: List[np.ndarray], targets: List[np.ndarray]
    ) -> MetricResult:
        matches, unmatched_preds, _ = self.match_predictions(predictions, targets)
        tp = len(matches)
        fp = len(unmatched_preds)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return MetricResult(name=self.name, value=precision)


class RecallMetric(DetectorMetric):
    """Compute recall only."""

    @property
    def name(self) -> str:
        return "recall"

    def compute(
        self, predictions: List[np.ndarray], targets: List[np.ndarray]
    ) -> MetricResult:
        matches, _, unmatched_targets = self.match_predictions(predictions, targets)
        tp = len(matches)
        fn = len(unmatched_targets)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return MetricResult(name=self.name, value=recall)


class F1Metric(DetectorMetric):
    """Compute F1 score only."""

    @property
    def name(self) -> str:
        return "f1"

    def compute(
        self, predictions: List[np.ndarray], targets: List[np.ndarray]
    ) -> MetricResult:
        matches, unmatched_preds, unmatched_targets = self.match_predictions(
            predictions, targets
        )
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_targets)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(name=self.name, value=f1)
