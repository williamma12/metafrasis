"""
Mean Average Precision (mAP) metric for detection.
"""

from typing import Any, List

import numpy as np

from ..base import MetricResult
from .base import DetectorMetric


class MeanAPMetric(DetectorMetric):
    """
    Compute Mean Average Precision (mAP) over multiple IoU thresholds.

    Standard thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    For each threshold, computes AP (average precision) and then averages
    across all thresholds.
    """

    def __init__(
        self,
        iou_thresholds: List[float] = None,
    ):
        """
        Initialize mAP metric.

        Args:
            iou_thresholds: List of IoU thresholds to evaluate at.
                           Default: [0.5, 0.55, ..., 0.95]
        """
        super().__init__(iou_threshold=0.5)

        if iou_thresholds is None:
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

    @property
    def name(self) -> str:
        return "mAP"

    def compute_ap_at_threshold(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        iou_threshold: float,
    ) -> float:
        """
        Compute AP at a single IoU threshold.

        Args:
            predictions: List of predicted polygons
            targets: List of target polygons
            iou_threshold: IoU threshold

        Returns:
            AP value
        """
        # Temporarily set threshold
        original_threshold = self.iou_threshold
        self.iou_threshold = iou_threshold

        matches, unmatched_preds, unmatched_targets = self.match_predictions(
            predictions, targets
        )

        # Restore threshold
        self.iou_threshold = original_threshold

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_targets)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # For single image, AP ≈ precision * recall
        # (simplified; full AP requires confidence ranking)
        return precision * recall if recall > 0 else 0.0

    def compute(
        self, predictions: List[np.ndarray], targets: List[np.ndarray]
    ) -> MetricResult:
        """
        Compute mAP across multiple IoU thresholds.

        Args:
            predictions: List of predicted polygons
            targets: List of target polygons

        Returns:
            MetricResult with mAP
        """
        if not predictions or not targets:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={"ap_per_threshold": {}},
            )

        ap_per_threshold = {}
        for threshold in self.iou_thresholds:
            ap = self.compute_ap_at_threshold(predictions, targets, threshold)
            ap_per_threshold[f"AP@{threshold:.2f}"] = ap

        mean_ap = sum(ap_per_threshold.values()) / len(ap_per_threshold)

        return MetricResult(
            name=self.name,
            value=mean_ap,
            details={
                "ap_per_threshold": ap_per_threshold,
                "AP@0.5": ap_per_threshold.get("AP@0.50", 0.0),
                "AP@0.75": ap_per_threshold.get("AP@0.75", 0.0),
            },
        )


class APAt50Metric(DetectorMetric):
    """Compute AP at IoU threshold 0.5."""

    def __init__(self):
        super().__init__(iou_threshold=0.5)

    @property
    def name(self) -> str:
        return "AP@0.5"

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
        ap = precision * recall if recall > 0 else 0.0

        return MetricResult(name=self.name, value=ap)


class APAt75Metric(DetectorMetric):
    """Compute AP at IoU threshold 0.75."""

    def __init__(self):
        super().__init__(iou_threshold=0.75)

    @property
    def name(self) -> str:
        return "AP@0.75"

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
        ap = precision * recall if recall > 0 else 0.0

        return MetricResult(name=self.name, value=ap)
