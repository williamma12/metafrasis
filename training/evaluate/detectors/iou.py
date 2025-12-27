"""
IoU (Intersection over Union) metric for detection.
"""

from typing import Any, List

import numpy as np

from ..base import MetricResult
from .base import DetectorMetric


class IoUMetric(DetectorMetric):
    """
    Compute mean IoU between predictions and targets.

    For each prediction, finds the best matching target and computes IoU.
    Reports mean IoU across all matched pairs.
    """

    @property
    def name(self) -> str:
        return "iou"

    def compute(
        self, predictions: List[np.ndarray], targets: List[np.ndarray]
    ) -> MetricResult:
        """
        Compute mean IoU.

        Args:
            predictions: List of predicted polygons [N, 2]
            targets: List of target polygons [M, 2]

        Returns:
            MetricResult with mean IoU
        """
        if not predictions or not targets:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={"num_predictions": len(predictions), "num_targets": len(targets)},
            )

        matches, unmatched_preds, unmatched_targets = self.match_predictions(
            predictions, targets
        )

        if not matches:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={
                    "num_matches": 0,
                    "num_predictions": len(predictions),
                    "num_targets": len(targets),
                },
            )

        ious = [iou for _, _, iou in matches]
        mean_iou = sum(ious) / len(ious)

        return MetricResult(
            name=self.name,
            value=mean_iou,
            details={
                "num_matches": len(matches),
                "num_predictions": len(predictions),
                "num_targets": len(targets),
                "unmatched_predictions": len(unmatched_preds),
                "unmatched_targets": len(unmatched_targets),
                "iou_values": ious,
            },
        )
