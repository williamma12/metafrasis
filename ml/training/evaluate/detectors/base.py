"""
Base class for detection evaluation metrics.

Provides shared functionality for computing IoU, matching predictions
to targets, and handling polygon-based detections.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import Metric, MetricResult


class DetectorMetric(Metric):
    """
    Base class for detection metrics.

    Works with polygon-based detections (list of [N, 2] arrays).
    Provides IoU computation and prediction-target matching.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize detector metric.

        Args:
            iou_threshold: IoU threshold for matching predictions to targets
        """
        super().__init__()
        self.iou_threshold = iou_threshold

    def compute_iou(
        self, poly1: np.ndarray, poly2: np.ndarray
    ) -> float:
        """
        Compute IoU between two polygons.

        Uses Shapely for accurate polygon intersection/union.

        Args:
            poly1: First polygon [N, 2]
            poly2: Second polygon [M, 2]

        Returns:
            IoU value in [0, 1]
        """
        try:
            from shapely.geometry import Polygon
            from shapely.validation import make_valid
        except ImportError:
            # Fallback to bounding box IoU
            return self._bbox_iou(poly1, poly2)

        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)

            if not p1.is_valid:
                p1 = make_valid(p1)
            if not p2.is_valid:
                p2 = make_valid(p2)

            intersection = p1.intersection(p2).area
            union = p1.union(p2).area

            if union == 0:
                return 0.0

            return intersection / union

        except Exception:
            return self._bbox_iou(poly1, poly2)

    def _bbox_iou(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """Compute bounding box IoU as fallback."""
        # Get bounding boxes
        x1_min, y1_min = poly1.min(axis=0)
        x1_max, y1_max = poly1.max(axis=0)
        x2_min, y2_min = poly2.min(axis=0)
        x2_max, y2_max = poly2.max(axis=0)

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def match_predictions(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        Match predictions to targets using greedy IoU matching.

        Args:
            predictions: List of predicted polygons
            targets: List of target polygons

        Returns:
            Tuple of:
                - matches: List of (pred_idx, target_idx, iou) tuples
                - unmatched_preds: List of unmatched prediction indices
                - unmatched_targets: List of unmatched target indices
        """
        if not predictions or not targets:
            return (
                [],
                list(range(len(predictions))),
                list(range(len(targets))),
            )

        # Compute IoU matrix
        iou_matrix = np.zeros((len(predictions), len(targets)))
        for i, pred in enumerate(predictions):
            for j, target in enumerate(targets):
                iou_matrix[i, j] = self.compute_iou(pred, target)

        # Greedy matching
        matches = []
        matched_preds = set()
        matched_targets = set()

        while True:
            # Find max IoU
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            pred_idx, target_idx = np.unravel_index(
                iou_matrix.argmax(), iou_matrix.shape
            )

            matches.append((pred_idx, target_idx, max_iou))
            matched_preds.add(pred_idx)
            matched_targets.add(target_idx)

            # Zero out matched row and column
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, target_idx] = 0

        unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
        unmatched_targets = [i for i in range(len(targets)) if i not in matched_targets]

        return matches, unmatched_preds, unmatched_targets

    def _flatten(self, data: List[Any]) -> List[np.ndarray]:
        """Flatten batches of polygon lists."""
        if not data:
            return []

        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result
