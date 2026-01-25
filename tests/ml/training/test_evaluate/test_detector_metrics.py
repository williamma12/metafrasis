"""Tests for detector evaluation metrics."""

import pytest
import numpy as np
from ml.training.evaluate.detectors import (
    DetectorMetric,
    IoUMetric,
    PrecisionRecallF1Metric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    MeanAPMetric,
    APAt50Metric,
    APAt75Metric,
)


class TestDetectorMetricBase:
    """Tests for DetectorMetric base class."""

    def test_compute_iou_perfect_overlap(self):
        """Test IoU for perfectly overlapping polygons."""
        metric = IoUMetric()
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        iou = metric.compute_iou(poly, poly)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        """Test IoU for non-overlapping polygons."""
        metric = IoUMetric()
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        poly2 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]])
        iou = metric.compute_iou(poly1, poly2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        """Test IoU for partially overlapping polygons."""
        metric = IoUMetric()
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        poly2 = np.array([[5, 0], [15, 0], [15, 10], [5, 10]])
        iou = metric.compute_iou(poly1, poly2)
        # Overlap area = 5x10 = 50, Union = 10x10 + 10x10 - 50 = 150
        # IoU = 50/150 = 0.333...
        assert 0.32 < iou < 0.35

    def test_match_predictions_perfect(self):
        """Test prediction matching with perfect matches."""
        metric = IoUMetric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]),
        ]
        matches, unmatched_preds, unmatched_targets = metric.match_predictions(
            predictions, targets
        )
        assert len(matches) == 2
        assert len(unmatched_preds) == 0
        assert len(unmatched_targets) == 0
        assert all(iou == 1.0 for _, _, iou in matches)

    def test_match_predictions_partial_overlap(self):
        """Test matching with partial overlap below threshold."""
        metric = IoUMetric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            # ~33% IoU - below 0.5 threshold
            np.array([[5, 0], [15, 0], [15, 10], [5, 10]]),
        ]
        matches, unmatched_preds, unmatched_targets = metric.match_predictions(
            predictions, targets
        )
        # IoU 0.33 < 0.5, so no match
        assert len(matches) == 0
        assert len(unmatched_preds) == 1
        assert len(unmatched_targets) == 1


class TestIoUMetric:
    """Tests for IoU metric."""

    def test_perfect_predictions(self):
        """Test IoU with perfect predictions."""
        metric = IoUMetric()
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.value == 1.0
        assert result.name == "iou"

    def test_high_overlap_match(self):
        """Test IoU with high overlap (above threshold)."""
        metric = IoUMetric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            # ~70% overlap
            np.array([[2, 0], [12, 0], [12, 10], [2, 10]]),
        ]
        result = metric.compute(predictions, targets)
        # Should match since IoU > 0.5
        assert result.value > 0.5

    def test_no_predictions(self):
        """Test IoU with no predictions."""
        metric = IoUMetric()
        predictions = []
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.value == 0.0

    def test_no_targets(self):
        """Test IoU with no targets."""
        metric = IoUMetric()
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = []
        result = metric.compute(predictions, targets)
        assert result.value == 0.0

    def test_multiple_predictions(self):
        """Test IoU with multiple predictions."""
        metric = IoUMetric()
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.value == 1.0


class TestPrecisionRecallF1Metric:
    """Tests for Precision, Recall, and F1 metrics."""

    def test_perfect_predictions(self):
        """Test P/R/F1 with perfect predictions."""
        metric = PrecisionRecallF1Metric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.details["precision"] == 1.0
        assert result.details["recall"] == 1.0
        assert result.details["f1"] == 1.0

    def test_false_positive(self):
        """Test with false positive (extra prediction)."""
        metric = PrecisionRecallF1Metric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]]),  # FP
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        # TP=1, FP=1, FN=0
        # Precision = 1/2 = 0.5
        # Recall = 1/1 = 1.0
        assert result.details["precision"] == 0.5
        assert result.details["recall"] == 1.0

    def test_false_negative(self):
        """Test with false negative (missed target)."""
        metric = PrecisionRecallF1Metric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]]),  # FN
        ]
        result = metric.compute(predictions, targets)
        # TP=1, FP=0, FN=1
        # Precision = 1/1 = 1.0
        # Recall = 1/2 = 0.5
        assert result.details["precision"] == 1.0
        assert result.details["recall"] == 0.5

    def test_no_predictions(self):
        """Test with no predictions."""
        metric = PrecisionRecallF1Metric(iou_threshold=0.5)
        predictions = []
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.details["precision"] == 0.0
        assert result.details["recall"] == 0.0
        assert result.details["f1"] == 0.0

    def test_no_targets(self):
        """Test with no targets."""
        metric = PrecisionRecallF1Metric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = []
        result = metric.compute(predictions, targets)
        assert result.details["precision"] == 0.0
        assert result.details["recall"] == 0.0


class TestPrecisionMetric:
    """Tests for Precision metric."""

    def test_precision_only(self):
        """Test that PrecisionMetric returns precision value."""
        metric = PrecisionMetric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]]),  # FP
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.name == "precision"
        assert result.value == 0.5


class TestRecallMetric:
    """Tests for Recall metric."""

    def test_recall_only(self):
        """Test that RecallMetric returns recall value."""
        metric = RecallMetric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]]),  # FN
        ]
        result = metric.compute(predictions, targets)
        assert result.name == "recall"
        assert result.value == 0.5


class TestF1Metric:
    """Tests for F1 metric."""

    def test_f1_calculation(self):
        """Test F1 score calculation."""
        metric = F1Metric(iou_threshold=0.5)
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]]),  # FP
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[100, 100], [110, 100], [110, 110], [100, 110]]),  # FN
        ]
        result = metric.compute(predictions, targets)
        assert result.name == "f1"
        # TP=1, FP=1, FN=1
        # Precision = 1/2 = 0.5
        # Recall = 1/2 = 0.5
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert result.value == 0.5


class TestMeanAPMetric:
    """Tests for Mean Average Precision metric."""

    def test_perfect_predictions(self):
        """Test mAP with perfect predictions."""
        metric = MeanAPMetric()
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.name == "mAP"
        assert result.value == 1.0

    def test_empty_predictions(self):
        """Test mAP with empty predictions."""
        metric = MeanAPMetric()
        predictions = []
        targets = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = metric.compute(predictions, targets)
        assert result.value == 0.0


class TestAPAtThresholds:
    """Tests for AP at specific IoU thresholds."""

    def test_ap_at_50(self):
        """Test AP@0.5."""
        metric = APAt50Metric()
        assert metric.iou_threshold == 0.5

    def test_ap_at_75(self):
        """Test AP@0.75."""
        metric = APAt75Metric()
        assert metric.iou_threshold == 0.75

    def test_stricter_threshold_lower_ap(self):
        """Test that stricter threshold gives lower AP for imperfect matches."""
        predictions = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        targets = [
            # Shifted right by 3 - IoU ≈ 0.54
            np.array([[3, 0], [13, 0], [13, 10], [3, 10]]),
        ]

        ap50 = APAt50Metric()
        ap75 = APAt75Metric()

        result50 = ap50.compute(predictions, targets)
        result75 = ap75.compute(predictions, targets)

        # IoU ≈ 0.54, so AP@0.5 should be 1.0, AP@0.75 should be 0.0
        assert result50.value == 1.0
        assert result75.value == 0.0
