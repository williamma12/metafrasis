"""
Accuracy metric for recognition.
"""

from typing import List

from ..base import MetricResult
from .base import RecognizerMetric


class AccuracyMetric(RecognizerMetric):
    """
    Compute exact match accuracy.

    Accuracy = Number of Exact Matches / Total Samples

    A prediction is considered correct only if it exactly matches
    the target (after normalization if case_sensitive=False).
    """

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(
        self, predictions: List[str], targets: List[str]
    ) -> MetricResult:
        """
        Compute accuracy.

        Args:
            predictions: List of predicted strings
            targets: List of target strings

        Returns:
            MetricResult with accuracy value
        """
        if not predictions or not targets:
            return MetricResult(name=self.name, value=0.0)

        correct = 0
        for pred, target in zip(predictions, targets):
            pred = self.normalize(pred)
            target = self.normalize(target)
            if pred == target:
                correct += 1

        accuracy = correct / len(targets)

        return MetricResult(
            name=self.name,
            value=accuracy,
            details={
                "correct": correct,
                "total": len(targets),
                "incorrect": len(targets) - correct,
            },
        )


class SequenceAccuracyMetric(RecognizerMetric):
    """
    Compute sequence-level accuracy with partial credit.

    For each sample, computes: 1 - (edit_distance / max_length)
    Then averages across all samples.
    """

    @property
    def name(self) -> str:
        return "sequence_accuracy"

    def compute(
        self, predictions: List[str], targets: List[str]
    ) -> MetricResult:
        """
        Compute sequence accuracy with partial credit.

        Args:
            predictions: List of predicted strings
            targets: List of target strings

        Returns:
            MetricResult with sequence accuracy
        """
        if not predictions or not targets:
            return MetricResult(name=self.name, value=0.0)

        scores = []
        for pred, target in zip(predictions, targets):
            pred = self.normalize(pred)
            target = self.normalize(target)

            max_len = max(len(pred), len(target))
            if max_len == 0:
                scores.append(1.0)  # Both empty = perfect match
            else:
                edit_dist = self.levenshtein_distance(pred, target)
                score = 1.0 - (edit_dist / max_len)
                scores.append(max(0.0, score))  # Clamp to [0, 1]

        accuracy = sum(scores) / len(scores)

        return MetricResult(
            name=self.name,
            value=accuracy,
            details={
                "num_samples": len(predictions),
                "scores": scores,
            },
        )
