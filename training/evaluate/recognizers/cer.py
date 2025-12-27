"""
Character Error Rate (CER) metric for recognition.
"""

from typing import List

from ..base import MetricResult
from .base import RecognizerMetric


class CERMetric(RecognizerMetric):
    """
    Compute Character Error Rate (CER).

    CER = (Substitutions + Insertions + Deletions) / Total Characters in Reference

    Lower is better. CER of 0 means perfect recognition.
    CER can exceed 1.0 if predictions are significantly longer than references.
    """

    @property
    def name(self) -> str:
        return "cer"

    def compute(
        self, predictions: List[str], targets: List[str]
    ) -> MetricResult:
        """
        Compute CER across all prediction-target pairs.

        Args:
            predictions: List of predicted strings
            targets: List of target strings

        Returns:
            MetricResult with CER value
        """
        if not predictions or not targets:
            return MetricResult(name=self.name, value=0.0)

        total_chars = 0
        total_errors = 0

        for pred, target in zip(predictions, targets):
            pred = self.normalize(pred)
            target = self.normalize(target)

            total_chars += len(target)
            total_errors += self.levenshtein_distance(pred, target)

        cer = total_errors / total_chars if total_chars > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=cer,
            details={
                "total_characters": total_chars,
                "total_errors": total_errors,
                "num_samples": len(predictions),
            },
        )


class NormalizedEditDistanceMetric(RecognizerMetric):
    """
    Compute Normalized Edit Distance (NED).

    NED = Edit Distance / max(len(prediction), len(target))

    Normalized to [0, 1] range where 0 is perfect match.
    """

    @property
    def name(self) -> str:
        return "ned"

    def compute(
        self, predictions: List[str], targets: List[str]
    ) -> MetricResult:
        """
        Compute mean NED across all pairs.

        Args:
            predictions: List of predicted strings
            targets: List of target strings

        Returns:
            MetricResult with mean NED
        """
        if not predictions or not targets:
            return MetricResult(name=self.name, value=0.0)

        neds = []
        for pred, target in zip(predictions, targets):
            pred = self.normalize(pred)
            target = self.normalize(target)

            max_len = max(len(pred), len(target))
            if max_len == 0:
                neds.append(0.0)
            else:
                ned = self.levenshtein_distance(pred, target) / max_len
                neds.append(ned)

        mean_ned = sum(neds) / len(neds)

        return MetricResult(
            name=self.name,
            value=mean_ned,
            details={
                "num_samples": len(predictions),
                "ned_values": neds,
            },
        )
