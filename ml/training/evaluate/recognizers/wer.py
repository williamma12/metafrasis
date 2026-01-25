"""
Word Error Rate (WER) metric for recognition.
"""

from typing import List

from ..base import MetricResult
from .base import RecognizerMetric


class WERMetric(RecognizerMetric):
    """
    Compute Word Error Rate (WER).

    WER = (Substitutions + Insertions + Deletions) / Total Words in Reference

    Words are split by whitespace. Lower is better.
    WER of 0 means perfect recognition at word level.
    """

    @property
    def name(self) -> str:
        return "wer"

    def word_levenshtein(self, words1: List[str], words2: List[str]) -> int:
        """
        Compute Levenshtein distance at word level.

        Args:
            words1: First word sequence
            words2: Second word sequence

        Returns:
            Edit distance in words
        """
        if len(words1) < len(words2):
            return self.word_levenshtein(words2, words1)

        if len(words2) == 0:
            return len(words1)

        prev_row = list(range(len(words2) + 1))
        for i, w1 in enumerate(words1):
            curr_row = [i + 1]
            for j, w2 in enumerate(words2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (w1 != w2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def compute(
        self, predictions: List[str], targets: List[str]
    ) -> MetricResult:
        """
        Compute WER across all prediction-target pairs.

        Args:
            predictions: List of predicted strings
            targets: List of target strings

        Returns:
            MetricResult with WER value
        """
        if not predictions or not targets:
            return MetricResult(name=self.name, value=0.0)

        total_words = 0
        total_errors = 0

        for pred, target in zip(predictions, targets):
            pred = self.normalize(pred)
            target = self.normalize(target)

            pred_words = pred.split()
            target_words = target.split()

            total_words += len(target_words)
            total_errors += self.word_levenshtein(pred_words, target_words)

        wer = total_errors / total_words if total_words > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=wer,
            details={
                "total_words": total_words,
                "total_errors": total_errors,
                "num_samples": len(predictions),
            },
        )
