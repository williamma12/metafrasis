"""
Base class for recognition evaluation metrics.

Provides shared functionality for string comparison, edit distance
computation, and text normalization.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import Metric, MetricResult


class RecognizerMetric(Metric):
    """
    Base class for recognition metrics.

    Works with string predictions and targets.
    Provides text normalization and edit distance computation.
    """

    def __init__(self, case_sensitive: bool = True):
        """
        Initialize recognizer metric.

        Args:
            case_sensitive: Whether to perform case-sensitive comparison
        """
        super().__init__()
        self.case_sensitive = case_sensitive

    def normalize(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if not self.case_sensitive:
            text = text.lower()
        return text

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein (edit) distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance (number of insertions, deletions, substitutions)
        """
        s1 = self.normalize(s1)
        s2 = self.normalize(s2)

        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def align_strings(
        self, s1: str, s2: str
    ) -> Tuple[str, str, List[Tuple[int, str, str]]]:
        """
        Align two strings using dynamic programming.

        Returns aligned strings with gaps (-) and list of differences.

        Args:
            s1: First string (prediction)
            s2: Second string (target)

        Returns:
            Tuple of (aligned_s1, aligned_s2, differences)
        """
        s1 = self.normalize(s1)
        s2 = self.normalize(s2)

        m, n = len(s1), len(s2)

        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],     # deletion
                        dp[i][j - 1],     # insertion
                        dp[i - 1][j - 1]  # substitution
                    )

        # Backtrack
        aligned1, aligned2 = [], []
        differences = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
                aligned1.append(s1[i - 1])
                aligned2.append(s2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                # Substitution
                aligned1.append(s1[i - 1])
                aligned2.append(s2[j - 1])
                differences.append((len(aligned1) - 1, s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                # Deletion (extra char in prediction)
                aligned1.append(s1[i - 1])
                aligned2.append("-")
                differences.append((len(aligned1) - 1, s1[i - 1], "-"))
                i -= 1
            else:
                # Insertion (missing char in prediction)
                aligned1.append("-")
                aligned2.append(s2[j - 1])
                differences.append((len(aligned1) - 1, "-", s2[j - 1]))
                j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), differences

    def _flatten(self, data: List[Any]) -> List[str]:
        """Flatten batches of string lists."""
        if not data:
            return []

        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(item)
            elif isinstance(item, str):
                result.append(item)
        return result
