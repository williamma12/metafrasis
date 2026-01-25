"""Tests for recognizer evaluation metrics."""

import pytest
from ml.training.evaluate.recognizers import (
    RecognizerMetric,
    CERMetric,
    NormalizedEditDistanceMetric,
    WERMetric,
    AccuracyMetric,
    SequenceAccuracyMetric,
)


class TestRecognizerMetricBase:
    """Tests for RecognizerMetric base class."""

    def test_levenshtein_identical(self):
        """Test Levenshtein distance for identical strings."""
        metric = CERMetric()
        assert metric.levenshtein_distance("hello", "hello") == 0

    def test_levenshtein_insertion(self):
        """Test Levenshtein distance for insertions."""
        metric = CERMetric()
        assert metric.levenshtein_distance("helo", "hello") == 1

    def test_levenshtein_deletion(self):
        """Test Levenshtein distance for deletions."""
        metric = CERMetric()
        assert metric.levenshtein_distance("hello", "helo") == 1

    def test_levenshtein_substitution(self):
        """Test Levenshtein distance for substitutions."""
        metric = CERMetric()
        assert metric.levenshtein_distance("hello", "hallo") == 1

    def test_levenshtein_complex(self):
        """Test Levenshtein distance for complex changes."""
        metric = CERMetric()
        # kitten -> sitten (1) -> sittin (1) -> sitting (1) = 3
        assert metric.levenshtein_distance("kitten", "sitting") == 3

    def test_normalize_case_sensitive(self):
        """Test normalization with case sensitivity."""
        metric = CERMetric(case_sensitive=True)
        assert metric.normalize("Hello World") == "Hello World"

    def test_normalize_case_insensitive(self):
        """Test normalization without case sensitivity."""
        metric = CERMetric(case_sensitive=False)
        assert metric.normalize("Hello World") == "hello world"


class TestCERMetric:
    """Tests for Character Error Rate metric."""

    def test_perfect_match(self):
        """Test CER for perfect match."""
        metric = CERMetric()
        result = metric.compute(
            predictions=["hello world"],
            targets=["hello world"]
        )
        assert result.value == 0.0
        assert result.name == "cer"

    def test_single_error(self):
        """Test CER for single character error."""
        metric = CERMetric()
        result = metric.compute(
            predictions=["helo world"],  # 1 error
            targets=["hello world"]  # 11 chars
        )
        # CER = 1/11 ≈ 0.0909
        assert 0.09 < result.value < 0.10

    def test_complete_mismatch(self):
        """Test CER for completely different strings."""
        metric = CERMetric()
        result = metric.compute(
            predictions=["abc"],
            targets=["xyz"]
        )
        # CER = 3/3 = 1.0
        assert result.value == 1.0

    def test_multiple_samples(self):
        """Test CER averaged over multiple samples."""
        metric = CERMetric()
        result = metric.compute(
            predictions=["hello", "world"],
            targets=["hello", "worlx"]  # 1 error in second
        )
        # First: 0/5, Second: 1/5
        # Average: 0.5/5 = 0.1
        assert result.value == 0.1

    def test_empty_prediction(self):
        """Test CER with empty prediction."""
        metric = CERMetric()
        result = metric.compute(
            predictions=[""],
            targets=["hello"]
        )
        # All characters are deletions: 5/5 = 1.0
        assert result.value == 1.0

    def test_batch_accumulation(self):
        """Test CER with batch accumulation."""
        metric = CERMetric()
        metric.update(predictions=["hello"], targets=["hello"])
        metric.update(predictions=["world"], targets=["worlx"])
        result = metric.aggregate()
        assert result.value == 0.1  # (0 + 1) / (5 + 5)

    def test_reset(self):
        """Test metric reset."""
        metric = CERMetric()
        metric.update(predictions=["hello"], targets=["helo"])
        metric.reset()
        metric.update(predictions=["world"], targets=["world"])
        result = metric.aggregate()
        assert result.value == 0.0


class TestNormalizedEditDistanceMetric:
    """Tests for Normalized Edit Distance metric."""

    def test_perfect_match(self):
        """Test NED for perfect match."""
        metric = NormalizedEditDistanceMetric()
        result = metric.compute(
            predictions=["hello"],
            targets=["hello"]
        )
        assert result.value == 0.0

    def test_partial_match(self):
        """Test NED for partial match."""
        metric = NormalizedEditDistanceMetric()
        result = metric.compute(
            predictions=["helo"],
            targets=["hello"]
        )
        # NED = 1 / max(4, 5) = 0.2
        assert result.value == 0.2

    def test_normalization_by_max_length(self):
        """Test that NED normalizes by max length."""
        metric = NormalizedEditDistanceMetric()
        # Prediction is longer than target
        result = metric.compute(
            predictions=["helloo"],
            targets=["hello"]
        )
        # NED = 1 / max(6, 5) = 1/6
        assert abs(result.value - 1/6) < 0.001


class TestWERMetric:
    """Tests for Word Error Rate metric."""

    def test_perfect_match(self):
        """Test WER for perfect match."""
        metric = WERMetric()
        result = metric.compute(
            predictions=["hello world"],
            targets=["hello world"]
        )
        assert result.value == 0.0

    def test_single_word_error(self):
        """Test WER for single word error."""
        metric = WERMetric()
        result = metric.compute(
            predictions=["hello world"],
            targets=["hello earth"]  # 2 words, 1 wrong
        )
        # WER = 1/2 = 0.5
        assert result.value == 0.5

    def test_insertion(self):
        """Test WER for word insertion."""
        metric = WERMetric()
        result = metric.compute(
            predictions=["hello beautiful world"],
            targets=["hello world"]  # 2 words
        )
        # WER = 1/2 = 0.5 (one insertion)
        assert result.value == 0.5

    def test_deletion(self):
        """Test WER for word deletion."""
        metric = WERMetric()
        result = metric.compute(
            predictions=["hello"],
            targets=["hello world"]  # 2 words
        )
        # WER = 1/2 = 0.5 (one deletion)
        assert result.value == 0.5

    def test_multiple_samples(self):
        """Test WER averaged over multiple samples."""
        metric = WERMetric()
        result = metric.compute(
            predictions=["hello world", "foo bar"],
            targets=["hello world", "foo baz"]
        )
        # First: 0/2, Second: 1/2
        # Total: 1/4 = 0.25
        assert result.value == 0.25


class TestAccuracyMetric:
    """Tests for exact match Accuracy metric."""

    def test_perfect_match(self):
        """Test accuracy for perfect match."""
        metric = AccuracyMetric()
        result = metric.compute(
            predictions=["hello"],
            targets=["hello"]
        )
        assert result.value == 1.0

    def test_no_match(self):
        """Test accuracy for no match."""
        metric = AccuracyMetric()
        result = metric.compute(
            predictions=["helo"],  # Not exact match
            targets=["hello"]
        )
        assert result.value == 0.0

    def test_partial_match(self):
        """Test accuracy for partial match (some exact, some not)."""
        metric = AccuracyMetric()
        result = metric.compute(
            predictions=["hello", "world"],
            targets=["hello", "worlx"]  # First exact, second not
        )
        # 1 out of 2 exact matches = 0.5
        assert result.value == 0.5

    def test_case_sensitivity(self):
        """Test accuracy with case sensitivity."""
        metric = AccuracyMetric(case_sensitive=True)
        result = metric.compute(
            predictions=["Hello"],
            targets=["hello"]
        )
        assert result.value == 0.0

        metric_insensitive = AccuracyMetric(case_sensitive=False)
        result = metric_insensitive.compute(
            predictions=["Hello"],
            targets=["hello"]
        )
        assert result.value == 1.0


class TestSequenceAccuracyMetric:
    """Tests for sequence-level Accuracy metric with partial credit."""

    def test_perfect_match(self):
        """Test sequence accuracy for perfect match."""
        metric = SequenceAccuracyMetric()
        result = metric.compute(
            predictions=["hello", "world"],
            targets=["hello", "world"]
        )
        assert result.value == 1.0

    def test_partial_sequence_match(self):
        """Test sequence accuracy for partial match."""
        metric = SequenceAccuracyMetric()
        result = metric.compute(
            predictions=["hello", "worlx"],  # 1 char diff in 2nd
            targets=["hello", "world"]
        )
        # First: 1.0 (perfect), Second: 1 - 1/5 = 0.8
        # Average: (1.0 + 0.8) / 2 = 0.9
        assert result.value == 0.9

    def test_no_match(self):
        """Test sequence accuracy for no character match."""
        metric = SequenceAccuracyMetric()
        result = metric.compute(
            predictions=["abc"],
            targets=["xyz"]
        )
        # 3 substitutions out of max(3, 3) = 1.0 - 1.0 = 0.0
        assert result.value == 0.0

    def test_case_sensitivity(self):
        """Test sequence accuracy with case sensitivity."""
        metric = SequenceAccuracyMetric(case_sensitive=True)
        result = metric.compute(
            predictions=["Hello"],
            targets=["hello"]
        )
        # 1 substitution (H->h) out of 5 chars = 1 - 0.2 = 0.8
        assert result.value == 0.8

        metric_insensitive = SequenceAccuracyMetric(case_sensitive=False)
        result = metric_insensitive.compute(
            predictions=["Hello"],
            targets=["hello"]
        )
        assert result.value == 1.0
