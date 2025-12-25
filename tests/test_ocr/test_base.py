"""
Tests for OCR base classes and data structures
"""
import pytest
import numpy as np
from services.ocr.base import (
    BoundingBox,
    Word,
    ConfidenceStats,
    OCRResult,
    DEFAULT_CONFIDENCE,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass"""

    def test_bounding_box_creation(self):
        """Test BoundingBox can be created with valid values"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        assert bbox.left == 10
        assert bbox.top == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_bounding_box_equality(self):
        """Test BoundingBox equality comparison"""
        bbox1 = BoundingBox(left=10, top=20, width=100, height=50)
        bbox2 = BoundingBox(left=10, top=20, width=100, height=50)
        bbox3 = BoundingBox(left=15, top=20, width=100, height=50)

        assert bbox1 == bbox2
        assert bbox1 != bbox3


class TestWord:
    """Tests for Word dataclass"""

    def test_word_creation_with_confidence(self):
        """Test Word creation with explicit confidence"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        word = Word(text="hello", bbox=bbox, confidence=0.95)

        assert word.text == "hello"
        assert word.bbox == bbox
        assert word.confidence == 0.95

    def test_word_creation_default_confidence(self):
        """Test Word creation with default confidence"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        word = Word(text="hello", bbox=bbox)

        assert word.text == "hello"
        assert word.confidence == DEFAULT_CONFIDENCE

    def test_word_equality(self):
        """Test Word equality comparison"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        word1 = Word(text="hello", bbox=bbox, confidence=0.95)
        word2 = Word(text="hello", bbox=bbox, confidence=0.95)
        word3 = Word(text="world", bbox=bbox, confidence=0.95)

        assert word1 == word2
        assert word1 != word3


class TestConfidenceStats:
    """Tests for ConfidenceStats dataclass"""

    def test_confidence_stats_creation(self):
        """Test ConfidenceStats creation"""
        stats = ConfidenceStats(mean=0.85, std=0.12, available=True)

        assert stats.mean == 0.85
        assert stats.std == 0.12
        assert stats.available is True

    def test_confidence_stats_unavailable(self):
        """Test ConfidenceStats when confidence is not available"""
        stats = ConfidenceStats(mean=DEFAULT_CONFIDENCE, std=0.0, available=False)

        assert stats.mean == DEFAULT_CONFIDENCE
        assert stats.std == 0.0
        assert stats.available is False


class TestOCRResult:
    """Tests for OCRResult dataclass and confidence statistics"""

    def test_ocr_result_creation(self):
        """Test OCRResult creation with words"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        words = [
            Word(text="hello", bbox=bbox, confidence=0.95),
            Word(text="world", bbox=bbox, confidence=0.87),
        ]

        result = OCRResult(
            words=words,
            engine_name="test_engine",
            processing_time=1.5,
            source="test.png"
        )

        assert len(result.words) == 2
        assert result.engine_name == "test_engine"
        assert result.processing_time == 1.5
        assert result.source == "test.png"

    def test_ocr_result_default_source(self):
        """Test OCRResult with default source"""
        result = OCRResult(
            words=[],
            engine_name="test_engine",
            processing_time=1.0
        )

        assert result.source == ""

    def test_confidence_stats_with_valid_confidences(self):
        """Test confidence statistics calculation with valid confidence values"""
        bbox = BoundingBox(left=0, top=0, width=50, height=20)
        words = [
            Word(text="word1", bbox=bbox, confidence=0.90),
            Word(text="word2", bbox=bbox, confidence=0.85),
            Word(text="word3", bbox=bbox, confidence=0.95),
        ]

        result = OCRResult(
            words=words,
            engine_name="test_engine",
            processing_time=1.0
        )

        stats = result.confidence_stats

        assert stats.available is True
        assert stats.mean == pytest.approx(0.90, abs=0.01)
        assert stats.std == pytest.approx(0.0408, abs=0.01)

    def test_confidence_stats_with_no_words(self):
        """Test confidence statistics when no words detected"""
        result = OCRResult(
            words=[],
            engine_name="test_engine",
            processing_time=1.0
        )

        stats = result.confidence_stats

        assert stats.available is False
        assert stats.mean == DEFAULT_CONFIDENCE
        assert stats.std == 0.0

    def test_confidence_stats_with_default_confidences(self):
        """Test confidence statistics when all words have DEFAULT_CONFIDENCE"""
        bbox = BoundingBox(left=0, top=0, width=50, height=20)
        words = [
            Word(text="word1", bbox=bbox),  # Uses DEFAULT_CONFIDENCE
            Word(text="word2", bbox=bbox),  # Uses DEFAULT_CONFIDENCE
        ]

        result = OCRResult(
            words=words,
            engine_name="test_engine",
            processing_time=1.0
        )

        stats = result.confidence_stats

        assert stats.available is False
        assert stats.mean == DEFAULT_CONFIDENCE
        assert stats.std == 0.0

    def test_confidence_stats_with_mixed_confidences(self):
        """Test confidence statistics with mix of valid and default confidences"""
        bbox = BoundingBox(left=0, top=0, width=50, height=20)
        words = [
            Word(text="word1", bbox=bbox, confidence=0.90),
            Word(text="word2", bbox=bbox),  # DEFAULT_CONFIDENCE
            Word(text="word3", bbox=bbox, confidence=0.85),
        ]

        result = OCRResult(
            words=words,
            engine_name="test_engine",
            processing_time=1.0
        )

        stats = result.confidence_stats

        # Should only calculate stats from valid confidences
        assert stats.available is True
        assert stats.mean == pytest.approx(0.875, abs=0.01)
        # Std of [0.90, 0.85] = 0.025
        assert stats.std == pytest.approx(0.025, abs=0.01)

    def test_confidence_stats_computation_uses_numpy(self):
        """Test that confidence statistics use numpy for calculation"""
        bbox = BoundingBox(left=0, top=0, width=50, height=20)
        words = [
            Word(text=f"word{i}", bbox=bbox, confidence=0.8 + i * 0.01)
            for i in range(10)
        ]

        result = OCRResult(
            words=words,
            engine_name="test_engine",
            processing_time=1.0
        )

        stats = result.confidence_stats

        # Calculate expected values using numpy
        confidences = [0.8 + i * 0.01 for i in range(10)]
        expected_mean = np.mean(confidences)
        expected_std = np.std(confidences)

        assert stats.available is True
        assert stats.mean == pytest.approx(expected_mean, abs=0.001)
        assert stats.std == pytest.approx(expected_std, abs=0.001)
