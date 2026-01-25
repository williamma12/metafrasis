"""
Evaluation metrics for OCR models.

Provides base classes and concrete metrics for:
- Detectors: IoU, Precision/Recall/F1, mAP
- Recognizers: CER, WER, Accuracy

Example usage:
    from training.evaluate import CERMetric, IoUMetric

    cer = CERMetric()
    result = cer.compute(predictions=["hello"], targets=["helo"])
    print(f"CER: {result.value}")
"""

from ml.training.evaluate.base import Metric, MetricResult
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
from ml.training.evaluate.recognizers import (
    RecognizerMetric,
    CERMetric,
    NormalizedEditDistanceMetric,
    WERMetric,
    AccuracyMetric,
    SequenceAccuracyMetric,
)

__all__ = [
    # Base classes
    "Metric",
    "MetricResult",
    "DetectorMetric",
    "RecognizerMetric",
    # Detector metrics
    "IoUMetric",
    "PrecisionRecallF1Metric",
    "PrecisionMetric",
    "RecallMetric",
    "F1Metric",
    "MeanAPMetric",
    "APAt50Metric",
    "APAt75Metric",
    # Recognizer metrics
    "CERMetric",
    "NormalizedEditDistanceMetric",
    "WERMetric",
    "AccuracyMetric",
    "SequenceAccuracyMetric",
]
