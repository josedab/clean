"""Detection module for Clean."""

from clean.detection.base import BaseDetector, DetectorResult
from clean.detection.bias import BiasDetector, analyze_bias
from clean.detection.duplicates import DuplicateDetector, find_duplicates
from clean.detection.factory import (
    ConfigurableDetectorFactory,
    DetectorFactory,
    DetectorFactoryProtocol,
)
from clean.detection.imbalance import ImbalanceDetector, analyze_imbalance
from clean.detection.label_errors import LabelErrorDetector, find_label_errors
from clean.detection.outliers import OutlierDetector, find_outliers

__all__ = [
    "BaseDetector",
    "BiasDetector",
    "ConfigurableDetectorFactory",
    "DetectorFactory",
    "DetectorFactoryProtocol",
    "DetectorResult",
    "DuplicateDetector",
    "ImbalanceDetector",
    "LabelErrorDetector",
    "OutlierDetector",
    "analyze_bias",
    "analyze_imbalance",
    "find_duplicates",
    "find_label_errors",
    "find_outliers",
]
