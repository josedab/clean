"""Detector factory for dependency injection.

This module provides a factory pattern for creating detectors,
enabling easier testing and customization of the DatasetCleaner.

Example:
    >>> factory = DetectorFactory()
    >>> detector = factory.create_outlier_detector(method="isolation_forest")
    >>>
    >>> # Custom factory for testing
    >>> mock_factory = MockDetectorFactory()
    >>> cleaner = DatasetCleaner(data=df, detector_factory=mock_factory)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clean.detection.base import BaseDetector


class DetectorFactoryProtocol(ABC):
    """Protocol for detector factories.

    Implement this protocol to customize detector creation,
    enabling dependency injection for testing or specialized configurations.
    """

    @abstractmethod
    def create_label_error_detector(self, **kwargs: Any) -> BaseDetector:
        """Create a label error detector."""
        ...

    @abstractmethod
    def create_duplicate_detector(self, **kwargs: Any) -> BaseDetector:
        """Create a duplicate detector."""
        ...

    @abstractmethod
    def create_outlier_detector(self, **kwargs: Any) -> BaseDetector:
        """Create an outlier detector."""
        ...

    @abstractmethod
    def create_imbalance_detector(self, **kwargs: Any) -> BaseDetector:
        """Create an imbalance detector."""
        ...

    @abstractmethod
    def create_bias_detector(self, **kwargs: Any) -> BaseDetector:
        """Create a bias detector."""
        ...


class DetectorFactory(DetectorFactoryProtocol):
    """Default factory for creating detector instances.

    Creates standard detector instances with configurable parameters.
    Can be subclassed to customize detector creation or replaced
    entirely for testing purposes.

    Example:
        >>> factory = DetectorFactory()
        >>> outlier_detector = factory.create_outlier_detector(
        ...     method="isolation_forest",
        ...     contamination=0.05
        ... )
    """

    def __init__(self, default_config: dict[str, Any] | None = None):
        """Initialize the factory.

        Args:
            default_config: Default configuration applied to all detectors
        """
        self.default_config = default_config or {}

    def create_label_error_detector(self, **kwargs: Any) -> BaseDetector:
        """Create a label error detector.

        Args:
            cv_folds: Number of cross-validation folds
            confidence_threshold: Threshold for error confidence

        Returns:
            LabelErrorDetector instance
        """
        from clean.detection.label_errors import LabelErrorDetector

        config = {**self.default_config, **kwargs}
        return LabelErrorDetector(
            cv_folds=config.get("cv_folds", 5),
            confidence_threshold=config.get("confidence_threshold", 0.5),
        )

    def create_duplicate_detector(self, **kwargs: Any) -> BaseDetector:
        """Create a duplicate detector.

        Args:
            methods: Detection methods ('hash', 'fuzzy', 'embedding')
            similarity_threshold: Threshold for near-duplicates
            text_column: Column for text embedding

        Returns:
            DuplicateDetector instance
        """
        from clean.detection.duplicates import DuplicateDetector

        config = {**self.default_config, **kwargs}
        return DuplicateDetector(
            methods=config.get("methods", ["hash"]),
            similarity_threshold=config.get("similarity_threshold", 0.9),
            text_column=config.get("text_column"),
            hash_columns=config.get("hash_columns"),
        )

    def create_outlier_detector(self, **kwargs: Any) -> BaseDetector:
        """Create an outlier detector.

        Args:
            method: Detection method ('isolation_forest', 'lof', 'zscore', etc.)
            contamination: Expected proportion of outliers

        Returns:
            OutlierDetector instance
        """
        from clean.detection.outliers import OutlierDetector

        config = {**self.default_config, **kwargs}
        return OutlierDetector(
            method=config.get("method", "isolation_forest"),
            contamination=config.get("contamination", 0.1),
        )

    def create_imbalance_detector(self, **kwargs: Any) -> BaseDetector:
        """Create an imbalance detector.

        Args:
            imbalance_threshold: Ratio threshold for imbalance

        Returns:
            ImbalanceDetector instance
        """
        from clean.detection.imbalance import ImbalanceDetector

        config = {**self.default_config, **kwargs}
        return ImbalanceDetector(
            imbalance_threshold=config.get("imbalance_threshold", 5.0),
        )

    def create_bias_detector(self, **kwargs: Any) -> BaseDetector:
        """Create a bias detector.

        Args:
            sensitive_features: Features to check for bias

        Returns:
            BiasDetector instance
        """
        from clean.detection.bias import BiasDetector

        config = {**self.default_config, **kwargs}
        return BiasDetector(
            sensitive_features=config.get("sensitive_features"),
        )


class ConfigurableDetectorFactory(DetectorFactory):
    """Factory with pre-configured detector settings.

    Allows setting default configurations for specific use cases
    like high-precision detection or fast scanning.

    Example:
        >>> # Factory optimized for high precision
        >>> factory = ConfigurableDetectorFactory.high_precision()
        >>> cleaner = DatasetCleaner(data=df, detector_factory=factory)
    """

    @classmethod
    def high_precision(cls) -> ConfigurableDetectorFactory:
        """Create factory optimized for high precision (fewer false positives)."""
        return cls(
            default_config={
                "cv_folds": 10,
                "confidence_threshold": 0.8,
                "contamination": 0.05,
                "similarity_threshold": 0.95,
            }
        )

    @classmethod
    def high_recall(cls) -> ConfigurableDetectorFactory:
        """Create factory optimized for high recall (fewer false negatives)."""
        return cls(
            default_config={
                "cv_folds": 3,
                "confidence_threshold": 0.3,
                "contamination": 0.15,
                "similarity_threshold": 0.85,
            }
        )

    @classmethod
    def fast_scan(cls) -> ConfigurableDetectorFactory:
        """Create factory optimized for fast scanning."""
        return cls(
            default_config={
                "cv_folds": 3,
                "methods": ["hash"],  # Fast hash-only duplicate detection
                "method": "zscore",  # Fast outlier detection
            }
        )


# Default factory instance
default_factory = DetectorFactory()
