"""Main DatasetCleaner class for comprehensive data quality analysis.

The DatasetCleaner is the primary entry point for analyzing ML datasets.
It orchestrates detection, scoring, and reporting of data quality issues.

Example:
    >>> from clean import DatasetCleaner
    >>>
    >>> # Initialize with DataFrame
    >>> cleaner = DatasetCleaner(data=df, label_column='target')
    >>>
    >>> # Run analysis
    >>> report = cleaner.analyze()
    >>>
    >>> # View results
    >>> print(report.summary())
    >>> errors = report.label_errors()
    >>>
    >>> # Get clean data
    >>> clean_df = cleaner.get_clean_data(remove_duplicates=True)

See Also:
    - :class:`QualityReport`: Analysis results container
    - :class:`FixEngine`: Auto-fix suggestions and application
    - :class:`DetectorFactory`: Customizable detector creation
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from clean.core.report import QualityReport
from clean.core.types import (
    DatasetInfo,
    DataType,
    TaskType,
)
from clean.detection.base import BaseDetector, DetectorResult
from clean.detection.factory import DetectorFactory, DetectorFactoryProtocol
from clean.loaders.base import BaseLoader
from clean.loaders.numpy_loader import NumpyLoader
from clean.loaders.pandas_loader import PandasLoader
from clean.scoring.quality_scorer import QualityScorer


class DatasetCleaner:
    """Main interface for data quality analysis and cleaning.

    Provides a comprehensive workflow for detecting and fixing data quality
    issues including label errors, duplicates, outliers, and bias.

    Supports dependency injection via the detector_factory parameter,
    enabling customization and easier testing.
    """

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray | None = None,
        labels: np.ndarray | None = None,
        label_column: str | None = None,
        task: str | TaskType = "classification",
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        loader: BaseLoader | None = None,
        detector_factory: DetectorFactoryProtocol | None = None,
    ):
        """Initialize the DatasetCleaner.

        Args:
            data: Input data (DataFrame or array)
            labels: Label array (if not in data)
            label_column: Name of label column in DataFrame
            task: ML task type ('classification' or 'regression')
            feature_columns: Columns to use as features
            exclude_columns: Columns to exclude from features
            loader: Custom data loader
            detector_factory: Factory for creating detectors (enables DI)
        """
        self.label_column = label_column
        self.task_type = TaskType(task) if isinstance(task, str) else task
        self._detector_factory = detector_factory or DetectorFactory()

        # Initialize data
        if loader is not None:
            self._loader = loader
            self._features, self._labels = loader.load()
            self._info = loader.get_info()
        elif isinstance(data, pd.DataFrame):
            self._loader = PandasLoader(
                data,
                label_column=label_column,
                feature_columns=feature_columns,
                exclude_columns=exclude_columns,
                task_type=self.task_type,
            )
            self._features, self._labels = self._loader.load()
            self._info = self._loader.get_info()
        elif isinstance(data, np.ndarray):
            self._loader = NumpyLoader(data, labels=labels, task_type=self.task_type)
            self._features, self._labels = self._loader.load()
            self._info = self._loader.get_info()
        else:
            raise ValueError("Must provide data (DataFrame/array) or loader")

        # Override labels if provided separately
        if labels is not None:
            self._labels = np.asarray(labels)

        # Detector instances (created via factory)
        self._label_detector: BaseDetector | None = None
        self._duplicate_detector: BaseDetector | None = None
        self._outlier_detector: BaseDetector | None = None
        self._imbalance_detector: BaseDetector | None = None
        self._bias_detector: BaseDetector | None = None

        # Results
        self._label_result: DetectorResult | None = None
        self._duplicate_result: DetectorResult | None = None
        self._outlier_result: DetectorResult | None = None
        self._imbalance_result: DetectorResult | None = None
        self._bias_result: DetectorResult | None = None

        # Report
        self._report: QualityReport | None = None

    @property
    def features(self) -> pd.DataFrame:
        """Get feature DataFrame."""
        return self._features

    @property
    def labels(self) -> np.ndarray | None:
        """Get label array."""
        return self._labels

    @property
    def info(self) -> DatasetInfo:
        """Get dataset info."""
        return self._info

    def analyze(
        self,
        detect_label_errors: bool = True,
        detect_duplicates: bool = True,
        detect_outliers: bool = True,
        detect_imbalance: bool = True,
        detect_bias: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> QualityReport:
        """Run comprehensive data quality analysis.

        Args:
            detect_label_errors: Whether to detect label errors
            detect_duplicates: Whether to detect duplicates
            detect_outliers: Whether to detect outliers
            detect_imbalance: Whether to detect class imbalance
            detect_bias: Whether to detect bias
            show_progress: Show progress bar
            **kwargs: Additional arguments for detectors

        Returns:
            QualityReport with all findings
        """
        steps = []
        if detect_label_errors and self._labels is not None:
            steps.append(("Label errors", self._detect_label_errors))
        if detect_duplicates:
            steps.append(("Duplicates", self._detect_duplicates))
        if detect_outliers:
            steps.append(("Outliers", self._detect_outliers))
        if detect_imbalance and self._labels is not None:
            steps.append(("Imbalance", self._detect_imbalance))
        if detect_bias and self._labels is not None:
            steps.append(("Bias", self._detect_bias))

        iterator = tqdm(steps, desc="Analyzing", disable=not show_progress)
        for name, func in iterator:
            iterator.set_description(f"Detecting {name}")
            try:
                func(**kwargs)
            except Exception as e:
                logging.warning("%s detection failed: %s", name, e)

        # Compute quality scores
        scorer = QualityScorer()
        quality_score = scorer.compute_score(
            n_samples=len(self._features),
            label_result=self._label_result,
            duplicate_result=self._duplicate_result,
            outlier_result=self._outlier_result,
            imbalance_result=self._imbalance_result,
            bias_result=self._bias_result,
            labels=self._labels,
            features=self._features,
        )

        # Get class distribution
        class_dist = None
        if self._imbalance_detector:
            class_dist = self._imbalance_detector.get_distribution()

        # Create report
        self._report = QualityReport(
            dataset_info=self._info,
            quality_score=quality_score,
            label_errors_result=self._label_result,
            duplicates_result=self._duplicate_result,
            outliers_result=self._outlier_result,
            imbalance_result=self._imbalance_result,
            bias_result=self._bias_result,
            class_distribution=class_dist,
        )

        return self._report

    def _detect_label_errors(self, **kwargs: Any) -> None:
        """Run label error detection."""
        if self._labels is None:
            return

        self._label_detector = self._detector_factory.create_label_error_detector(
            cv_folds=kwargs.get("cv_folds", 5),
            confidence_threshold=kwargs.get("label_confidence_threshold", 0.5),
        )
        self._label_result = self._label_detector.fit_detect(self._features, self._labels)

    def _detect_duplicates(self, **kwargs: Any) -> None:
        """Run duplicate detection."""
        methods = kwargs.get("duplicate_methods", ["hash"])
        if self._info.data_type == DataType.TEXT:
            methods = ["hash", "embedding"]

        self._duplicate_detector = self._detector_factory.create_duplicate_detector(
            methods=methods,
            similarity_threshold=kwargs.get("similarity_threshold", 0.9),
            text_column=kwargs.get("text_column"),
        )
        self._duplicate_result = self._duplicate_detector.fit_detect(self._features)

    def _detect_outliers(self, **kwargs: Any) -> None:
        """Run outlier detection."""
        self._outlier_detector = self._detector_factory.create_outlier_detector(
            method=kwargs.get("outlier_method", "isolation_forest"),
            contamination=kwargs.get("contamination", 0.1),
        )
        self._outlier_result = self._outlier_detector.fit_detect(self._features)

    def _detect_imbalance(self, **kwargs: Any) -> None:
        """Run class imbalance detection."""
        if self._labels is None:
            return

        self._imbalance_detector = self._detector_factory.create_imbalance_detector(
            imbalance_threshold=kwargs.get("imbalance_threshold", 5.0),
        )
        self._imbalance_result = self._imbalance_detector.fit_detect(
            self._features, self._labels
        )

    def _detect_bias(self, **kwargs: Any) -> None:
        """Run bias detection."""
        if self._labels is None:
            return

        self._bias_detector = self._detector_factory.create_bias_detector(
            sensitive_features=kwargs.get("sensitive_features"),
        )
        self._bias_result = self._bias_detector.fit_detect(self._features, self._labels)

    def get_clean_data(
        self,
        remove_duplicates: bool = True,
        remove_outliers: bool | str = False,
        remove_label_errors: bool = False,
        keep_first_duplicate: bool = True,
    ) -> pd.DataFrame:
        """Get cleaned dataset with issues removed.

        Args:
            remove_duplicates: Remove duplicate samples
            remove_outliers: Remove outliers (False, 'conservative', 'aggressive')
            remove_label_errors: Remove samples with label errors
            keep_first_duplicate: Keep first sample in duplicate groups

        Returns:
            Cleaned DataFrame
        """
        indices_to_remove: set[int] = set()

        # Remove duplicates
        if remove_duplicates and self._duplicate_result:
            for dup in self._duplicate_result.issues:
                # Keep first, remove second
                if keep_first_duplicate:
                    indices_to_remove.add(dup.index2)
                else:
                    indices_to_remove.add(dup.index1)
                    indices_to_remove.add(dup.index2)

        # Remove outliers
        if remove_outliers and self._outlier_result:
            threshold = 0.0
            if remove_outliers == "conservative":
                # Only remove high-confidence outliers
                scores = [o.score for o in self._outlier_result.issues]
                if scores:
                    threshold = np.percentile(scores, 75)
            elif remove_outliers == "aggressive":
                threshold = 0.0

            for outlier in self._outlier_result.issues:
                if outlier.score >= threshold:
                    indices_to_remove.add(outlier.index)

        # Remove label errors
        if remove_label_errors and self._label_result:
            for error in self._label_result.issues:
                indices_to_remove.add(error.index)

        # Create clean DataFrame
        mask = ~self._features.index.isin(indices_to_remove)
        clean_df = self._features[mask].copy()

        # Add labels if available
        if self._labels is not None and self.label_column:
            clean_labels = self._labels[mask]
            clean_df[self.label_column] = clean_labels

        return clean_df.reset_index(drop=True)

    def get_review_queue(
        self,
        max_items: int = 100,
        include_label_errors: bool = True,
        include_outliers: bool = True,
        include_duplicates: bool = False,
    ) -> pd.DataFrame:
        """Get prioritized queue of samples for manual review.

        Args:
            max_items: Maximum items to return
            include_label_errors: Include label errors
            include_outliers: Include outliers
            include_duplicates: Include duplicates

        Returns:
            DataFrame with samples to review, sorted by priority
        """
        items: list[dict[str, Any]] = []

        if include_label_errors and self._label_result:
            items.extend(
                {
                    "index": error.index,
                    "issue_type": "label_error",
                    "priority_score": error.confidence,
                    "given_label": error.given_label,
                    "suggested_label": error.predicted_label,
                    "confidence": error.confidence,
                }
                for error in self._label_result.issues
            )

        if include_outliers and self._outlier_result:
            items.extend(
                {
                    "index": outlier.index,
                    "issue_type": "outlier",
                    "priority_score": outlier.score * 0.5,  # Lower priority than label errors
                    "method": outlier.method,
                    "score": outlier.score,
                }
                for outlier in self._outlier_result.issues
            )

        if include_duplicates and self._duplicate_result:
            items.extend(
                {
                    "index": dup.index1,
                    "issue_type": "duplicate",
                    "priority_score": dup.similarity * 0.3,
                    "duplicate_of": dup.index2,
                    "similarity": dup.similarity,
                }
                for dup in self._duplicate_result.issues
                if not dup.is_exact  # Exact duplicates don't need review
            )

        # Sort by priority and limit
        items.sort(key=lambda x: x["priority_score"], reverse=True)
        items = items[:max_items]

        return pd.DataFrame(items)

    def relabel(
        self,
        apply_suggestions: bool = False,
        confidence_threshold: float = 0.9,
    ) -> pd.DataFrame:
        """Get relabeling suggestions or apply them.

        Args:
            apply_suggestions: Actually change labels (if False, just returns suggestions)
            confidence_threshold: Minimum confidence to apply automatic relabeling

        Returns:
            DataFrame with relabeling information
        """
        if self._label_result is None or self._labels is None:
            return pd.DataFrame()

        suggestions = []
        new_labels = self._labels.copy()

        for error in self._label_result.issues:
            suggestion = {
                "index": error.index,
                "current_label": error.given_label,
                "suggested_label": error.predicted_label,
                "confidence": error.confidence,
                "will_apply": error.confidence >= confidence_threshold and apply_suggestions,
            }
            suggestions.append(suggestion)

            if apply_suggestions and error.confidence >= confidence_threshold:
                new_labels[error.index] = error.predicted_label

        if apply_suggestions:
            self._labels = new_labels

        return pd.DataFrame(suggestions)

    def get_report(self) -> QualityReport | None:
        """Get the quality report from last analysis.

        Returns:
            QualityReport or None if analyze() not called
        """
        return self._report

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DatasetCleaner(samples={self._info.n_samples}, "
            f"features={self._info.n_features}, "
            f"task={self._info.task_type.value if self._info.task_type else 'unknown'})"
        )
