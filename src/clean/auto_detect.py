"""Zero-Config Auto-Detection for Clean.

This module provides automatic inference of data modality, label columns,
and optimal detection settings from data inspection.

Example:
    >>> from clean.auto_detect import AutoDetector, auto_analyze
    >>>
    >>> # Automatic analysis with zero configuration
    >>> report = auto_analyze(df)  # No label_column needed!
    >>>
    >>> # Or use the detector directly
    >>> detector = AutoDetector()
    >>> config = detector.detect(df)
    >>> print(f"Label column: {config.label_column}")
    >>> print(f"Modality: {config.modality}")
    >>> print(f"Task type: {config.task_type}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from clean.core.types import DataType
from clean.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class DataModality(Enum):
    """Detected data modality."""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE_PATHS = "image_paths"
    EMBEDDINGS = "embeddings"
    TIME_SERIES = "time_series"
    MIXED = "mixed"


class TaskType(Enum):
    """Detected task type."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    dtype: str
    n_unique: int
    n_missing: int
    n_total: int
    sample_values: list[Any]
    is_numeric: bool
    is_categorical: bool
    is_text: bool
    is_datetime: bool
    is_path: bool
    is_embedding: bool
    avg_text_length: float | None = None
    label_score: float = 0.0  # How likely this is a label column


@dataclass
class AutoConfig:
    """Auto-detected configuration."""

    label_column: str | None
    label_confidence: float
    feature_columns: list[str]
    modality: DataModality
    task_type: TaskType
    data_type: DataType
    text_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    embedding_columns: list[str]
    path_columns: list[str]
    suggested_detectors: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label_column": self.label_column,
            "label_confidence": self.label_confidence,
            "feature_columns": self.feature_columns,
            "modality": self.modality.value,
            "task_type": self.task_type.value,
            "data_type": self.data_type.value,
            "text_columns": self.text_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "embedding_columns": self.embedding_columns,
            "path_columns": self.path_columns,
            "suggested_detectors": self.suggested_detectors,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Auto-Detected Configuration",
            "=" * 40,
            f"Modality: {self.modality.value}",
            f"Task Type: {self.task_type.value}",
            f"Data Type: {self.data_type.value}",
            "",
        ]

        if self.label_column:
            lines.append(f"Label Column: {self.label_column} (confidence: {self.label_confidence:.0%})")
        else:
            lines.append("Label Column: Not detected (unsupervised mode)")

        lines.extend([
            f"Feature Columns: {len(self.feature_columns)}",
            f"  Numeric: {len(self.numeric_columns)}",
            f"  Categorical: {len(self.categorical_columns)}",
            f"  Text: {len(self.text_columns)}",
        ])

        if self.warnings:
            lines.extend(["", "⚠️  Warnings:"])
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.extend(["", "Suggested Detectors:"])
        for det in self.suggested_detectors:
            lines.append(f"  ✓ {det}")

        return "\n".join(lines)


class AutoDetector:
    """Automatic data configuration detector."""

    # Common label column names
    LABEL_COLUMN_PATTERNS = [
        r"^label$",
        r"^labels$",
        r"^target$",
        r"^class$",
        r"^y$",
        r"^output$",
        r"^category$",
        r"^sentiment$",
        r".*_label$",
        r".*_target$",
        r".*_class$",
        r"^is_.*",
        r"^has_.*",
    ]

    # Common text column names
    TEXT_COLUMN_PATTERNS = [
        r".*text.*",
        r".*content.*",
        r".*description.*",
        r".*body.*",
        r".*message.*",
        r".*comment.*",
        r".*review.*",
        r".*title.*",
        r".*instruction.*",
        r".*prompt.*",
        r".*response.*",
        r".*question.*",
        r".*answer.*",
    ]

    # Common path column names
    PATH_COLUMN_PATTERNS = [
        r".*path.*",
        r".*file.*",
        r".*image.*",
        r".*url.*",
        r".*uri.*",
    ]

    def __init__(
        self,
        sample_size: int = 1000,
        text_length_threshold: int = 50,
        categorical_threshold: float = 0.05,
    ):
        """Initialize auto-detector.

        Args:
            sample_size: Number of rows to sample for detection
            text_length_threshold: Min avg length to consider a column as text
            categorical_threshold: Max unique ratio to consider categorical
        """
        self.sample_size = sample_size
        self.text_length_threshold = text_length_threshold
        self.categorical_threshold = categorical_threshold

    def detect(self, data: pd.DataFrame) -> AutoConfig:
        """Detect optimal configuration for the dataset.

        Args:
            data: Input DataFrame

        Returns:
            Auto-detected configuration
        """
        # Sample if large
        if len(data) > self.sample_size:
            sample = data.sample(n=self.sample_size, random_state=42)
        else:
            sample = data

        # Profile columns
        profiles = [self._profile_column(sample, col) for col in data.columns]

        # Detect label column
        label_col, label_confidence = self._detect_label_column(profiles, sample)

        # Categorize columns
        text_cols = [p.name for p in profiles if p.is_text]
        numeric_cols = [p.name for p in profiles if p.is_numeric]
        categorical_cols = [p.name for p in profiles if p.is_categorical and not p.is_text]
        datetime_cols = [p.name for p in profiles if p.is_datetime]
        embedding_cols = [p.name for p in profiles if p.is_embedding]
        path_cols = [p.name for p in profiles if p.is_path]

        # Feature columns (exclude label)
        feature_cols = [p.name for p in profiles if p.name != label_col]

        # Detect modality
        modality = self._detect_modality(profiles, text_cols, path_cols, embedding_cols)

        # Detect task type
        task_type = self._detect_task_type(label_col, sample)

        # Detect data type
        data_type = self._detect_data_type(modality, text_cols, numeric_cols)

        # Suggest detectors
        detectors = self._suggest_detectors(
            modality, task_type, label_col is not None, text_cols
        )

        # Collect warnings
        warnings = self._collect_warnings(
            data, profiles, label_col, label_confidence, modality
        )

        return AutoConfig(
            label_column=label_col,
            label_confidence=label_confidence,
            feature_columns=feature_cols,
            modality=modality,
            task_type=task_type,
            data_type=data_type,
            text_columns=text_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            embedding_columns=embedding_cols,
            path_columns=path_cols,
            suggested_detectors=detectors,
            warnings=warnings,
        )

    def _profile_column(self, data: pd.DataFrame, column: str) -> ColumnProfile:
        """Profile a single column."""
        col_data = data[column]
        dtype = str(col_data.dtype)
        n_total = len(col_data)
        n_missing = col_data.isna().sum()
        n_unique = col_data.nunique()

        # Sample values
        non_null = col_data.dropna()
        sample_values = non_null.head(5).tolist() if len(non_null) > 0 else []

        # Detect types
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        is_datetime = pd.api.types.is_datetime64_any_dtype(col_data)

        # Check for text
        is_text = False
        avg_text_length = None
        if col_data.dtype == "object" and len(non_null) > 0:
            # Check if string and long enough
            str_vals = non_null.astype(str)
            avg_text_length = str_vals.str.len().mean()
            is_text = (
                avg_text_length > self.text_length_threshold
                or self._matches_patterns(column, self.TEXT_COLUMN_PATTERNS)
            )

        # Check for categorical
        is_categorical = (
            not is_numeric
            and not is_datetime
            and not is_text
            and n_unique / max(n_total - n_missing, 1) <= self.categorical_threshold
        )

        # Check for paths
        is_path = False
        if col_data.dtype == "object" and len(non_null) > 0:
            is_path = self._matches_patterns(column, self.PATH_COLUMN_PATTERNS)
            if not is_path:
                # Check content for path-like strings
                sample_str = str(non_null.iloc[0])
                is_path = (
                    sample_str.endswith((".jpg", ".png", ".jpeg", ".gif", ".bmp"))
                    or "/" in sample_str
                    or "\\" in sample_str
                )

        # Check for embeddings (list of floats)
        is_embedding = False
        if col_data.dtype == "object" and len(non_null) > 0:
            first_val = non_null.iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                is_embedding = all(isinstance(x, (int, float)) for x in first_val[:10])

        # Calculate label score
        label_score = self._calculate_label_score(column, n_unique, n_total, is_categorical)

        return ColumnProfile(
            name=column,
            dtype=dtype,
            n_unique=n_unique,
            n_missing=n_missing,
            n_total=n_total,
            sample_values=sample_values,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_text=is_text,
            is_datetime=is_datetime,
            is_path=is_path,
            is_embedding=is_embedding,
            avg_text_length=avg_text_length,
            label_score=label_score,
        )

    def _matches_patterns(self, name: str, patterns: list[str]) -> bool:
        """Check if column name matches any pattern."""
        name_lower = name.lower()
        return any(re.match(p, name_lower) for p in patterns)

    def _calculate_label_score(
        self, column: str, n_unique: int, n_total: int, is_categorical: bool
    ) -> float:
        """Calculate how likely a column is a label column."""
        score = 0.0

        # Name matching
        if self._matches_patterns(column, self.LABEL_COLUMN_PATTERNS):
            score += 0.5

        # Categorical with reasonable cardinality
        if is_categorical and 2 <= n_unique <= 100:
            score += 0.3

        # Binary columns are often labels
        if n_unique == 2:
            score += 0.2

        # Very low cardinality relative to size
        if n_unique < 20 and n_total > 100:
            score += 0.1

        return min(score, 1.0)

    def _detect_label_column(
        self, profiles: list[ColumnProfile], data: pd.DataFrame
    ) -> tuple[str | None, float]:
        """Detect the most likely label column."""
        # Sort by label score
        candidates = sorted(profiles, key=lambda p: p.label_score, reverse=True)

        for profile in candidates:
            if profile.label_score > 0.3:
                return profile.name, profile.label_score

        # No clear label column found
        return None, 0.0

    def _detect_modality(
        self,
        profiles: list[ColumnProfile],
        text_cols: list[str],
        path_cols: list[str],
        embedding_cols: list[str],
    ) -> DataModality:
        """Detect the primary data modality."""
        if embedding_cols:
            return DataModality.EMBEDDINGS

        if path_cols:
            return DataModality.IMAGE_PATHS

        text_ratio = len(text_cols) / max(len(profiles), 1)
        if text_ratio > 0.3:
            return DataModality.TEXT

        # Check for time series
        datetime_cols = [p for p in profiles if p.is_datetime]
        numeric_cols = [p for p in profiles if p.is_numeric]
        if datetime_cols and numeric_cols:
            # Could be time series
            pass

        if text_cols and len([p for p in profiles if p.is_numeric]) > 0:
            return DataModality.MIXED

        return DataModality.TABULAR

    def _detect_task_type(
        self, label_column: str | None, data: pd.DataFrame
    ) -> TaskType:
        """Detect the task type based on label column."""
        if label_column is None:
            return TaskType.UNSUPERVISED

        label_data = data[label_column].dropna()
        n_unique = label_data.nunique()

        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION

        if n_unique <= 20:
            return TaskType.MULTICLASS_CLASSIFICATION

        if pd.api.types.is_numeric_dtype(label_data):
            return TaskType.REGRESSION

        return TaskType.MULTICLASS_CLASSIFICATION

    def _detect_data_type(
        self, modality: DataModality, text_cols: list[str], numeric_cols: list[str]
    ) -> DataType:
        """Detect the data type."""
        if modality == DataModality.TEXT:
            return DataType.TEXT

        if modality == DataModality.IMAGE_PATHS:
            return DataType.IMAGE

        if modality == DataModality.EMBEDDINGS:
            return DataType.EMBEDDING

        if text_cols and numeric_cols:
            return DataType.MIXED

        return DataType.TABULAR

    def _suggest_detectors(
        self,
        modality: DataModality,
        task_type: TaskType,
        has_labels: bool,
        text_cols: list[str],
    ) -> list[str]:
        """Suggest appropriate detectors."""
        detectors = []

        # Always suggest duplicates and outliers
        detectors.append("DuplicateDetector")
        detectors.append("OutlierDetector")

        # Label errors only if supervised
        if has_labels and task_type != TaskType.UNSUPERVISED:
            detectors.append("LabelErrorDetector")

        # Imbalance for classification
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            detectors.append("ImbalanceDetector")

        # Bias detection
        detectors.append("BiasDetector")

        # Text-specific
        if text_cols or modality == DataModality.TEXT:
            detectors.append("LLMEvaluator")

        return detectors

    def _collect_warnings(
        self,
        data: pd.DataFrame,
        profiles: list[ColumnProfile],
        label_column: str | None,
        label_confidence: float,
        modality: DataModality,
    ) -> list[str]:
        """Collect warnings about the data."""
        warnings = []

        # Low label confidence
        if label_column and label_confidence < 0.5:
            warnings.append(
                f"Label column '{label_column}' detected with low confidence ({label_confidence:.0%}). "
                "Consider specifying explicitly."
            )

        # No label column
        if label_column is None:
            warnings.append(
                "No label column detected. Running in unsupervised mode. "
                "Label error detection will be skipped."
            )

        # High missing rate columns
        for p in profiles:
            missing_rate = p.n_missing / p.n_total
            if missing_rate > 0.2:
                warnings.append(
                    f"Column '{p.name}' has {missing_rate:.0%} missing values."
                )

        # Small dataset
        if len(data) < 100:
            warnings.append(
                f"Small dataset ({len(data)} rows). Some detectors may be unreliable."
            )

        return warnings


def auto_analyze(
    data: pd.DataFrame,
    label_column: str | None = None,
    **kwargs: Any,
) -> Any:
    """Analyze data with automatic configuration.

    Args:
        data: Input DataFrame
        label_column: Optional label column (auto-detected if not provided)
        **kwargs: Additional arguments passed to DatasetCleaner

    Returns:
        QualityReport from analysis

    Example:
        >>> report = auto_analyze(df)  # Zero config!
        >>> print(report.summary())
    """
    from clean.core.cleaner import DatasetCleaner

    # Auto-detect configuration
    detector = AutoDetector()
    config = detector.detect(data)

    logger.info(f"Auto-detected configuration:\n{config.summary()}")

    # Use provided label column or auto-detected
    final_label_column = label_column or config.label_column

    # Map our TaskType to core module's TaskType
    task_mapping = {
        TaskType.BINARY_CLASSIFICATION: "classification",
        TaskType.MULTICLASS_CLASSIFICATION: "classification",
        TaskType.REGRESSION: "regression",
        TaskType.UNSUPERVISED: "classification",  # default
    }
    task = task_mapping.get(config.task_type, "classification")

    # Create cleaner with detected settings
    cleaner = DatasetCleaner(
        data=data,
        label_column=final_label_column,
        task=task if final_label_column else "classification",
        **kwargs,
    )

    return cleaner.analyze()


def detect_config(data: pd.DataFrame) -> AutoConfig:
    """Detect optimal configuration for a dataset.

    Args:
        data: Input DataFrame

    Returns:
        Auto-detected configuration

    Example:
        >>> config = detect_config(df)
        >>> print(f"Detected label: {config.label_column}")
        >>> print(f"Modality: {config.modality}")
    """
    detector = AutoDetector()
    return detector.detect(data)


__all__ = [
    "DataModality",
    "ColumnProfile",
    "AutoConfig",
    "AutoDetector",
    "auto_analyze",
    "detect_config",
]
