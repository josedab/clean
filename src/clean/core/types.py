"""Core type definitions for Clean."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class TaskType(Enum):
    """Type of ML task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_LABEL = "multi_label"


class DataType(Enum):
    """Type of data being processed."""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


class IssueType(Enum):
    """Type of data quality issue."""

    LABEL_ERROR = "label_error"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    OUTLIER = "outlier"
    CLASS_IMBALANCE = "class_imbalance"
    BIAS = "bias"


class IssueSeverity(Enum):
    """Severity level of a data quality issue."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OutlierRemovalStrategy(Enum):
    """Strategy for removing outliers."""

    CONSERVATIVE = "conservative"  # Only high-confidence outliers
    MODERATE = "moderate"  # Medium and high confidence
    AGGRESSIVE = "aggressive"  # All detected outliers


@dataclass
class LabelError:
    """Represents a detected label error."""

    index: int
    given_label: Any
    predicted_label: Any
    confidence: float
    self_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "given_label": self.given_label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "self_confidence": self.self_confidence,
        }


@dataclass
class DuplicatePair:
    """Represents a pair of duplicate samples."""

    index1: int
    index2: int
    similarity: float
    is_exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index1": self.index1,
            "index2": self.index2,
            "similarity": self.similarity,
            "is_exact": self.is_exact,
        }


@dataclass
class Outlier:
    """Represents a detected outlier."""

    index: int
    score: float
    method: str
    features_contributing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "score": self.score,
            "method": self.method,
            "features_contributing": self.features_contributing,
        }


@dataclass
class BiasIssue:
    """Represents a detected bias issue."""

    feature: str
    metric: str
    value: float
    threshold: float
    affected_groups: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "affected_groups": self.affected_groups,
            "description": self.description,
        }


@dataclass
class ClassDistribution:
    """Class distribution statistics."""

    class_counts: dict[Any, int]
    class_ratios: dict[Any, float]
    imbalance_ratio: float
    majority_class: Any
    minority_class: Any

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        # Convert numpy types in keys to Python native types
        def convert_key(k: Any) -> str | int | float:
            if isinstance(k, np.integer):
                return int(k)
            if isinstance(k, np.floating):
                return float(k)
            return k

        return {
            "class_counts": {convert_key(k): int(v) for k, v in self.class_counts.items()},
            "class_ratios": {convert_key(k): float(v) for k, v in self.class_ratios.items()},
            "imbalance_ratio": float(self.imbalance_ratio),
            "majority_class": convert_key(self.majority_class),
            "minority_class": convert_key(self.minority_class),
        }


@dataclass
class QualityScore:
    """Quality score for the dataset."""

    overall: float
    label_quality: float
    duplicate_quality: float
    outlier_quality: float
    imbalance_quality: float
    bias_quality: float
    per_class: dict[Any, float] = field(default_factory=dict)
    per_feature: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        # Convert numpy types in keys to Python native types
        def convert_key(k: Any) -> str | int | float:
            if isinstance(k, np.integer):
                return int(k)
            if isinstance(k, np.floating):
                return float(k)
            return k

        return {
            "overall": float(self.overall),
            "label_quality": float(self.label_quality),
            "duplicate_quality": float(self.duplicate_quality),
            "outlier_quality": float(self.outlier_quality),
            "imbalance_quality": float(self.imbalance_quality),
            "bias_quality": float(self.bias_quality),
            "per_class": {convert_key(k): float(v) for k, v in self.per_class.items()},
            "per_feature": {str(k): float(v) for k, v in self.per_feature.items()},
        }


@dataclass
class DatasetInfo:
    """Information about the dataset."""

    n_samples: int
    n_features: int
    n_classes: int | None
    feature_names: list[str]
    label_column: str | None
    data_type: DataType
    task_type: TaskType | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_names": self.feature_names,
            "label_column": self.label_column,
            "data_type": self.data_type.value if self.data_type else None,
            "task_type": self.task_type.value if self.task_type else None,
        }


def issues_to_dataframe(issues: list[LabelError | DuplicatePair | Outlier]) -> pd.DataFrame:
    """Convert a list of issues to a DataFrame."""
    if not issues:
        return pd.DataFrame()
    return pd.DataFrame([issue.to_dict() for issue in issues])


def numpy_to_list(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj


# Type hints use string annotations to avoid circular imports


@dataclass
class DetectionResults:
    """Container for all detection results from analysis.

    This parameter object aggregates results from all detectors,
    reducing the number of parameters needed in scoring and reporting functions.

    Example:
        >>> results = DetectionResults(
        ...     label=label_detector_result,
        ...     duplicate=dup_detector_result,
        ...     outlier=outlier_detector_result,
        ... )
        >>> score = scorer.compute_score(n_samples=1000, results=results)
    """

    label: Any | None = None  # DetectorResult | None
    duplicate: Any | None = None  # DetectorResult | None
    outlier: Any | None = None  # DetectorResult | None
    imbalance: Any | None = None  # DetectorResult | None
    bias: Any | None = None  # DetectorResult | None

    @property
    def has_any_issues(self) -> bool:
        """Check if any detector found issues."""
        results = [self.label, self.duplicate, self.outlier, self.imbalance, self.bias]
        return any(r is not None and r.n_issues > 0 for r in results if r is not None)

    @property
    def total_issues(self) -> int:
        """Get total number of issues across all detectors."""
        total = 0
        for result in [self.label, self.duplicate, self.outlier, self.imbalance, self.bias]:
            if result is not None:
                total += result.n_issues
        return total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "label": {
                "n_issues": self.label.n_issues if self.label else 0,
                "metadata": self.label.metadata if self.label else {},
            },
            "duplicate": {
                "n_issues": self.duplicate.n_issues if self.duplicate else 0,
                "metadata": self.duplicate.metadata if self.duplicate else {},
            },
            "outlier": {
                "n_issues": self.outlier.n_issues if self.outlier else 0,
                "metadata": self.outlier.metadata if self.outlier else {},
            },
            "imbalance": {
                "n_issues": self.imbalance.n_issues if self.imbalance else 0,
                "metadata": self.imbalance.metadata if self.imbalance else {},
            },
            "bias": {
                "n_issues": self.bias.n_issues if self.bias else 0,
                "metadata": self.bias.metadata if self.bias else {},
            },
        }
