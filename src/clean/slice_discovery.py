"""Data Slice Discovery - Find underperforming data subgroups.

This module automatically discovers data slices (subgroups) where
model performance or data quality is significantly worse than average.

Example:
    >>> from clean.slice_discovery import SliceDiscoverer
    >>>
    >>> discoverer = SliceDiscoverer()
    >>> slices = discoverer.discover(
    ...     X=df,
    ...     y=labels,
    ...     predictions=model.predict(df),
    ... )
    >>> print(slices.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    pass


class SliceType(Enum):
    """Types of slice definitions."""

    SINGLE_FEATURE = "single_feature"  # One feature condition
    CONJUNCTION = "conjunction"  # AND of multiple conditions
    DISJUNCTION = "disjunction"  # OR of multiple conditions
    RANGE = "range"  # Numeric range
    CLUSTER = "cluster"  # Clustering-based


class IssueType(Enum):
    """Types of issues found in slices."""

    LOW_ACCURACY = "low_accuracy"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_CONFUSION = "high_confusion"
    DISTRIBUTION_SHIFT = "distribution_shift"
    UNDERREPRESENTED = "underrepresented"


@dataclass
class SliceCondition:
    """A condition defining a slice."""

    feature: str
    operator: str  # "==", "!=", "<", ">", "<=", ">=", "in", "between"
    value: Any
    value_end: Any | None = None  # For "between" operator

    def __str__(self) -> str:
        if self.operator == "between":
            return f"{self.feature} ∈ [{self.value}, {self.value_end}]"
        elif self.operator == "in":
            return f"{self.feature} ∈ {self.value}"
        else:
            return f"{self.feature} {self.operator} {self.value}"

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        """Evaluate condition on DataFrame.

        Args:
            df: DataFrame to evaluate

        Returns:
            Boolean mask of matching rows
        """
        if self.feature not in df.columns:
            return np.zeros(len(df), dtype=bool)

        col = df[self.feature]

        if self.operator == "==":
            return (col == self.value).values
        elif self.operator == "!=":
            return (col != self.value).values
        elif self.operator == "<":
            return (col < self.value).values
        elif self.operator == "<=":
            return (col <= self.value).values
        elif self.operator == ">":
            return (col > self.value).values
        elif self.operator == ">=":
            return (col >= self.value).values
        elif self.operator == "in":
            return col.isin(self.value).values
        elif self.operator == "between":
            return ((col >= self.value) & (col <= self.value_end)).values
        else:
            return np.zeros(len(df), dtype=bool)


@dataclass
class DataSlice:
    """A discovered data slice with performance issues."""

    slice_id: str
    conditions: list[SliceCondition]
    slice_type: SliceType

    # Statistics
    n_samples: int
    n_total: int  # Total samples in dataset
    support: float  # Fraction of data in this slice

    # Performance metrics
    accuracy: float
    baseline_accuracy: float
    accuracy_gap: float  # baseline - slice accuracy
    error_rate: float
    avg_confidence: float

    # Issue characterization
    issue_types: list[IssueType]
    severity_score: float  # 0-1, higher = more severe

    # Sample information
    sample_indices: np.ndarray

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        conditions_str = " AND ".join(str(c) for c in self.conditions)
        return f"Slice[{conditions_str}]: n={self.n_samples}, acc={self.accuracy:.1%}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slice_id": self.slice_id,
            "conditions": [str(c) for c in self.conditions],
            "slice_type": self.slice_type.value,
            "n_samples": self.n_samples,
            "support": self.support,
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "accuracy_gap": self.accuracy_gap,
            "error_rate": self.error_rate,
            "issue_types": [i.value for i in self.issue_types],
            "severity_score": self.severity_score,
        }


@dataclass
class SliceDiscoveryReport:
    """Report from slice discovery analysis."""

    n_samples: int
    n_slices_found: int
    slices: list[DataSlice]
    baseline_accuracy: float
    worst_slice_accuracy: float
    total_affected_samples: int
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Data Slice Discovery Report",
            "=" * 50,
            "",
            f"Dataset size: {self.n_samples:,}",
            f"Slices discovered: {self.n_slices_found}",
            f"Baseline accuracy: {self.baseline_accuracy:.1%}",
            f"Worst slice accuracy: {self.worst_slice_accuracy:.1%}",
            f"Total affected samples: {self.total_affected_samples:,}",
            "",
        ]

        if self.slices:
            lines.append("Top Problem Slices:")
            lines.append("-" * 40)
            for i, slice_ in enumerate(self.slices[:5], 1):
                conditions = " AND ".join(str(c) for c in slice_.conditions)
                lines.append(f"{i}. {conditions}")
                lines.append(f"   Samples: {slice_.n_samples:,} ({slice_.support:.1%})")
                lines.append(f"   Accuracy: {slice_.accuracy:.1%} (gap: {slice_.accuracy_gap:.1%})")
                lines.append(f"   Issues: {', '.join(i.value for i in slice_.issue_types)}")
                lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            lines.append("-" * 40)
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_slices_found": self.n_slices_found,
            "slices": [s.to_dict() for s in self.slices],
            "baseline_accuracy": self.baseline_accuracy,
            "recommendations": self.recommendations,
        }

    def get_slice_samples(self, slice_id: str) -> np.ndarray | None:
        """Get sample indices for a specific slice."""
        for slice_ in self.slices:
            if slice_.slice_id == slice_id:
                return slice_.sample_indices
        return None


class SliceDiscoverer:
    """Discover underperforming data slices.

    Uses automatic feature binning and combinatorial search to find
    data subgroups with significantly worse performance.
    """

    def __init__(
        self,
        min_slice_size: int = 50,
        min_support: float = 0.01,
        max_conditions: int = 3,
        significance_level: float = 0.05,
        accuracy_gap_threshold: float = 0.05,
        n_bins: int = 5,
    ):
        """Initialize the discoverer.

        Args:
            min_slice_size: Minimum samples in a valid slice
            min_support: Minimum fraction of data for valid slice
            max_conditions: Maximum conditions in a slice definition
            significance_level: P-value threshold for significance
            accuracy_gap_threshold: Minimum accuracy gap to report
            n_bins: Number of bins for numeric features
        """
        self.min_slice_size = min_slice_size
        self.min_support = min_support
        self.max_conditions = max_conditions
        self.significance_level = significance_level
        self.accuracy_gap_threshold = accuracy_gap_threshold
        self.n_bins = n_bins

    def discover(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        predictions: np.ndarray | None = None,
        pred_proba: np.ndarray | None = None,
        feature_columns: list[str] | None = None,
    ) -> SliceDiscoveryReport:
        """Discover underperforming data slices.

        Args:
            X: Feature DataFrame
            y: True labels
            predictions: Model predictions (optional)
            pred_proba: Prediction probabilities (optional)
            feature_columns: Features to analyze (default: all)

        Returns:
            SliceDiscoveryReport with discovered slices
        """
        n_samples = len(X)

        # Calculate baseline metrics
        if predictions is not None:
            baseline_accuracy = float((predictions == y).mean())
            is_correct = predictions == y
        else:
            baseline_accuracy = 1.0  # No predictions = assume all correct
            is_correct = np.ones(n_samples, dtype=bool)

        # Select features to analyze
        if feature_columns is None:
            feature_columns = X.columns.tolist()

        # Generate candidate conditions
        conditions = self._generate_conditions(X, feature_columns)

        # Find slices with single conditions
        single_slices = self._evaluate_single_conditions(
            X, y, is_correct, conditions, baseline_accuracy, pred_proba
        )

        # Find slices with multiple conditions (conjunctions)
        multi_slices = []
        if self.max_conditions >= 2:
            multi_slices = self._evaluate_conjunctions(
                X, y, is_correct, single_slices, baseline_accuracy, pred_proba
            )

        # Combine and sort by severity
        all_slices = single_slices + multi_slices
        all_slices.sort(key=lambda s: s.severity_score, reverse=True)

        # Remove redundant slices
        all_slices = self._remove_redundant_slices(all_slices)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_slices, baseline_accuracy, n_samples
        )

        worst_accuracy = (
            min(s.accuracy for s in all_slices) if all_slices else baseline_accuracy
        )
        total_affected = len(set(
            idx for s in all_slices for idx in s.sample_indices
        ))

        return SliceDiscoveryReport(
            n_samples=n_samples,
            n_slices_found=len(all_slices),
            slices=all_slices,
            baseline_accuracy=baseline_accuracy,
            worst_slice_accuracy=worst_accuracy,
            total_affected_samples=total_affected,
            recommendations=recommendations,
        )

    def _generate_conditions(
        self,
        X: pd.DataFrame,
        feature_columns: list[str],
    ) -> list[SliceCondition]:
        """Generate candidate slice conditions from data."""
        conditions = []

        for col in feature_columns:
            if col not in X.columns:
                continue

            dtype = X[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                # Numeric: create range conditions
                conditions.extend(self._create_numeric_conditions(X[col], col))
            else:
                # Categorical: create equality conditions
                conditions.extend(self._create_categorical_conditions(X[col], col))

        return conditions

    def _create_numeric_conditions(
        self,
        series: pd.Series,
        col_name: str,
    ) -> list[SliceCondition]:
        """Create conditions for numeric feature."""
        conditions = []

        # Remove NaN for binning
        valid = series.dropna()
        if len(valid) < self.min_slice_size:
            return conditions

        # Create quantile-based bins
        try:
            bins = pd.qcut(valid, q=self.n_bins, duplicates='drop')
            for interval in bins.unique():
                if pd.isna(interval):
                    continue

                conditions.append(SliceCondition(
                    feature=col_name,
                    operator="between",
                    value=interval.left,
                    value_end=interval.right,
                ))
        except Exception:
            # Fallback to equal-width bins
            bins = np.linspace(valid.min(), valid.max(), self.n_bins + 1)
            for i in range(len(bins) - 1):
                conditions.append(SliceCondition(
                    feature=col_name,
                    operator="between",
                    value=bins[i],
                    value_end=bins[i + 1],
                ))

        return conditions

    def _create_categorical_conditions(
        self,
        series: pd.Series,
        col_name: str,
    ) -> list[SliceCondition]:
        """Create conditions for categorical feature."""
        conditions = []

        # Get value counts
        value_counts = series.value_counts()

        for value, count in value_counts.items():
            if count >= self.min_slice_size:
                conditions.append(SliceCondition(
                    feature=col_name,
                    operator="==",
                    value=value,
                ))

        return conditions

    def _evaluate_single_conditions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        is_correct: np.ndarray,
        conditions: list[SliceCondition],
        baseline_accuracy: float,
        pred_proba: np.ndarray | None,
    ) -> list[DataSlice]:
        """Evaluate single-condition slices."""
        slices = []
        n_total = len(X)

        for i, condition in enumerate(conditions):
            mask = condition.evaluate(X)
            n_samples = mask.sum()

            if n_samples < self.min_slice_size:
                continue

            support = n_samples / n_total
            if support < self.min_support:
                continue

            slice_accuracy = is_correct[mask].mean() if n_samples > 0 else 0
            accuracy_gap = baseline_accuracy - slice_accuracy

            # Statistical test
            p_value = self._test_significance(
                is_correct[mask].sum(), n_samples,
                is_correct.sum(), n_total
            )

            if p_value > self.significance_level:
                continue

            if accuracy_gap < self.accuracy_gap_threshold:
                continue

            # Characterize issues
            issue_types, severity = self._characterize_issues(
                mask, is_correct, pred_proba, baseline_accuracy
            )

            slice_ = DataSlice(
                slice_id=f"slice_{i}",
                conditions=[condition],
                slice_type=SliceType.SINGLE_FEATURE,
                n_samples=int(n_samples),
                n_total=n_total,
                support=float(support),
                accuracy=float(slice_accuracy),
                baseline_accuracy=baseline_accuracy,
                accuracy_gap=float(accuracy_gap),
                error_rate=float(1 - slice_accuracy),
                avg_confidence=self._avg_confidence(mask, pred_proba),
                issue_types=issue_types,
                severity_score=severity,
                sample_indices=np.where(mask)[0],
            )

            slices.append(slice_)

        return slices

    def _evaluate_conjunctions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        is_correct: np.ndarray,
        single_slices: list[DataSlice],
        baseline_accuracy: float,
        pred_proba: np.ndarray | None,
    ) -> list[DataSlice]:
        """Evaluate multi-condition (conjunction) slices."""
        slices = []
        n_total = len(X)

        # Get significant single conditions
        significant_conditions = [
            s.conditions[0] for s in single_slices
            if s.severity_score > 0.5
        ]

        # Generate pairs
        for depth in range(2, self.max_conditions + 1):
            for combo in combinations(significant_conditions, depth):
                # Check if conditions are on different features
                features = set(c.feature for c in combo)
                if len(features) < depth:
                    continue  # Skip if same feature appears twice

                # Evaluate conjunction
                mask = np.ones(n_total, dtype=bool)
                for condition in combo:
                    mask &= condition.evaluate(X)

                n_samples = mask.sum()
                if n_samples < self.min_slice_size:
                    continue

                support = n_samples / n_total
                if support < self.min_support:
                    continue

                slice_accuracy = is_correct[mask].mean() if n_samples > 0 else 0
                accuracy_gap = baseline_accuracy - slice_accuracy

                if accuracy_gap < self.accuracy_gap_threshold:
                    continue

                # Check if this is better than parent slices
                parent_gaps = [
                    s.accuracy_gap for s in single_slices
                    if any(str(c) == str(s.conditions[0]) for c in combo)
                ]
                if parent_gaps and accuracy_gap <= max(parent_gaps) * 1.1:
                    continue  # Not significantly better than parents

                # Characterize issues
                issue_types, severity = self._characterize_issues(
                    mask, is_correct, pred_proba, baseline_accuracy
                )

                slice_ = DataSlice(
                    slice_id=f"slice_conj_{len(slices)}",
                    conditions=list(combo),
                    slice_type=SliceType.CONJUNCTION,
                    n_samples=int(n_samples),
                    n_total=n_total,
                    support=float(support),
                    accuracy=float(slice_accuracy),
                    baseline_accuracy=baseline_accuracy,
                    accuracy_gap=float(accuracy_gap),
                    error_rate=float(1 - slice_accuracy),
                    avg_confidence=self._avg_confidence(mask, pred_proba),
                    issue_types=issue_types,
                    severity_score=severity,
                    sample_indices=np.where(mask)[0],
                )

                slices.append(slice_)

        return slices

    def _test_significance(
        self,
        slice_correct: int,
        slice_total: int,
        total_correct: int,
        total_n: int,
    ) -> float:
        """Test if slice accuracy is significantly different from baseline."""
        # Use Fisher's exact test
        # Contingency table:
        # [[slice_correct, slice_incorrect], [other_correct, other_incorrect]]

        slice_incorrect = slice_total - slice_correct
        other_correct = total_correct - slice_correct
        other_incorrect = (total_n - total_correct) - slice_incorrect

        table = [
            [slice_correct, slice_incorrect],
            [other_correct, other_incorrect],
        ]

        try:
            _, p_value = stats.fisher_exact(table)
        except Exception:
            p_value = 1.0

        return p_value

    def _characterize_issues(
        self,
        mask: np.ndarray,
        is_correct: np.ndarray,
        pred_proba: np.ndarray | None,
        baseline_accuracy: float,
    ) -> tuple[list[IssueType], float]:
        """Characterize the types of issues in a slice."""
        issues = []
        severity_components = []

        slice_accuracy = is_correct[mask].mean()

        # Low accuracy
        if slice_accuracy < baseline_accuracy - 0.1:
            issues.append(IssueType.LOW_ACCURACY)
            severity_components.append(baseline_accuracy - slice_accuracy)

        # High error rate
        error_rate = 1 - slice_accuracy
        if error_rate > 0.3:
            issues.append(IssueType.HIGH_ERROR_RATE)
            severity_components.append(error_rate)

        # Low confidence
        if pred_proba is not None:
            slice_confidence = pred_proba[mask].max(axis=1).mean()
            baseline_confidence = pred_proba.max(axis=1).mean()

            if slice_confidence < baseline_confidence - 0.1:
                issues.append(IssueType.LOW_CONFIDENCE)
                severity_components.append(baseline_confidence - slice_confidence)

        # Underrepresented
        support = mask.mean()
        if support < 0.05:
            issues.append(IssueType.UNDERREPRESENTED)
            severity_components.append(0.3)

        # Calculate overall severity
        if severity_components:
            severity = min(1.0, sum(severity_components) / len(severity_components) * 2)
        else:
            severity = 0.0

        if not issues:
            issues.append(IssueType.LOW_ACCURACY)

        return issues, severity

    def _avg_confidence(
        self,
        mask: np.ndarray,
        pred_proba: np.ndarray | None,
    ) -> float:
        """Calculate average confidence for a slice."""
        if pred_proba is None:
            return 0.0

        return float(pred_proba[mask].max(axis=1).mean())

    def _remove_redundant_slices(
        self,
        slices: list[DataSlice],
    ) -> list[DataSlice]:
        """Remove slices that are subsets of others with similar metrics."""
        if not slices:
            return slices

        keep = []
        for slice_ in slices:
            is_redundant = False

            for kept in keep:
                # Check if slice_ is a subset of kept
                overlap = len(set(slice_.sample_indices) & set(kept.sample_indices))
                if overlap > 0.9 * len(slice_.sample_indices):
                    # Slice is mostly contained in kept
                    if abs(slice_.accuracy_gap - kept.accuracy_gap) < 0.02:
                        is_redundant = True
                        break

            if not is_redundant:
                keep.append(slice_)

        return keep

    def _generate_recommendations(
        self,
        slices: list[DataSlice],
        baseline_accuracy: float,
        n_samples: int,
    ) -> list[str]:
        """Generate recommendations based on discovered slices."""
        recommendations = []

        if not slices:
            recommendations.append(
                "No significant underperforming slices found. "
                "Model performance is relatively consistent across subgroups."
            )
            return recommendations

        # Most severe slice
        worst = slices[0]
        conditions_str = " AND ".join(str(c) for c in worst.conditions)
        recommendations.append(
            f"Priority: Investigate slice '{conditions_str}' with "
            f"{worst.accuracy_gap:.1%} accuracy gap ({worst.n_samples:,} samples)."
        )

        # Group by issue type
        issue_counts = {}
        for slice_ in slices:
            for issue in slice_.issue_types:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if IssueType.LOW_CONFIDENCE in issue_counts:
            recommendations.append(
                f"{issue_counts[IssueType.LOW_CONFIDENCE]} slices have low model confidence. "
                "Consider collecting more training data for these subgroups."
            )

        if IssueType.UNDERREPRESENTED in issue_counts:
            recommendations.append(
                f"{issue_counts[IssueType.UNDERREPRESENTED]} underrepresented slices found. "
                "These may benefit from targeted data collection."
            )

        # Feature-specific recommendations
        feature_issues = {}
        for slice_ in slices:
            for cond in slice_.conditions:
                feature_issues[cond.feature] = feature_issues.get(cond.feature, 0) + 1

        if feature_issues:
            worst_feature = max(feature_issues.items(), key=lambda x: x[1])
            recommendations.append(
                f"Feature '{worst_feature[0]}' appears in {worst_feature[1]} problem slices. "
                "Consider feature engineering or additional data collection."
            )

        return recommendations


def discover_slices(
    X: pd.DataFrame,
    y: np.ndarray,
    predictions: np.ndarray | None = None,
    **kwargs: Any,
) -> SliceDiscoveryReport:
    """Convenience function for slice discovery.

    Args:
        X: Feature DataFrame
        y: True labels
        predictions: Model predictions
        **kwargs: Additional arguments for SliceDiscoverer

    Returns:
        SliceDiscoveryReport with discovered slices
    """
    discoverer = SliceDiscoverer(**kwargs)
    return discoverer.discover(X, y, predictions)
