"""Automated Root Cause Analysis for data quality issues.

This module provides ML-powered drill-down capabilities to identify
why quality issues occur (annotator, time period, data source, etc.).

Example:
    >>> from clean.root_cause import RootCauseAnalyzer
    >>>
    >>> analyzer = RootCauseAnalyzer()
    >>> report = analyzer.analyze(
    ...     data=df,
    ...     issues=quality_report.issues,
    ...     metadata_columns=["annotator", "source", "collection_date"],
    ... )
    >>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    pass


class RootCauseType(Enum):
    """Types of root causes for quality issues."""

    ANNOTATOR = "annotator"
    DATA_SOURCE = "data_source"
    TIME_PERIOD = "time_period"
    FEATURE_RANGE = "feature_range"
    CLASS_LABEL = "class_label"
    COLLECTION_METHOD = "collection_method"
    UNKNOWN = "unknown"


class CorrelationStrength(Enum):
    """Strength of correlation between factor and issues."""

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class RootCause:
    """Identified root cause for quality issues."""

    cause_type: RootCauseType
    factor_name: str
    factor_value: Any
    issue_count: int
    issue_rate: float
    baseline_rate: float
    lift: float  # How much more likely issues are for this factor
    confidence: float  # Statistical confidence
    affected_indices: list[int]
    correlation_strength: CorrelationStrength
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cause_type": self.cause_type.value,
            "factor_name": self.factor_name,
            "factor_value": self.factor_value,
            "issue_count": self.issue_count,
            "issue_rate": self.issue_rate,
            "baseline_rate": self.baseline_rate,
            "lift": self.lift,
            "confidence": self.confidence,
            "affected_samples": len(self.affected_indices),
            "correlation_strength": self.correlation_strength.value,
            "explanation": self.explanation,
        }


@dataclass
class FeatureCorrelation:
    """Correlation between a feature and issue occurrence."""

    feature_name: str
    correlation: float
    p_value: float
    issue_concentration: dict[str, float]  # Feature range -> issue rate
    thresholds: list[float]  # Identified thresholds where issues spike


@dataclass
class TemporalPattern:
    """Temporal pattern in issue occurrence."""

    period: str  # "daily", "weekly", "monthly"
    pattern_type: str  # "increasing", "decreasing", "cyclic", "spike"
    peak_periods: list[str]
    trend_coefficient: float
    seasonality_strength: float


@dataclass
class RootCauseReport:
    """Complete root cause analysis report."""

    n_samples: int
    n_issues: int
    overall_issue_rate: float
    root_causes: list[RootCause]
    feature_correlations: list[FeatureCorrelation]
    temporal_patterns: list[TemporalPattern]
    recommendations: list[str]
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Root Cause Analysis Report",
            "=" * 50,
            "",
            f"Total samples: {self.n_samples:,}",
            f"Issues found: {self.n_issues:,} ({self.overall_issue_rate:.1%})",
            "",
        ]

        if self.root_causes:
            lines.append(f"Top Root Causes (found {len(self.root_causes)}):")
            lines.append("-" * 40)
            for i, cause in enumerate(self.root_causes[:5], 1):
                lines.append(f"{i}. {cause.factor_name} = '{cause.factor_value}'")
                lines.append(f"   Issue rate: {cause.issue_rate:.1%} (vs baseline {cause.baseline_rate:.1%})")
                lines.append(f"   Lift: {cause.lift:.1f}x, Confidence: {cause.confidence:.0%}")
                lines.append(f"   → {cause.explanation}")
                lines.append("")
        else:
            lines.append("No significant root causes identified.")
            lines.append("")

        if self.feature_correlations:
            lines.append("Feature Correlations:")
            lines.append("-" * 40)
            for fc in self.feature_correlations[:3]:
                lines.append(f"  • {fc.feature_name}: r={fc.correlation:.3f} (p={fc.p_value:.3f})")

        if self.temporal_patterns:
            lines.append("")
            lines.append("Temporal Patterns:")
            lines.append("-" * 40)
            for tp in self.temporal_patterns:
                lines.append(f"  • {tp.period}: {tp.pattern_type}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            lines.append("-" * 40)
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_issues": self.n_issues,
            "overall_issue_rate": self.overall_issue_rate,
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "recommendations": self.recommendations,
        }

    def get_top_causes(self, n: int = 5) -> list[RootCause]:
        """Get top N root causes by lift."""
        return sorted(self.root_causes, key=lambda x: x.lift, reverse=True)[:n]


class RootCauseAnalyzer:
    """Analyzer for identifying root causes of data quality issues.

    Uses statistical analysis and ML to identify factors correlated
    with quality issues.
    """

    def __init__(
        self,
        min_sample_size: int = 30,
        significance_level: float = 0.05,
        min_lift: float = 1.5,
        min_confidence: float = 0.8,
    ):
        """Initialize the analyzer.

        Args:
            min_sample_size: Minimum samples for a factor to be considered
            significance_level: P-value threshold for significance
            min_lift: Minimum lift to report as root cause
            min_confidence: Minimum confidence to report
        """
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.min_lift = min_lift
        self.min_confidence = min_confidence

    def analyze(
        self,
        data: pd.DataFrame,
        issue_indices: list[int] | np.ndarray | None = None,
        issue_column: str | None = None,
        metadata_columns: list[str] | None = None,
        temporal_column: str | None = None,
        label_column: str | None = None,
    ) -> RootCauseReport:
        """Analyze root causes of data quality issues.

        Args:
            data: DataFrame with data and metadata
            issue_indices: Indices of samples with issues
            issue_column: Column indicating issue (bool or 0/1)
            metadata_columns: Columns to analyze as potential causes
            temporal_column: Column with timestamps for temporal analysis
            label_column: Label column for class-based analysis

        Returns:
            RootCauseReport with identified causes
        """
        # Create issue mask
        if issue_indices is not None:
            issue_mask = np.zeros(len(data), dtype=bool)
            issue_mask[issue_indices] = True
        elif issue_column and issue_column in data.columns:
            issue_mask = data[issue_column].astype(bool).values
        else:
            raise ValueError("Must provide issue_indices or issue_column")

        n_samples = len(data)
        n_issues = int(issue_mask.sum())
        overall_rate = n_issues / n_samples if n_samples > 0 else 0

        # Identify metadata columns if not provided
        if metadata_columns is None:
            metadata_columns = self._identify_metadata_columns(data, label_column)

        root_causes = []
        feature_correlations = []
        temporal_patterns = []

        # Analyze categorical factors
        for col in metadata_columns:
            if col in data.columns:
                causes = self._analyze_categorical_factor(
                    data, col, issue_mask, overall_rate
                )
                root_causes.extend(causes)

        # Analyze numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != label_column and col not in metadata_columns:
                correlation = self._analyze_numeric_correlation(
                    data[col].values, issue_mask
                )
                if correlation:
                    feature_correlations.append(correlation)

        # Analyze temporal patterns
        if temporal_column and temporal_column in data.columns:
            patterns = self._analyze_temporal_patterns(
                data, temporal_column, issue_mask
            )
            temporal_patterns.extend(patterns)

        # Analyze by class label
        if label_column and label_column in data.columns:
            label_causes = self._analyze_categorical_factor(
                data, label_column, issue_mask, overall_rate,
                cause_type=RootCauseType.CLASS_LABEL
            )
            root_causes.extend(label_causes)

        # Sort by lift
        root_causes.sort(key=lambda x: x.lift, reverse=True)
        feature_correlations.sort(key=lambda x: abs(x.correlation), reverse=True)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            root_causes, feature_correlations, temporal_patterns, overall_rate
        )

        return RootCauseReport(
            n_samples=n_samples,
            n_issues=n_issues,
            overall_issue_rate=overall_rate,
            root_causes=root_causes,
            feature_correlations=feature_correlations,
            temporal_patterns=temporal_patterns,
            recommendations=recommendations,
        )

    def _identify_metadata_columns(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
    ) -> list[str]:
        """Identify likely metadata columns automatically."""
        metadata_cols = []

        for col in data.columns:
            if col == label_column:
                continue

            # Check for low cardinality (likely categorical)
            n_unique = data[col].nunique()
            n_total = len(data)

            # Metadata typically has low cardinality relative to data size
            if (n_unique < 50 and n_unique < n_total * 0.1) or any(kw in col.lower() for kw in [
                "annotator", "labeler", "source", "batch", "date",
                "method", "version", "split", "origin", "type"
            ]):
                metadata_cols.append(col)

        return metadata_cols

    def _analyze_categorical_factor(
        self,
        data: pd.DataFrame,
        column: str,
        issue_mask: np.ndarray,
        baseline_rate: float,
        cause_type: RootCauseType | None = None,
    ) -> list[RootCause]:
        """Analyze a categorical factor for correlation with issues."""
        causes = []

        # Infer cause type from column name if not provided
        if cause_type is None:
            cause_type = self._infer_cause_type(column)

        # Get value counts and issue rates
        values = data[column].values
        unique_values = pd.unique(values)

        for value in unique_values:
            value_mask = values == value
            n_with_value = value_mask.sum()

            if n_with_value < self.min_sample_size:
                continue

            # Calculate issue rate for this value
            issues_with_value = (value_mask & issue_mask).sum()
            issue_rate = issues_with_value / n_with_value if n_with_value > 0 else 0

            # Calculate lift
            lift = issue_rate / baseline_rate if baseline_rate > 0 else 0

            # Statistical test (chi-squared)
            contingency = [
                [issues_with_value, n_with_value - issues_with_value],
                [issue_mask.sum() - issues_with_value, (~issue_mask).sum() - (n_with_value - issues_with_value)],
            ]
            contingency = np.array(contingency)

            # Ensure non-negative values
            if (contingency < 0).any():
                continue

            try:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                confidence = 1 - p_value
            except Exception:
                confidence = 0
                p_value = 1

            # Check significance
            if p_value > self.significance_level or lift < self.min_lift:
                continue

            # Determine correlation strength
            correlation_strength = self._get_correlation_strength(lift)

            # Generate explanation
            explanation = self._generate_cause_explanation(
                cause_type, column, value, issue_rate, baseline_rate, lift
            )

            # Get affected indices
            affected = np.where(value_mask & issue_mask)[0].tolist()

            causes.append(RootCause(
                cause_type=cause_type,
                factor_name=column,
                factor_value=value,
                issue_count=issues_with_value,
                issue_rate=issue_rate,
                baseline_rate=baseline_rate,
                lift=lift,
                confidence=confidence,
                affected_indices=affected,
                correlation_strength=correlation_strength,
                explanation=explanation,
            ))

        return causes

    def _analyze_numeric_correlation(
        self,
        values: np.ndarray,
        issue_mask: np.ndarray,
    ) -> FeatureCorrelation | None:
        """Analyze correlation between numeric feature and issues."""
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < self.min_sample_size:
            return None

        valid_values = values[valid_mask]
        valid_issues = issue_mask[valid_mask]

        # Point-biserial correlation
        try:
            correlation, p_value = stats.pointbiserialr(valid_issues, valid_values)
        except Exception:
            return None

        if np.isnan(correlation) or p_value > self.significance_level:
            return None

        # Find concentration of issues in value ranges
        percentiles = [0, 25, 50, 75, 100]
        thresholds = np.percentile(valid_values, percentiles)

        concentration = {}
        identified_thresholds = []

        for i in range(len(percentiles) - 1):
            low, high = thresholds[i], thresholds[i + 1]
            range_mask = (valid_values >= low) & (valid_values <= high)
            n_in_range = range_mask.sum()

            if n_in_range > 0:
                issue_rate = (range_mask & valid_issues).sum() / n_in_range
                range_label = f"{percentiles[i]}-{percentiles[i+1]}%"
                concentration[range_label] = issue_rate

                # Check if this range has elevated issues
                baseline = valid_issues.mean()
                if issue_rate > baseline * 1.5:
                    identified_thresholds.append((low + high) / 2)

        return FeatureCorrelation(
            feature_name="unknown",  # Will be set by caller
            correlation=float(correlation),
            p_value=float(p_value),
            issue_concentration=concentration,
            thresholds=identified_thresholds,
        )

    def _analyze_temporal_patterns(
        self,
        data: pd.DataFrame,
        temporal_column: str,
        issue_mask: np.ndarray,
    ) -> list[TemporalPattern]:
        """Analyze temporal patterns in issue occurrence."""
        patterns = []

        try:
            timestamps = pd.to_datetime(data[temporal_column])
        except Exception:
            return patterns

        # Create time series of issue rates
        df_temporal = pd.DataFrame({
            "timestamp": timestamps,
            "is_issue": issue_mask,
        })

        # Daily aggregation
        df_temporal["date"] = df_temporal["timestamp"].dt.date
        daily = df_temporal.groupby("date").agg(
            n_samples=("is_issue", "count"),
            n_issues=("is_issue", "sum"),
        )
        daily["issue_rate"] = daily["n_issues"] / daily["n_samples"]

        if len(daily) >= 7:
            # Detect trend
            x = np.arange(len(daily))
            y = daily["issue_rate"].values
            valid = ~np.isnan(y)

            if valid.sum() >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x[valid], y[valid]
                )

                if abs(slope) > 0.001 and p_value < 0.1:
                    pattern_type = "increasing" if slope > 0 else "decreasing"
                    patterns.append(TemporalPattern(
                        period="daily",
                        pattern_type=pattern_type,
                        peak_periods=[],
                        trend_coefficient=float(slope),
                        seasonality_strength=0,
                    ))

                # Find spike periods
                mean_rate = y[valid].mean()
                std_rate = y[valid].std()
                spike_threshold = mean_rate + 2 * std_rate

                spike_dates = daily.index[daily["issue_rate"] > spike_threshold].tolist()
                if spike_dates:
                    patterns.append(TemporalPattern(
                        period="daily",
                        pattern_type="spike",
                        peak_periods=[str(d) for d in spike_dates[:5]],
                        trend_coefficient=0,
                        seasonality_strength=0,
                    ))

        # Weekly aggregation
        df_temporal["week"] = df_temporal["timestamp"].dt.isocalendar().week
        weekly = df_temporal.groupby("week").agg(
            n_samples=("is_issue", "count"),
            n_issues=("is_issue", "sum"),
        )
        weekly["issue_rate"] = weekly["n_issues"] / weekly["n_samples"]

        # Day of week pattern
        df_temporal["day_of_week"] = df_temporal["timestamp"].dt.dayofweek
        dow = df_temporal.groupby("day_of_week").agg(
            n_samples=("is_issue", "count"),
            n_issues=("is_issue", "sum"),
        )
        dow["issue_rate"] = dow["n_issues"] / dow["n_samples"]

        if len(dow) >= 5:
            # Check for day-of-week pattern
            dow_rates = dow["issue_rate"].values
            if dow_rates.std() > dow_rates.mean() * 0.2:
                peak_days = dow.nlargest(2, "issue_rate").index.tolist()
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                peak_day_names = [day_names[d] for d in peak_days if d < 7]

                patterns.append(TemporalPattern(
                    period="weekly",
                    pattern_type="cyclic",
                    peak_periods=peak_day_names,
                    trend_coefficient=0,
                    seasonality_strength=float(dow_rates.std() / max(dow_rates.mean(), 0.001)),
                ))

        return patterns

    def _infer_cause_type(self, column_name: str) -> RootCauseType:
        """Infer root cause type from column name."""
        name_lower = column_name.lower()

        if any(kw in name_lower for kw in ["annotator", "labeler", "rater", "worker"]):
            return RootCauseType.ANNOTATOR
        elif any(kw in name_lower for kw in ["source", "origin", "dataset", "corpus"]):
            return RootCauseType.DATA_SOURCE
        elif any(kw in name_lower for kw in ["date", "time", "timestamp", "created"]):
            return RootCauseType.TIME_PERIOD
        elif any(kw in name_lower for kw in ["method", "technique", "approach"]):
            return RootCauseType.COLLECTION_METHOD
        else:
            return RootCauseType.UNKNOWN

    def _get_correlation_strength(self, lift: float) -> CorrelationStrength:
        """Determine correlation strength from lift value."""
        if lift < 1.2:
            return CorrelationStrength.NONE
        elif lift < 1.5:
            return CorrelationStrength.WEAK
        elif lift < 2.0:
            return CorrelationStrength.MODERATE
        elif lift < 3.0:
            return CorrelationStrength.STRONG
        else:
            return CorrelationStrength.VERY_STRONG

    def _generate_cause_explanation(
        self,
        cause_type: RootCauseType,
        column: str,
        value: Any,
        issue_rate: float,
        baseline_rate: float,
        lift: float,
    ) -> str:
        """Generate human-readable explanation for a root cause."""
        if cause_type == RootCauseType.ANNOTATOR:
            return (
                f"Annotator '{value}' has {lift:.1f}x higher issue rate "
                f"({issue_rate:.1%} vs {baseline_rate:.1%}). "
                "Consider additional training or quality review."
            )
        elif cause_type == RootCauseType.DATA_SOURCE:
            return (
                f"Data from source '{value}' shows {lift:.1f}x more issues. "
                "Review data collection process for this source."
            )
        elif cause_type == RootCauseType.TIME_PERIOD:
            return (
                f"Period '{value}' has elevated issues ({issue_rate:.1%}). "
                "Investigate what changed during this time."
            )
        elif cause_type == RootCauseType.CLASS_LABEL:
            return (
                f"Class '{value}' has {lift:.1f}x higher error rate. "
                "May need more training examples or clearer labeling guidelines."
            )
        else:
            return (
                f"{column}='{value}' strongly correlates with issues "
                f"(lift={lift:.1f}x). Investigate this pattern."
            )

    def _generate_recommendations(
        self,
        root_causes: list[RootCause],
        feature_correlations: list[FeatureCorrelation],
        temporal_patterns: list[TemporalPattern],
        overall_rate: float,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Annotator-related recommendations
        annotator_causes = [
            rc for rc in root_causes
            if rc.cause_type == RootCauseType.ANNOTATOR and rc.lift > 2
        ]
        if annotator_causes:
            worst = max(annotator_causes, key=lambda x: x.lift)
            recommendations.append(
                f"Review work from annotator '{worst.factor_value}' - "
                f"they have {worst.lift:.1f}x higher error rate. "
                f"Consider additional training or removing their annotations."
            )

        # Source-related recommendations
        source_causes = [
            rc for rc in root_causes
            if rc.cause_type == RootCauseType.DATA_SOURCE and rc.lift > 1.5
        ]
        if source_causes:
            worst = max(source_causes, key=lambda x: x.issue_count)
            recommendations.append(
                f"Data source '{worst.factor_value}' contributes significantly "
                f"to quality issues. Consider improving collection process "
                f"or filtering this source."
            )

        # Class-related recommendations
        class_causes = [
            rc for rc in root_causes
            if rc.cause_type == RootCauseType.CLASS_LABEL and rc.lift > 1.5
        ]
        if class_causes:
            problem_classes = [rc.factor_value for rc in class_causes[:3]]
            recommendations.append(
                f"Classes {problem_classes} have elevated error rates. "
                f"Review labeling guidelines and consider collecting more examples."
            )

        # Temporal recommendations
        increasing_trends = [
            tp for tp in temporal_patterns
            if tp.pattern_type == "increasing"
        ]
        if increasing_trends:
            recommendations.append(
                "Quality issues are increasing over time. "
                "Investigate recent changes to data collection or annotation process."
            )

        spike_patterns = [
            tp for tp in temporal_patterns
            if tp.pattern_type == "spike"
        ]
        if spike_patterns:
            dates = spike_patterns[0].peak_periods
            recommendations.append(
                f"Issue spikes detected on {dates[:3]}. "
                f"Review what happened during these periods."
            )

        # Feature correlation recommendations
        strong_correlations = [
            fc for fc in feature_correlations
            if abs(fc.correlation) > 0.3
        ]
        if strong_correlations:
            fc = strong_correlations[0]
            recommendations.append(
                f"Feature '{fc.feature_name}' correlates with issues "
                f"(r={fc.correlation:.2f}). Consider feature engineering or filtering."
            )

        # Overall recommendations
        if overall_rate > 0.1:
            recommendations.append(
                f"Overall issue rate ({overall_rate:.1%}) is high. "
                f"Consider systematic review of annotation guidelines."
            )
        elif overall_rate < 0.01 and not root_causes:
            recommendations.append(
                "Data quality looks good! Continue monitoring for changes."
            )

        return recommendations


def analyze_root_causes(
    data: pd.DataFrame,
    issue_indices: list[int] | np.ndarray | None = None,
    issue_column: str | None = None,
    **kwargs: Any,
) -> RootCauseReport:
    """Convenience function for root cause analysis.

    Args:
        data: DataFrame with data and metadata
        issue_indices: Indices of samples with issues
        issue_column: Column indicating issue
        **kwargs: Additional arguments for analyzer

    Returns:
        RootCauseReport with analysis results
    """
    analyzer = RootCauseAnalyzer()
    return analyzer.analyze(
        data=data,
        issue_indices=issue_indices,
        issue_column=issue_column,
        **kwargs,
    )
