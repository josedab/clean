"""Data drift detection for monitoring distribution changes.

This module provides tools for detecting drift between datasets:
- Statistical tests (KS test, Chi-squared, PSI)
- Feature-level drift detection
- Label drift detection
- Embedding drift for text/image data

Example:
    >>> from clean.drift import DriftDetector
    >>>
    >>> detector = DriftDetector()
    >>> detector.fit(reference_df)
    >>> report = detector.detect(current_df)
    >>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats


class DriftType(Enum):
    """Type of drift detected."""

    NONE = "none"
    COVARIATE = "covariate"  # Feature distribution change
    LABEL = "label"  # Label distribution change
    CONCEPT = "concept"  # Relationship between features and labels changed


class DriftSeverity(Enum):
    """Severity level of detected drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDrift:
    """Drift information for a single feature."""

    feature_name: str
    drift_score: float
    p_value: float | None
    test_method: str
    has_drift: bool
    severity: DriftSeverity
    reference_stats: dict[str, float] = field(default_factory=dict)
    current_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "test_method": self.test_method,
            "has_drift": self.has_drift,
            "severity": self.severity.value,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""

    timestamp: datetime
    reference_size: int
    current_size: int
    overall_drift_score: float
    overall_severity: DriftSeverity
    has_drift: bool
    drift_type: DriftType
    feature_drifts: list[FeatureDrift]
    label_drift: FeatureDrift | None
    drifted_features: list[str]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "⚠️ DRIFT DETECTED" if self.has_drift else "✅ NO DRIFT"
        lines = [
            "Data Drift Report",
            "=" * 50,
            f"Status: {status}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Reference Size: {self.reference_size:,}",
            f"Current Size: {self.current_size:,}",
            f"Overall Drift Score: {self.overall_drift_score:.3f}",
            f"Severity: {self.overall_severity.value.upper()}",
            f"Drift Type: {self.drift_type.value}",
            "",
        ]

        if self.drifted_features:
            lines.append(f"Drifted Features ({len(self.drifted_features)}):")
            for feat in self.drifted_features[:10]:
                drift = next(f for f in self.feature_drifts if f.feature_name == feat)
                lines.append(f"  - {feat}: score={drift.drift_score:.3f}, severity={drift.severity.value}")
            if len(self.drifted_features) > 10:
                lines.append(f"  ... and {len(self.drifted_features) - 10} more")
            lines.append("")

        if self.label_drift and self.label_drift.has_drift:
            lines.append("Label Drift:")
            lines.append(f"  Score: {self.label_drift.drift_score:.3f}")
            lines.append(f"  Severity: {self.label_drift.severity.value}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "overall_drift_score": self.overall_drift_score,
            "overall_severity": self.overall_severity.value,
            "has_drift": self.has_drift,
            "drift_type": self.drift_type.value,
            "feature_drifts": [f.to_dict() for f in self.feature_drifts],
            "label_drift": self.label_drift.to_dict() if self.label_drift else None,
            "drifted_features": self.drifted_features,
            "recommendations": self.recommendations,
        }


class DriftDetector:
    """Detector for data distribution drift.

    Monitors changes in feature and label distributions between
    a reference dataset and current data.
    """

    def __init__(
        self,
        drift_threshold: float = 0.1,
        p_value_threshold: float = 0.05,
        method: str = "auto",
        categorical_method: str = "chi2",
        numerical_method: str = "ks",
    ):
        """Initialize the drift detector.

        Args:
            drift_threshold: Threshold for drift score to flag drift
            p_value_threshold: P-value threshold for statistical tests
            method: Detection method ('auto', 'statistical', 'psi')
            categorical_method: Method for categorical features ('chi2', 'psi')
            numerical_method: Method for numerical features ('ks', 'psi', 'wasserstein')
        """
        self.drift_threshold = drift_threshold
        self.p_value_threshold = p_value_threshold
        self.method = method
        self.categorical_method = categorical_method
        self.numerical_method = numerical_method

        self._reference: pd.DataFrame | None = None
        self._reference_labels: np.ndarray | None = None
        self._feature_types: dict[str, str] = {}
        self._is_fitted = False

    def fit(
        self,
        reference: pd.DataFrame,
        labels: np.ndarray | None = None,
        label_column: str | None = None,
    ) -> "DriftDetector":
        """Fit the detector with reference data.

        Args:
            reference: Reference DataFrame
            labels: Optional label array
            label_column: Optional column name for labels in DataFrame

        Returns:
            Self for chaining
        """
        self._reference = reference.copy()

        # Extract labels if specified
        if label_column and label_column in reference.columns:
            self._reference_labels = reference[label_column].values
            self._reference = reference.drop(columns=[label_column])
        elif labels is not None:
            self._reference_labels = np.asarray(labels)

        # Determine feature types
        self._feature_types = {}
        for col in self._reference.columns:
            if pd.api.types.is_numeric_dtype(self._reference[col]):
                self._feature_types[col] = "numerical"
            else:
                self._feature_types[col] = "categorical"

        self._is_fitted = True
        return self

    def detect(
        self,
        current: pd.DataFrame,
        labels: np.ndarray | None = None,
        label_column: str | None = None,
    ) -> DriftReport:
        """Detect drift between reference and current data.

        Args:
            current: Current DataFrame to compare
            labels: Optional current label array
            label_column: Optional column name for labels

        Returns:
            DriftReport with detection results
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        assert self._reference is not None

        current_df = current.copy()
        current_labels = None

        # Extract labels if specified
        if label_column and label_column in current.columns:
            current_labels = current[label_column].values
            current_df = current.drop(columns=[label_column])
        elif labels is not None:
            current_labels = np.asarray(labels)

        # Detect feature drift
        feature_drifts = []
        for col in self._reference.columns:
            if col not in current_df.columns:
                continue

            drift = self._detect_feature_drift(
                self._reference[col].values,
                current_df[col].values,
                col,
                self._feature_types.get(col, "numerical"),
            )
            feature_drifts.append(drift)

        # Detect label drift
        label_drift = None
        if self._reference_labels is not None and current_labels is not None:
            label_drift = self._detect_feature_drift(
                self._reference_labels,
                current_labels,
                "_label",
                "categorical",
            )

        # Compute overall metrics
        drifted_features = [f.feature_name for f in feature_drifts if f.has_drift]
        drift_scores = [f.drift_score for f in feature_drifts]
        overall_score = np.mean(drift_scores) if drift_scores else 0.0

        # Determine severity
        overall_severity = self._score_to_severity(overall_score)

        # Determine drift type
        drift_type = DriftType.NONE
        if drifted_features:
            drift_type = DriftType.COVARIATE
        if label_drift and label_drift.has_drift:
            drift_type = DriftType.LABEL if drift_type == DriftType.NONE else DriftType.CONCEPT

        has_drift = len(drifted_features) > 0 or (label_drift and label_drift.has_drift)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            feature_drifts, label_drift, drifted_features
        )

        return DriftReport(
            timestamp=datetime.now(),
            reference_size=len(self._reference),
            current_size=len(current_df),
            overall_drift_score=overall_score,
            overall_severity=overall_severity,
            has_drift=has_drift,
            drift_type=drift_type,
            feature_drifts=feature_drifts,
            label_drift=label_drift,
            drifted_features=drifted_features,
            recommendations=recommendations,
        )

    def _detect_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        feature_type: str,
    ) -> FeatureDrift:
        """Detect drift for a single feature."""
        # Clean data
        ref_clean = reference[~pd.isna(reference)]
        cur_clean = current[~pd.isna(current)]

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return FeatureDrift(
                feature_name=feature_name,
                drift_score=0.0,
                p_value=None,
                test_method="none",
                has_drift=False,
                severity=DriftSeverity.NONE,
            )

        if feature_type == "numerical":
            return self._detect_numerical_drift(ref_clean, cur_clean, feature_name)
        else:
            return self._detect_categorical_drift(ref_clean, cur_clean, feature_name)

    def _detect_numerical_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
    ) -> FeatureDrift:
        """Detect drift for numerical feature."""
        method = self.numerical_method

        if method == "ks":
            statistic, p_value = stats.ks_2samp(reference, current)
            drift_score = statistic
        elif method == "wasserstein":
            drift_score = stats.wasserstein_distance(reference, current)
            # Normalize by range
            combined = np.concatenate([reference, current])
            data_range = np.ptp(combined)
            if data_range > 0:
                drift_score = drift_score / data_range
            p_value = None
        elif method == "psi":
            drift_score = self._compute_psi(reference, current)
            p_value = None
        else:
            # Default to KS test
            statistic, p_value = stats.ks_2samp(reference, current)
            drift_score = statistic

        has_drift = drift_score > self.drift_threshold
        if p_value is not None:
            has_drift = has_drift or p_value < self.p_value_threshold

        severity = self._score_to_severity(drift_score)

        return FeatureDrift(
            feature_name=feature_name,
            drift_score=float(drift_score),
            p_value=float(p_value) if p_value is not None else None,
            test_method=method,
            has_drift=has_drift,
            severity=severity,
            reference_stats={
                "mean": float(np.mean(reference)),
                "std": float(np.std(reference)),
                "min": float(np.min(reference)),
                "max": float(np.max(reference)),
            },
            current_stats={
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
                "min": float(np.min(current)),
                "max": float(np.max(current)),
            },
        )

    def _detect_categorical_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
    ) -> FeatureDrift:
        """Detect drift for categorical feature."""
        method = self.categorical_method

        # Get unique categories
        all_categories = set(reference) | set(current)

        if method == "chi2":
            # Build contingency table
            ref_counts = pd.Series(reference).value_counts()
            cur_counts = pd.Series(current).value_counts()

            # Align categories
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]

            # Perform chi-squared test
            try:
                # Normalize to same scale
                ref_norm = np.array(ref_freq) / sum(ref_freq) * 1000
                cur_norm = np.array(cur_freq) / sum(cur_freq) * 1000

                statistic, p_value = stats.chisquare(cur_norm, ref_norm)
                drift_score = statistic / 1000  # Normalize
            except Exception:
                drift_score = 0.0
                p_value = 1.0

        elif method == "psi":
            drift_score = self._compute_psi_categorical(reference, current, all_categories)
            p_value = None
        else:
            # Default to PSI
            drift_score = self._compute_psi_categorical(reference, current, all_categories)
            p_value = None

        has_drift = drift_score > self.drift_threshold
        if p_value is not None:
            has_drift = has_drift or p_value < self.p_value_threshold

        severity = self._score_to_severity(drift_score)

        # Compute distribution stats
        ref_dist = pd.Series(reference).value_counts(normalize=True).to_dict()
        cur_dist = pd.Series(current).value_counts(normalize=True).to_dict()

        return FeatureDrift(
            feature_name=feature_name,
            drift_score=float(drift_score),
            p_value=float(p_value) if p_value is not None else None,
            test_method=method,
            has_drift=has_drift,
            severity=severity,
            reference_stats={str(k): float(v) for k, v in ref_dist.items()},
            current_stats={str(k): float(v) for k, v in cur_dist.items()},
        )

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index for numerical data."""
        # Create bins from reference data
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        # Normalize
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)

        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

        # Compute PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _compute_psi_categorical(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        categories: set,
    ) -> float:
        """Compute PSI for categorical data."""
        ref_counts = pd.Series(reference).value_counts()
        cur_counts = pd.Series(current).value_counts()

        psi = 0.0
        for cat in categories:
            ref_pct = ref_counts.get(cat, 0) / len(reference)
            cur_pct = cur_counts.get(cat, 0) / len(current)

            # Avoid division by zero
            ref_pct = max(ref_pct, 0.0001)
            cur_pct = max(cur_pct, 0.0001)

            psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

        return float(psi)

    def _score_to_severity(self, score: float) -> DriftSeverity:
        """Convert drift score to severity level."""
        if score < 0.05:
            return DriftSeverity.NONE
        elif score < 0.1:
            return DriftSeverity.LOW
        elif score < 0.2:
            return DriftSeverity.MEDIUM
        elif score < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _generate_recommendations(
        self,
        feature_drifts: list[FeatureDrift],
        label_drift: FeatureDrift | None,
        drifted_features: list[str],
    ) -> list[str]:
        """Generate recommendations based on drift results."""
        recommendations = []

        if not drifted_features and (not label_drift or not label_drift.has_drift):
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations

        # Feature drift recommendations
        high_severity_features = [
            f for f in feature_drifts
            if f.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
        ]

        if high_severity_features:
            feat_names = [f.feature_name for f in high_severity_features[:3]]
            recommendations.append(
                f"High drift in features: {', '.join(feat_names)}. "
                "Consider retraining the model."
            )

        if len(drifted_features) > len(feature_drifts) * 0.5:
            recommendations.append(
                "More than 50% of features show drift. "
                "This may indicate a fundamental change in data collection."
            )

        # Label drift recommendations
        if label_drift and label_drift.has_drift:
            if label_drift.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
                recommendations.append(
                    "Significant label drift detected. "
                    "Review labeling consistency and consider retraining."
                )
            else:
                recommendations.append(
                    "Moderate label drift detected. Monitor closely."
                )

        if not recommendations:
            recommendations.append(
                "Minor drift detected. Increase monitoring frequency."
            )

        return recommendations


class DriftMonitor:
    """Continuous drift monitoring with history tracking.

    Tracks drift over time and provides alerting capabilities.
    """

    def __init__(
        self,
        detector: DriftDetector | None = None,
        alert_threshold: DriftSeverity = DriftSeverity.MEDIUM,
        max_history: int = 100,
    ):
        """Initialize the drift monitor.

        Args:
            detector: DriftDetector instance (created if not provided)
            alert_threshold: Severity threshold for alerts
            max_history: Maximum number of reports to keep in history
        """
        self.detector = detector or DriftDetector()
        self.alert_threshold = alert_threshold
        self.max_history = max_history

        self._history: list[DriftReport] = []
        self._alert_callbacks: list[Callable[[DriftReport], None]] = []

    def set_reference(
        self,
        reference: pd.DataFrame,
        labels: np.ndarray | None = None,
        label_column: str | None = None,
    ) -> "DriftMonitor":
        """Set the reference data for monitoring.

        Args:
            reference: Reference DataFrame
            labels: Optional label array
            label_column: Optional label column name

        Returns:
            Self for chaining
        """
        self.detector.fit(reference, labels, label_column)
        return self

    def check(
        self,
        current: pd.DataFrame,
        labels: np.ndarray | None = None,
        label_column: str | None = None,
    ) -> DriftReport:
        """Check current data for drift.

        Args:
            current: Current data to check
            labels: Optional label array
            label_column: Optional label column name

        Returns:
            DriftReport with results
        """
        report = self.detector.detect(current, labels, label_column)

        # Add to history
        self._history.append(report)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        # Check for alerts
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]

        if severity_order.index(report.overall_severity) >= severity_order.index(
            self.alert_threshold
        ):
            self._trigger_alerts(report)

        return report

    def add_alert_callback(
        self, callback: Callable[[DriftReport], None]
    ) -> "DriftMonitor":
        """Add a callback function for drift alerts.

        Args:
            callback: Function to call when drift exceeds threshold

        Returns:
            Self for chaining
        """
        self._alert_callbacks.append(callback)
        return self

    def _trigger_alerts(self, report: DriftReport) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(report)
            except Exception:
                pass  # Don't let callback errors break monitoring

    def get_history(self) -> list[DriftReport]:
        """Get drift detection history."""
        return self._history.copy()

    def get_trend(self, feature: str | None = None) -> pd.DataFrame:
        """Get drift score trend over time.

        Args:
            feature: Optional specific feature to track

        Returns:
            DataFrame with drift scores over time
        """
        rows = []
        for report in self._history:
            row = {
                "timestamp": report.timestamp,
                "overall_score": report.overall_drift_score,
                "severity": report.overall_severity.value,
                "has_drift": report.has_drift,
            }

            if feature:
                feat_drift = next(
                    (f for f in report.feature_drifts if f.feature_name == feature),
                    None,
                )
                if feat_drift:
                    row["feature_score"] = feat_drift.drift_score
                    row["feature_has_drift"] = feat_drift.has_drift

            rows.append(row)

        return pd.DataFrame(rows)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    label_column: str | None = None,
    **kwargs: Any,
) -> DriftReport:
    """Detect drift between reference and current datasets.

    Args:
        reference: Reference DataFrame
        current: Current DataFrame to compare
        label_column: Optional label column name
        **kwargs: Additional arguments for DriftDetector

    Returns:
        DriftReport with detection results
    """
    detector = DriftDetector(**kwargs)
    detector.fit(reference, label_column=label_column)
    return detector.detect(current, label_column=label_column)
