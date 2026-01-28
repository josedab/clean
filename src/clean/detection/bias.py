"""Bias detection for fairness analysis."""

from typing import Any

import numpy as np
import pandas as pd

from clean.core.types import BiasIssue
from clean.detection.base import BaseDetector, DetectorResult


class BiasDetector(BaseDetector):
    """Detect bias and fairness issues in datasets.

    Analyzes relationships between sensitive attributes and labels/predictions
    to identify potential bias issues.
    """

    def __init__(
        self,
        sensitive_features: list[str] | None = None,
        demographic_parity_threshold: float = 0.1,
        equalized_odds_threshold: float = 0.1,
        correlation_threshold: float = 0.3,
    ):
        """Initialize the bias detector.

        Args:
            sensitive_features: List of sensitive feature names to analyze
            demographic_parity_threshold: Max allowed difference in positive rates
            equalized_odds_threshold: Max allowed difference in TPR/FPR
            correlation_threshold: Threshold for flagging correlated features
        """
        super().__init__(
            sensitive_features=sensitive_features,
            demographic_parity_threshold=demographic_parity_threshold,
            equalized_odds_threshold=equalized_odds_threshold,
            correlation_threshold=correlation_threshold,
        )
        self.sensitive_features = sensitive_features or []
        self.demographic_parity_threshold = demographic_parity_threshold
        self.equalized_odds_threshold = equalized_odds_threshold
        self.correlation_threshold = correlation_threshold

        self._feature_stats: dict[str, dict[str, Any]] = {}

    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> "BiasDetector":
        """Fit the detector by analyzing feature distributions.

        Args:
            features: Feature data
            labels: Label data

        Returns:
            Self for chaining
        """
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        # Auto-detect sensitive features if not specified
        if not self.sensitive_features:
            self.sensitive_features = self._detect_sensitive_features(features)

        # Compute statistics for each sensitive feature
        for feat in self.sensitive_features:
            if feat in features.columns:
                self._feature_stats[feat] = self._analyze_feature(
                    features[feat], labels
                )

        self._is_fitted = True
        return self

    def _detect_sensitive_features(self, df: pd.DataFrame) -> list[str]:
        """Auto-detect potentially sensitive features."""
        sensitive_keywords = [
            "gender", "sex", "age", "race", "ethnicity", "religion",
            "disability", "marital", "nationality", "country", "region",
        ]

        detected = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in sensitive_keywords) or (df[col].dtype == object and df[col].nunique() < 10):
                detected.append(col)

        return detected[:5]  # Limit to 5 features

    def _analyze_feature(
        self, feature: pd.Series, labels: np.ndarray | None
    ) -> dict[str, Any]:
        """Analyze a single feature for bias indicators."""
        stats: dict[str, Any] = {
            "unique_values": feature.nunique(),
            "value_counts": feature.value_counts().to_dict(),
        }

        if labels is not None:
            # Compute positive rate per group
            df = pd.DataFrame({"feature": feature, "label": labels})
            group_stats = df.groupby("feature")["label"].agg(["mean", "count"])
            stats["positive_rate_by_group"] = group_stats["mean"].to_dict()
            stats["count_by_group"] = group_stats["count"].to_dict()

            # Check for correlation
            if feature.dtype in [np.int64, np.float64, int, float]:
                correlation = np.corrcoef(
                    feature.fillna(0).values, labels
                )[0, 1]
                stats["correlation"] = float(correlation) if not np.isnan(correlation) else 0.0

        return stats

    def detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Detect bias issues in the data.

        Args:
            features: Feature data
            labels: Label data

        Returns:
            DetectorResult with BiasIssue objects
        """
        self._check_fitted()

        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        issues: list[BiasIssue] = []

        for feat, stats in self._feature_stats.items():
            # Check demographic parity
            if "positive_rate_by_group" in stats:
                rates = list(stats["positive_rate_by_group"].values())
                if len(rates) >= 2:
                    max_diff = max(rates) - min(rates)
                    if max_diff > self.demographic_parity_threshold:
                        groups = list(stats["positive_rate_by_group"].keys())
                        issues.append(BiasIssue(
                            feature=feat,
                            metric="demographic_parity",
                            value=float(max_diff),
                            threshold=self.demographic_parity_threshold,
                            affected_groups=[str(g) for g in groups],
                            description=(
                                f"Positive rate differs by {max_diff:.2%} across groups"
                            ),
                        ))

            # Check correlation with label
            if "correlation" in stats:
                corr = abs(stats["correlation"])
                if corr > self.correlation_threshold:
                    issues.append(BiasIssue(
                        feature=feat,
                        metric="label_correlation",
                        value=float(corr),
                        threshold=self.correlation_threshold,
                        affected_groups=[],
                        description=(
                            f"Feature has {corr:.2f} correlation with label"
                        ),
                    ))

            # Check for underrepresented groups
            if "count_by_group" in stats:
                counts = stats["count_by_group"]
                total = sum(counts.values())
                for group, count in counts.items():
                    if count / total < 0.01:  # Less than 1%
                        issues.append(BiasIssue(
                            feature=feat,
                            metric="representation",
                            value=float(count / total),
                            threshold=0.01,
                            affected_groups=[str(group)],
                            description=(
                                f"Group '{group}' is underrepresented ({count} samples)"
                            ),
                        ))

        # Sort by value (higher = more severe)
        issues.sort(key=lambda i: i.value, reverse=True)

        metadata = {
            "sensitive_features": self.sensitive_features,
            "n_features_analyzed": len(self._feature_stats),
            "n_issues": len(issues),
            "has_demographic_parity_issues": any(
                i.metric == "demographic_parity" for i in issues
            ),
            "has_correlation_issues": any(
                i.metric == "label_correlation" for i in issues
            ),
        }

        return DetectorResult(issues=issues, metadata=metadata)

    def compute_demographic_parity(
        self, feature: pd.Series, labels: np.ndarray
    ) -> dict[str, float]:
        """Compute demographic parity for a feature.

        Args:
            feature: Sensitive feature values
            labels: Label values

        Returns:
            Dictionary with positive rate per group
        """
        df = pd.DataFrame({"feature": feature, "label": labels})
        return df.groupby("feature")["label"].mean().to_dict()

    def compute_equalized_odds(
        self,
        feature: pd.Series,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Compute equalized odds metrics.

        Args:
            feature: Sensitive feature values
            labels: True labels
            predictions: Predicted labels

        Returns:
            Dictionary with TPR and FPR per group
        """
        df = pd.DataFrame({
            "feature": feature,
            "label": labels,
            "pred": predictions,
        })

        results: dict[str, dict[str, float]] = {}
        for group in df["feature"].unique():
            group_df = df[df["feature"] == group]

            # True positive rate
            positives = group_df[group_df["label"] == 1]
            tpr = (
                positives["pred"].sum() / len(positives)
                if len(positives) > 0 else 0.0
            )

            # False positive rate
            negatives = group_df[group_df["label"] == 0]
            fpr = (
                negatives["pred"].sum() / len(negatives)
                if len(negatives) > 0 else 0.0
            )

            results[str(group)] = {"tpr": float(tpr), "fpr": float(fpr)}

        return results


def analyze_bias(
    features: pd.DataFrame,
    labels: np.ndarray,
    sensitive_features: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze bias in a dataset.

    Args:
        features: Feature data
        labels: Label data
        sensitive_features: Features to analyze
        **kwargs: Additional arguments

    Returns:
        Dictionary with bias analysis
    """
    detector = BiasDetector(sensitive_features=sensitive_features, **kwargs)
    result = detector.fit_detect(features, labels)

    return {
        "issues": [i.to_dict() for i in result.issues],
        "metadata": result.metadata,
        "feature_stats": detector._feature_stats,
    }
