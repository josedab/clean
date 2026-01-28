"""Quality scoring system for datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.core.types import DetectionResults, QualityScore
from clean.detection.base import DetectorResult
from clean.scoring.metrics import (
    ScoringWeights,
    compute_bias_quality_score,
    compute_duplicate_quality_score,
    compute_imbalance_quality_score,
    compute_label_quality_score,
    compute_outlier_quality_score,
    severity_from_score,
)

if TYPE_CHECKING:
    pass


class QualityScorer:
    """Compute overall and component quality scores for a dataset."""

    def __init__(
        self,
        weights: ScoringWeights | None = None,
    ):
        """Initialize the quality scorer.

        Args:
            weights: Custom weights for score components
        """
        self.weights = (weights or ScoringWeights()).normalize()
        self._scores: QualityScore | None = None

    def compute_score(
        self,
        n_samples: int,
        results: DetectionResults | None = None,
        labels: np.ndarray | None = None,
        features: pd.DataFrame | None = None,
        *,
        # Legacy parameters for backward compatibility
        label_result: DetectorResult | None = None,
        duplicate_result: DetectorResult | None = None,
        outlier_result: DetectorResult | None = None,
        imbalance_result: DetectorResult | None = None,
        bias_result: DetectorResult | None = None,
    ) -> QualityScore:
        """Compute comprehensive quality scores.

        Args:
            n_samples: Number of samples in dataset
            results: DetectionResults container with all detector results (preferred)
            labels: Label array for per-class scoring
            features: Feature DataFrame for per-feature scoring
            label_result: (Legacy) Result from label error detection
            duplicate_result: (Legacy) Result from duplicate detection
            outlier_result: (Legacy) Result from outlier detection
            imbalance_result: (Legacy) Result from imbalance detection
            bias_result: (Legacy) Result from bias detection

        Returns:
            QualityScore with all components

        Note:
            You can either pass a DetectionResults object via `results`,
            or pass individual results via the legacy parameters.
            The `results` parameter takes precedence if both are provided.
        """
        # Use DetectionResults if provided, otherwise fall back to individual params
        if results is not None:
            label_result = results.label
            duplicate_result = results.duplicate
            outlier_result = results.outlier
            imbalance_result = results.imbalance
            bias_result = results.bias

        # Compute component scores
        label_score = self._compute_label_score(n_samples, label_result)
        duplicate_score = self._compute_duplicate_score(n_samples, duplicate_result)
        outlier_score = self._compute_outlier_score(n_samples, outlier_result)
        imbalance_score = self._compute_imbalance_score(imbalance_result)
        bias_score = self._compute_bias_score(bias_result)

        # Compute weighted overall score
        overall = (
            label_score * self.weights.label_errors
            + duplicate_score * self.weights.duplicates
            + outlier_score * self.weights.outliers
            + imbalance_score * self.weights.imbalance
            + bias_score * self.weights.bias
        )

        # Compute per-class scores
        per_class = {}
        if labels is not None and label_result is not None:
            per_class = self._compute_per_class_scores(labels, label_result)

        # Compute per-feature scores
        per_feature = {}
        if features is not None and outlier_result is not None:
            per_feature = self._compute_per_feature_scores(features, outlier_result)

        self._scores = QualityScore(
            overall=round(overall, 2),
            label_quality=round(label_score, 2),
            duplicate_quality=round(duplicate_score, 2),
            outlier_quality=round(outlier_score, 2),
            imbalance_quality=round(imbalance_score, 2),
            bias_quality=round(bias_score, 2),
            per_class={k: round(v, 2) for k, v in per_class.items()},
            per_feature={k: round(v, 2) for k, v in per_feature.items()},
        )

        return self._scores

    def _compute_label_score(
        self, n_samples: int, result: DetectorResult | None
    ) -> float:
        """Compute label quality score."""
        if result is None:
            return 100.0

        n_errors = result.n_issues
        avg_confidence = 0.0
        if n_errors > 0:
            confidences = [issue.confidence for issue in result.issues]
            avg_confidence = np.mean(confidences)

        return compute_label_quality_score(n_samples, n_errors, avg_confidence)

    def _compute_duplicate_score(
        self, n_samples: int, result: DetectorResult | None
    ) -> float:
        """Compute duplicate quality score."""
        if result is None:
            return 100.0

        n_pairs = result.n_issues
        n_exact = result.metadata.get("n_exact", 0)

        return compute_duplicate_quality_score(n_samples, n_pairs, n_exact)

    def _compute_outlier_score(
        self, n_samples: int, result: DetectorResult | None
    ) -> float:
        """Compute outlier quality score."""
        if result is None:
            return 100.0

        n_outliers = result.n_issues
        contamination = result.metadata.get("contamination", 0.1)

        return compute_outlier_quality_score(n_samples, n_outliers, contamination)

    def _compute_imbalance_score(self, result: DetectorResult | None) -> float:
        """Compute imbalance quality score."""
        if result is None:
            return 100.0

        imbalance_ratio = result.metadata.get("imbalance_ratio", 1.0)
        n_classes = result.metadata.get("n_classes", 2)

        return compute_imbalance_quality_score(imbalance_ratio, n_classes)

    def _compute_bias_score(self, result: DetectorResult | None) -> float:
        """Compute bias quality score."""
        if result is None:
            return 100.0

        n_issues = result.n_issues
        max_dp_diff = 0.0
        has_corr = False

        for issue in result.issues:
            if hasattr(issue, "metric"):
                if issue.metric == "demographic_parity":
                    max_dp_diff = max(max_dp_diff, issue.value)
                elif issue.metric == "label_correlation":
                    has_corr = True

        return compute_bias_quality_score(n_issues, max_dp_diff, has_corr)

    def _compute_per_class_scores(
        self, labels: np.ndarray, label_result: DetectorResult
    ) -> dict[Any, float]:
        """Compute quality scores per class."""
        classes, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(classes, counts))

        # Count errors per class
        class_errors: dict[Any, int] = dict.fromkeys(classes, 0)
        for issue in label_result.issues:
            given = issue.given_label
            if given in class_errors:
                class_errors[given] += 1

        # Compute per-class score
        per_class = {}
        for cls in classes:
            n = class_counts[cls]
            n_err = class_errors[cls]
            per_class[cls] = compute_label_quality_score(n, n_err)

        return per_class

    def _compute_per_feature_scores(
        self, features: pd.DataFrame, outlier_result: DetectorResult
    ) -> dict[str, float]:
        """Compute quality scores per feature."""
        feature_issues: dict[str, int] = dict.fromkeys(features.columns, 0)

        # Count outliers affecting each feature
        for issue in outlier_result.issues:
            if hasattr(issue, "features_contributing"):
                for feat in issue.features_contributing:
                    if feat in feature_issues:
                        feature_issues[feat] += 1

        # Compute per-feature score
        n_samples = len(features)
        per_feature = {}
        for col in features.columns:
            n_issues = feature_issues.get(col, 0)
            per_feature[col] = compute_outlier_quality_score(n_samples, n_issues, 0.1)

        return per_feature

    def get_severity(self) -> str:
        """Get overall severity level.

        Returns:
            Severity string: excellent, good, moderate, poor, critical
        """
        if self._scores is None:
            return "unknown"
        return severity_from_score(self._scores.overall)

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Get recommendations for improving data quality.

        Returns:
            List of recommendation dictionaries
        """
        if self._scores is None:
            return []

        recommendations = []

        if self._scores.label_quality < 80:
            recommendations.append({
                "priority": "high" if self._scores.label_quality < 60 else "medium",
                "category": "labels",
                "action": "Review and correct label errors",
                "impact": "Significant accuracy improvement expected",
            })

        if self._scores.duplicate_quality < 80:
            recommendations.append({
                "priority": "high" if self._scores.duplicate_quality < 60 else "medium",
                "category": "duplicates",
                "action": "Remove or deduplicate similar samples",
                "impact": "Reduce overfitting and data leakage",
            })

        if self._scores.outlier_quality < 80:
            recommendations.append({
                "priority": "medium",
                "category": "outliers",
                "action": "Review and handle outliers",
                "impact": "Improve model robustness",
            })

        if self._scores.imbalance_quality < 70:
            recommendations.append({
                "priority": "medium",
                "category": "imbalance",
                "action": "Apply resampling or class weights",
                "impact": "Better minority class performance",
            })

        if self._scores.bias_quality < 80:
            recommendations.append({
                "priority": "high",
                "category": "bias",
                "action": "Investigate and mitigate bias",
                "impact": "Improved fairness and compliance",
            })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r["priority"], 2))

        return recommendations
