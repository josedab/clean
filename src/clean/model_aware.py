"""Model-Aware Quality Scoring.

This module provides quality scoring relative to a specific model's
confusion patterns and failure modes, making quality metrics actionable
for specific use cases.

Example:
    >>> from clean.model_aware import ModelAwareScorer
    >>>
    >>> scorer = ModelAwareScorer(model=my_classifier)
    >>> scorer.fit(X_train, y_train)
    >>> report = scorer.score(X_test, y_test)
    >>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict

if TYPE_CHECKING:
    pass


class ImpactLevel(Enum):
    """Impact level of a quality issue on model performance."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class SampleQuality:
    """Quality assessment for a single sample."""

    index: int
    quality_score: float  # 0-100
    model_confidence: float  # 0-1
    is_correctly_predicted: bool
    is_in_confusion_zone: bool
    predicted_label: Any
    true_label: Any
    impact_on_model: ImpactLevel
    issues: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "quality_score": self.quality_score,
            "model_confidence": self.model_confidence,
            "is_correctly_predicted": self.is_correctly_predicted,
            "is_in_confusion_zone": self.is_in_confusion_zone,
            "predicted_label": self.predicted_label,
            "true_label": self.true_label,
            "impact_on_model": self.impact_on_model.value,
            "issues": self.issues,
        }


@dataclass
class ClassMetrics:
    """Quality metrics for a single class."""

    label: Any
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_rate: float  # How often confused with other classes
    avg_confidence: float
    quality_score: float
    most_confused_with: list[tuple[Any, float]]  # [(class, rate), ...]


@dataclass
class ModelAwareReport:
    """Complete model-aware quality report."""

    n_samples: int
    overall_quality_score: float  # 0-100
    model_accuracy: float
    model_confidence_mean: float
    model_confidence_std: float

    # Per-sample scores
    sample_scores: list[SampleQuality]

    # Per-class metrics
    class_metrics: dict[Any, ClassMetrics]

    # Confusion analysis
    high_confusion_pairs: list[tuple[Any, Any, float]]  # [(class1, class2, rate), ...]

    # Impact analysis
    critical_samples: list[int]  # Indices of samples with critical impact
    high_impact_samples: list[int]

    # Recommendations
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Model-Aware Quality Report",
            "=" * 50,
            "",
            f"Samples analyzed: {self.n_samples:,}",
            f"Overall Quality Score: {self.overall_quality_score:.1f}/100",
            "",
            "Model Performance:",
            f"  Accuracy: {self.model_accuracy:.1%}",
            f"  Mean Confidence: {self.model_confidence_mean:.3f}",
            f"  Confidence Std: {self.model_confidence_std:.3f}",
            "",
        ]

        if self.high_confusion_pairs:
            lines.append("High Confusion Pairs:")
            for c1, c2, rate in self.high_confusion_pairs[:5]:
                lines.append(f"  • '{c1}' ↔ '{c2}': {rate:.1%}")
            lines.append("")

        if self.critical_samples:
            lines.append(f"Critical samples: {len(self.critical_samples)}")
        if self.high_impact_samples:
            lines.append(f"High-impact samples: {len(self.high_impact_samples)}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "overall_quality_score": self.overall_quality_score,
            "model_accuracy": self.model_accuracy,
            "model_confidence_mean": self.model_confidence_mean,
            "critical_samples": len(self.critical_samples),
            "high_impact_samples": len(self.high_impact_samples),
        }

    def get_samples_by_impact(self, level: ImpactLevel) -> list[SampleQuality]:
        """Get samples with specific impact level."""
        return [s for s in self.sample_scores if s.impact_on_model == level]

    def get_worst_classes(self, n: int = 5) -> list[ClassMetrics]:
        """Get classes with worst quality scores."""
        classes = list(self.class_metrics.values())
        classes.sort(key=lambda x: x.quality_score)
        return classes[:n]


class ModelAwareScorer:
    """Score data quality relative to a specific model's behavior.

    Identifies samples that are particularly problematic for the given
    model, based on confusion patterns and confidence calibration.
    """

    def __init__(
        self,
        model: BaseEstimator | None = None,
        cv_folds: int = 5,
        confidence_threshold: float = 0.7,
        confusion_threshold: float = 0.1,
    ):
        """Initialize the scorer.

        Args:
            model: Sklearn classifier (must have predict_proba)
            cv_folds: Cross-validation folds for scoring
            confidence_threshold: Threshold for low confidence flagging
            confusion_threshold: Threshold for high confusion flagging
        """
        self.model = model
        self.cv_folds = cv_folds
        self.confidence_threshold = confidence_threshold
        self.confusion_threshold = confusion_threshold

        self._fitted = False
        self._classes: np.ndarray | None = None
        self._confusion_matrix: np.ndarray | None = None
        self._class_priors: dict[Any, float] = {}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> ModelAwareScorer:
        """Fit the scorer to training data.

        Args:
            X: Feature data
            y: Labels

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number]).values

        self._classes = np.unique(y)

        # Calculate class priors
        for cls in self._classes:
            self._class_priors[cls] = (y == cls).mean()

        # Fit model and get cross-validated predictions
        if self.model is None:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000, random_state=42)

        # Get cross-validated probabilities
        cv = min(self.cv_folds, len(np.unique(y)))
        pred_proba = cross_val_predict(
            clone(self.model),
            X, y,
            cv=cv,
            method="predict_proba",
        )

        # Build confusion matrix from CV predictions
        pred = self._classes[pred_proba.argmax(axis=1)]
        self._confusion_matrix = self._build_confusion_matrix(y, pred)

        # Fit the model on full data for inference
        self.model.fit(X, y)

        self._fitted = True
        return self

    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> ModelAwareReport:
        """Score samples for model-aware quality.

        Args:
            X: Feature data
            y: Labels

        Returns:
            ModelAwareReport with quality assessments
        """
        if not self._fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        # Get predictions and probabilities
        pred_proba = self.model.predict_proba(X_arr)
        pred = self.model.predict(X_arr)

        # Calculate per-sample scores
        sample_scores = []
        for i in range(len(X_arr)):
            score = self._score_sample(
                index=i,
                true_label=y_arr[i],
                predicted_label=pred[i],
                probabilities=pred_proba[i],
            )
            sample_scores.append(score)

        # Calculate class metrics
        class_metrics = self._calculate_class_metrics(y_arr, pred, pred_proba)

        # Calculate confusion pairs
        confusion_pairs = self._get_high_confusion_pairs()

        # Identify critical and high-impact samples
        critical_samples = [
            s.index for s in sample_scores
            if s.impact_on_model == ImpactLevel.CRITICAL
        ]
        high_impact_samples = [
            s.index for s in sample_scores
            if s.impact_on_model == ImpactLevel.HIGH
        ]

        # Overall metrics
        n_samples = len(X_arr)
        accuracy = (pred == y_arr).mean()
        confidence_mean = pred_proba.max(axis=1).mean()
        confidence_std = pred_proba.max(axis=1).std()

        # Overall quality score
        quality_scores = [s.quality_score for s in sample_scores]
        overall_quality = np.mean(quality_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            sample_scores, class_metrics, confusion_pairs, accuracy
        )

        return ModelAwareReport(
            n_samples=n_samples,
            overall_quality_score=float(overall_quality),
            model_accuracy=float(accuracy),
            model_confidence_mean=float(confidence_mean),
            model_confidence_std=float(confidence_std),
            sample_scores=sample_scores,
            class_metrics=class_metrics,
            high_confusion_pairs=confusion_pairs,
            critical_samples=critical_samples,
            high_impact_samples=high_impact_samples,
            recommendations=recommendations,
        )

    def _score_sample(
        self,
        index: int,
        true_label: Any,
        predicted_label: Any,
        probabilities: np.ndarray,
    ) -> SampleQuality:
        """Score a single sample."""
        confidence = float(probabilities.max())
        is_correct = true_label == predicted_label

        issues = []

        # Check if in confusion zone
        sorted_probs = np.sort(probabilities)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        is_confusion_zone = margin < 0.2

        if is_confusion_zone:
            issues.append("Sample in model confusion zone")

        # Check confidence
        if confidence < self.confidence_threshold:
            issues.append(f"Low model confidence ({confidence:.2f})")

        # Check if this class pair is commonly confused
        if not is_correct and self._confusion_matrix is not None:
            true_idx = list(self._classes).index(true_label)
            pred_idx = list(self._classes).index(predicted_label)
            confusion_rate = self._confusion_matrix[true_idx, pred_idx]

            if confusion_rate > self.confusion_threshold:
                issues.append(f"Common confusion: {true_label} → {predicted_label}")

        # Calculate quality score
        quality_score = self._calculate_sample_quality(
            is_correct, confidence, is_confusion_zone, len(issues)
        )

        # Determine impact level
        impact = self._determine_impact(
            is_correct, confidence, is_confusion_zone, true_label
        )

        return SampleQuality(
            index=index,
            quality_score=quality_score,
            model_confidence=confidence,
            is_correctly_predicted=is_correct,
            is_in_confusion_zone=is_confusion_zone,
            predicted_label=predicted_label,
            true_label=true_label,
            impact_on_model=impact,
            issues=issues,
        )

    def _calculate_sample_quality(
        self,
        is_correct: bool,
        confidence: float,
        is_confusion_zone: bool,
        n_issues: int,
    ) -> float:
        """Calculate quality score for a sample."""
        score = 100.0

        # Penalty for incorrect prediction
        if not is_correct:
            score -= 30.0

        # Penalty for low confidence
        if confidence < self.confidence_threshold:
            score -= (self.confidence_threshold - confidence) * 50

        # Penalty for confusion zone
        if is_confusion_zone:
            score -= 15.0

        # Penalty per issue
        score -= n_issues * 5.0

        return max(0.0, min(100.0, score))

    def _determine_impact(
        self,
        is_correct: bool,
        confidence: float,
        is_confusion_zone: bool,
        true_label: Any,
    ) -> ImpactLevel:
        """Determine the impact level of a sample on model performance."""
        # High-confidence wrong prediction is critical
        if not is_correct and confidence > 0.8:
            return ImpactLevel.CRITICAL

        # Wrong prediction on rare class is high impact
        if not is_correct:
            class_prior = self._class_priors.get(true_label, 0.5)
            if class_prior < 0.1:
                return ImpactLevel.HIGH
            return ImpactLevel.HIGH if confidence > 0.5 else ImpactLevel.MEDIUM

        # Correct but in confusion zone
        if is_confusion_zone:
            return ImpactLevel.MEDIUM

        # Correct but low confidence
        if confidence < self.confidence_threshold:
            return ImpactLevel.LOW

        return ImpactLevel.NEGLIGIBLE

    def _calculate_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pred_proba: np.ndarray,
    ) -> dict[Any, ClassMetrics]:
        """Calculate metrics per class."""
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self._classes, zero_division=0
        )

        class_metrics = {}

        for i, cls in enumerate(self._classes):
            mask = y_true == cls
            n_samples = int(mask.sum())

            if n_samples == 0:
                continue

            class_accuracy = (y_pred[mask] == y_true[mask]).mean()
            class_confidence = pred_proba[mask, i].mean() if i < pred_proba.shape[1] else 0

            # Calculate confusion rate
            wrong_mask = mask & (y_pred != y_true)
            confusion_rate = wrong_mask.sum() / n_samples if n_samples > 0 else 0

            # Find most confused classes
            confused_with = []
            if self._confusion_matrix is not None:
                row = self._confusion_matrix[i]
                for j, rate in enumerate(row):
                    if j != i and rate > 0:
                        confused_with.append((self._classes[j], rate))
                confused_with.sort(key=lambda x: x[1], reverse=True)

            # Quality score for class
            quality_score = (
                class_accuracy * 40 +
                precision[i] * 20 +
                recall[i] * 20 +
                (1 - confusion_rate) * 20
            )

            class_metrics[cls] = ClassMetrics(
                label=cls,
                n_samples=n_samples,
                accuracy=float(class_accuracy),
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1_score=float(f1[i]),
                confusion_rate=float(confusion_rate),
                avg_confidence=float(class_confidence),
                quality_score=float(quality_score),
                most_confused_with=confused_with[:3],
            )

        return class_metrics

    def _build_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Build normalized confusion matrix."""
        n_classes = len(self._classes)
        matrix = np.zeros((n_classes, n_classes))

        class_to_idx = {cls: i for i, cls in enumerate(self._classes)}

        for true, pred in zip(y_true, y_pred):
            i = class_to_idx.get(true)
            j = class_to_idx.get(pred)
            if i is not None and j is not None:
                matrix[i, j] += 1

        # Normalize by row (true class)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums

        return matrix

    def _get_high_confusion_pairs(self) -> list[tuple[Any, Any, float]]:
        """Get class pairs with high confusion rates."""
        if self._confusion_matrix is None or self._classes is None:
            return []

        pairs = []
        n_classes = len(self._classes)

        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    rate = self._confusion_matrix[i, j]
                    if rate > self.confusion_threshold:
                        pairs.append((
                            self._classes[i],
                            self._classes[j],
                            float(rate),
                        ))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _generate_recommendations(
        self,
        sample_scores: list[SampleQuality],
        class_metrics: dict[Any, ClassMetrics],
        confusion_pairs: list[tuple[Any, Any, float]],
        accuracy: float,
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Critical samples
        critical = [s for s in sample_scores if s.impact_on_model == ImpactLevel.CRITICAL]
        if len(critical) > 0:
            recommendations.append(
                f"Review {len(critical)} critical samples with high-confidence errors - "
                f"these may be label errors or important edge cases."
            )

        # High confusion pairs
        if confusion_pairs:
            c1, c2, rate = confusion_pairs[0]
            recommendations.append(
                f"Classes '{c1}' and '{c2}' are frequently confused ({rate:.1%}). "
                f"Consider clarifying labeling guidelines or adding more examples."
            )

        # Low-performing classes
        worst_classes = sorted(class_metrics.values(), key=lambda x: x.quality_score)[:3]
        for cls in worst_classes:
            if cls.quality_score < 70:
                recommendations.append(
                    f"Class '{cls.label}' has low quality ({cls.quality_score:.0f}/100). "
                    f"It may need more training examples or cleaner labels."
                )

        # Overall accuracy
        if accuracy < 0.8:
            low_confidence = [s for s in sample_scores if s.model_confidence < 0.5]
            recommendations.append(
                f"Model accuracy ({accuracy:.1%}) suggests data quality issues. "
                f"{len(low_confidence)} samples have very low confidence."
            )

        # Class imbalance impact
        class_sizes = [m.n_samples for m in class_metrics.values()]
        if max(class_sizes) > min(class_sizes) * 5:
            recommendations.append(
                "Significant class imbalance detected. "
                "Consider oversampling minority classes or collecting more data."
            )

        if not recommendations:
            recommendations.append("Data quality looks good for this model!")

        return recommendations


def score_with_model(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    model: BaseEstimator | None = None,
    **kwargs: Any,
) -> ModelAwareReport:
    """Convenience function for model-aware scoring.

    Args:
        X: Feature data
        y: Labels
        model: Sklearn classifier
        **kwargs: Additional arguments for ModelAwareScorer

    Returns:
        ModelAwareReport with quality assessments
    """
    scorer = ModelAwareScorer(model=model, **kwargs)
    scorer.fit(X, y)
    return scorer.score(X, y)


class ModelComparisonScorer:
    """Compare data quality across multiple models.

    Identifies samples that are problematic for all models vs
    model-specific issues.
    """

    def __init__(self, models: list[BaseEstimator]):
        """Initialize comparison scorer.

        Args:
            models: List of sklearn classifiers
        """
        self.models = models
        self.scorers = [ModelAwareScorer(model=m) for m in models]

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> ModelComparisonScorer:
        """Fit all scorers."""
        for scorer in self.scorers:
            scorer.fit(X, y)
        return self

    def compare(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Compare quality across models.

        Args:
            X: Feature data
            y: Labels

        Returns:
            Comparison results
        """
        reports = [scorer.score(X, y) for scorer in self.scorers]

        # Find samples problematic for all models
        all_critical = set(reports[0].critical_samples)
        for report in reports[1:]:
            all_critical &= set(report.critical_samples)

        # Model-specific issues
        model_specific = {}
        for i, report in enumerate(reports):
            specific = set(report.critical_samples) - all_critical
            model_specific[f"model_{i}"] = list(specific)

        return {
            "n_models": len(self.models),
            "model_accuracies": [r.model_accuracy for r in reports],
            "quality_scores": [r.overall_quality_score for r in reports],
            "universal_issues": list(all_critical),
            "model_specific_issues": model_specific,
            "recommendation": (
                f"{len(all_critical)} samples are problematic for all models - "
                f"likely label errors or inherently difficult cases."
            ),
        }
