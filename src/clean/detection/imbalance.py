"""Class imbalance detection."""

from typing import Any

import numpy as np
import pandas as pd

from clean.core.types import ClassDistribution
from clean.detection.base import BaseDetector, DetectorResult


class ImbalanceDetector(BaseDetector):
    """Detect class imbalance in classification datasets.

    Analyzes class distribution and flags severe imbalances that
    may affect model performance.
    """

    def __init__(
        self,
        imbalance_threshold: float = 5.0,
        minority_threshold: float = 0.05,
    ):
        """Initialize the imbalance detector.

        Args:
            imbalance_threshold: Ratio threshold to flag imbalance
            minority_threshold: Minimum ratio for minority class
        """
        super().__init__(
            imbalance_threshold=imbalance_threshold,
            minority_threshold=minority_threshold,
        )
        self.imbalance_threshold = imbalance_threshold
        self.minority_threshold = minority_threshold

        self._distribution: ClassDistribution | None = None

    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> "ImbalanceDetector":
        """Fit the detector by computing class distribution.

        Args:
            features: Feature data (not used)
            labels: Label data (required)

        Returns:
            Self for chaining

        Raises:
            ValueError: If labels not provided
        """
        if labels is None:
            raise ValueError("Labels required for imbalance detection")

        self._compute_distribution(labels)
        self._is_fitted = True
        return self

    def _compute_distribution(self, labels: np.ndarray) -> None:
        """Compute class distribution statistics."""
        y = np.asarray(labels)
        unique, counts = np.unique(y, return_counts=True)

        total = len(y)
        class_counts = dict(zip(unique, counts))
        class_ratios = {k: v / total for k, v in class_counts.items()}

        max_count = max(counts)
        min_count = min(counts)

        majority_idx = np.argmax(counts)
        minority_idx = np.argmin(counts)

        self._distribution = ClassDistribution(
            class_counts={k: int(v) for k, v in class_counts.items()},
            class_ratios={k: float(v) for k, v in class_ratios.items()},
            imbalance_ratio=float(max_count / min_count) if min_count > 0 else float("inf"),
            majority_class=unique[majority_idx],
            minority_class=unique[minority_idx],
        )

    def detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Detect class imbalance issues.

        Args:
            features: Not used
            labels: Label data

        Returns:
            DetectorResult with imbalance information
        """
        self._check_fitted()

        assert self._distribution is not None
        dist = self._distribution

        issues: list[dict[str, Any]] = []

        # Check overall imbalance
        if dist.imbalance_ratio >= self.imbalance_threshold:
            issues.append({
                "type": "severe_imbalance",
                "imbalance_ratio": dist.imbalance_ratio,
                "majority_class": dist.majority_class,
                "minority_class": dist.minority_class,
                "severity": "high" if dist.imbalance_ratio >= 10 else "medium",
            })

        # Check minority class proportion
        minority_ratio = dist.class_ratios.get(dist.minority_class, 0)
        if minority_ratio < self.minority_threshold:
            issues.append({
                "type": "minority_underrepresented",
                "class": dist.minority_class,
                "ratio": minority_ratio,
                "threshold": self.minority_threshold,
                "severity": "high" if minority_ratio < 0.01 else "medium",
            })

        # Check for classes with very few samples
        for cls, count in dist.class_counts.items():
            if count < 10:
                issues.append({
                    "type": "insufficient_samples",
                    "class": cls,
                    "count": count,
                    "severity": "high",
                })

        # Helper to convert numpy types to native Python types
        def convert_val(v: Any) -> Any:
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            return v

        def convert_dict(d: dict) -> dict:
            return {convert_val(k): convert_val(v) for k, v in d.items()}

        metadata = {
            "n_classes": len(dist.class_counts),
            "imbalance_ratio": float(dist.imbalance_ratio),
            "majority_class": convert_val(dist.majority_class),
            "minority_class": convert_val(dist.minority_class),
            "class_counts": convert_dict(dist.class_counts),
            "class_ratios": convert_dict(dist.class_ratios),
            "is_imbalanced": dist.imbalance_ratio >= self.imbalance_threshold,
        }

        # Return issues as simple objects for DataFrame conversion
        return DetectorResult(issues=issues, metadata=metadata)

    def get_distribution(self) -> ClassDistribution | None:
        """Get the computed class distribution."""
        return self._distribution

    def get_resampling_suggestion(self) -> dict[str, Any]:
        """Get suggestions for handling imbalance.

        Returns:
            Dictionary with resampling recommendations
        """
        self._check_fitted()
        assert self._distribution is not None
        dist = self._distribution

        suggestions: dict[str, Any] = {"strategies": []}

        if dist.imbalance_ratio >= self.imbalance_threshold:
            if dist.imbalance_ratio >= 10:
                suggestions["strategies"].append({
                    "name": "SMOTE",
                    "description": "Synthetic Minority Over-sampling Technique",
                    "reason": "Severe imbalance - synthetic oversampling recommended",
                })
                suggestions["strategies"].append({
                    "name": "class_weights",
                    "description": "Use class weights in model training",
                    "weights": {
                        k: dist.class_counts[dist.majority_class] / v
                        for k, v in dist.class_counts.items()
                    },
                })
            else:
                suggestions["strategies"].append({
                    "name": "random_oversample",
                    "description": "Random oversampling of minority classes",
                    "reason": "Moderate imbalance",
                })

        suggestions["target_distribution"] = {
            k: 1.0 / len(dist.class_counts) for k in dist.class_counts
        }

        return suggestions


def analyze_imbalance(
    labels: np.ndarray,
    imbalance_threshold: float = 5.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze class imbalance in labels.

    Args:
        labels: Label array
        imbalance_threshold: Ratio threshold
        **kwargs: Additional arguments

    Returns:
        Dictionary with imbalance analysis
    """
    detector = ImbalanceDetector(imbalance_threshold=imbalance_threshold, **kwargs)
    detector.fit(None, labels)  # type: ignore

    dist = detector.get_distribution()
    result = detector.detect(None, labels)  # type: ignore
    suggestions = detector.get_resampling_suggestion()

    return {
        "distribution": dist.to_dict() if dist else {},
        "issues": result.issues,
        "metadata": result.metadata,
        "suggestions": suggestions,
    }
