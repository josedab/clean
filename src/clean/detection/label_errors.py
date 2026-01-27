"""Label error detection using confident learning."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from clean.core.types import LabelError
from clean.detection.base import BaseDetector, DetectorResult

# Optional cleanlab import
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False


class LabelErrorDetector(BaseDetector):
    """Detect label errors using confident learning.

    Uses cross-validation to obtain out-of-sample predicted probabilities,
    then applies confident learning to find samples where the given label
    likely differs from the true label.
    """

    def __init__(
        self,
        method: str = "confident_learning",
        classifier: Any | None = None,
        cv_folds: int = 5,
        confidence_threshold: float = 0.5,
        n_jobs: int = -1,
    ):
        """Initialize the label error detector.

        Args:
            method: Detection method ('confident_learning', 'self_confidence', 'both')
            classifier: Sklearn-compatible classifier (default: LogisticRegression)
            cv_folds: Number of cross-validation folds
            confidence_threshold: Minimum confidence to flag as error (0-1)
            n_jobs: Number of parallel jobs for cross-validation
        """
        super().__init__(
            method=method,
            cv_folds=cv_folds,
            confidence_threshold=confidence_threshold,
            n_jobs=n_jobs,
        )
        self.method = method
        self.classifier = classifier or LogisticRegression(
            max_iter=1000, n_jobs=n_jobs, random_state=42
        )
        self.cv_folds = cv_folds
        self.confidence_threshold = confidence_threshold
        self.n_jobs = n_jobs

        self._pred_probs: np.ndarray | None = None
        self._classes: np.ndarray | None = None

    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> "LabelErrorDetector":
        """Fit the detector by computing cross-validated predictions.

        Args:
            features: Feature data
            labels: Label data (required for label error detection)

        Returns:
            Self for chaining

        Raises:
            ValueError: If labels not provided
        """
        if labels is None:
            raise ValueError("Labels required for label error detection")

        # Convert to numpy, handling mixed types
        if isinstance(features, pd.DataFrame):
            # Use only numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric features available for label error detection")
            X = features[numeric_cols].values
        else:
            X = np.asarray(features)

        y = np.asarray(labels)

        # Handle any NaN in features (only for numeric data)
        if np.issubdtype(X.dtype, np.number) and np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)

        # Get unique classes
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes for label error detection")

        # Adjust cv_folds if needed
        cv_folds = min(self.cv_folds, len(y))

        # Check minimum samples per class
        class_counts = np.bincount(
            np.searchsorted(self._classes, y), minlength=n_classes
        )
        min_samples = class_counts.min()
        if min_samples < cv_folds:
            cv_folds = max(2, min_samples)

        # Get cross-validated predicted probabilities
        self._pred_probs = cross_val_predict(
            self.classifier, X, y, cv=cv_folds, method="predict_proba", n_jobs=self.n_jobs
        )

        self._is_fitted = True
        return self

    def detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Detect label errors in the data.

        Args:
            features: Feature data
            labels: Label data (required)

        Returns:
            DetectorResult with LabelError objects
        """
        self._check_fitted()

        if labels is None:
            raise ValueError("Labels required for label error detection")

        y = np.asarray(labels)
        pred_probs = self._pred_probs

        assert pred_probs is not None
        assert self._classes is not None

        errors: list[LabelError] = []

        if HAS_CLEANLAB and self.method in ("confident_learning", "both"):
            # Use cleanlab for confident learning
            issue_mask = find_label_issues(
                labels=y,
                pred_probs=pred_probs,
                return_indices_ranked_by="self_confidence",
            )

            # Get quality scores
            quality_scores = get_label_quality_scores(labels=y, pred_probs=pred_probs)

            for idx in issue_mask:
                if quality_scores[idx] < (1 - self.confidence_threshold):
                    given_label = y[idx]
                    pred_class_idx = np.argmax(pred_probs[idx])
                    predicted_label = self._classes[pred_class_idx]
                    confidence = 1 - quality_scores[idx]
                    self_conf = pred_probs[idx][
                        np.searchsorted(self._classes, given_label)
                    ]

                    errors.append(
                        LabelError(
                            index=int(idx),
                            given_label=given_label,
                            predicted_label=predicted_label,
                            confidence=float(confidence),
                            self_confidence=float(self_conf),
                        )
                    )

        elif self.method == "self_confidence" or not HAS_CLEANLAB:
            # Fallback: use self-confidence method
            for idx in range(len(y)):
                given_label = y[idx]
                class_idx = np.searchsorted(self._classes, given_label)
                self_conf = pred_probs[idx][class_idx]
                pred_class_idx = np.argmax(pred_probs[idx])
                predicted_label = self._classes[pred_class_idx]

                # Flag if model is confident about a different class
                if self_conf < (1 - self.confidence_threshold):
                    max_prob = pred_probs[idx][pred_class_idx]
                    if max_prob > self.confidence_threshold and pred_class_idx != class_idx:
                        errors.append(
                            LabelError(
                                index=int(idx),
                                given_label=given_label,
                                predicted_label=predicted_label,
                                confidence=float(max_prob),
                                self_confidence=float(self_conf),
                            )
                        )

        # Sort by confidence (highest first)
        errors.sort(key=lambda e: e.confidence, reverse=True)

        metadata = {
            "method": self.method,
            "cv_folds": self.cv_folds,
            "confidence_threshold": self.confidence_threshold,
            "n_samples": len(y),
            "n_classes": len(self._classes),
            "error_rate": len(errors) / len(y) if len(y) > 0 else 0,
        }

        return DetectorResult(issues=errors, metadata=metadata)

    def get_pred_probs(self) -> np.ndarray | None:
        """Get the cross-validated predicted probabilities."""
        return self._pred_probs

    def get_label_quality_scores(self, labels: np.ndarray) -> np.ndarray:
        """Get per-sample label quality scores.

        Args:
            labels: Label array

        Returns:
            Array of quality scores (0=low quality, 1=high quality)
        """
        self._check_fitted()
        assert self._pred_probs is not None
        assert self._classes is not None

        if HAS_CLEANLAB:
            return get_label_quality_scores(labels=labels, pred_probs=self._pred_probs)

        # Fallback: use self-confidence
        scores = np.zeros(len(labels))
        for idx, label in enumerate(labels):
            class_idx = np.searchsorted(self._classes, label)
            scores[idx] = self._pred_probs[idx][class_idx]
        return scores


def find_label_errors(
    features: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """Find label errors in a dataset.

    Args:
        features: Feature data
        labels: Label data
        confidence_threshold: Minimum confidence to flag as error
        **kwargs: Additional arguments for LabelErrorDetector

    Returns:
        DataFrame with columns: index, given_label, predicted_label, confidence
    """
    detector = LabelErrorDetector(confidence_threshold=confidence_threshold, **kwargs)
    result = detector.fit_detect(features, labels)
    return result.to_dataframe()
