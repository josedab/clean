"""Outlier detection using multiple methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clean.core.types import Outlier
from clean.detection.base import BaseDetector, DetectorResult
from clean.detection.strategies import (
    FeatureStats,
    OutlierStrategy,
    create_strategy,
)


class OutlierDetector(BaseDetector):
    """Detect outliers using statistical and ML methods.

    Supports multiple detection methods:
    - isolation_forest: Isolation Forest algorithm
    - lof: Local Outlier Factor
    - zscore: Z-score based detection
    - iqr: Interquartile range method
    - ensemble: Combine multiple methods with voting

    The detector uses the Strategy pattern internally, allowing each
    detection algorithm to be implemented independently.

    Example:
        >>> detector = OutlierDetector(method="ensemble")
        >>> result = detector.fit_detect(features)
        >>> print(f"Found {result.n_issues} outliers")
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        ensemble_methods: list[str] | None = None,
        ensemble_threshold: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """Initialize the outlier detector.

        Args:
            method: Detection method ('isolation_forest', 'lof', 'zscore', 'iqr', 'ensemble')
            contamination: Expected proportion of outliers (for IF, LOF)
            zscore_threshold: Threshold for z-score method
            iqr_multiplier: Multiplier for IQR method
            ensemble_methods: Methods to combine for ensemble
            ensemble_threshold: Minimum votes to flag as outlier
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        super().__init__(
            method=method,
            contamination=contamination,
            zscore_threshold=zscore_threshold,
            iqr_multiplier=iqr_multiplier,
        )
        self.method = method
        self.contamination = contamination
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.ensemble_methods = ensemble_methods or ["isolation_forest", "zscore", "iqr"]
        self.ensemble_threshold = ensemble_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._feature_stats: dict[str, FeatureStats] = {}
        self._feature_names: list[str] = []
        self._strategy: OutlierStrategy | None = None

    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> OutlierDetector:
        """Fit the outlier detector.

        Args:
            features: Feature data
            labels: Not used

        Returns:
            Self for chaining
        """
        X = self._to_numpy(features)
        feature_names = self._get_feature_names(features)

        # Handle non-numeric data
        if not np.issubdtype(X.dtype, np.number):
            if isinstance(features, pd.DataFrame):
                X = features.select_dtypes(include=[np.number]).values
                feature_names = list(features.select_dtypes(include=[np.number]).columns)
            else:
                raise ValueError("Non-numeric data without column info")

        if X.shape[1] == 0:
            raise ValueError("No numeric features for outlier detection")

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Compute feature statistics
        self._feature_names = feature_names
        self._feature_stats = {}
        for i, name in enumerate(feature_names):
            col = X[:, i]
            q1, q3 = np.percentile(col, [25, 75])
            self._feature_stats[name] = FeatureStats(
                mean=float(np.mean(col)),
                std=float(np.std(col)),
                q1=float(q1),
                q3=float(q3),
                iqr=float(q3 - q1),
            )

        # Create and fit the detection strategy
        self._strategy = create_strategy(
            method=self.method,
            contamination=self.contamination,
            zscore_threshold=self.zscore_threshold,
            iqr_multiplier=self.iqr_multiplier,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            ensemble_methods=self.ensemble_methods,
            ensemble_threshold=self.ensemble_threshold,
        )
        self._strategy.fit(X_scaled, self._feature_stats)

        self._is_fitted = True
        return self

    def detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Detect outliers in the data.

        Args:
            features: Feature data
            labels: Not used

        Returns:
            DetectorResult with Outlier objects
        """
        self._check_fitted()

        X = self._to_numpy(features)
        feature_names = self._get_feature_names(features)

        # Handle non-numeric
        if not np.issubdtype(X.dtype, np.number):
            if isinstance(features, pd.DataFrame):
                X = features.select_dtypes(include=[np.number]).values
                feature_names = list(features.select_dtypes(include=[np.number]).columns)
            else:
                return DetectorResult(issues=[], metadata={"error": "No numeric features"})

        if X.shape[1] == 0:
            return DetectorResult(issues=[], metadata={"error": "No numeric features"})

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        assert self._scaler is not None
        assert self._strategy is not None
        X_scaled = self._scaler.transform(X)

        # Use strategy to detect outliers
        candidates, _scores = self._strategy.detect(X_scaled, feature_names)

        # Convert candidates to Outlier objects
        outliers = [
            Outlier(
                index=c.index,
                score=c.score,
                method=c.method,
                features_contributing=c.contributing_features,
            )
            for c in candidates
        ]

        # Sort by score
        outliers.sort(key=lambda o: o.score, reverse=True)

        n_samples = len(X)
        metadata = {
            "method": self.method,
            "contamination": self.contamination,
            "n_samples": n_samples,
            "n_outliers": len(outliers),
            "outlier_rate": len(outliers) / n_samples if n_samples > 0 else 0,
        }

        return DetectorResult(issues=outliers, metadata=metadata)

    def get_outlier_scores(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Get outlier scores for all samples.

        Args:
            features: Feature data

        Returns:
            Array of outlier scores (higher = more outlier-like)
        """
        self._check_fitted()

        X = self._to_numpy(features)
        if isinstance(features, pd.DataFrame):
            X = features.select_dtypes(include=[np.number]).values

        X = np.nan_to_num(X, nan=0.0)

        assert self._scaler is not None
        X_scaled = self._scaler.transform(X)

        # Try to get scores from Isolation Forest strategy
        from clean.detection.strategies import IsolationForestStrategy

        if isinstance(self._strategy, IsolationForestStrategy):
            return self._strategy.get_scores(X_scaled)

        # Fallback to z-score based
        return np.max(np.abs(X_scaled), axis=1)


def find_outliers(
    features: pd.DataFrame | np.ndarray,
    method: str = "isolation_forest",
    contamination: float = 0.1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Find outliers in a dataset.

    Args:
        features: Feature data
        method: Detection method
        contamination: Expected outlier proportion
        **kwargs: Additional arguments for OutlierDetector

    Returns:
        DataFrame with outlier information
    """
    detector = OutlierDetector(method=method, contamination=contamination, **kwargs)
    result = detector.fit_detect(features)
    return result.to_dataframe()
