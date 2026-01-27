"""Outlier detection strategies.

This module provides strategy classes for different outlier detection algorithms,
implementing the Strategy pattern to reduce complexity in the main OutlierDetector.

Each strategy handles a specific detection algorithm (Isolation Forest, LOF,
Z-score, IQR) and can be used independently or combined in ensemble detection.

Example:
    >>> strategy = IsolationForestStrategy(contamination=0.1)
    >>> strategy.fit(X_scaled, feature_stats)
    >>> outliers, scores = strategy.detect(X_scaled, feature_names)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class FeatureStats:
    """Statistics for a single feature used in outlier detection."""

    mean: float
    std: float
    q1: float
    q3: float
    iqr: float


@dataclass
class OutlierCandidate:
    """A candidate outlier identified by a detection strategy."""

    index: int
    score: float
    method: str
    contributing_features: list[str]


class OutlierStrategy(ABC):
    """Abstract base class for outlier detection strategies.

    Each strategy implements a specific outlier detection algorithm.
    Strategies can be composed for ensemble detection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name for identification."""
        ...

    @abstractmethod
    def fit(
        self,
        x_scaled: np.ndarray,
        feature_stats: dict[str, FeatureStats],
    ) -> OutlierStrategy:
        """Fit the strategy on scaled data.

        Args:
            x_scaled: Scaled feature array
            feature_stats: Pre-computed statistics for each feature

        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def detect(
        self,
        x_scaled: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[OutlierCandidate], dict[int, dict[str, float]]]:
        """Detect outliers using this strategy.

        Args:
            x_scaled: Scaled feature array
            feature_names: Names of features

        Returns:
            Tuple of (list of outlier candidates, dict of per-sample scores)
        """
        ...


class IsolationForestStrategy(OutlierStrategy):
    """Isolation Forest outlier detection strategy.

    Uses the Isolation Forest algorithm which isolates outliers by
    randomly selecting a feature and split value. Outliers require
    fewer splits to isolate.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """Initialize the Isolation Forest strategy.

        Args:
            contamination: Expected proportion of outliers
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: IsolationForest | None = None

    @property
    def name(self) -> str:
        return "isolation_forest"

    def fit(
        self,
        x_scaled: np.ndarray,
        feature_stats: dict[str, FeatureStats],
    ) -> IsolationForestStrategy:
        """Fit the Isolation Forest model."""
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(x_scaled)
        return self

    def detect(
        self,
        x_scaled: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[OutlierCandidate], dict[int, dict[str, float]]]:
        """Detect outliers using Isolation Forest."""
        if self._model is None:
            raise RuntimeError("Strategy must be fitted before detection")

        predictions = self._model.predict(x_scaled)
        anomaly_scores = -self._model.score_samples(x_scaled)

        candidates = []
        scores: dict[int, dict[str, float]] = {}

        for idx in range(len(x_scaled)):
            scores[idx] = {self.name: float(anomaly_scores[idx])}
            if predictions[idx] == -1:  # Outlier
                candidates.append(
                    OutlierCandidate(
                        index=idx,
                        score=float(anomaly_scores[idx]),
                        method=self.name,
                        contributing_features=[],
                    )
                )

        return candidates, scores

    def get_scores(self, x_scaled: np.ndarray) -> np.ndarray:
        """Get anomaly scores for all samples."""
        if self._model is None:
            raise RuntimeError("Strategy must be fitted before scoring")
        return -self._model.score_samples(x_scaled)


class LOFStrategy(OutlierStrategy):
    """Local Outlier Factor detection strategy.

    Uses LOF which measures local density deviation. Outliers have
    substantially lower density than their neighbors.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_neighbors: int = 20,
    ):
        """Initialize the LOF strategy.

        Args:
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors for density estimation
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self._params: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "lof"

    def fit(
        self,
        x_scaled: np.ndarray,
        feature_stats: dict[str, FeatureStats],
    ) -> LOFStrategy:
        """Store parameters for LOF (fitted during detection)."""
        n_samples = len(x_scaled)
        self._params = {
            "contamination": self.contamination,
            "n_neighbors": min(self.n_neighbors, n_samples - 1),
        }
        return self

    def detect(
        self,
        x_scaled: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[OutlierCandidate], dict[int, dict[str, float]]]:
        """Detect outliers using LOF."""
        lof = LocalOutlierFactor(
            contamination=self._params.get("contamination", self.contamination),
            n_neighbors=self._params.get("n_neighbors", min(20, len(x_scaled) - 1)),
            novelty=False,
        )
        predictions = lof.fit_predict(x_scaled)
        lof_scores = -lof.negative_outlier_factor_

        candidates = []
        scores: dict[int, dict[str, float]] = {}

        for idx in range(len(x_scaled)):
            scores[idx] = {self.name: float(lof_scores[idx])}
            if predictions[idx] == -1:
                candidates.append(
                    OutlierCandidate(
                        index=idx,
                        score=float(lof_scores[idx]),
                        method=self.name,
                        contributing_features=[],
                    )
                )

        return candidates, scores


class ZScoreStrategy(OutlierStrategy):
    """Z-score based outlier detection strategy.

    Identifies outliers as points with z-scores exceeding a threshold.
    Works on a per-feature basis.
    """

    def __init__(self, threshold: float = 3.0):
        """Initialize the Z-score strategy.

        Args:
            threshold: Z-score threshold for outlier detection
        """
        self.threshold = threshold
        self._feature_stats: dict[str, FeatureStats] = {}

    @property
    def name(self) -> str:
        return "zscore"

    def fit(
        self,
        x_scaled: np.ndarray,
        feature_stats: dict[str, FeatureStats],
    ) -> ZScoreStrategy:
        """Store feature statistics for z-score computation."""
        self._feature_stats = feature_stats
        return self

    def detect(
        self,
        x_scaled: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[OutlierCandidate], dict[int, dict[str, float]]]:
        """Detect outliers using z-score method."""
        n_samples = len(x_scaled)
        candidates = []
        scores: dict[int, dict[str, float]] = {i: {} for i in range(n_samples)}
        outlier_indices: set[int] = set()

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name not in self._feature_stats:
                continue

            stats = self._feature_stats[feat_name]
            if stats.std <= 0:
                continue

            # Compute z-scores for this feature
            # Note: x_scaled is already scaled, but we use original stats
            col = x_scaled[:, feat_idx]
            z_scores = np.abs(col)  # Already standardized

            for idx in range(n_samples):
                if z_scores[idx] > self.threshold:
                    outlier_indices.add(idx)
                    scores[idx][f"zscore_{feat_name}"] = float(z_scores[idx])

        # Create candidates for outlier indices
        for idx in outlier_indices:
            # Calculate aggregate score
            zscore_values = [v for k, v in scores[idx].items() if k.startswith("zscore_")]
            if zscore_values:
                candidates.append(
                    OutlierCandidate(
                        index=idx,
                        score=float(np.mean(zscore_values)),
                        method=self.name,
                        contributing_features=[
                            k.replace("zscore_", "") for k in scores[idx]
                        ],
                    )
                )

        return candidates, scores


class IQRStrategy(OutlierStrategy):
    """Interquartile Range outlier detection strategy.

    Identifies outliers as points outside Q1 - k*IQR to Q3 + k*IQR range,
    where k is typically 1.5 (default) or 3.0 (for extreme outliers).
    """

    def __init__(self, multiplier: float = 1.5):
        """Initialize the IQR strategy.

        Args:
            multiplier: IQR multiplier for boundary calculation
        """
        self.multiplier = multiplier
        self._feature_stats: dict[str, FeatureStats] = {}

    @property
    def name(self) -> str:
        return "iqr"

    def fit(
        self,
        x_scaled: np.ndarray,
        feature_stats: dict[str, FeatureStats],
    ) -> IQRStrategy:
        """Store feature statistics for IQR computation."""
        self._feature_stats = feature_stats
        return self

    def detect(
        self,
        x_scaled: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[OutlierCandidate], dict[int, dict[str, float]]]:
        """Detect outliers using IQR method."""
        n_samples = len(x_scaled)
        candidates = []
        scores: dict[int, dict[str, float]] = {i: {} for i in range(n_samples)}
        outlier_indices: set[int] = set()

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name not in self._feature_stats:
                continue

            stats = self._feature_stats[feat_name]
            if stats.iqr <= 0:
                continue

            lower = stats.q1 - self.multiplier * stats.iqr
            upper = stats.q3 + self.multiplier * stats.iqr

            # Need to unscale to compare with original bounds
            # Since we're working with scaled data, recompute bounds from scaled data
            col = x_scaled[:, feat_idx]

            for idx in range(n_samples):
                val = col[idx]
                # Check if outside bounds (using scaled values)
                if val < lower or val > upper:
                    outlier_indices.add(idx)
                    # Distance from boundary as score
                    distance = max(
                        lower - val if val < lower else 0,
                        val - upper if val > upper else 0,
                    )
                    scores[idx][f"iqr_{feat_name}"] = float(distance / stats.iqr)

        # Create candidates for outlier indices
        for idx in outlier_indices:
            iqr_values = [v for k, v in scores[idx].items() if k.startswith("iqr_")]
            if iqr_values:
                candidates.append(
                    OutlierCandidate(
                        index=idx,
                        score=float(np.mean(iqr_values)),
                        method=self.name,
                        contributing_features=[
                            k.replace("iqr_", "") for k in scores[idx]
                        ],
                    )
                )

        return candidates, scores


class EnsembleStrategy(OutlierStrategy):
    """Ensemble outlier detection combining multiple strategies.

    Combines results from multiple detection strategies using voting.
    A sample is flagged as an outlier if at least `min_votes` strategies
    agree.
    """

    def __init__(
        self,
        strategies: list[OutlierStrategy],
        min_votes: int = 2,
    ):
        """Initialize ensemble strategy.

        Args:
            strategies: List of strategies to combine
            min_votes: Minimum votes required to flag as outlier
        """
        self.strategies = strategies
        self.min_votes = min_votes

    @property
    def name(self) -> str:
        return "ensemble"

    def fit(
        self,
        x_scaled: np.ndarray,
        feature_stats: dict[str, FeatureStats],
    ) -> EnsembleStrategy:
        """Fit all constituent strategies."""
        for strategy in self.strategies:
            strategy.fit(x_scaled, feature_stats)
        return self

    def detect(
        self,
        x_scaled: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[OutlierCandidate], dict[int, dict[str, float]]]:
        """Detect outliers using ensemble voting."""
        n_samples = len(x_scaled)
        votes: dict[int, list[str]] = {i: [] for i in range(n_samples)}
        all_scores: dict[int, dict[str, float]] = {i: {} for i in range(n_samples)}

        # Collect votes from each strategy
        for strategy in self.strategies:
            candidates, scores = strategy.detect(x_scaled, feature_names)

            # Record votes
            for candidate in candidates:
                votes[candidate.index].append(strategy.name)

            # Merge scores
            for idx, score_dict in scores.items():
                all_scores[idx].update(score_dict)

        # Create candidates based on voting threshold
        candidates = []
        for idx in range(n_samples):
            if len(votes[idx]) >= self.min_votes:
                score_values = list(all_scores[idx].values())
                avg_score = np.mean(score_values) if score_values else 0.0

                # Collect contributing features
                contributing = set()
                for k in all_scores[idx]:
                    for prefix in ["zscore_", "iqr_"]:
                        if k.startswith(prefix):
                            contributing.add(k[len(prefix) :])

                candidates.append(
                    OutlierCandidate(
                        index=idx,
                        score=float(avg_score),
                        method=",".join(votes[idx]),
                        contributing_features=list(contributing)[:5],
                    )
                )

        return candidates, all_scores


def create_strategy(
    method: str,
    contamination: float = 0.1,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    n_jobs: int = -1,
    random_state: int = 42,
    ensemble_methods: list[str] | None = None,
    ensemble_threshold: int = 2,
) -> OutlierStrategy:
    """Factory function to create outlier detection strategies.

    Args:
        method: Detection method name
        contamination: Expected outlier proportion (for IF, LOF)
        zscore_threshold: Z-score threshold
        iqr_multiplier: IQR multiplier
        n_jobs: Parallel jobs (for IF)
        random_state: Random seed
        ensemble_methods: Methods for ensemble (if method='ensemble')
        ensemble_threshold: Minimum votes for ensemble

    Returns:
        Configured OutlierStrategy instance
    """
    if method == "isolation_forest":
        return IsolationForestStrategy(
            contamination=contamination,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif method == "lof":
        return LOFStrategy(contamination=contamination)
    elif method == "zscore":
        return ZScoreStrategy(threshold=zscore_threshold)
    elif method == "iqr":
        return IQRStrategy(multiplier=iqr_multiplier)
    elif method == "ensemble":
        methods = ensemble_methods or ["isolation_forest", "zscore", "iqr"]
        strategies = [
            create_strategy(
                m,
                contamination=contamination,
                zscore_threshold=zscore_threshold,
                iqr_multiplier=iqr_multiplier,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            for m in methods
        ]
        return EnsembleStrategy(strategies, min_votes=ensemble_threshold)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


__all__ = [
    "OutlierStrategy",
    "IsolationForestStrategy",
    "LOFStrategy",
    "ZScoreStrategy",
    "IQRStrategy",
    "EnsembleStrategy",
    "FeatureStats",
    "OutlierCandidate",
    "create_strategy",
]
