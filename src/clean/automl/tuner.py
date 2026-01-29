"""AutoML Quality Tuning - Automatic threshold optimization for data quality.

This module provides automatic tuning of detection thresholds using
model performance feedback and optimization techniques.

Example:
    >>> from clean.automl import QualityTuner, TuningConfig
    >>>
    >>> tuner = QualityTuner(config=TuningConfig(metric="accuracy"))
    >>> result = tuner.tune(X, y, model=LogisticRegression())
    >>> print(f"Best thresholds: {result.best_params}")
    >>> print(f"Accuracy improvement: {result.improvement:.2%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score

from clean.automl.optimizers import (
    OptimizationState,
    OptimizationStrategy,
    create_optimizer,
)

if TYPE_CHECKING:
    pass


class OptimizationMethod(Enum):
    """Optimization methods for threshold tuning."""

    GRID_SEARCH = "grid"
    RANDOM_SEARCH = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"


class TuningMetric(Enum):
    """Metrics to optimize during tuning."""

    ACCURACY = "accuracy"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    PRECISION = "precision"
    RECALL = "recall"
    CUSTOM = "custom"


@dataclass
class TuningConfig:
    """Configuration for AutoML quality tuning."""

    metric: TuningMetric | str = TuningMetric.ACCURACY
    method: OptimizationMethod | str = OptimizationMethod.RANDOM_SEARCH
    n_trials: int = 50
    cv_folds: int = 5
    n_jobs: int = -1
    random_state: int = 42
    timeout_seconds: float | None = None
    early_stopping_rounds: int | None = 10
    verbose: bool = True

    # Search space bounds
    label_error_threshold_range: tuple[float, float] = (0.3, 0.95)
    outlier_contamination_range: tuple[float, float] = (0.01, 0.2)
    duplicate_threshold_range: tuple[float, float] = (0.8, 0.99)

    def __post_init__(self) -> None:
        """Convert string enums."""
        if isinstance(self.metric, str):
            self.metric = TuningMetric(self.metric)
        if isinstance(self.method, str):
            self.method = OptimizationMethod(self.method)


@dataclass
class ThresholdParams:
    """Parameters for quality thresholds."""

    label_error_threshold: float = 0.5
    outlier_contamination: float = 0.1
    duplicate_threshold: float = 0.9
    remove_outliers: bool = True
    remove_duplicates: bool = True
    relabel_errors: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label_error_threshold": self.label_error_threshold,
            "outlier_contamination": self.outlier_contamination,
            "duplicate_threshold": self.duplicate_threshold,
            "remove_outliers": self.remove_outliers,
            "remove_duplicates": self.remove_duplicates,
            "relabel_errors": self.relabel_errors,
        }


@dataclass
class TuningResult:
    """Result of AutoML tuning."""

    best_params: ThresholdParams
    best_score: float
    baseline_score: float
    improvement: float
    all_trials: list[dict[str, Any]]
    n_trials_completed: int
    optimization_time_seconds: float
    samples_removed: int
    samples_relabeled: int
    convergence_history: list[float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "AutoML Quality Tuning Results",
            "=" * 50,
            "",
            f"Optimization completed in {self.optimization_time_seconds:.1f}s",
            f"Trials completed: {self.n_trials_completed}",
            "",
            "Performance:",
            f"  Baseline score: {self.baseline_score:.4f}",
            f"  Best score:     {self.best_score:.4f}",
            f"  Improvement:    {self.improvement:+.2%}",
            "",
            "Best Thresholds:",
            f"  Label error threshold:   {self.best_params.label_error_threshold:.3f}",
            f"  Outlier contamination:   {self.best_params.outlier_contamination:.3f}",
            f"  Duplicate threshold:     {self.best_params.duplicate_threshold:.3f}",
            "",
            "Data Changes:",
            f"  Samples removed:   {self.samples_removed:,}",
            f"  Samples relabeled: {self.samples_relabeled:,}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params.to_dict(),
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "n_trials": self.n_trials_completed,
            "optimization_time": self.optimization_time_seconds,
            "samples_removed": self.samples_removed,
            "samples_relabeled": self.samples_relabeled,
        }


class QualityTuner:
    """AutoML tuner for optimizing data quality thresholds.

    Automatically finds optimal thresholds for label error detection,
    outlier removal, and duplicate detection by optimizing for model
    performance on cleaned data.

    Uses the Strategy pattern for optimization algorithms, allowing
    easy extension with new optimization methods.
    """

    def __init__(
        self,
        config: TuningConfig | None = None,
        custom_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
        optimizer: OptimizationStrategy | None = None,
    ):
        """Initialize the tuner.

        Args:
            config: Tuning configuration
            custom_metric: Custom metric function(y_true, y_pred) -> score
            optimizer: Custom optimization strategy (overrides config.method)
        """
        self.config = config or TuningConfig()
        self.custom_metric = custom_metric
        self._optimizer = optimizer
        self._rng = np.random.RandomState(self.config.random_state)

    def _get_optimizer(self) -> OptimizationStrategy:
        """Get the optimization strategy to use."""
        if self._optimizer is not None:
            return self._optimizer
        return create_optimizer(self.config.method.value)

    def tune(
        self,
        X: pd.DataFrame | np.ndarray,  # noqa: N803
        y: np.ndarray,
        model: BaseEstimator | None = None,
        label_column: str | None = None,  # noqa: ARG002
    ) -> TuningResult:
        """Run AutoML tuning to find optimal thresholds.

        Args:
            X: Feature data
            y: Labels
            model: Sklearn classifier to use for evaluation
            label_column: Name of label column if X is DataFrame with labels

        Returns:
            TuningResult with optimal thresholds and metrics
        """
        start_time = time.time()

        # Convert to appropriate format
        if isinstance(X, pd.DataFrame):
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        # Default model
        if model is None:
            model = LogisticRegression(max_iter=1000, random_state=42)

        # Calculate baseline score
        baseline_score = self._evaluate_model(X_arr, y_arr, model)

        # Create evaluation function for optimizer
        def evaluate_params(params: ThresholdParams) -> float:
            return self._evaluate_params(X_arr, y_arr, model, params)

        # Create params factory
        def create_params(le: float, oc: float, dt: float) -> ThresholdParams:
            return ThresholdParams(
                label_error_threshold=le,
                outlier_contamination=oc,
                duplicate_threshold=dt,
                remove_outliers=self._rng.choice([True, False]),
                remove_duplicates=self._rng.choice([True, False]),
                relabel_errors=False,
            )

        # Initialize state and run optimization
        state = OptimizationState.create(ThresholdParams())
        optimizer = self._get_optimizer()
        best_params = optimizer.optimize(
            config=self.config,
            evaluate_fn=evaluate_params,
            create_params_fn=create_params,
            state=state,
            rng=self._rng,
        )

        # Get final clean data stats
        X_clean, y_clean, stats = self._apply_cleaning(X_arr, y_arr, best_params)
        best_score = self._evaluate_model(X_clean, y_clean, model)

        elapsed = time.time() - start_time

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            baseline_score=baseline_score,
            improvement=(best_score - baseline_score) / max(baseline_score, 1e-10),
            all_trials=state.trials,
            n_trials_completed=len(state.trials),
            optimization_time_seconds=elapsed,
            samples_removed=stats.get("removed", 0),
            samples_relabeled=stats.get("relabeled", 0),
            convergence_history=state.convergence_history,
        )

    def _evaluate_model(
        self,
        x_data: np.ndarray,
        y: np.ndarray,
        model: BaseEstimator,
    ) -> float:
        """Evaluate model performance using cross-validation."""
        if len(x_data) < self.config.cv_folds:
            return 0.0

        # Handle small datasets
        n_unique = len(np.unique(y))
        cv = min(self.config.cv_folds, n_unique, len(x_data) // 2)
        cv = max(cv, 2)

        metric_name = self._get_sklearn_metric_name()

        try:
            scores = cross_val_score(
                clone(model),
                x_data,
                y,
                cv=cv,
                scoring=metric_name,
                n_jobs=self.config.n_jobs,
            )
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def _get_sklearn_metric_name(self) -> str:
        """Get sklearn scoring metric name."""
        metric_map = {
            TuningMetric.ACCURACY: "accuracy",
            TuningMetric.F1: "f1",
            TuningMetric.F1_MACRO: "f1_macro",
            TuningMetric.F1_WEIGHTED: "f1_weighted",
            TuningMetric.PRECISION: "precision",
            TuningMetric.RECALL: "recall",
        }
        return metric_map.get(self.config.metric, "accuracy")

    def _evaluate_params(
        self,
        x_data: np.ndarray,
        y: np.ndarray,
        model: BaseEstimator,
        params: ThresholdParams,
    ) -> float:
        """Evaluate a parameter configuration."""
        x_clean, y_clean, _ = self._apply_cleaning(x_data, y, params)

        if len(x_clean) < self.config.cv_folds * 2:
            return 0.0  # Not enough data after cleaning

        return self._evaluate_model(x_clean, y_clean, model)

    def _apply_cleaning(
        self,
        x_data: np.ndarray,
        y: np.ndarray,
        params: ThresholdParams,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """Apply cleaning with given parameters."""
        mask = np.ones(len(x_data), dtype=bool)
        stats: dict[str, int] = {"removed": 0, "relabeled": 0}
        y_clean = y.copy()

        # Remove outliers
        if params.remove_outliers:
            outlier_mask = self._detect_outliers(x_data, params.outlier_contamination)
            mask &= ~outlier_mask
            stats["removed"] += int(outlier_mask.sum())

        # Remove duplicates
        if params.remove_duplicates:
            dup_mask = self._detect_duplicates(x_data, params.duplicate_threshold)
            mask &= ~dup_mask
            stats["removed"] += int(dup_mask.sum())

        x_clean = x_data[mask]
        y_clean = y_clean[mask]

        return x_clean, y_clean, stats

    def _detect_outliers(
        self,
        x_data: np.ndarray,
        contamination: float,
    ) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        if len(x_data) < 10:
            return np.zeros(len(x_data), dtype=bool)

        iso = IsolationForest(
            contamination=contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        try:
            predictions = iso.fit_predict(x_data)
            return predictions == -1
        except Exception:
            return np.zeros(len(x_data), dtype=bool)

    def _detect_duplicates(
        self,
        x_data: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Detect near-duplicates using cosine similarity."""
        if len(x_data) < 2:
            return np.zeros(len(x_data), dtype=bool)

        # Normalize for cosine similarity
        norms = np.linalg.norm(x_data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        x_norm = x_data / norms

        duplicates = np.zeros(len(x_data), dtype=bool)

        # For small datasets, compute full similarity matrix
        if len(x_data) <= 1000:
            sim = cosine_similarity(x_norm)
            # Zero out diagonal and lower triangle
            np.fill_diagonal(sim, 0)
            sim = np.triu(sim)

            # Find duplicates (mark the later occurrence)
            for i in range(len(x_data)):
                for j in range(i + 1, len(x_data)):
                    if sim[i, j] > threshold:
                        duplicates[j] = True
            return duplicates

        # For large datasets, check in batches
        batch_size = 1000
        for i in range(batch_size, len(x_data), batch_size):
            batch_end = min(i + batch_size, len(x_data))
            sim = cosine_similarity(x_norm[i:batch_end], x_norm[:i])

            max_sim = sim.max(axis=1)
            duplicates[i:batch_end] = max_sim > threshold

        return duplicates


def tune_quality_thresholds(
    features: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    model: BaseEstimator | None = None,
    metric: str = "accuracy",
    n_trials: int = 50,
    **kwargs: Any,
) -> TuningResult:
    """Convenience function for tuning quality thresholds.

    Args:
        features: Feature data
        y: Labels
        model: Model to optimize for
        metric: Metric to optimize
        n_trials: Number of optimization trials
        **kwargs: Additional config options

    Returns:
        TuningResult with optimized thresholds
    """
    config = TuningConfig(metric=metric, n_trials=n_trials, **kwargs)
    tuner = QualityTuner(config=config)
    return tuner.tune(features, y, model=model)


class AdaptiveThresholdManager:
    """Manages adaptive thresholds that adjust over time.

    Monitors model performance and adjusts thresholds when
    performance degrades.
    """

    def __init__(
        self,
        initial_params: ThresholdParams | None = None,
        performance_window: int = 100,
        degradation_threshold: float = 0.05,
        auto_retune: bool = True,
    ):
        """Initialize the manager.

        Args:
            initial_params: Starting threshold parameters
            performance_window: Window size for performance monitoring
            degradation_threshold: Performance drop to trigger retuning
            auto_retune: Whether to automatically retune on degradation
        """
        self.params = initial_params or ThresholdParams()
        self.performance_window = performance_window
        self.degradation_threshold = degradation_threshold
        self.auto_retune = auto_retune

        self._performance_history: list[float] = []
        self._baseline_performance: float | None = None
        self._tuner = QualityTuner()

    def record_performance(self, score: float) -> None:
        """Record a performance observation.

        Args:
            score: Model performance score
        """
        self._performance_history.append(score)

        if len(self._performance_history) > self.performance_window:
            self._performance_history = self._performance_history[
                -self.performance_window :
            ]

        if self._baseline_performance is None and len(self._performance_history) >= 10:
            self._baseline_performance = float(np.mean(self._performance_history))

    def check_degradation(self) -> bool:
        """Check if performance has degraded.

        Returns:
            True if performance has dropped below threshold
        """
        if self._baseline_performance is None:
            return False

        if len(self._performance_history) < 10:
            return False

        recent_avg = float(np.mean(self._performance_history[-10:]))
        degradation = (
            self._baseline_performance - recent_avg
        ) / self._baseline_performance

        return degradation > self.degradation_threshold

    def update_thresholds(
        self,
        x_data: np.ndarray,
        y: np.ndarray,
        model: BaseEstimator,
    ) -> ThresholdParams:
        """Update thresholds based on current data.

        Args:
            x_data: Feature data
            y: Labels
            model: Model to optimize for

        Returns:
            Updated threshold parameters
        """
        result = self._tuner.tune(x_data, y, model)
        self.params = result.best_params
        self._baseline_performance = result.best_score
        self._performance_history = []

        return self.params

    def get_current_params(self) -> ThresholdParams:
        """Get current threshold parameters."""
        return self.params
