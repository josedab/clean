"""AutoML Quality Tuning - Automatic threshold optimization for data quality.

This module re-exports from the automl package for backward compatibility.

Example:
    >>> from clean.automl import QualityTuner, TuningConfig
    >>>
    >>> tuner = QualityTuner(config=TuningConfig(metric="accuracy"))
    >>> result = tuner.tune(X, y, model=LogisticRegression())
    >>> print(f"Best thresholds: {result.best_params}")
    >>> print(f"Accuracy improvement: {result.improvement:.2%}")
"""

# Re-export everything from the automl package for backward compatibility
from clean.automl import (
    AdaptiveThresholdManager,
    BayesianOptimizer,
    EvolutionaryOptimizer,
    GridSearchOptimizer,
    OptimizationMethod,
    OptimizationState,
    OptimizationStrategy,
    QualityTuner,
    RandomSearchOptimizer,
    ThresholdParams,
    TuningConfig,
    TuningMetric,
    TuningResult,
    create_optimizer,
    tune_quality_thresholds,
)

__all__ = [
    # Tuner
    "QualityTuner",
    "TuningConfig",
    "TuningResult",
    "TuningMetric",
    "OptimizationMethod",
    "ThresholdParams",
    "AdaptiveThresholdManager",
    "tune_quality_thresholds",
    # Optimizers
    "OptimizationStrategy",
    "OptimizationState",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "EvolutionaryOptimizer",
    "create_optimizer",
]
