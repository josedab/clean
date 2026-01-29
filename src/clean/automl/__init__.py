"""AutoML module for data quality optimization.

This module provides automatic tuning of detection thresholds using
model performance feedback and various optimization techniques.
"""

from clean.automl.optimizers import (
    BayesianOptimizer,
    EvolutionaryOptimizer,
    GridSearchOptimizer,
    OptimizationState,
    OptimizationStrategy,
    RandomSearchOptimizer,
    create_optimizer,
)
from clean.automl.tuner import (
    AdaptiveThresholdManager,
    OptimizationMethod,
    QualityTuner,
    ThresholdParams,
    TuningConfig,
    TuningMetric,
    TuningResult,
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
