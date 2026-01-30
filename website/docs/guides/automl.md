---
sidebar_position: 11
title: AutoML Tuning
---

# AutoML Quality Threshold Tuning

Automatically find optimal thresholds for each quality detector.

## The Problem

Clean's detectors have configurable thresholds:
- `label_error_threshold`: Confidence cutoff for flagging mislabels
- `outlier_threshold`: Standard deviations for outlier detection
- `duplicate_threshold`: Similarity score for duplicates

Manual tuning is tedious. Set thresholds too low and you get false positives. Too high and you miss real issues.

## The Solution

AutoML tuning uses optimization algorithms to find thresholds that maximize detection accuracy on your validation data.

## Quick Start

```python
from clean.automl import QualityTuner, TuningConfig

# Configure tuning
config = TuningConfig(
    method="bayesian",     # Efficient optimization
    n_trials=50,           # Number of combinations to try
    metric="f1",           # Optimize for F1 score
)

tuner = QualityTuner(config=config)

# Run optimization
result = tuner.tune(
    X=features,
    y=labels,
    validation_labels=known_issues,  # Ground truth quality labels
)

print(f"Best score: {result.best_score:.3f}")
print(f"Best thresholds: {result.best_params}")

# Use optimized thresholds
from clean import DatasetCleaner

cleaner = DatasetCleaner(
    data=df,
    label_column="label",
    **result.best_params  # Apply tuned thresholds
)
```

## Optimization Methods

### Bayesian Optimization (Recommended)

Uses Gaussian Process to model the objective function. Most efficient for 20-100 trials.

```python
config = TuningConfig(
    method="bayesian",
    n_trials=50,
    acquisition_function="ei",  # Expected improvement
)
```

### Random Search

Good baseline. Samples randomly from parameter space.

```python
config = TuningConfig(
    method="random",
    n_trials=100,
)
```

### Grid Search

Exhaustive search over a predefined grid. Use when you have specific values to try.

```python
config = TuningConfig(
    method="grid",
    param_grid={
        "label_error_threshold": [0.8, 0.85, 0.9, 0.95],
        "outlier_threshold": [2.0, 2.5, 3.0],
        "duplicate_threshold": [0.9, 0.95, 0.99],
    }
)
```

### Evolutionary Algorithm

Genetic algorithm for complex search spaces with many parameters.

```python
config = TuningConfig(
    method="evolutionary",
    n_trials=100,
    population_size=20,
    mutation_rate=0.1,
)
```

## Validation Data

The tuner needs ground truth labels to optimize against:

```python
# Option 1: Binary mask of known issues
validation_labels = np.array([0, 0, 1, 0, 1, 1, 0, ...])  # 1 = issue

# Option 2: DataFrame with issue column
validation_df = pd.DataFrame({
    "feature1": [...],
    "label": [...],
    "is_issue": [False, False, True, ...]  # Known quality issues
})

result = tuner.tune(
    X=validation_df.drop(columns=["is_issue"]),
    y=validation_df["label"],
    validation_labels=validation_df["is_issue"],
)
```

### Creating Validation Data

If you don't have labeled issues, create them:

```python
# Manual review
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Review top 100 suspected issues
candidates = report.label_errors().head(100)
# Manually label which are true errors
reviewed = manual_review(candidates)  # Your review process

# Use reviewed labels for tuning
result = tuner.tune(X, y, validation_labels=reviewed["is_error"])
```

## Metrics

Choose the metric that matches your goal:

| Metric | Best When |
|--------|-----------|
| `f1` | Balance precision and recall (default) |
| `precision` | False positives are costly (reviewing takes time) |
| `recall` | Missing issues is costly (bad data hurts model) |
| `balanced_accuracy` | Classes are imbalanced |

```python
config = TuningConfig(
    method="bayesian",
    metric="precision",  # Minimize false positives
)
```

## Tunable Parameters

```python
from clean.automl import ThresholdParams

# View default search space
print(ThresholdParams.default_ranges())
# {
#   "label_error_threshold": (0.5, 0.99),
#   "outlier_threshold": (1.5, 4.0),
#   "duplicate_threshold": (0.8, 0.99),
#   "imbalance_threshold": (2.0, 20.0),
#   "bias_threshold": (0.05, 0.3),
# }

# Customize search space
config = TuningConfig(
    method="bayesian",
    param_ranges={
        "label_error_threshold": (0.7, 0.95),  # Narrower range
        "duplicate_threshold": (0.9, 0.999),   # Higher values only
    }
)
```

## Results

```python
result = tuner.tune(X, y, validation_labels)

# Best configuration
print(result.best_params)
# {'label_error_threshold': 0.87, 'outlier_threshold': 2.3, ...}

print(result.best_score)  # 0.923

# View all trials
for trial in result.trials[:10]:
    print(f"Trial {trial.number}: {trial.params} -> {trial.score:.3f}")

# Optimization history
result.plot_optimization_history()

# Parameter importance
result.plot_param_importance()
```

## Saving and Loading

```python
import json

# Save optimized thresholds
with open("quality_thresholds.json", "w") as f:
    json.dump(result.best_params, f)

# Load in production
with open("quality_thresholds.json") as f:
    thresholds = json.load(f)

cleaner = DatasetCleaner(data=df, **thresholds)
```

## Cross-Validation

Enable cross-validation for more robust threshold selection:

```python
config = TuningConfig(
    method="bayesian",
    n_trials=50,
    cv_folds=5,  # 5-fold cross-validation
)
```

## Parallel Tuning

Speed up tuning with parallel workers:

```python
config = TuningConfig(
    method="bayesian",
    n_trials=100,
    n_jobs=-1,  # Use all CPU cores
)
```

## Timeout

Set a maximum tuning time:

```python
config = TuningConfig(
    method="bayesian",
    n_trials=1000,
    timeout=3600,  # Stop after 1 hour
)
```

## Convenience Function

```python
from clean.automl import tune_quality_thresholds

best_params = tune_quality_thresholds(
    X=features,
    y=labels,
    validation_labels=known_issues,
    method="bayesian",
    n_trials=50,
)
```

## Best Practices

1. **Start with validation data**: Even 100 manually reviewed samples help significantly
2. **Use Bayesian for most cases**: Most efficient method
3. **Don't overtune**: 50-100 trials is usually enough
4. **Save your thresholds**: Version control optimized parameters
5. **Re-tune periodically**: Data characteristics may shift over time

## Next Steps

- [Model-Aware Scoring](/docs/guides/model-aware) - Tailor thresholds to your ML model
- [Root Cause Analysis](/docs/guides/root-cause) - Understand what drives quality issues
- [API Reference](/docs/guides/automl) - Full API documentation
