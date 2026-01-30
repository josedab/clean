---
sidebar_position: 14
title: Model-Aware Scoring
---

# Model-Aware Quality Scoring

Get quality scores tailored to your specific ML model architecture.

## Why Model-Aware?

Different models have different sensitivities to data issues:

| Issue | Neural Network | Random Forest | SVM |
|-------|---------------|---------------|-----|
| Label noise | 🔴 High impact | 🟢 Low impact | 🟡 Medium |
| Duplicates | 🔴 Overfits easily | 🟢 Robust | 🟡 Medium |
| Outliers | 🟡 Can memorize | 🟢 Isolated in leaves | 🔴 Distorts margin |
| Imbalance | 🔴 Predicts majority | 🟡 Somewhat affected | 🔴 Poor boundary |

A generic quality score doesn't reflect this. Model-aware scoring weights issues by how much they'll actually hurt *your* model.

## Quick Start

```python
from clean.model_aware import ModelAwareScorer

# Specify your target model
scorer = ModelAwareScorer(
    model_type="neural_network",
    task="classification",
)

# Get model-specific score
result = scorer.score(
    data=df,
    labels=labels,
    quality_report=report,
)

print(f"Quality Score for Neural Network: {result.score:.1f}/100")
print(f"Training Viability: {result.viability}")
print(f"Estimated Performance Impact: -{result.performance_impact:.1%}")
```

Example output:
```
Quality Score for Neural Network: 67.3/100
Training Viability: fair
Estimated Performance Impact: -8.5%
```

## Supported Model Types

```python
# Neural networks (most sensitive)
scorer = ModelAwareScorer(model_type="neural_network")

# Tree-based (most robust)
scorer = ModelAwareScorer(model_type="random_forest")
scorer = ModelAwareScorer(model_type="gradient_boosting")
scorer = ModelAwareScorer(model_type="decision_tree")

# Linear models
scorer = ModelAwareScorer(model_type="linear")
scorer = ModelAwareScorer(model_type="logistic_regression")

# Instance-based
scorer = ModelAwareScorer(model_type="knn")
scorer = ModelAwareScorer(model_type="svm")
```

## Model Configuration

Provide details about your architecture for more accurate scoring:

```python
# Neural network with specific architecture
scorer = ModelAwareScorer(
    model_type="neural_network",
    model_config={
        "architecture": "transformer",  # cnn, rnn, mlp, transformer
        "size": "large",                # small, medium, large
        "regularization": "dropout",    # none, dropout, weight_decay
    }
)

# Gradient boosting with hyperparameters
scorer = ModelAwareScorer(
    model_type="gradient_boosting",
    model_config={
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
    }
)
```

## Understanding Results

### Score Components

```python
result = scorer.score(data, labels, report)

# Overall score (0-100)
result.score  # 67.3

# Viability assessment
result.viability  # "excellent", "good", "fair", "poor", "untrainable"

# Estimated impact on model performance
result.performance_impact  # 0.085 (8.5% accuracy reduction)
```

### Issue Weights

See how each issue type is weighted for your model:

```python
for issue, weight in result.issue_weights.items():
    print(f"{issue}: {weight:.1f}x impact")

# Output for neural_network:
# label_errors: 2.5x impact
# duplicates: 1.8x impact
# outliers: 1.2x impact
# imbalance: 1.5x impact

# Output for random_forest:
# label_errors: 0.8x impact
# duplicates: 0.5x impact
# outliers: 0.3x impact
# imbalance: 1.2x impact
```

### Risk Factors

```python
for risk in result.risk_factors:
    print(f"⚠️ {risk.description}")
    print(f"   Severity: {risk.severity}")

# Output:
# ⚠️ High label error rate (3.5%) will cause memorization
#    Severity: high
# ⚠️ 15:1 class imbalance will bias predictions
#    Severity: medium
```

## Recommendations

Get prioritized cleanup recommendations for your model:

```python
recommendations = scorer.get_recommendations(
    data=df,
    labels=labels,
    quality_report=report,
    budget="medium",  # low, medium, high effort
)

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec.action}")
    print(f"   Priority: {rec.priority}")
    print(f"   Effort: {rec.effort}")
    print(f"   Expected improvement: +{rec.expected_improvement:.1%}")
```

Example output:
```
1. Fix top 100 label errors (confidence > 0.95)
   Priority: critical
   Effort: 2 hours
   Expected improvement: +3.2%

2. Remove 234 duplicate pairs
   Priority: high
   Effort: 30 minutes
   Expected improvement: +1.8%

3. Address class imbalance with SMOTE
   Priority: medium
   Effort: 1 hour
   Expected improvement: +2.1%
```

## Impact Simulation

Simulate how fixing issues would improve model performance:

```python
simulation = scorer.simulate_impact(
    data=df,
    labels=labels,
    quality_report=report,
    fix_scenarios=[
        {"fix": "label_errors", "fraction": 0.5},   # Fix 50% of label errors
        {"fix": "duplicates", "fraction": 1.0},     # Remove all duplicates
        {"fix": "outliers", "fraction": 0.8},       # Remove 80% of outliers
    ]
)

for scenario in simulation.scenarios:
    print(f"\nScenario: {scenario.description}")
    print(f"  Score: {scenario.before_score:.1f} → {scenario.after_score:.1f}")
    print(f"  Estimated accuracy gain: +{scenario.accuracy_gain:.1%}")
```

## Model Sensitivity Reference

### Label Errors

| Model Type | Sensitivity | Why |
|------------|-------------|-----|
| Neural Network | 🔴 Very High | Memorizes noisy labels |
| Gradient Boosting | 🟡 Medium | Sequential correction helps |
| Random Forest | 🟢 Low | Averaging reduces noise |
| KNN | 🔴 Very High | Directly affects neighbors |
| SVM | 🟡 Medium | Affects margin definition |

### Duplicates

| Model Type | Sensitivity | Why |
|------------|-------------|-----|
| Neural Network | 🔴 High | Overfits to repeated samples |
| Tree-based | 🟢 Low | Only affects split points |
| KNN | 🔴 Very High | Skews nearest neighbor voting |
| Linear | 🟢 Low | Just shifts centroids slightly |

### Outliers

| Model Type | Sensitivity | Why |
|------------|-------------|-----|
| SVM | 🔴 Very High | Distorts margin |
| Linear | 🔴 High | High leverage points |
| Neural Network | 🟡 Medium | Can memorize outliers |
| Tree-based | 🟢 Low | Isolated in leaf nodes |

## Convenience Function

```python
from clean.model_aware import score_for_model

result = score_for_model(
    data=df,
    labels=labels,
    model_type="neural_network",
    quality_report=report,
)
```

## Integration Example

Full workflow with model-aware scoring:

```python
from clean import DatasetCleaner
from clean.model_aware import ModelAwareScorer

# 1. Analyze data quality
cleaner = DatasetCleaner(data=train_df, label_column="label")
report = cleaner.analyze()

# 2. Score for your target model
scorer = ModelAwareScorer(
    model_type="neural_network",
    model_config={"architecture": "transformer", "size": "large"}
)
result = scorer.score(train_df, train_df["label"], report)

# 3. Decide based on score
if result.viability == "untrainable":
    print("❌ Data quality too low for reliable training")
    print(f"   Issues: {result.risk_factors}")
    
elif result.viability in ["poor", "fair"]:
    print("⚠️ Data quality will limit model performance")
    recommendations = scorer.get_recommendations(train_df, labels, report)
    print(f"   Top recommendation: {recommendations[0].action}")
    
else:
    print("✅ Data quality sufficient for training")
    print(f"   Expected impact: -{result.performance_impact:.1%}")
```

## Best Practices

1. **Always specify your model**: Generic scores may not reflect actual impact
2. **Use simulation before cleaning**: Estimate ROI of cleanup efforts
3. **Consider model robustness**: Sometimes a more robust model is easier than cleaning
4. **Re-score after fixes**: Verify improvements before training
5. **Track scores over time**: Monitor data quality in production

## Next Steps

- [AutoML Tuning](/docs/guides/automl) - Optimize thresholds for your model
- [Slice Discovery](/docs/guides/slice-discovery) - Find where your model will fail
- [API Reference](/docs/guides/model-aware) - Full API documentation
