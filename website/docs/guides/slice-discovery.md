---
sidebar_position: 13
title: Slice Discovery
---

# Data Slice Discovery

Automatically find data subgroups where quality or model performance degrades.

## The Problem

Your model might have 95% accuracy overall, but only 60% accuracy on:
- Images taken at night
- Text shorter than 20 characters
- Users from a specific region

These "slices" are hidden in aggregate metrics. Slice discovery finds them automatically.

## Quick Start

```python
from clean.slice_discovery import SliceDiscovery

# Initialize discoverer
discoverer = SliceDiscovery(
    method="decision_tree",
    min_slice_size=50,
)

# Discover problematic slices
result = discoverer.discover(
    data=df,
    predictions=y_pred,
    targets=y_true,
    metric="accuracy",
)

# View discovered slices
for slice in result.top_slices[:5]:
    print(f"\nSlice: {slice.description}")
    print(f"  Size: {slice.size} samples ({slice.fraction:.1%} of data)")
    print(f"  Accuracy: {slice.metric_value:.1%} (overall: {result.overall_metric:.1%})")
    print(f"  Gap: {slice.gap:+.1%}")
```

Example output:
```
Slice: age >= 65 AND income < 30000
  Size: 234 samples (2.3% of data)
  Accuracy: 61.5% (overall: 94.2%)
  Gap: -32.7%

Slice: text_length < 15
  Size: 567 samples (5.7% of data)
  Accuracy: 72.3% (overall: 94.2%)
  Gap: -21.9%

Slice: source = 'mobile' AND hour >= 22
  Size: 189 samples (1.9% of data)
  Accuracy: 78.1% (overall: 94.2%)
  Gap: -16.1%
```

## Discovery Methods

### Decision Tree Slicing

Uses decision trees to find feature combinations that predict errors:

```python
discoverer = SliceDiscovery(
    method="decision_tree",
    max_depth=4,          # Complexity of slice definitions
    min_samples_leaf=50,  # Minimum slice size
)
```

Best for: Interpretable, rule-based slices

### Clustering-Based

Groups errors and finds common characteristics:

```python
discoverer = SliceDiscovery(
    method="clustering",
    n_clusters="auto",
    cluster_method="hdbscan",
)
```

Best for: Finding natural groupings of errors

### Rule Mining

Discovers association rules that define slices:

```python
discoverer = SliceDiscovery(
    method="rule_mining",
    max_rules=100,
    min_support=0.01,
)
```

Best for: Complex, multi-condition slices

### Domino Slicing

State-of-the-art algorithm from research:

```python
discoverer = SliceDiscovery(
    method="domino",
    representation="embeddings",
)
```

Best for: High-dimensional data, embeddings

## Metrics

Choose the metric that matters:

```python
# Classification
result = discoverer.discover(df, y_pred, y_true, metric="accuracy")
result = discoverer.discover(df, y_pred, y_true, metric="f1")
result = discoverer.discover(df, y_pred, y_true, metric="precision")
result = discoverer.discover(df, y_pred, y_true, metric="recall")

# Regression
result = discoverer.discover(df, y_pred, y_true, metric="mse")
result = discoverer.discover(df, y_pred, y_true, metric="mae")

# Data quality
result = discoverer.discover(df, issue_mask, None, metric="error_rate")
```

## Working with Slices

### Slice Properties

```python
slice = result.top_slices[0]

# Definition
slice.description    # "age >= 65 AND income < 30000"
slice.predicate      # Callable: slice.predicate(row) -> bool
slice.features       # {"age": (">=", 65), "income": ("<", 30000)}

# Size
slice.size           # 234
slice.fraction       # 0.023
slice.indices        # array([42, 67, 89, ...])

# Performance
slice.metric_value   # 0.615
slice.gap            # -0.327 (difference from overall)
```

### Filtering Slices

```python
# Get slices with significant gap
significant = result.filter(min_gap=0.1)

# Get large slices only
large = result.filter(min_size=100)

# Combined
important = result.filter(min_gap=0.1, min_size=100)
```

### Applying Slices

```python
# Get samples in a slice
slice_data = df.iloc[slice.indices]

# Use predicate on new data
mask = df.apply(slice.predicate, axis=1)
slice_data = df[mask]
```

## Quality-Based Slice Discovery

Find slices with high rates of quality issues:

```python
from clean import DatasetCleaner
from clean.slice_discovery import SliceDiscovery

# Run quality analysis
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Create issue mask
issue_mask = np.zeros(len(df), dtype=bool)
issue_mask[report.label_errors().index] = True

# Find slices with high error rates
discoverer = SliceDiscovery(method="decision_tree")
result = discoverer.discover(
    data=df,
    predictions=issue_mask.astype(int),
    targets=None,
    metric="error_rate",
)

print("Slices with high label error rates:")
for slice in result.top_slices[:3]:
    print(f"  {slice.description}: {slice.metric_value:.1%} error rate")
```

## Slice-Aware Scoring

Adjust quality scores to account for problematic slices:

```python
from clean.slice_discovery import SliceAwareScorer

scorer = SliceAwareScorer(
    min_slice_quality=0.7,  # Flag slices below 70%
    max_slice_gap=0.2,      # Flag gaps > 20%
)

result = scorer.score(
    data=df,
    quality_report=report,
)

print(f"Overall score: {result.overall_score:.1f}")
print(f"Worst slice score: {result.min_slice_score:.1f}")
print(f"Slice-adjusted score: {result.adjusted_score:.1f}")
```

## Explaining Slices

Understand why a slice underperforms:

```python
explanation = discoverer.explain_slice(
    slice=worst_slice,
    data=df,
    predictions=y_pred,
    targets=y_true,
)

print(f"Why '{worst_slice.description}' underperforms:")
for factor in explanation.factors[:3]:
    print(f"  • {factor.description}: {factor.contribution:.1%}")

print(f"\nExample errors in this slice:")
for example in explanation.examples[:3]:
    print(f"  • Predicted: {example.prediction}, Actual: {example.target}")
```

## Visualization

```python
# Bar chart of slice performance
result.plot_slice_comparison()

# Feature distributions in worst slice vs overall
result.plot_slice_distributions(worst_slice, features=["age", "income"])

# Slice tree (for decision tree method)
result.plot_slice_tree()
```

## Export Results

```python
# To DataFrame
df_slices = result.to_dataframe()

# To JSON
result.to_json("slices.json")

# To HTML report
result.to_html("slice_report.html")
```

## Convenience Function

```python
from clean.slice_discovery import discover_slices

slices = discover_slices(
    data=df,
    predictions=y_pred,
    targets=y_true,
    metric="accuracy",
    method="decision_tree",
    top_k=10,
)
```

## Best Practices

1. **Set appropriate minimum size**: Too small = noise, too large = misses problems
2. **Prioritize by size × gap**: Large underperforming slices matter most
3. **Validate on held-out data**: Confirm slices aren't artifacts of train set
4. **Act on findings**: Collect more data, improve labels, add features
5. **Re-discover after fixes**: Verify improvements and find new slices

## Common Actions

| Slice Finding | Typical Action |
|---------------|----------------|
| Demographic underperformance | Collect more diverse training data |
| Edge case errors | Add specific training examples |
| Data source issues | Audit or drop problematic source |
| Temporal patterns | Account for time-based drift |
| Feature value gaps | Improve feature engineering |

## Next Steps

- [Root Cause Analysis](/docs/guides/root-cause) - Understand why slices fail
- [Model-Aware Scoring](/docs/guides/model-aware) - Score quality for your model
- [API Reference](/docs/guides/slice-discovery) - Full API documentation
