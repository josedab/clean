---
sidebar_position: 18
title: Intelligent Sampling
---

# Intelligent Sampling for Labeling

Maximize labeling efficiency by selecting the most informative samples.

## Why Intelligent Sampling?

Labeling data is expensive. Random sampling wastes budget on easy samples. Intelligent sampling selects samples that will most improve your model:

| Strategy | What it does | Best for |
|----------|--------------|----------|
| Random | Uniform selection | Baseline |
| Uncertainty | Model is least confident | General use |
| Diversity | Cover data distribution | Cold start |
| Hybrid | Balance uncertainty + diversity | Most scenarios |

Studies show intelligent sampling can achieve the same model accuracy with **50-80% fewer labels**.

## Quick Start

```python
from clean.active_learning import IntelligentSampler

# Initialize sampler
sampler = IntelligentSampler(
    strategy="hybrid",
    exploration_weight=0.3,
)

# Fit on labeled data
sampler.fit(X_labeled, y_labeled)

# Select samples for labeling
indices = sampler.select(
    X_unlabeled,
    n_samples=100,
)

# Get samples to label
samples_to_label = X_unlabeled[indices]

# After labeling, update model
sampler.update(samples_to_label, new_labels)
```

## Query Strategies

### Uncertainty Sampling

Select samples where the model is least confident:

```python
sampler = IntelligentSampler(strategy="uncertainty")

# Variants
sampler = IntelligentSampler(
    strategy="uncertainty",
    uncertainty_method="entropy",       # Maximum entropy
    # or "margin"                       # Smallest margin between top 2 classes
    # or "least_confident"              # Lowest max probability
)
```

### Query by Committee

Use disagreement between multiple models:

```python
from clean.active_learning import QueryByCommittee

committee = QueryByCommittee(
    n_estimators=5,
    base_estimator="random_forest",
)

committee.fit(X_labeled, y_labeled)
indices = committee.select(X_unlabeled, n_samples=100)

# View disagreement scores
disagreements = committee.get_disagreement_scores(X_unlabeled)
```

### Expected Model Change

Select samples that would most change the model:

```python
from clean.active_learning import ExpectedModelChange

emc = ExpectedModelChange(
    model="logistic_regression",
    gradient_method="empirical",
)

emc.fit(X_labeled, y_labeled)
indices = emc.select(X_unlabeled, n_samples=100)
```

### Diversity Sampling

Ensure selected samples cover the data distribution:

```python
sampler = IntelligentSampler(
    strategy="diversity",
    diversity_method="kmeans",      # Cluster centers
    # or "coreset"                  # Core-set selection
    # or "determinantal"            # DPP for diversity
)
```

### Hybrid Strategy

Combine uncertainty and diversity (recommended):

```python
sampler = IntelligentSampler(
    strategy="hybrid",
    uncertainty_weight=0.6,
    diversity_weight=0.4,
)
```

## Batch Selection

### Diverse Batches

Ensure samples in each batch are diverse:

```python
indices = sampler.select(
    X_unlabeled,
    n_samples=100,
    batch_mode="diverse",
    batch_diversity_threshold=0.5,
)
```

### Clustered Batches

Select representative samples from each cluster:

```python
indices = sampler.select(
    X_unlabeled,
    n_samples=100,
    batch_mode="clustered",
    n_clusters=10,
)
```

## Iterative Labeling Workflow

```python
from clean.active_learning import LabelingSession

# Create session
session = LabelingSession(
    sampler=sampler,
    X_unlabeled=X_unlabeled,
)

# Iteration 1
batch1 = session.get_next_batch(n_samples=50)
labels1 = label_samples(batch1)  # Your labeling function
session.submit_labels(batch1.indices, labels1)

# Iteration 2 (model updated automatically)
batch2 = session.get_next_batch(n_samples=50)
labels2 = label_samples(batch2)
session.submit_labels(batch2.indices, labels2)

# Check progress
print(f"Samples labeled: {session.n_labeled}")
print(f"Estimated accuracy: {session.estimated_accuracy:.1%}")
```

## Stopping Criteria

Know when to stop labeling:

```python
session = LabelingSession(
    sampler=sampler,
    stopping_criteria={
        "min_accuracy": 0.95,           # Stop at 95% accuracy
        "accuracy_plateau_rounds": 3,    # Stop if no improvement for 3 rounds
        "max_samples": 1000,             # Hard limit
    }
)

while not session.should_stop():
    batch = session.get_next_batch(n_samples=50)
    labels = label_samples(batch)
    session.submit_labels(batch, labels)

print(f"Stopped after {session.n_labeled} samples")
print(f"Final accuracy: {session.estimated_accuracy:.1%}")
```

## Export to Labeling Tools

### Label Studio

```python
from clean.active_learning import LabelStudioExporter

exporter = LabelStudioExporter(
    project_name="Quality Review",
    label_config_type="classification",
)

exporter.export(
    samples=X_unlabeled[indices],
    sample_ids=indices,
    output_path="label_studio_tasks.json",
)
```

### CVAT

```python
from clean.active_learning import CVATExporter

exporter = CVATExporter(
    task_name="Image Classification",
    labels=["cat", "dog", "bird"],
)

exporter.export(
    image_paths=image_paths[indices],
    output_path="cvat_tasks.xml",
)
```

### Prodigy

```python
from clean.active_learning import ProdigyExporter

exporter = ProdigyExporter()

exporter.export(
    texts=texts[indices],
    output_path="prodigy_tasks.jsonl",
    task_type="textcat",
    labels=["positive", "negative"],
)
```

## Metrics and Analysis

```python
# Sampling efficiency
metrics = sampler.get_metrics()

print(f"Selection efficiency: {metrics.efficiency:.2f}")
print(f"Diversity score: {metrics.diversity:.2f}")
print(f"Estimated label savings: {metrics.label_savings:.1%}")
```

### Compare Strategies

```python
from clean.active_learning import compare_strategies

comparison = compare_strategies(
    X_train, y_train, X_test, y_test,
    strategies=["uncertainty", "diversity", "hybrid", "random"],
    n_iterations=10,
    samples_per_iteration=50,
)

# Plot learning curves
comparison.plot_learning_curves()

# Best strategy
print(f"Best strategy: {comparison.best_strategy}")
print(f"Label savings vs random: {comparison.label_savings:.1%}")
```

## Integration with Clean

Use intelligent sampling for quality review:

```python
from clean import DatasetCleaner
from clean.active_learning import IntelligentSampler

# Find potential issues
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Get suspected errors
suspected_errors = report.label_errors()

# Sample most informative ones for review
sampler = IntelligentSampler(strategy="uncertainty")
sampler.fit(X, y)

review_indices = sampler.select(
    X[suspected_errors.index],
    n_samples=100,
)

# These 100 samples will be most valuable to review
to_review = suspected_errors.iloc[review_indices]
```

## Convenience Function

```python
from clean.active_learning import select_for_labeling

indices = select_for_labeling(
    X_unlabeled,
    strategy="hybrid",
    n_samples=100,
    model=trained_model,  # Optional: use existing model
)
```

## Best Practices

1. **Start with uncertainty**: Most effective for initial rounds
2. **Add diversity later**: Prevents sampling similar instances
3. **Use batch diversity**: Avoid redundant samples per batch
4. **Monitor stopping criteria**: Don't over-label
5. **Validate with held-out set**: Track true improvement
6. **Save intermediate models**: Track progress over time

## Next Steps

- [Collaborative Review](/docs/guides/collaboration) - Team-based labeling
- [Vector DB](/docs/guides/vectordb) - Scale similarity search
- [API Reference](/docs/guides/intelligent-sampling) - Full API documentation
