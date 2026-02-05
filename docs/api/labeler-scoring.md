# Labeler Performance Scoring

Automated evaluation and scoring of human labelers.

## Quick Example

```python
from clean.labeler_scoring import LabelerEvaluator, evaluate_labelers

# Evaluate labelers
evaluator = LabelerEvaluator()
evaluator.fit(
    labels=annotation_labels,
    labeler_ids=annotator_ids,
    ground_truth=gold_labels
)

# Get rankings
ranking = evaluator.get_labeler_ranking(metric="accuracy")
print("Top labelers by accuracy:")
for labeler_id, score in ranking[:5]:
    print(f"  {labeler_id}: {score:.2%}")
```

## API Reference

### LabelerEvaluator

Main evaluator class.

#### `__init__(window_size=100, min_labels_for_evaluation=10)`

Initialize the evaluator.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `100` | Window for recent performance |
| `min_labels_for_evaluation` | `int` | `10` | Min labels to evaluate |

#### `fit(labels, labeler_ids, ground_truth=None, categories=None, timestamps=None, task_ids=None) -> LabelerEvaluator`

Fit evaluator on labeling data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `labels` | `list \| np.ndarray` | Labels assigned by labelers |
| `labeler_ids` | `list \| np.ndarray` | Labeler identifiers |
| `ground_truth` | `list \| np.ndarray \| None` | Ground truth labels |
| `categories` | `list \| np.ndarray \| None` | Category for each task |
| `timestamps` | `list[datetime] \| None` | Labeling timestamps |
| `task_ids` | `list \| None` | Task identifiers |

#### `get_labeler_metrics(labeler_id: str) -> LabelerMetrics | None`

Get metrics for a specific labeler.

#### `get_all_labeler_metrics() -> dict[str, LabelerMetrics]`

Get metrics for all evaluated labelers.

#### `get_labeler_ranking(metric="accuracy", ascending=False) -> list[tuple[str, float]]`

Get labelers ranked by a metric.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | `"accuracy"` | Metric to rank by |
| `ascending` | `bool` | `False` | Ascending order |

#### `get_labeler_report(labeler_id: str) -> LabelerReport | None`

Get detailed report for a labeler.

### LabelerMetrics

Labeler performance metrics dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `labeler_id` | str | Labeler identifier |
| `n_labels` | int | Total labels assigned |
| `n_tasks` | int | Unique tasks labeled |
| `accuracy` | float | Overall accuracy (0-1) |
| `error_rate` | float | Error rate (0-1) |
| `agreement_rate` | float | Agreement with others |
| `consistency_score` | float | Self-consistency score |
| `self_agreement` | float | Agreement on repeated tasks |
| `avg_labels_per_day` | float | Labeling velocity |
| `completion_rate` | float | Task completion rate |
| `expertise_level` | str | `"novice"`, `"intermediate"`, `"expert"` |
| `strong_categories` | list[str] | Categories with high accuracy |
| `weak_categories` | list[str] | Categories with low accuracy |
| `performance_trend` | str | `"improving"`, `"stable"`, `"declining"` |
| `recent_accuracy` | float | Recent window accuracy |
| `accuracy_change` | float | Change from baseline |
| `first_label_date` | datetime | First labeling date |
| `last_label_date` | datetime | Most recent labeling |
| `active_days` | int | Days with labeling activity |
| `metadata` | dict | Additional metadata |

#### `to_dict() -> dict`

Convert metrics to dictionary.

### LabelerReport

Detailed labeler report dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `labeler_id` | str | Labeler identifier |
| `metrics` | LabelerMetrics | Core metrics |
| `generated_at` | datetime | Report generation time |
| `category_performance` | dict | Performance by category |
| `common_error_patterns` | list | Frequent error patterns |
| `confused_label_pairs` | list | Often-confused label pairs |
| `training_recommendations` | list[str] | Training suggestions |
| `suitable_task_types` | list[str] | Good task matches |
| `unsuitable_task_types` | list[str] | Poor task matches |
| `percentile_rank` | float | Rank vs other labelers |
| `comparison_to_avg` | dict | Comparison to average |

### SmartRouter

Intelligent task routing based on labeler performance.

#### `__init__(evaluator: LabelerEvaluator)`

Initialize router with an evaluator.

#### `route(category=None, difficulty=None, n_labelers=1) -> list[str]`

Route task to best labelers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | `str \| None` | `None` | Task category |
| `difficulty` | `str \| None` | `None` | Task difficulty |
| `n_labelers` | `int` | `1` | Number of labelers needed |

### Convenience Functions

#### `evaluate_labelers(labels, labeler_ids, ground_truth=None, **kwargs) -> LabelerEvaluator`

Quick labeler evaluation.

#### `get_labeler_report(evaluator, labeler_id) -> LabelerReport | None`

Get report from existing evaluator.

## Example Workflows

### Basic Evaluation

```python
from clean.labeler_scoring import LabelerEvaluator

evaluator = LabelerEvaluator(min_labels_for_evaluation=20)
evaluator.fit(
    labels=all_labels,
    labeler_ids=labeler_ids,
    ground_truth=gold_labels
)

# Check all metrics
for labeler_id, metrics in evaluator.get_all_labeler_metrics().items():
    print(f"{labeler_id}:")
    print(f"  Accuracy: {metrics.accuracy:.2%}")
    print(f"  Level: {metrics.expertise_level}")
    print(f"  Trend: {metrics.performance_trend}")
```

### Category-Based Analysis

```python
evaluator = LabelerEvaluator()
evaluator.fit(
    labels=labels,
    labeler_ids=labeler_ids,
    ground_truth=gold_labels,
    categories=task_categories
)

# Get detailed report
report = evaluator.get_labeler_report("labeler_123")

if report:
    print(f"Category Performance for {report.labeler_id}:")
    for cat, perf in report.category_performance.items():
        print(f"  {cat}: {perf['accuracy']:.2%}")
    
    print(f"\nTraining Recommendations:")
    for rec in report.training_recommendations:
        print(f"  - {rec}")
```

### Smart Task Routing

```python
from clean.labeler_scoring import LabelerEvaluator, SmartRouter

# Train evaluator on historical data
evaluator = LabelerEvaluator()
evaluator.fit(labels, labeler_ids, ground_truth, categories)

# Create router
router = SmartRouter(evaluator)

# Route new tasks
for task in new_tasks:
    suggested_labelers = router.route(
        category=task.category,
        difficulty=task.difficulty,
        n_labelers=3
    )
    print(f"Task {task.id}: Assign to {suggested_labelers}")
```

### Performance Tracking Over Time

```python
from clean.labeler_scoring import LabelerEvaluator
from datetime import datetime, timedelta

evaluator = LabelerEvaluator(window_size=50)
evaluator.fit(
    labels=labels,
    labeler_ids=labeler_ids,
    ground_truth=gold_labels,
    timestamps=timestamps
)

# Check performance trends
for labeler_id in evaluator.get_all_labeler_metrics().keys():
    metrics = evaluator.get_labeler_metrics(labeler_id)
    if metrics:
        print(f"{labeler_id}:")
        print(f"  Overall: {metrics.accuracy:.2%}")
        print(f"  Recent: {metrics.recent_accuracy:.2%}")
        print(f"  Trend: {metrics.performance_trend}")
        print(f"  Change: {metrics.accuracy_change:+.2%}")
```

### Quality-Based Labeler Filtering

```python
from clean.labeler_scoring import LabelerEvaluator

evaluator = LabelerEvaluator()
evaluator.fit(labels, labeler_ids, ground_truth)

# Filter to high-quality labelers
high_quality = []
for labeler_id, metrics in evaluator.get_all_labeler_metrics().items():
    if metrics.accuracy >= 0.9 and metrics.consistency_score >= 0.85:
        high_quality.append(labeler_id)

print(f"High-quality labelers: {high_quality}")

# Use only their labels
quality_mask = [lid in high_quality for lid in labeler_ids]
filtered_labels = [l for l, m in zip(labels, quality_mask) if m]
```

### Consensus Building

```python
from clean.labeler_scoring import LabelerEvaluator
import numpy as np

evaluator = LabelerEvaluator()
evaluator.fit(labels, labeler_ids, ground_truth)

# Weight labels by labeler accuracy
all_metrics = evaluator.get_all_labeler_metrics()

def weighted_consensus(task_labels, task_labelers):
    weights = []
    for lid in task_labelers:
        if lid in all_metrics:
            weights.append(all_metrics[lid].accuracy)
        else:
            weights.append(0.5)  # Default weight
    
    # Weighted voting
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # For binary classification
    weighted_vote = sum(w * l for w, l in zip(weights, task_labels))
    return 1 if weighted_vote >= 0.5 else 0
```
