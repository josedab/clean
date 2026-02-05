# Curriculum Learning Optimizer

Optimize training sample order for improved model learning.

## Quick Example

```python
from clean.curriculum import CurriculumOptimizer

# Create optimizer
optimizer = CurriculumOptimizer()

# Generate curriculum schedule
schedule = optimizer.optimize(X, y, quality_scores=quality_scores)

print(f"Samples: {schedule.n_samples}")
print(f"Epochs: {schedule.n_epochs}")
print(f"First epoch samples: {schedule.samples_per_epoch[0]}")
```

## API Reference

### CurriculumConfig

Configuration for curriculum learning.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | CurriculumStrategy | `EASY_TO_HARD` | Learning strategy |
| `n_epochs` | int | `10` | Number of training epochs |
| `warmup_epochs` | int | `2` | Warmup epochs (easy samples only) |
| `initial_fraction` | float | `0.3` | Initial fraction of samples |
| `growth_rate` | float | `0.1` | Sample growth rate per epoch |
| `pace_function` | str | `"linear"` | Pacing function |
| `pace_parameter` | float | `1.0` | Pacing function parameter |
| `quality_weight` | float | `0.4` | Weight for quality scores |
| `confidence_weight` | float | `0.3` | Weight for confidence |
| `neighbor_weight` | float | `0.2` | Weight for neighbor agreement |
| `outlier_weight` | float | `0.1` | Weight for outlier scores |
| `diversity_factor` | float | `0.2` | Diversity in sample selection |

### CurriculumStrategy (Enum)

Available curriculum strategies.

| Value | Description |
|-------|-------------|
| `EASY_TO_HARD` | Start with easy samples, progress to hard |
| `SELF_PACED` | Model decides sample difficulty |
| `DIVERSITY` | Maximize sample diversity |

### DifficultyMetric (Enum)

Metrics for computing sample difficulty.

| Value | Description |
|-------|-------------|
| `QUALITY_SCORE` | Use quality scores as difficulty |
| `MODEL_CONFIDENCE` | Use model prediction confidence |
| `LABEL_CERTAINTY` | Label certainty from quality analysis |
| `OUTLIER_SCORE` | Outlier score (outliers are harder) |
| `NEIGHBOR_AGREEMENT` | k-NN neighbor label agreement |
| `LOSS_VALUE` | Training loss value |

### CurriculumOptimizer

Main optimizer class.

#### `__init__(config=None, strategy=None)`

Initialize with optional configuration or strategy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `CurriculumConfig \| None` | Full configuration |
| `strategy` | `str \| CurriculumStrategy \| None` | Quick strategy selection |

#### `optimize(X, y, quality_scores=None, model=None) -> CurriculumSchedule`

Generate an optimized curriculum schedule.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray \| pd.DataFrame` | Feature matrix |
| `y` | `np.ndarray` | Labels |
| `quality_scores` | `np.ndarray \| None` | Per-sample quality scores |
| `model` | `BaseEstimator \| None` | Model for confidence-based difficulty |

### CurriculumSchedule

Generated curriculum schedule.

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | CurriculumStrategy | Strategy used |
| `n_samples` | int | Total samples |
| `n_epochs` | int | Number of epochs |
| `sample_order` | list[int] | Global sample ordering |
| `epoch_schedules` | dict | Per-epoch sample indices |
| `samples_per_epoch` | list[int] | Sample count per epoch |
| `difficulty_scores` | list[float] | Sample difficulty scores |
| `metadata` | dict | Additional metadata |

#### `to_dict() -> dict`

Convert schedule to dictionary.

### CurriculumDataLoader

Data loader that follows curriculum schedule.

#### `create_curriculum_loader(X, y, schedule, batch_size=32) -> CurriculumDataLoader`

Create a data loader from a schedule.

```python
from clean.curriculum import create_curriculum_loader

loader = create_curriculum_loader(X, y, schedule, batch_size=32)

for epoch in range(schedule.n_epochs):
    for X_batch, y_batch in loader.get_epoch(epoch):
        # Train on batch
        pass
```

## Example Workflows

### Basic Easy-to-Hard Curriculum

```python
from clean.curriculum import CurriculumOptimizer, CurriculumStrategy

optimizer = CurriculumOptimizer(strategy=CurriculumStrategy.EASY_TO_HARD)
schedule = optimizer.optimize(X, y)

# Use schedule for training
for epoch in range(schedule.n_epochs):
    epoch_indices = schedule.epoch_schedules.get(epoch, schedule.sample_order)
    X_epoch = X[epoch_indices]
    y_epoch = y[epoch_indices]
    # Train for one epoch
```

### Quality-Guided Curriculum

```python
from clean import DatasetCleaner
from clean.curriculum import CurriculumOptimizer

# Get quality scores from analysis
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Use quality scores for curriculum
quality_scores = report.get_quality_scores()  # If available

optimizer = CurriculumOptimizer()
schedule = optimizer.optimize(X, y, quality_scores=quality_scores)
```

### Self-Paced Learning

```python
from clean.curriculum import CurriculumOptimizer, CurriculumStrategy, CurriculumConfig
from sklearn.ensemble import RandomForestClassifier

config = CurriculumConfig(
    strategy=CurriculumStrategy.SELF_PACED,
    n_epochs=20,
    initial_fraction=0.2
)

# Initial model fit
model = RandomForestClassifier()
model.fit(X[:100], y[:100])  # Fit on subset

optimizer = CurriculumOptimizer(config=config)
schedule = optimizer.optimize(X, y, model=model)
```

### Integration with PyTorch

```python
from clean.curriculum import CurriculumOptimizer, create_curriculum_loader
import torch

optimizer = CurriculumOptimizer()
schedule = optimizer.optimize(X, y)

for epoch in range(schedule.n_epochs):
    epoch_indices = schedule.epoch_schedules.get(epoch, schedule.sample_order)
    
    # Create PyTorch tensors
    X_tensor = torch.tensor(X[epoch_indices], dtype=torch.float32)
    y_tensor = torch.tensor(y[epoch_indices], dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    for batch_X, batch_y in loader:
        # Train step
        pass
```
