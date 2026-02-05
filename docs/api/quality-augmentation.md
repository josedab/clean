# Quality-Aware Augmentation

Intelligent data augmentation that improves dataset quality by addressing gaps.

## Quick Example

```python
from clean.quality_augmentation import QualityAwareAugmenter

# Create augmenter
augmenter = QualityAwareAugmenter()

# Augment imbalanced data
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["label"].values

result = augmenter.augment(X, y)

print(f"Original samples: {result.n_samples_original}")
print(f"Generated samples: {result.n_samples_generated}")
print(f"Quality improvement: {result.quality_improvement:.2%}")
```

## API Reference

### AugmentationConfig

Configuration for augmentation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_balance_ratio` | float | `0.8` | Target minority/majority ratio |
| `min_samples_per_class` | int | `10` | Minimum samples per class |
| `max_augmentation_factor` | float | `3.0` | Maximum augmentation multiplier |
| `quality_threshold` | float | `0.5` | Minimum quality for generated samples |
| `diversity_weight` | float | `0.3` | Weight for diversity in selection |
| `preferred_methods` | list | `None` | Preferred augmentation methods |
| `validate_samples` | bool | `True` | Whether to validate generated samples |
| `use_quality_filter` | bool | `True` | Filter low-quality augmentations |
| `max_rejection_rate` | float | `0.5` | Maximum rejection rate before stopping |

### AugmentationMethod (Enum)

Available augmentation methods.

| Value | Description |
|-------|-------------|
| `SMOTE` | Synthetic Minority Over-sampling |
| `MIXUP` | Mixup interpolation |
| `NOISE_INJECTION` | Random noise injection |

### GapType (Enum)

Types of quality gaps addressed.

| Value | Description |
|-------|-------------|
| `CLASS_IMBALANCE` | Imbalanced class distribution |
| `LOW_COVERAGE` | Low feature space coverage |
| `BOUNDARY_SPARSE` | Sparse decision boundaries |

### QualityAwareAugmenter

Main augmenter class.

#### `__init__(config: AugmentationConfig | None = None)`

Initialize with optional configuration.

#### `augment(X, y, report=None, label_column=None) -> AugmentationResult`

Perform quality-aware augmentation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray \| pd.DataFrame` | Feature matrix |
| `y` | `np.ndarray` | Labels |
| `report` | `QualityReport \| None` | Quality report for gap analysis |
| `label_column` | `str \| None` | Label column name (if X is DataFrame) |

### AugmentationResult

Result of augmentation operation.

| Field | Type | Description |
|-------|------|-------------|
| `n_samples_original` | int | Original sample count |
| `n_samples_generated` | int | Generated sample count |
| `n_samples_accepted` | int | Accepted after filtering |
| `n_samples_rejected` | int | Rejected by quality filter |
| `gaps_addressed` | list | Quality gaps addressed |
| `samples` | list \| None | Generated sample objects |
| `quality_improvement` | float | Estimated quality improvement |
| `class_balance_improvement` | float | Balance improvement |
| `diversity_improvement` | float | Diversity improvement |
| `rejection_reasons` | dict | Reasons for rejections |
| `method_breakdown` | dict | Samples by method |

#### `to_dict() -> dict`

Convert result to dictionary.

### Convenience Functions

#### `augment_for_quality(X, y, config=None, report=None) -> AugmentationResult`

One-liner augmentation function.

#### `create_augmenter(config=None) -> QualityAwareAugmenter`

Create an augmenter instance.

## Example Workflows

### Address Class Imbalance

```python
from clean.quality_augmentation import QualityAwareAugmenter, AugmentationConfig

config = AugmentationConfig(
    target_balance_ratio=1.0,  # Fully balanced
    max_augmentation_factor=5.0
)

augmenter = QualityAwareAugmenter(config=config)
result = augmenter.augment(X, y)

if result.samples:
    X_aug = np.array([s.features for s in result.samples])
    y_aug = np.array([s.label for s in result.samples])
    
    X_combined = np.vstack([X, X_aug])
    y_combined = np.concatenate([y, y_aug])
```

### Using Quality Report for Targeted Augmentation

```python
from clean import DatasetCleaner
from clean.quality_augmentation import QualityAwareAugmenter

# Analyze data first
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Augment based on identified gaps
augmenter = QualityAwareAugmenter()
result = augmenter.augment(X, y, report=report)

print(f"Gaps addressed: {result.gaps_addressed}")
```

### Quality-Filtered Augmentation

```python
config = AugmentationConfig(
    quality_threshold=0.7,  # Only accept high-quality samples
    validate_samples=True,
    use_quality_filter=True
)

augmenter = QualityAwareAugmenter(config=config)
result = augmenter.augment(X, y)

print(f"Accepted: {result.n_samples_accepted}")
print(f"Rejected: {result.n_samples_rejected}")
print(f"Rejection reasons: {result.rejection_reasons}")
```
