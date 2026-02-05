# Contamination Detection

Cross-dataset contamination and data leakage detection.

## Quick Example

```python
from clean.contamination import detect_contamination

# Check for train/test leakage
report = detect_contamination(train_df, test_df)

print(f"Contamination rate: {report.contamination_rate:.2%}")
print(f"Contaminated samples: {report.n_contaminated}")

for rec in report.recommendations:
    print(f"- {rec}")
```

## API Reference

### ContaminationConfig

Configuration for contamination detection.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `exact_match` | bool | `True` | Check for exact duplicates |
| `fuzzy_match` | bool | `True` | Check for near-duplicates |
| `fuzzy_threshold` | float | `0.95` | Similarity threshold for fuzzy matching |
| `text_similarity` | bool | `True` | Check text column similarity |
| `embedding_similarity` | bool | `False` | Use embeddings for similarity |
| `sample_size` | int \| None | `None` | Sample size for large datasets |

### SeverityLevel (Enum)

Contamination severity levels.

| Value | Description |
|-------|-------------|
| `CRITICAL` | >10% contamination, major concern |
| `HIGH` | 5-10% contamination |
| `MEDIUM` | 1-5% contamination |
| `LOW` | <1% contamination |

### ContaminationDetector

Main detector class.

#### `__init__(config: ContaminationConfig | None = None)`

Initialize with optional configuration.

#### `detect(dataset1, dataset2, name1="dataset_1", name2="dataset_2", text_columns=None) -> ContaminationReport`

Detect contamination between two datasets.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset1` | `pd.DataFrame \| str` | First dataset or path |
| `dataset2` | `pd.DataFrame \| str` | Second dataset or path |
| `name1` | `str` | Name for first dataset |
| `name2` | `str` | Name for second dataset |
| `text_columns` | `list[str] \| None` | Text columns for similarity |

### ContaminationReport

Detection result dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | Detection timestamp |
| `datasets_compared` | int | Number of datasets compared |
| `n_contaminated` | int | Count of contaminated samples |
| `contamination_rate` | float | Rate as fraction (0-1) |
| `contaminated_pairs` | list | List of contaminated sample pairs |
| `contamination_by_type` | dict | Breakdown by contamination type |
| `contamination_by_severity` | dict | Breakdown by severity |
| `recommendations` | list[str] | Remediation recommendations |

#### `to_dict() -> dict`

Convert report to dictionary.

### Convenience Functions

#### `detect_contamination(train, test, text_columns=None, config=None) -> ContaminationReport`

Quick contamination check between train and test sets.

```python
from clean.contamination import detect_contamination

report = detect_contamination(
    train_df, 
    test_df, 
    text_columns=["text_field"]
)
```

## Example Workflows

### Basic Train/Test Leakage Check

```python
from clean.contamination import ContaminationDetector

detector = ContaminationDetector()
report = detector.detect(
    train_data, 
    test_data,
    name1="training",
    name2="test"
)

if report.contamination_rate > 0.01:  # >1%
    print("WARNING: Significant data leakage detected!")
    print(f"Found {report.n_contaminated} leaked samples")
```

### Text Data Contamination

```python
from clean.contamination import ContaminationDetector, ContaminationConfig

config = ContaminationConfig(
    text_similarity=True,
    fuzzy_threshold=0.9  # Catch near-duplicates
)

detector = ContaminationDetector(config=config)
report = detector.detect(
    train_df, 
    test_df,
    text_columns=["text", "description"]
)
```

### Multi-Dataset Comparison

```python
from clean.contamination import ContaminationDetector

detector = ContaminationDetector()
datasets = [train_df, val_df, test_df]
names = ["train", "validation", "test"]

# Check all pairs
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        report = detector.detect(
            datasets[i], 
            datasets[j],
            name1=names[i],
            name2=names[j]
        )
        if report.n_contaminated > 0:
            print(f"{names[i]} <-> {names[j]}: {report.n_contaminated} contaminated")
```

### CI/CD Integration

```python
from clean.contamination import detect_contamination

report = detect_contamination(train_df, test_df)

# Fail CI if contamination exceeds threshold
assert report.contamination_rate < 0.001, \
    f"Data leakage detected: {report.contamination_rate:.2%}"
```
