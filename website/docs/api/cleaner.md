---
sidebar_position: 1
title: DatasetCleaner
---

# DatasetCleaner

The main entry point for data quality analysis.

```python
from clean import DatasetCleaner
```

## Constructor

```python
DatasetCleaner(
    data: Union[pd.DataFrame, np.ndarray, str, Path, Dataset],
    labels: Optional[Union[np.ndarray, str]] = None,
    feature_columns: Optional[List[str]] = None,
    task_type: Optional[str] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame, ndarray, str, Path, Dataset | Input data |
| `labels` | ndarray or str | Labels or column name |
| `feature_columns` | list | Features to analyze |
| `task_type` | str | "classification" or "regression" |

### Example

```python
import pandas as pd
from clean import DatasetCleaner

df = pd.read_csv("data.csv")
cleaner = DatasetCleaner(df, labels="target")
```

## Methods

### analyze()

Run quality analysis on the dataset.

```python
analyze(
    detectors: Optional[List[str]] = None,
    show_progress: bool = True,
) -> QualityReport
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detectors` | list | None | Detectors to run (default: all) |
| `show_progress` | bool | True | Show progress bar |

#### Returns

[`QualityReport`](/docs/api/report) with analysis results.

#### Example

```python
# Run all detectors
report = cleaner.analyze()

# Run specific detectors
report = cleaner.analyze(detectors=["label_errors", "duplicates"])
```

### get_clean_data()

Get dataset with issues removed.

```python
get_clean_data(
    remove_duplicates: bool = True,
    remove_outliers: bool = False,
    remove_label_errors: bool = False,
    outlier_threshold: float = 0.05,
) -> pd.DataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_duplicates` | bool | True | Remove duplicates |
| `remove_outliers` | bool | False | Remove outliers |
| `remove_label_errors` | bool | False | Remove label errors |
| `outlier_threshold` | float | 0.05 | Outlier contamination |

#### Returns

Cleaned pandas DataFrame.

### get_review_queue()

Get prioritized review queue.

```python
get_review_queue(
    issue_types: Optional[List[str]] = None,
    limit: int = 100,
) -> pd.DataFrame
```

#### Returns

DataFrame with columns: index, issue_type, confidence, details.

## Properties

### data_type

```python
@property
def data_type(self) -> str
```

Returns detected data type: "tabular", "text", or "image".

### n_samples

```python
@property
def n_samples(self) -> int
```

Number of samples in the dataset.

### n_features

```python
@property
def n_features(self) -> int
```

Number of features.

### class_distribution

```python
@property
def class_distribution(self) -> Dict[Any, int]
```

Class label distribution.

## Example: Full Workflow

```python
import pandas as pd
from clean import DatasetCleaner

# Load data
df = pd.read_csv("training_data.csv")

# Create cleaner
cleaner = DatasetCleaner(df, labels="target")

# Analyze
report = cleaner.analyze()
print(report.summary())

# Get clean data
clean_df = cleaner.get_clean_data(
    remove_duplicates=True,
    remove_outliers=True
)

# Export
clean_df.to_csv("clean_data.csv", index=False)
```
