# Label Error Detection

Label errors are one of the most impactful data quality issues. Even a small percentage of mislabeled samples can significantly hurt model performance.

## How It Works

Clean uses **confident learning** (via the [cleanlab](https://github.com/cleanlab/cleanlab) library) to identify potential label errors. The algorithm:

1. Trains a cross-validated classifier on your data
2. Gets out-of-sample predicted probabilities for each sample
3. Estimates the confusion matrix between true and given labels
4. Identifies samples where the given label is unlikely to be correct

## Basic Usage

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get label errors
label_errors = report.label_errors()
print(label_errors.head())
```

**Output:**
```
   index  given_label  predicted_label  confidence
0     42          cat              dog        0.94
1    187          cat             bird        0.89
2    523          dog              cat        0.87
```

## Understanding the Output

| Column | Description |
|--------|-------------|
| `index` | Row index in original data |
| `given_label` | The label currently assigned |
| `predicted_label` | The model's predicted label |
| `confidence` | Confidence that this is an error (0-1) |

## Configuration

```python
from clean import LabelErrorDetector

detector = LabelErrorDetector(
    n_folds=5,                    # Cross-validation folds (default: 5)
    confidence_threshold=0.5,      # Minimum confidence to flag (default: 0.5)
    model=None,                    # Custom sklearn classifier (default: LogisticRegression)
)

cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    label_error_detector=detector
)
```

### Adjusting Sensitivity

- **Higher `n_folds`** (e.g., 10): More stable estimates, slower
- **Higher `confidence_threshold`** (e.g., 0.8): Fewer, higher-confidence errors
- **Lower `confidence_threshold`** (e.g., 0.3): More errors, may include false positives

## Custom Classifier

You can provide your own classifier for better performance:

```python
from sklearn.ensemble import RandomForestClassifier

detector = LabelErrorDetector(
    model=RandomForestClassifier(n_estimators=100, random_state=42)
)
```

!!! tip
    Use a model similar to what you'll train on the cleaned data. This helps the detector find errors that would confuse your final model.

## Handling Label Errors

### Option 1: Manual Review

```python
# Get review queue prioritized by confidence
review_queue = cleaner.get_review_queue(
    include_label_errors=True,
    include_outliers=False,
    include_duplicates=False,
    max_items=100
)
```

### Option 2: Automatic Relabeling

```python
# Get relabeling suggestions
suggestions = cleaner.relabel(apply_suggestions=False)

# Apply with high confidence threshold
df_relabeled = cleaner.relabel(
    apply_suggestions=True,
    confidence_threshold=0.9  # Only relabel if very confident
)
```

!!! warning
    Automatic relabeling should only be used for exploratory purposes or when you have very high confidence. Always validate on a held-out set.

### Option 3: Exclusion

```python
# Get clean data excluding potential errors
clean_df = cleaner.get_clean_data(
    remove_duplicates=False,
    remove_outliers=False
)
# Note: Label errors are NOT removed by default, only flagged
```

## Multiclass Classification

Clean handles multiclass classification automatically:

```python
# Works with any number of classes
cleaner = DatasetCleaner(
    data=df,
    label_column='category',  # e.g., 10 categories
    task='classification'
)
```

## Regression Tasks

Currently, label error detection is only available for classification tasks. For regression, consider using outlier detection on the target variable.

## Best Practices

1. **Start with a subset**: Test on a small sample first
2. **Validate detections**: Manually check a random sample of detected errors
3. **Iterate**: Run detection → fix obvious errors → re-run
4. **Track metrics**: Compare model performance before/after cleaning

## Example: Finding Label Errors in MNIST

```python
from sklearn.datasets import load_digits
import pandas as pd
from clean import DatasetCleaner

# Load data
digits = load_digits()
df = pd.DataFrame(digits.data)
df['label'] = digits.target

# Introduce some label errors
import numpy as np
np.random.seed(42)
error_idx = np.random.choice(len(df), size=50, replace=False)
df.loc[error_idx, 'label'] = (df.loc[error_idx, 'label'] + 1) % 10

# Detect errors
cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Check how many we found
detected = set(report.label_errors()['index'])
actual = set(error_idx)
print(f"Recall: {len(detected & actual) / len(actual):.2%}")
```
