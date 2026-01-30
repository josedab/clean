---
sidebar_position: 4
title: Outliers
---

# Outlier Detection

Outliers are samples that differ significantly from the rest of your data.

## Types of Outliers

### Global Outliers

Samples unusual compared to the entire dataset:

```
Feature 1: [1.0, 1.2, 1.1, 1.3, 100.0]  ← 100.0 is a global outlier
```

### Contextual Outliers

Samples unusual within their class:

```
Class A: [1.0, 1.2, 1.1, 1.3]
Class B: [5.0, 5.2, 5.1, 1.0]  ← 1.0 is unusual for Class B
```

### Collective Outliers

Groups of samples that are unusual together:

```
Normal: daily sales between $1K-$10K
Outlier group: 5 consecutive days with $0 sales (system down?)
```

## Why Outliers Matter

- **Skew learned representations**: Models overfit to unusual patterns
- **Distort statistics**: Mean, variance become unreliable
- **Indicate data quality issues**: Entry errors, sensor failures
- **Sometimes important**: Fraud, rare diseases, edge cases

## Detection Methods

Clean supports multiple outlier detection methods:

### Isolation Forest (Default)

Best for high-dimensional data. Isolates anomalies by random partitioning.

```python
from clean.detection import OutlierDetector

detector = OutlierDetector(method='isolation_forest')
```

### Local Outlier Factor (LOF)

Compares local density to neighbors. Good for clustered data.

```python
detector = OutlierDetector(method='lof', n_neighbors=20)
```

### Statistical Methods

Simple, interpretable methods:

```python
# IQR method
detector = OutlierDetector(method='iqr', multiplier=1.5)

# Z-score method  
detector = OutlierDetector(method='zscore', threshold=3.0)
```

### Ensemble (Most Robust)

Combines multiple methods with voting:

```python
detector = OutlierDetector(
    method='ensemble',
    ensemble_methods=['isolation_forest', 'lof', 'zscore'],
    voting='majority'  # or 'all', 'any'
)
```

## How to Detect Outliers

### Basic Usage

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get outlier indices
outlier_indices = report.outliers()
print(f"Found {len(outlier_indices)} outliers")
```

### Understanding Results

```python
outlier_result = report.outliers_result

# Outlier scores (higher = more anomalous)
scores = outlier_result.metadata['scores']

# Method used
method = outlier_result.metadata['method']

# Top 10 most anomalous
import numpy as np
top_10 = np.argsort(scores)[-10:]
```

## Configuration

### Contamination (Expected Outlier Rate)

```python
detector = OutlierDetector(
    method='isolation_forest',
    contamination=0.05,  # Expect ~5% outliers
)
```

If you don't know the contamination rate, use `'auto'`:

```python
detector = OutlierDetector(contamination='auto')
```

### Per-Class Detection

Detect outliers within each class:

```python
detector = OutlierDetector(
    method='isolation_forest',
    per_class=True,  # Detect outliers within each class
)
```

### Feature Selection

Focus on specific features:

```python
detector = OutlierDetector(
    feature_columns=['feature_1', 'feature_2'],
    ignore_columns=['id', 'timestamp'],
)
```

## Handling Outliers

### Option 1: Remove

```python
clean_df = cleaner.get_clean_data(remove_outliers='aggressive')
```

### Option 2: Flag for Review

```python
df['is_outlier'] = df.index.isin(outlier_indices)
review_df = df[df['is_outlier']]
```

### Option 3: Winsorize (Cap Values)

```python
import numpy as np

for col in numeric_columns:
    p01, p99 = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(p01, p99)
```

### Option 4: Investigate Root Cause

```python
# Which features drive outlier-ness?
from clean.detection import OutlierDetector

detector = OutlierDetector(method='isolation_forest')
detector.fit(df[numeric_cols])

# Feature importance (Isolation Forest)
importances = detector.model.feature_importances_
for col, imp in zip(numeric_cols, importances):
    print(f"{col}: {imp:.3f}")
```

## Comparison of Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Isolation Forest** | High-dimensional | Fast, scalable | Less interpretable |
| **LOF** | Clustered data | Captures local patterns | Slow on large data |
| **IQR** | Univariate | Simple, interpretable | Misses multivariate |
| **Z-score** | Normal distributions | Fast | Assumes normality |
| **Ensemble** | General | Robust | Slower |

## Best Practices

### 1. Start with Visualization

```python
import matplotlib.pyplot as plt

# Scatter plot with outliers highlighted
plt.scatter(df['x'], df['y'], c='blue', alpha=0.5)
plt.scatter(
    df.loc[outlier_indices, 'x'],
    df.loc[outlier_indices, 'y'],
    c='red', label='Outliers'
)
plt.legend()
```

### 2. Don't Remove Without Understanding

- Outliers might be your most valuable data (rare cases)
- Investigate root cause before removing
- Document decisions for reproducibility

### 3. Consider Domain Context

```python
# Time series: outliers might be events, not errors
# Fraud detection: outliers ARE what you're looking for
# Sensor data: outliers might indicate calibration issues
```

### 4. Use Conservative Settings Initially

```python
# Start conservative
detector = OutlierDetector(contamination=0.01)  # 1% expected

# Review flagged samples
# Increase contamination if too few detected
```

## Next Steps

- [Imbalance](/docs/concepts/imbalance) - Class distribution analysis
- [Auto-Fix Engine](/docs/guides/auto-fix) - Handle outliers automatically
- [API Reference](/docs/api/detectors) - OutlierDetector API
