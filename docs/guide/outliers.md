# Outlier Detection

Outliers are samples that differ significantly from the rest of your data. They can be:

- **Genuine anomalies**: Rare but valid data points
- **Errors**: Data entry mistakes, sensor malfunctions
- **Mislabeled**: Samples in the wrong class

## Detection Methods

Clean uses an ensemble of methods for robust outlier detection:

### Isolation Forest

Tree-based method that isolates anomalies by random partitioning. Effective for high-dimensional data.

### Local Outlier Factor (LOF)

Density-based method that compares local density of a sample to its neighbors.

### Statistical Methods

- **Z-score**: Samples beyond a threshold number of standard deviations
- **IQR**: Samples outside the interquartile range

## Basic Usage

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get outliers
outliers = report.outliers()
print(outliers.head())
```

**Output:**
```
   index     method     score
0    234  isolation    0.823
1     56        lof    0.756
2    189  isolation    0.698
```

## Understanding the Output

| Column | Description |
|--------|-------------|
| `index` | Row index in original data |
| `method` | Detection method that flagged it |
| `score` | Outlier score (higher = more anomalous) |

## Configuration

```python
from clean import OutlierDetector

detector = OutlierDetector(
    methods=['isolation_forest', 'lof', 'zscore'],  # Methods to use
    contamination=0.1,                               # Expected outlier fraction
    ensemble_method='any',                           # 'any', 'all', or 'majority'
)

cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    outlier_detector=detector
)
```

### Ensemble Methods

| Method | Description |
|--------|-------------|
| `'any'` | Flagged if ANY method detects it |
| `'all'` | Flagged only if ALL methods agree |
| `'majority'` | Flagged if majority of methods agree |

## Contamination Parameter

The `contamination` parameter sets the expected proportion of outliers:

- **0.01 (1%)**: Very selective, only flags extreme outliers
- **0.1 (10%)**: Moderate sensitivity
- **0.2 (20%)**: Aggressive, flags more samples

```python
# Conservative: fewer outliers
detector = OutlierDetector(contamination=0.01)

# Aggressive: more outliers
detector = OutlierDetector(contamination=0.2)
```

## Per-Class Outliers

Detect outliers within each class separately:

```python
detector = OutlierDetector(
    per_class=True  # Fit separate models per class
)
```

This is useful when classes have different distributions.

## Handling Outliers

### Option 1: Conservative Removal

```python
clean_df = cleaner.get_clean_data(
    remove_outliers='conservative'  # Remove only high-score outliers
)
```

### Option 2: Aggressive Removal

```python
clean_df = cleaner.get_clean_data(
    remove_outliers='aggressive'  # Remove all detected outliers
)
```

### Option 3: Manual Review

```python
# Add outlier flag to DataFrame
outlier_indices = set(report.outliers()['index'])
df['is_outlier'] = df.index.isin(outlier_indices)

# Review outliers
print(df[df['is_outlier']])
```

### Option 4: Score-Based Filtering

```python
# Only remove outliers with high scores
outliers = report.outliers()
high_score_outliers = outliers[outliers['score'] > 0.8]
df_clean = df.drop(high_score_outliers['index'])
```

## Visualization

```python
from clean.visualization import plot_outlier_distribution

# Scatter plot with outliers highlighted
fig = plot_outlier_distribution(report, features=df)
fig.savefig('outliers.png')
```

## Feature Importance

Understand which features contribute most to outlier scores:

```python
# Get feature contributions for a specific outlier
idx = 234  # outlier index
sample = df.loc[idx]

# Compare to population statistics
for col in df.columns:
    mean = df[col].mean()
    std = df[col].std()
    z = (sample[col] - mean) / std
    if abs(z) > 2:
        print(f"{col}: z-score = {z:.2f}")
```

## Best Practices

1. **Understand your data**: Not all outliers should be removed
2. **Domain expertise**: Some "outliers" may be important edge cases
3. **Use conservative settings initially**: Start strict, relax if needed
4. **Visualize**: Always plot outliers to understand what's being flagged
5. **Document decisions**: Record why samples were kept or removed

## Example: Detecting Anomalies in Sensor Data

```python
import numpy as np
import pandas as pd
from clean import DatasetCleaner

# Simulate sensor data with anomalies
np.random.seed(42)
normal_data = np.random.normal(100, 10, (1000, 5))
anomalies = np.random.normal(150, 5, (20, 5))  # Sensor malfunction
data = np.vstack([normal_data, anomalies])

df = pd.DataFrame(data, columns=[f'sensor_{i}' for i in range(5)])
df['status'] = 'normal'

# Detect
cleaner = DatasetCleaner(data=df, label_column='status', task='classification')
report = cleaner.analyze()

# Check detection
outliers = report.outliers()
print(f"Detected {len(outliers)} outliers out of 20 actual anomalies")
```
