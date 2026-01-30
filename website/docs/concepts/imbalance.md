---
sidebar_position: 5
title: Class Imbalance
---

# Class Imbalance Detection

Class imbalance occurs when classes have unequal representation in your dataset.

## Why Imbalance Matters

| Imbalance Ratio | Impact |
|-----------------|--------|
| 1:1 to 1:3 | Minimal - standard training works |
| 1:3 to 1:10 | Moderate - consider resampling |
| 1:10 to 1:100 | Severe - special handling required |
| >1:100 | Extreme - anomaly detection territory |

Effects of imbalance:
- **Models ignore minority classes**: Easier to predict majority
- **Inflated accuracy**: 99% accuracy if 99% is one class
- **Poor recall on rare classes**: Misses important cases

## How Clean Detects Imbalance

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get imbalance analysis
imbalance = report.imbalance_result
print(imbalance)
```

### Results Explained

```python
# Class counts
print(imbalance.metadata['class_counts'])
# {'positive': 9000, 'negative': 1000}

# Imbalance ratio
print(imbalance.metadata['imbalance_ratio'])
# 9.0 (majority / minority)

# Majority and minority classes
print(imbalance.metadata['majority_class'])  # 'positive'
print(imbalance.metadata['minority_class'])  # 'negative'

# Severity
print(imbalance.metadata['severity'])  # 'moderate'
```

### Severity Levels

| Ratio | Severity | Recommendation |
|-------|----------|----------------|
| Under 2:1 | None | No action needed |
| 2:1 - 5:1 | Low | Consider class weights |
| 5:1 - 10:1 | Moderate | Use resampling or weights |
| >10:1 | High | Major intervention needed |

## Visualization

```python
# Built-in visualization
from clean.visualization import plot_class_distribution

plot_class_distribution(report)
```

Or manually:

```python
import matplotlib.pyplot as plt

counts = report.class_distribution.class_counts
plt.bar(counts.keys(), counts.values())
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
```

## Handling Imbalance

### Option 1: Class Weights

Most models support class weighting:

```python
from sklearn.ensemble import RandomForestClassifier

# Calculate weights
class_weights = 'balanced'  # Auto-compute
# or
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=classes, y=y)

model = RandomForestClassifier(class_weight='balanced')
```

### Option 2: Oversampling (SMOTE)

Generate synthetic minority samples:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Before: {len(X)}, After: {len(X_resampled)}")
```

### Option 3: Undersampling

Remove majority class samples:

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

### Option 4: Combination

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)
```

### Option 5: Threshold Tuning

Instead of resampling, adjust the classification threshold:

```python
# Get predicted probabilities
probs = model.predict_proba(X_test)[:, 1]

# Find optimal threshold for F1
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = max(thresholds, key=lambda t: f1_score(y_test, probs > t))

# Use custom threshold
y_pred = (probs > best_threshold).astype(int)
```

## Metrics for Imbalanced Data

Avoid accuracy! Use these instead:

| Metric | When to Use |
|--------|-------------|
| **F1 Score** | Balance precision and recall |
| **PR-AUC** | Precision-Recall curve area |
| **Balanced Accuracy** | Average recall per class |
| **Cohen's Kappa** | Agreement correcting for chance |
| **MCC** | Matthews Correlation Coefficient |

```python
from sklearn.metrics import (
    f1_score, 
    balanced_accuracy_score,
    average_precision_score,
    matthews_corrcoef,
)

print(f"F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
print(f"Balanced Acc: {balanced_accuracy_score(y_true, y_pred):.3f}")
print(f"PR-AUC: {average_precision_score(y_true, y_proba):.3f}")
print(f"MCC: {matthews_corrcoef(y_true, y_pred):.3f}")
```

## Multi-Class Imbalance

For multi-class problems:

```python
# Get per-class metrics
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

Check:
- **Per-class recall**: Low recall = class being ignored
- **Support**: Number of samples per class
- **Macro vs Weighted F1**: Macro treats all classes equally

## Best Practices

### 1. Always Check Distribution

```python
import pandas as pd
print(pd.Series(y).value_counts(normalize=True))
```

### 2. Stratified Splits

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # <-- Important!
)
```

### 3. Combine Strategies

```python
# Use SMOTE + class weights + custom threshold
smote = SMOTE(sampling_strategy=0.5)  # Partial resampling
X_res, y_res = smote.fit_resample(X, y)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_res, y_res)

probs = model.predict_proba(X_test)[:, 1]
y_pred = probs > optimal_threshold
```

### 4. Validate on Original Distribution

```python
# Train on resampled, validate on original
model.fit(X_train_resampled, y_train_resampled)
score = model.score(X_test, y_test)  # Original distribution
```

## Next Steps

- [Bias Detection](/docs/concepts/bias) - Fairness analysis
- [Auto-Fix Engine](/docs/guides/auto-fix) - Automated handling
- [API Reference](/docs/api/detectors) - ImbalanceDetector API
