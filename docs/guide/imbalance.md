# Class Imbalance

Class imbalance occurs when some classes have significantly more samples than others. This can cause:

- **Bias toward majority class**: Model predicts common classes more often
- **Poor minority class performance**: Rare classes are under-represented
- **Misleading metrics**: High accuracy despite poor performance on minority classes

## Detection

Clean automatically analyzes class distribution:

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get class distribution
distribution = report.class_distribution
print(f"Imbalance ratio: {distribution.imbalance_ratio:.2f}")
```

## Understanding the Output

```python
distribution = report.class_distribution

print(f"Classes: {distribution.class_names}")
print(f"Counts: {distribution.counts}")
print(f"Imbalance ratio: {distribution.imbalance_ratio}")
print(f"Majority class: {distribution.majority_class}")
print(f"Minority class: {distribution.minority_class}")
```

| Attribute | Description |
|-----------|-------------|
| `class_names` | List of class labels |
| `counts` | Dict of class → count |
| `imbalance_ratio` | Ratio of largest to smallest class |
| `majority_class` | Class with most samples |
| `minority_class` | Class with fewest samples |

## Imbalance Severity

| Ratio | Severity | Recommendation |
|-------|----------|----------------|
| 1-3 | Mild | Usually acceptable |
| 3-10 | Moderate | Consider resampling |
| 10-50 | Severe | Resampling or class weights needed |
| 50+ | Extreme | Specialized techniques required |

## Visualization

```python
from clean.visualization import plot_class_distribution

fig = plot_class_distribution(report.labels)
fig.savefig('class_distribution.png')
```

## Handling Imbalance

Clean doesn't automatically fix imbalance, but provides information to guide your strategy.

### Option 1: Oversampling Minority Classes

```python
from imblearn.over_sampling import SMOTE

# Get clean features and labels
X = df.drop('label', axis=1)
y = df['label']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

### Option 2: Undersampling Majority Classes

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)
```

### Option 3: Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
weight_dict = dict(zip(np.unique(y), class_weights))

# Use in model training
model.fit(X, y, class_weight=weight_dict)
```

### Option 4: Stratified Sampling

Ensure train/test splits preserve class distribution:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,  # Preserve class distribution
    random_state=42
)
```

## Quality Score Impact

The imbalance quality score in Clean's report:

- **1.0**: Perfectly balanced (equal class sizes)
- **0.5**: Moderate imbalance
- **0.0**: Extreme imbalance

```python
print(f"Balance quality: {report.quality_score.balance:.2f}")
```

## Best Practices

1. **Understand the context**: Some imbalance is natural (e.g., fraud detection)
2. **Use appropriate metrics**: Precision, recall, F1 instead of accuracy
3. **Stratify splits**: Always use stratified sampling for imbalanced data
4. **Consider cost**: Misclassifying minority class may be more costly
5. **Combine techniques**: Often multiple strategies work best together

## Example: Analyzing Imbalanced Dataset

```python
from sklearn.datasets import make_classification
import pandas as pd
from clean import DatasetCleaner

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_classes=3,
    weights=[0.7, 0.2, 0.1],  # 70%, 20%, 10%
    n_informative=5,
    random_state=42
)

df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
df['label'] = y

# Analyze
cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Check imbalance
dist = report.class_distribution
print(f"Class counts: {dist.counts}")
print(f"Imbalance ratio: {dist.imbalance_ratio:.1f}:1")
print(f"Balance quality score: {report.quality_score.balance:.2f}")
```
