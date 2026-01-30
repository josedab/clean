---
sidebar_position: 2
title: Label Errors
---

# Label Error Detection

Label errors are mislabeled samples in your training data. They're surprisingly common—studies show **3-10% of labels are wrong** in popular ML benchmarks.

## Why Label Errors Matter

- **Degrade model performance**: Models learn wrong patterns
- **Hard to detect**: Models can memorize noise
- **Compound over time**: Mislabeled predictions become mislabeled training data

## How Clean Detects Label Errors

Clean uses **confident learning**, a technique that identifies samples where a trained model strongly disagrees with the given label.

### The Algorithm

1. **Cross-validation training**: Train a classifier using k-fold CV
2. **Out-of-fold predictions**: Get probability estimates for each sample
3. **Confident joint estimation**: Build a matrix of (given label × predicted label)
4. **Pruning**: Identify samples likely mislabeled

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get label errors sorted by confidence
errors = report.label_errors()
```

### Understanding the Results

```python
print(errors.head())
```

| index | given_label | predicted_label | confidence | self_confidence |
|-------|-------------|-----------------|------------|-----------------|
| 42 | cat | dog | 0.94 | 0.12 |
| 187 | cat | bird | 0.89 | 0.08 |
| 523 | dog | cat | 0.87 | 0.15 |

- **given_label**: The label in your dataset
- **predicted_label**: What the model thinks it should be
- **confidence**: How confident Clean is this is an error (0-1)
- **self_confidence**: Model's confidence in the given label (low = likely wrong)

## Configuration

### Basic Usage

```python
cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    task='classification'
)
```

### Advanced Configuration

```python
from clean.detection import LabelErrorDetector

detector = LabelErrorDetector(
    # Cross-validation settings
    cv_n_folds=5,           # Number of folds (default: 5)
    
    # Classifier to use
    classifier='logistic',   # 'logistic', 'random_forest', or sklearn estimator
    
    # Pruning method
    prune_method='prune_by_noise_rate',  # How to select errors
    
    # Minimum confidence to report
    min_confidence=0.5,
)

result = detector.fit_detect(features, labels)
```

### Prune Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `prune_by_noise_rate` | Based on noise rate per class | Balanced datasets |
| `prune_by_class` | Prune from noisiest classes | Imbalanced datasets |
| `both` | Intersection of above methods | Conservative |
| `confident_learning` | Full CL approach | Most datasets |

## Best Practices

### 1. Review High-Confidence Errors First

```python
# Top 50 most confident errors
top_errors = errors[errors['confidence'] > 0.9]
print(f"High-confidence errors: {len(top_errors)}")
```

### 2. Manual Verification

Don't auto-correct all errors. Review a sample:

```python
import random

# Sample 20 errors for review
sample = errors.sample(n=min(20, len(errors)))
for _, row in sample.iterrows():
    print(f"Index {row['index']}: {row['given_label']} → {row['predicted_label']}")
    # Show the actual data
    print(df.iloc[row['index']])
```

### 3. Track Correction Rates

If you're manually reviewing, track your correction rate:

```python
# If 80% of flagged errors are actual errors, the detector is working well
true_positives = confirmed_errors / reviewed_errors
print(f"Precision: {true_positives:.1%}")
```

### 4. Use with FixEngine

```python
from clean import FixEngine, FixConfig

config = FixConfig(
    label_error_threshold=0.95,  # Very conservative
    auto_relabel=False,          # Suggest only
)

engine = FixEngine(report=report, features=X, labels=y, config=config)
fixes = engine.suggest_fixes(include_label_errors=True)
```

## Common Questions

### How many errors should I expect?

Typically 1-10% of labels, depending on:
- How labels were collected
- Annotator agreement
- Class ambiguity

### What if my model is bad?

Confident learning is robust to model quality because:
- It uses cross-validation (no overfitting)
- It looks at disagreement patterns, not raw accuracy
- Errors are flagged based on confident disagreement

### Can I use my own classifier?

Yes:

```python
from sklearn.ensemble import GradientBoostingClassifier

detector = LabelErrorDetector(
    classifier=GradientBoostingClassifier(n_estimators=100)
)
```

## Next Steps

- [Auto-Fix Engine](/docs/guides/auto-fix) - Apply corrections automatically
- [Duplicates](/docs/concepts/duplicates) - Find redundant samples
- [API Reference](/docs/api/detectors) - Full detector API
