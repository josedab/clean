# Quality Reports

Clean generates comprehensive quality reports that summarize all detected issues.

## Getting a Report

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()
```

## Report Summary

```python
print(report.summary())
```

**Output:**
```
================================================================================
                           DATA QUALITY REPORT
================================================================================

Dataset Information:
  Samples: 10,000
  Features: 25
  Task: classification
  Classes: 5

Quality Scores:
  Overall:     0.82
  Label:       0.95 (347 errors, 3.5%)
  Duplicate:   0.93 (234 pairs, 4.7%)
  Outlier:     0.98 (156 outliers, 1.6%)
  Balance:     0.65 (15:1 ratio)

Issues Found:
  🏷️  Label Errors:    347 (3.5%) - HIGH PRIORITY
  🔍  Duplicates:      234 pairs (4.7%)
  📊  Outliers:        156 (1.6%)
  ⚖️  Class Imbalance: Ratio 15:1

Recommendations:
  1. Review label errors with confidence > 0.9 (42 samples)
  2. Remove exact duplicates (12 pairs)
  3. Investigate outliers in features: income, age
  4. Consider resampling for class imbalance

================================================================================
```

## Quality Scores

The report includes a `QualityScore` object:

```python
score = report.quality_score

print(f"Overall: {score.overall:.2f}")
print(f"Label quality: {score.label:.2f}")
print(f"Duplicate quality: {score.duplicate:.2f}")
print(f"Outlier quality: {score.outlier:.2f}")
print(f"Balance quality: {score.balance:.2f}")
```

### Score Interpretation

| Score | Quality | Action |
|-------|---------|--------|
| 0.9+ | Excellent | Ready to use |
| 0.7-0.9 | Good | Minor cleanup recommended |
| 0.5-0.7 | Fair | Significant issues present |
| < 0.5 | Poor | Major cleanup required |

## Accessing Issues

### Label Errors

```python
label_errors = report.label_errors()
# Returns DataFrame with: index, given_label, predicted_label, confidence
```

### Duplicates

```python
duplicates = report.duplicates()
# Returns DataFrame with: index1, index2, similarity, is_exact
```

### Outliers

```python
outliers = report.outliers()
# Returns DataFrame with: index, method, score
```

### All Issues

```python
all_issues = report.all_issues()
# Combined DataFrame with all issues
```

## Export Options

### JSON

```python
report.save_json('report.json')

# Or get as string
json_str = report.to_json()
```

**JSON structure:**
```json
{
  "quality_score": {
    "overall": 0.82,
    "label": 0.95,
    "duplicate": 0.93,
    "outlier": 0.98,
    "balance": 0.65
  },
  "dataset_info": {
    "n_samples": 10000,
    "n_features": 25,
    "task": "classification",
    "n_classes": 5
  },
  "label_errors": [...],
  "duplicates": [...],
  "outliers": [...]
}
```

### HTML

```python
report.save_html('report.html')

# Or get as string
html_str = report.to_html()
```

The HTML report includes:
- Interactive tables
- Charts and visualizations
- Collapsible sections
- Print-friendly styling

### CSV

Export issues to CSV for external tools:

```python
report.label_errors().to_csv('label_errors.csv', index=False)
report.duplicates().to_csv('duplicates.csv', index=False)
report.outliers().to_csv('outliers.csv', index=False)
```

## Report Metadata

```python
info = report.dataset_info

print(f"Samples: {info.n_samples}")
print(f"Features: {info.n_features}")
print(f"Task: {info.task}")
print(f"Classes: {info.n_classes}")
print(f"Class names: {info.class_names}")
```

## Class Distribution

```python
dist = report.class_distribution

print(f"Counts: {dist.counts}")
print(f"Imbalance ratio: {dist.imbalance_ratio:.2f}")
print(f"Majority class: {dist.majority_class}")
print(f"Minority class: {dist.minority_class}")
```

## Programmatic Analysis

```python
# Check if dataset needs cleaning
if report.quality_score.overall < 0.7:
    print("Dataset needs significant cleaning")

# Count high-priority issues
high_confidence_errors = len(
    report.label_errors()[report.label_errors()['confidence'] > 0.9]
)
print(f"High-confidence label errors: {high_confidence_errors}")

# Check specific thresholds
if report.class_distribution.imbalance_ratio > 10:
    print("Severe class imbalance detected")
```

## Comparing Reports

Track quality improvements over time:

```python
import json

# Save baseline
report1 = cleaner.analyze()
report1.save_json('baseline.json')

# ... make improvements ...

# Compare
report2 = cleaner.analyze()

print(f"Overall improvement: {report2.quality_score.overall - report1.quality_score.overall:+.2f}")
print(f"Label errors: {len(report1.label_errors())} → {len(report2.label_errors())}")
```

## Custom Reports

Extend reports with your own analysis:

```python
# Add custom fields
report_dict = json.loads(report.to_json())
report_dict['custom_metrics'] = {
    'reviewed_samples': 150,
    'corrected_labels': 42,
    'review_date': '2024-01-15'
}

with open('custom_report.json', 'w') as f:
    json.dump(report_dict, f, indent=2)
```
