---
sidebar_position: 2
title: Getting Started
---

# Getting Started

Get Clean installed and run your first data quality analysis in under 5 minutes.

## Installation

```bash
pip install clean-data-quality
```

### Optional Dependencies

```bash
# Text data (sentence embeddings for duplicate detection)
pip install clean-data-quality[text]

# Image data (CLIP embeddings)
pip install clean-data-quality[image]

# Interactive visualizations (plotly, ipywidgets)
pip install clean-data-quality[interactive]

# REST API server (FastAPI)
pip install clean-data-quality[api]

# Everything
pip install clean-data-quality[all]
```

## Quick Start

### 1. Import and Initialize

```python
import pandas as pd
from clean import DatasetCleaner

# Load your data
df = pd.read_csv('training_data.csv')

# Initialize cleaner
cleaner = DatasetCleaner(
    data=df,
    label_column='label'  # Name of your target column
)
```

### 2. Run Analysis

```python
# Analyze all quality dimensions
report = cleaner.analyze()

# View summary
print(report.summary())
```

Output:
```
Data Quality Report
==================
Samples analyzed: 10,000
Quality Score: 82.5/100

Issues Found:
  - Label errors: 347 (3.5%) - HIGH PRIORITY
  - Near-duplicates: 234 pairs (4.7%)
  - Outliers: 156 (1.6%)
  - Class imbalance: 15:1 ratio

Recommendations:
  1. Review and correct the 50 highest-confidence label errors
  2. Remove duplicate samples to prevent data leakage
  3. Consider SMOTE or class weighting for imbalance
```

### 3. Explore Issues

```python
# Get label errors with suggested corrections
label_errors = report.label_errors()
print(label_errors.head(10))
```

| index | given_label | predicted_label | confidence |
|-------|-------------|-----------------|------------|
| 42 | cat | dog | 0.94 |
| 187 | cat | bird | 0.89 |
| 523 | dog | cat | 0.87 |

```python
# Get duplicate pairs
duplicates = report.duplicates()

# Get outlier indices
outliers = report.outliers()

# Get class distribution
print(report.class_distribution)
```

### 4. Apply Fixes

```python
from clean import FixEngine, FixConfig

# Configure fix strategy
config = FixConfig(
    label_error_threshold=0.9,  # Only fix high-confidence errors
    auto_relabel=False,         # Suggest, don't auto-apply
)

# Create fix engine
features = df.drop(columns=['label'])
labels = df['label'].values

engine = FixEngine(
    report=report,
    features=features,
    labels=labels,
    config=config
)

# Get suggestions
fixes = engine.suggest_fixes()
print(f"Found {len(fixes)} suggested fixes")

# Apply fixes
result = engine.apply_fixes(fixes)
print(result.summary())
```

### 5. Export Clean Data

```python
# Get cleaned DataFrame
clean_df = cleaner.get_clean_data(
    remove_duplicates=True,
    remove_outliers='conservative'  # or 'aggressive'
)

clean_df.to_csv('clean_training_data.csv', index=False)
```

## Command-Line Interface

Clean also works from the terminal:

```bash
# Analyze a dataset
clean analyze data.csv --label-column target

# Export report to JSON
clean analyze data.csv -l target -o report.json -f json

# Apply fixes
clean fix data.csv -o cleaned.csv --strategy conservative

# Get dataset info
clean info data.csv
```

## What's Next?

Now that you've run your first analysis:

1. **[Core Concepts](/docs/concepts/overview)** - Understand how each detector works
2. **[LLM Data Quality](/docs/guides/llm-data)** - Clean instruction-tuning datasets
3. **[Streaming](/docs/guides/streaming)** - Handle large datasets
4. **[API Reference](/docs/api/cleaner)** - Full DatasetCleaner API

## Common Issues

### "No module named 'cleanlab'"

Clean depends on cleanlab for label error detection:

```bash
pip install cleanlab
```

### "Import error for sentence_transformers"

Text embedding features require the text extra:

```bash
pip install clean-data-quality[text]
```

### Slow analysis on large datasets

Use streaming mode for datasets over 100K rows:

```python
from clean import StreamingCleaner

cleaner = StreamingCleaner(label_column='label', chunk_size=10000)
for result in cleaner.analyze_file('large_data.csv'):
    print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
```
