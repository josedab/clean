# Quick Start

This guide will get you up and running with Clean in minutes.

## Installation

```bash
pip install clean-data-quality
```

For additional features:

```bash
# Text data support
pip install clean-data-quality[text]

# Interactive visualizations
pip install clean-data-quality[interactive]

# All features
pip install clean-data-quality[all]
```

## Basic Usage

```python
import pandas as pd
from clean import DatasetCleaner

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize the cleaner
cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    task='classification'
)

# Run analysis
report = cleaner.analyze()

# View summary
print(report.summary())
```

## Understanding the Report

The report provides:

1. **Overall Quality Score** (0-100): Higher is better
2. **Component Scores**:
   - Label Quality: How clean are your labels?
   - Duplicate Quality: How much redundancy?
   - Outlier Quality: How many anomalies?
   - Imbalance Quality: How balanced are classes?
   - Bias Quality: Any fairness concerns?

3. **Issue Details**: Specific problematic samples

## Getting Specific Issues

```python
# Get label errors
label_errors = report.label_errors()
print(label_errors.head())

# Get duplicates
duplicates = report.duplicates()

# Get outliers
outliers = report.outliers()
```

## Cleaning Your Data

```python
# Get cleaned dataset
clean_df = cleaner.get_clean_data(
    remove_duplicates=True,
    remove_outliers='conservative'
)

# Or get a review queue for manual inspection
queue = cleaner.get_review_queue(max_items=100)
```

## Exporting Results

```python
# Save as JSON
report.save_json('quality_report.json')

# Save as HTML
report.save_html('quality_report.html')
```

## Next Steps

- [Label Error Detection Deep Dive](examples/label_errors.md)
- [Working with Text Data](examples/text_data.md)
- [API Reference](api/cleaner.md)
