# DatasetCleaner

The `DatasetCleaner` class is the main interface for data quality analysis.

## Quick Example

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()
print(report.summary())
```

## API Reference

::: clean.core.cleaner.DatasetCleaner
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - from_csv
        - from_arrays
        - analyze
        - get_clean_data
        - get_review_queue
        - relabel
        - features
        - labels
        - info
