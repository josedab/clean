# QualityReport

The `QualityReport` class contains all analysis results.

## Quick Example

```python
report = cleaner.analyze()

# Summary
print(report.summary())

# Access issues
label_errors = report.label_errors()
duplicates = report.duplicates()
outliers = report.outliers()

# Export
report.save_json('report.json')
report.save_html('report.html')
```

## API Reference

::: clean.core.report.QualityReport
    options:
      show_root_heading: true
      show_source: false

## Related Types

### QualityScore

::: clean.core.types.QualityScore
    options:
      show_root_heading: true
      show_source: false

### DatasetInfo

::: clean.core.types.DatasetInfo
    options:
      show_root_heading: true
      show_source: false

### ClassDistribution

::: clean.core.types.ClassDistribution
    options:
      show_root_heading: true
      show_source: false

### Issue Types

::: clean.core.types.LabelError
    options:
      show_root_heading: true
      show_source: false

::: clean.core.types.DuplicatePair
    options:
      show_root_heading: true
      show_source: false

::: clean.core.types.Outlier
    options:
      show_root_heading: true
      show_source: false

::: clean.core.types.BiasIssue
    options:
      show_root_heading: true
      show_source: false
