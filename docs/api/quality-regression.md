# Quality Regression Testing

Automated quality regression detection for CI/CD pipelines.

## Quick Example

```python
from clean import DatasetCleaner
from clean.quality_regression import QualityRegressionTester

# Create baseline
baseline_cleaner = DatasetCleaner(data=baseline_df, label_column="label")
baseline_report = baseline_cleaner.analyze()

# Test new data against baseline
current_cleaner = DatasetCleaner(data=current_df, label_column="label")
current_report = current_cleaner.analyze()

tester = QualityRegressionTester()
tester.set_baseline(baseline_report)

result = tester.test(current_report)

if not result.passed:
    print("Quality regression detected!")
    for warning in result.warnings:
        print(f"  - {warning}")
```

## API Reference

### RegressionTestConfig

Configuration for regression testing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quality_score_warning` | float | `0.05` | Warning threshold for quality drop |
| `quality_score_critical` | float | `0.10` | Critical threshold for quality drop |
| `label_error_rate_warning` | float | `0.02` | Warning threshold for error rate increase |
| `label_error_rate_critical` | float | `0.05` | Critical threshold for error rate increase |
| `duplicate_rate_warning` | float | `0.05` | Warning for duplicate rate increase |
| `duplicate_rate_critical` | float | `0.10` | Critical for duplicate rate increase |
| `outlier_rate_warning` | float | `0.05` | Warning for outlier rate increase |
| `outlier_rate_critical` | float | `0.10` | Critical for outlier rate increase |
| `fail_on_warning` | bool | `False` | Fail test on warnings |
| `fail_on_critical` | bool | `True` | Fail test on critical issues |
| `use_rolling_baseline` | bool | `False` | Use rolling window baseline |
| `rolling_window` | int | `5` | Rolling window size |

### QualitySnapshot

Point-in-time quality snapshot.

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique snapshot identifier |
| `timestamp` | datetime | Snapshot creation time |
| `dataset_name` | str | Dataset identifier |
| `n_samples` | int | Number of samples |
| `metrics` | dict[str, float] | Quality metrics |
| `metadata` | dict | Additional metadata |

#### `to_dict() -> dict`

Convert snapshot to dictionary.

### MetricThreshold

Custom metric threshold definition.

| Field | Type | Description |
|-------|------|-------------|
| `metric` | str | Metric name |
| `warning_threshold` | float | Warning level threshold |
| `critical_threshold` | float | Critical level threshold |
| `direction` | str | `"increase"` or `"decrease"` |

### QualityRegressionTester

Main regression tester class.

#### `__init__(baseline=None, config=None, store=None)`

Initialize with optional baseline and configuration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `baseline` | `QualitySnapshot \| QualityReport \| None` | Initial baseline |
| `config` | `RegressionTestConfig \| None` | Test configuration |
| `store` | `QualityHistoryStore \| None` | Historical storage |

#### `set_baseline(baseline: QualitySnapshot | QualityReport) -> None`

Set or update the baseline for comparison.

#### `add_threshold(threshold: MetricThreshold) -> None`

Add a custom metric threshold.

#### `test(current, dataset_name="default") -> QualityTestResult`

Test current quality against baseline.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current` | `QualitySnapshot \| QualityReport` | Current quality state |
| `dataset_name` | `str` | Dataset identifier |

### QualityTestResult

Regression test result.

| Field | Type | Description |
|-------|------|-------------|
| `passed` | bool | Whether test passed |
| `status` | str | `"pass"`, `"warning"`, or `"critical"` |
| `timestamp` | datetime | Test execution time |
| `baseline_metrics` | dict | Baseline metric values |
| `current_metrics` | dict | Current metric values |
| `differences` | dict | Metric differences |
| `warnings` | list[str] | Warning messages |
| `critical_issues` | list[str] | Critical issue messages |

## Example Workflows

### CI/CD Integration

```python
# In CI/CD pipeline
from clean import DatasetCleaner
from clean.quality_regression import QualityRegressionTester, RegressionTestConfig
import json

# Load baseline from artifact
with open("quality_baseline.json") as f:
    baseline_data = json.load(f)

config = RegressionTestConfig(
    fail_on_warning=False,
    fail_on_critical=True
)

tester = QualityRegressionTester(config=config)
# ... set baseline from loaded data

# Test current data
current_cleaner = DatasetCleaner(data=new_df, label_column="label")
current_report = current_cleaner.analyze()

result = tester.test(current_report)

if not result.passed:
    print("::error::Quality regression detected")
    for issue in result.critical_issues:
        print(f"::error::{issue}")
    exit(1)
```

### Custom Thresholds

```python
from clean.quality_regression import QualityRegressionTester, MetricThreshold

tester = QualityRegressionTester()
tester.set_baseline(baseline_report)

# Add custom metric threshold
tester.add_threshold(MetricThreshold(
    metric="custom_score",
    warning_threshold=0.1,
    critical_threshold=0.2,
    direction="decrease"
))

result = tester.test(current_report)
```

### Rolling Baseline

```python
from clean.quality_regression import QualityRegressionTester, RegressionTestConfig

config = RegressionTestConfig(
    use_rolling_baseline=True,
    rolling_window=5  # Compare against last 5 versions
)

tester = QualityRegressionTester(config=config)

# Add historical snapshots
for historical_report in historical_reports:
    tester.add_snapshot(historical_report)

# Test against rolling baseline
result = tester.test(current_report)
```

### GitHub Actions Example

```yaml
- name: Quality Regression Test
  run: |
    python -c "
    from clean import DatasetCleaner
    from clean.quality_regression import QualityRegressionTester
    import pandas as pd
    
    baseline = pd.read_csv('data/baseline.csv')
    current = pd.read_csv('data/current.csv')
    
    baseline_report = DatasetCleaner(baseline, 'label').analyze()
    current_report = DatasetCleaner(current, 'label').analyze()
    
    tester = QualityRegressionTester()
    tester.set_baseline(baseline_report)
    result = tester.test(current_report)
    
    assert result.passed, f'Regression: {result.critical_issues}'
    "
```
