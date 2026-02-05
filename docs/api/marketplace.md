# Data Quality Marketplace

Multi-organization benchmark sharing and industry comparison.

## Quick Example

```python
from clean import DatasetCleaner
from clean.marketplace import QualityMarketplace, Domain

# Create marketplace connection
marketplace = QualityMarketplace(org_id="my-organization")

# Contribute a benchmark (anonymized)
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

benchmark = marketplace.contribute_benchmark(
    report=report,
    domain=Domain.FINANCE,
)

# Compare to industry
percentile = marketplace.get_percentile(
    report.overall_score,
    domain=Domain.FINANCE
)
print(f"Your data quality is in the {percentile.percentile:.0f}th percentile")
```

## API Reference

### Domain (Enum)

Industry domains for benchmark categorization.

| Value | Description |
|-------|-------------|
| `HEALTHCARE` | Healthcare and medical data |
| `FINANCE` | Financial services data |
| `RETAIL` | Retail and e-commerce data |
| `MANUFACTURING` | Manufacturing and IoT data |
| `TECHNOLOGY` | Tech industry data |
| `EDUCATION` | Educational data |
| `GOVERNMENT` | Government and public sector |
| `GENERAL` | General/cross-industry |

### DataType (Enum)

Data type categories.

| Value | Description |
|-------|-------------|
| `TABULAR` | Structured tabular data |
| `TEXT` | Text/NLP data |
| `IMAGE` | Image data |
| `TIME_SERIES` | Time series data |
| `MIXED` | Mixed data types |

### PrivacyLevel (Enum)

Benchmark privacy levels.

| Value | Description |
|-------|-------------|
| `PUBLIC` | Public benchmark |
| `ORGANIZATION` | Organization-only |
| `PRIVATE` | Private (not shared) |

### QualityMarketplace

Main marketplace class.

#### `__init__(store=None, org_id=None)`

Initialize marketplace connection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | `BenchmarkStore \| None` | Custom storage backend |
| `org_id` | `str \| None` | Organization identifier |

#### `contribute_benchmark(report=None, quality_score=None, n_samples=None, ...) -> AnonymizedBenchmark`

Contribute an anonymized benchmark.

| Parameter | Type | Description |
|-----------|------|-------------|
| `report` | `QualityReport \| None` | Quality report to contribute |
| `quality_score` | `float \| None` | Manual quality score |
| `n_samples` | `int \| None` | Sample count |
| `label_error_rate` | `float` | Label error rate |
| `duplicate_rate` | `float` | Duplicate rate |
| `outlier_rate` | `float` | Outlier rate |
| `domain` | `Domain` | Industry domain |
| `data_type` | `DataType` | Data type |
| `privacy_level` | `PrivacyLevel` | Sharing privacy level |
| `task_type` | `str \| None` | ML task type |

#### `get_percentile(score, domain=Domain.GENERAL, data_type=None) -> PercentileResult`

Get industry percentile for a quality score.

| Parameter | Type | Description |
|-----------|------|-------------|
| `score` | `float` | Quality score to compare |
| `domain` | `Domain` | Industry domain |
| `data_type` | `DataType \| None` | Data type filter |

#### `get_industry_benchmark(domain, data_type=None) -> IndustryBenchmark | None`

Get aggregated industry benchmark statistics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain` | `Domain` | Industry domain |
| `data_type` | `DataType \| None` | Data type filter |

#### `compare_to_industry(report, domain=Domain.GENERAL) -> dict[str, Any]`

Compare a quality report to industry benchmarks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `report` | `QualityReport` | Report to compare |
| `domain` | `Domain` | Industry domain |

#### `get_leaderboard(domain=None, top_n=10) -> list[dict[str, Any]]`

Get quality leaderboard for a domain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain` | `Domain \| None` | Domain filter |
| `top_n` | `int` | Number of entries |

### IndustryBenchmark

Aggregated industry benchmark dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `domain` | Domain | Industry domain |
| `data_type` | DataType | Data type |
| `n_contributions` | int | Number of contributions |
| `last_updated` | datetime | Last update time |
| `quality_score_mean` | float | Mean quality score |
| `quality_score_median` | float | Median quality score |
| `quality_score_std` | float | Standard deviation |
| `quality_score_p25` | float | 25th percentile |
| `quality_score_p75` | float | 75th percentile |
| `quality_score_p90` | float | 90th percentile |
| `label_error_rate_mean` | float | Mean label error rate |
| `label_error_rate_median` | float | Median label error rate |
| `duplicate_rate_mean` | float | Mean duplicate rate |
| `outlier_rate_mean` | float | Mean outlier rate |
| `percentile_thresholds` | dict | Score thresholds by percentile |

### PercentileResult

Percentile comparison result.

| Field | Type | Description |
|-------|------|-------------|
| `score` | float | Input score |
| `percentile` | float | Percentile rank (0-100) |
| `domain` | Domain | Comparison domain |
| `n_comparisons` | int | Number of benchmarks compared |

### Convenience Functions

#### `create_marketplace(store=None, org_id=None) -> QualityMarketplace`

Create a marketplace instance.

#### `get_industry_percentile(score, marketplace, domain=Domain.GENERAL) -> PercentileResult`

Quick percentile lookup.

## Example Workflows

### Contributing Benchmarks

```python
from clean import DatasetCleaner
from clean.marketplace import QualityMarketplace, Domain, DataType

marketplace = QualityMarketplace(org_id="acme-corp")

# Analyze and contribute
cleaner = DatasetCleaner(data=df, label_column="target")
report = cleaner.analyze()

benchmark = marketplace.contribute_benchmark(
    report=report,
    domain=Domain.FINANCE,
    data_type=DataType.TABULAR,
    task_type="classification"
)

print(f"Benchmark ID: {benchmark.benchmark_id}")
```

### Industry Comparison

```python
from clean.marketplace import QualityMarketplace, Domain

marketplace = QualityMarketplace()

# Get industry statistics
benchmark = marketplace.get_industry_benchmark(
    Domain.HEALTHCARE, 
    DataType.TABULAR
)

if benchmark:
    print(f"Healthcare Industry Benchmarks ({benchmark.n_contributions} organizations):")
    print(f"  Mean quality: {benchmark.quality_score_mean:.2f}")
    print(f"  Median quality: {benchmark.quality_score_median:.2f}")
    print(f"  Top 10%: >{benchmark.quality_score_p90:.2f}")
```

### Quality Scorecard

```python
from clean import DatasetCleaner
from clean.marketplace import QualityMarketplace, Domain

marketplace = QualityMarketplace()
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

comparison = marketplace.compare_to_industry(report, domain=Domain.TECHNOLOGY)

print("Quality Scorecard vs Industry:")
for metric, values in comparison.items():
    print(f"  {metric}:")
    print(f"    Your value: {values['your_value']:.2f}")
    print(f"    Industry avg: {values['industry_avg']:.2f}")
    print(f"    Percentile: {values['percentile']:.0f}th")
```

### Leaderboard Tracking

```python
from clean.marketplace import QualityMarketplace, Domain

marketplace = QualityMarketplace()

# Get top performers
leaderboard = marketplace.get_leaderboard(domain=Domain.RETAIL, top_n=10)

print("Retail Data Quality Leaderboard:")
for i, entry in enumerate(leaderboard, 1):
    print(f"  {i}. Score: {entry['quality_score']:.2f} "
          f"(Error rate: {entry['label_error_rate']:.1%})")
```

### Multi-Domain Analysis

```python
from clean.marketplace import QualityMarketplace, Domain

marketplace = QualityMarketplace()

# Compare across domains
domains = [Domain.HEALTHCARE, Domain.FINANCE, Domain.RETAIL, Domain.TECHNOLOGY]

print("Cross-Industry Quality Comparison:")
for domain in domains:
    benchmark = marketplace.get_industry_benchmark(domain)
    if benchmark:
        print(f"  {domain.value}: "
              f"mean={benchmark.quality_score_mean:.2f}, "
              f"n={benchmark.n_contributions}")
```
