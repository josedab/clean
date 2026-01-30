---
sidebar_position: 13
title: FAQ
---

# Frequently Asked Questions

## General

### What is Clean?

Clean is an AI-powered data quality platform that automatically detects and fixes issues in machine learning datasets. It finds label errors, duplicates, outliers, class imbalance, and bias.

### How is Clean different from cleanlab?

Clean uses cleanlab internally for label error detection, but extends it with:
- Duplicate detection (exact and semantic)
- Outlier detection
- Bias detection
- Auto-fix engine
- Streaming support
- LLM data quality tools
- REST API and CLI

See the [Comparison](/docs/comparison) page for details.

### What data types are supported?

- **Tabular**: pandas DataFrames, CSV files
- **Text**: Text columns, sentence/document datasets
- **Image**: Image folders organized by class
- **LLM**: Instruction-response pairs, RAG data

### What's the minimum dataset size?

Clean works best with at least 100 samples. Smaller datasets may not have enough signal for reliable detection.

---

## Installation

### How do I install Clean?

```bash
pip install clean-data-quality
```

For additional features:

```bash
pip install clean-data-quality[all]  # All features
pip install clean-data-quality[text]  # Text embeddings
pip install clean-data-quality[image]  # Image support
```

### What Python versions are supported?

Python 3.9, 3.10, 3.11, and 3.12.

### I get "No module named 'sentence_transformers'"

Install text dependencies:

```bash
pip install clean-data-quality[text]
```

### I get CUDA/GPU errors

Clean works on CPU by default. For GPU acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### How do I analyze a dataset?

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(df, labels="target")
report = cleaner.analyze()
print(report.summary())
```

### How do I fix detected issues?

```python
from clean.fix import FixEngine, FixConfig

config = FixConfig(
    remove_duplicates=True,
    relabel_errors=True,
)
engine = FixEngine(config)
result = engine.apply_fixes(df, report)
fixed_df = result.data
```

### How do I analyze a large file?

Use streaming mode:

```python
from clean.streaming import StreamingCleaner

cleaner = StreamingCleaner(label_column="label", chunk_size=50000)
report = await cleaner.analyze_file("large.csv")
```

### How do I run from command line?

```bash
clean analyze data.csv --label-column target
```

---

## Detection

### How accurate is label error detection?

In benchmarks, Clean detects approximately 95% of synthetic label errors with under 5% false positives. Real-world accuracy depends on data quality.

### Why are some correct labels flagged as errors?

Possible reasons:
- Ambiguous examples that could belong to multiple classes
- Outliers that don't fit typical patterns
- Model limitations

Review flagged labels and adjust `min_confidence`:

```python
# Only high-confidence errors
errors = report.label_errors(min_confidence=0.95)
```

### How does duplicate detection work?

Three methods:
1. **Hash**: Exact row matches
2. **Fuzzy**: Similarity threshold on features
3. **Semantic**: Embedding-based for text/images

### What outlier methods are available?

- `isolation_forest`: Good for high-dimensional data
- `lof`: Local Outlier Factor, for local anomalies
- `zscore`: Statistical, for simple distributions
- `ensemble`: Combines all methods

```python
from clean.detection import OutlierDetector

detector = OutlierDetector(method="ensemble", contamination=0.05)
```

---

## Performance

### How long does analysis take?

Rough estimates for 100K samples:

| Detector | Time |
|----------|------|
| Label errors | 30-60s |
| Duplicates | 5-15s |
| Outliers | 5-10s |
| Full analysis | 45-90s |

### How can I speed up analysis?

1. Run specific detectors:
   ```python
   report = cleaner.analyze(detectors=["duplicates", "outliers"])
   ```

2. Use sampling for exploration:
   ```python
   sample_df = df.sample(10000)
   ```

3. Reduce cross-validation folds:
   ```python
   from clean.detection import LabelErrorDetector
   detector = LabelErrorDetector(cv_folds=3)  # Default is 5
   ```

### How much memory does Clean use?

Approximately 5-10x the size of your DataFrame. For a 1GB CSV:
- Peak memory: 5-10 GB
- Use streaming mode for larger files

---

## Integration

### How do I use Clean in CI/CD?

```yaml
# .github/workflows/data-quality.yml
- run: |
    clean analyze data.csv -l target -f json -o report.json
    score=$(jq '.quality_score.overall' report.json)
    if (( $(echo "$score < 80" | bc -l) )); then
      exit 1
    fi
```

### Can I use Clean with MLflow/W&B?

Yes, log the report:

```python
import mlflow

report = cleaner.analyze()
mlflow.log_metric("data_quality_score", report.quality_score.overall)
mlflow.log_dict(report.to_dict(), "quality_report.json")
```

### Does Clean support distributed processing?

Not yet. For very large datasets, use:
- Streaming mode for sequential processing
- Sample-based analysis for exploration

---

## Troubleshooting

### "ValueError: Not enough samples"

Your dataset is too small. Minimum ~100 samples recommended.

### "MemoryError during analysis"

Dataset too large for memory. Use streaming:

```python
from clean.streaming import StreamingCleaner
cleaner = StreamingCleaner(chunk_size=10000)
```

### "No issues found" but I know there are problems

1. Check if labels are correct:
   ```python
   print(df["label"].value_counts())
   ```

2. Try different detector settings:
   ```python
   from clean.detection import LabelErrorDetector
   detector = LabelErrorDetector(threshold=0.3)  # More sensitive
   ```

3. Verify data loading:
   ```python
   print(cleaner.n_samples, cleaner.n_features)
   ```

### Analysis hangs or is very slow

1. Check dataset size:
   ```python
   print(len(df))
   ```

2. Use progress bar:
   ```python
   report = cleaner.analyze(show_progress=True)
   ```

3. Run simpler detectors first:
   ```python
   report = cleaner.analyze(detectors=["duplicates"])
   ```

---

## Contributing

### How do I report a bug?

Open an issue on GitHub with:
- Clean version (`clean --version`)
- Python version
- Minimal reproducible example
- Error message/traceback

### How do I request a feature?

Open an issue with:
- Use case description
- Proposed solution (if any)
- Priority for your workflow

### How do I contribute code?

See the [Contributing](/docs/contributing) guide.
