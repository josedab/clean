# Benchmark Results

This directory contains benchmark results from Clean detection tests.

## Running Benchmarks

```bash
# From repository root
python -m benchmarks.run_benchmarks
```

## Benchmark Types

1. **Label Error Detection**: Tests detection of synthetic label errors
2. **Duplicate Detection**: Tests detection of injected duplicates  
3. **Outlier Detection**: Tests detection of injected outliers

## Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Runtime**: Time to run detection
