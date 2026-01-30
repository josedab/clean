---
sidebar_position: 2
title: Streaming Analysis
---

# Streaming Analysis

Process datasets that don't fit in memory using chunk-by-chunk analysis.

## When to Use Streaming

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100K rows | Standard `DatasetCleaner` |
| 100K - 1M rows | Consider streaming |
| > 1M rows | Definitely use streaming |

## Quick Start

```python
from clean import StreamingCleaner
import asyncio

async def analyze():
    cleaner = StreamingCleaner(
        label_column="label",
        chunk_size=50000,  # 50K rows per chunk
    )
    
    async for result in cleaner.analyze_file("large_dataset.csv"):
        print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
    
    summary = cleaner.get_summary()
    print(f"Total: {summary.total_issues} issues in {summary.total_rows:,} rows")

asyncio.run(analyze())
```

## Sync Alternative

For simpler use cases:

```python
from clean import stream_analyze
import pandas as pd

df = pd.read_csv("data.csv")

for result in stream_analyze(df, label_column="label", chunk_size=10000):
    print(f"Chunk {result.chunk_id}: quality {result.quality_score:.1f}")
```

## Configuration

### StreamingCleaner Options

```python
cleaner = StreamingCleaner(
    label_column="label",       # Label column name
    chunk_size=50000,           # Rows per chunk
    detectors=["duplicates", "outliers"],  # Detectors to run
)
```

### Available Detectors

| Detector | Streaming Support | Notes |
|----------|-------------------|-------|
| `duplicates` | ✅ Within-chunk | Cross-chunk duplicates not detected |
| `outliers` | ✅ Within-chunk | Uses IQR method |
| `imbalance` | ✅ Aggregated | Final summary aggregates all chunks |
| `label_errors` | ❌ | Requires full dataset for CV |
| `bias` | ❌ | Requires full dataset |

### Memory Management

```python
# Lower chunk size = lower memory
cleaner = StreamingCleaner(chunk_size=10000)  # ~100MB for 1000-column data

# Estimate memory
rows_per_chunk = 10000
columns = 100
bytes_per_cell = 8  # float64
memory_mb = (rows_per_chunk * columns * bytes_per_cell) / 1024 / 1024
print(f"Estimated memory per chunk: {memory_mb:.1f} MB")
```

## ChunkResult

Each chunk returns a `ChunkResult`:

```python
@dataclass
class ChunkResult:
    chunk_id: int          # Sequential chunk number
    start_row: int         # First row index
    end_row: int           # Last row index (exclusive)
    issues: dict           # Issues by type
    quality_score: float   # Chunk quality (0-100)
    stats: dict            # Additional statistics
```

Usage:

```python
async for result in cleaner.analyze_file("data.csv"):
    print(f"Chunk {result.chunk_id}")
    print(f"  Rows: {result.start_row} - {result.end_row}")
    print(f"  Quality: {result.quality_score:.1f}")
    print(f"  Issues: {result.total_issues}")
    
    # Issue breakdown
    for issue_type, indices in result.issues.items():
        print(f"    {issue_type}: {len(indices)}")
```

## StreamingSummary

After processing all chunks:

```python
summary = cleaner.get_summary()

print(f"Total rows: {summary.total_rows:,}")
print(f"Total chunks: {summary.total_chunks}")
print(f"Total issues: {summary.total_issues:,}")
print(f"Average quality: {summary.average_quality_score:.1f}")

# Issue breakdown
for issue_type, count in summary.issue_breakdown.items():
    print(f"  {issue_type}: {count:,}")
```

## Processing Large Files

### From File

```python
async for result in cleaner.analyze_file("data.csv"):
    # Process each chunk
    pass
```

Supports:
- CSV files (automatic chunked reading)
- Gzipped CSV (`.csv.gz`)

### From DataFrame

```python
df = pd.read_parquet("data.parquet")

async for result in cleaner.analyze_dataframe(df):
    # Process each chunk
    pass
```

### From Generator

For custom data sources:

```python
def data_generator():
    for i in range(100):
        chunk = fetch_data_chunk(offset=i*10000, limit=10000)
        yield chunk

# Process with stream_analyze
for chunk_df in data_generator():
    for result in stream_analyze(chunk_df, chunk_size=10000):
        process(result)
```

## Parallel Processing

Process chunks in parallel for faster analysis:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def analyze_chunk(chunk_df, chunk_id):
    cleaner = StreamingCleaner(label_column="label", chunk_size=len(chunk_df))
    async for result in cleaner.analyze_dataframe(chunk_df):
        return result

async def parallel_analyze(df, n_workers=4):
    chunks = [df.iloc[i:i+50000] for i in range(0, len(df), 50000)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            asyncio.get_event_loop().run_in_executor(
                executor, analyze_chunk, chunk, i
            )
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*futures)
    
    return results
```

## Progress Tracking

### With tqdm

```python
from tqdm import tqdm

cleaner = StreamingCleaner(label_column="label", chunk_size=50000)

# Estimate total chunks
total_rows = 1_000_000
total_chunks = total_rows // 50000

with tqdm(total=total_chunks, desc="Analyzing") as pbar:
    async for result in cleaner.analyze_file("data.csv"):
        pbar.update(1)
        pbar.set_postfix(issues=result.total_issues)
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

async for result in cleaner.analyze_file("data.csv"):
    logging.info(f"Processed chunk {result.chunk_id}: {result.total_issues} issues")
```

## Saving Results

### Incremental Save

```python
import json

results = []
async for result in cleaner.analyze_file("data.csv"):
    results.append({
        "chunk_id": result.chunk_id,
        "issues": result.issues,
        "quality": result.quality_score,
    })
    
    # Save incrementally
    with open("results.json", "w") as f:
        json.dump(results, f)
```

### To DataFrame

```python
import pandas as pd

all_issues = []
async for result in cleaner.analyze_file("data.csv"):
    for issue_type, indices in result.issues.items():
        for idx in indices:
            all_issues.append({
                "chunk_id": result.chunk_id,
                "index": idx,
                "issue_type": issue_type,
            })

issues_df = pd.DataFrame(all_issues)
issues_df.to_csv("all_issues.csv", index=False)
```

## Limitations

1. **No cross-chunk duplicates**: Duplicates spanning chunks not detected
2. **No label errors**: Confident learning requires full dataset
3. **Approximate outliers**: Per-chunk statistics may differ from global
4. **No bias detection**: Requires all data for fairness metrics

### Workaround for Cross-Chunk Duplicates

```python
# After streaming, sample for cross-chunk duplicates
sample_size = 100000
sample = df.sample(n=min(sample_size, len(df)))

from clean import DatasetCleaner
mini_cleaner = DatasetCleaner(data=sample, label_column="label")
sample_report = mini_cleaner.analyze()

# Extrapolate duplicate rate
dup_rate = len(sample_report.duplicates()) / len(sample)
estimated_dups = int(dup_rate * len(df))
print(f"Estimated total duplicates: {estimated_dups:,}")
```

## Next Steps

- [REST API](/docs/guides/rest-api) - Streaming via HTTP
- [CLI](/docs/guides/cli) - Streaming from command line
- [API Reference](/docs/api/streaming) - StreamingCleaner API
