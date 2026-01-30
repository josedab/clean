# Streaming Analysis

Clean supports streaming analysis for large datasets that don't fit in memory.

## Overview

The streaming module allows you to analyze datasets in chunks, enabling quality
analysis on arbitrarily large files with controlled memory usage.

## Quick Start

```python
from clean import StreamingCleaner, stream_analyze
import asyncio

# Async streaming
async def analyze_large_file():
    cleaner = StreamingCleaner(label_column="label", chunk_size=10000)

    async for result in cleaner.analyze_file("large_dataset.csv"):
        print(f"Chunk {result.chunk_id}: {result.total_issues} issues")

    summary = cleaner.get_summary()
    print(f"Total issues: {summary.total_issues}")

asyncio.run(analyze_large_file())
```

## Sync Helper

For simpler use cases, use the synchronous helper:

```python
from clean import stream_analyze
import pandas as pd

df = pd.read_csv("large_file.csv")

for result in stream_analyze(df, label_column="label", chunk_size=5000):
    print(f"Chunk {result.chunk_id}: quality score {result.quality_score:.1f}")
```

## Configuration

### StreamingCleaner Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_column` | str | "label" | Name of the label column |
| `chunk_size` | int | 10000 | Number of rows per chunk |
| `detectors` | list | ["duplicates", "outliers", "imbalance"] | Detectors to run |

## ChunkResult

Each chunk returns a `ChunkResult` object:

```python
@dataclass
class ChunkResult:
    chunk_id: int          # Sequential chunk number
    start_row: int         # First row index
    end_row: int           # Last row index
    issues: dict           # Issues found by type
    quality_score: float   # Chunk quality score
    stats: dict            # Additional statistics
```

## Summary

Get an overall summary after processing:

```python
summary = cleaner.get_summary()

print(f"Total rows: {summary.total_rows}")
print(f"Total chunks: {summary.total_chunks}")
print(f"Total issues: {summary.total_issues}")
print(f"Average quality: {summary.average_quality_score:.1f}")
```

## Memory Considerations

- Chunk size determines peak memory usage
- Smaller chunks = lower memory, more overhead
- Recommended: 5,000 - 50,000 rows per chunk
- For very wide datasets, use smaller chunks

## Limitations

- Label error detection requires full dataset (skipped in streaming)
- Cross-chunk duplicates not detected
- Results are approximate per-chunk analysis
