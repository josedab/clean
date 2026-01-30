---
sidebar_position: 6
title: StreamingCleaner
---

# StreamingCleaner

Process large datasets in streaming mode.

```python
from clean.streaming import StreamingCleaner
```

## Overview

`StreamingCleaner` enables analysis of datasets that don't fit in memory by processing data in chunks and aggregating results.

## Constructor

```python
StreamingCleaner(
    label_column: Optional[str] = None,
    chunk_size: int = 10000,
    detectors: Optional[List[str]] = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_column` | str | None | Label column name |
| `chunk_size` | int | 10000 | Rows per chunk |
| `detectors` | list | None | Detectors to run |

## Methods

### analyze_file()

Analyze a CSV file in streaming mode.

```python
async def analyze_file(
    path: Union[str, Path],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> QualityReport
```

### analyze_chunks()

Analyze an iterator of DataFrames.

```python
async def analyze_chunks(
    chunks: Iterator[pd.DataFrame],
    total_chunks: Optional[int] = None,
) -> QualityReport
```

### stream()

Async generator yielding progress updates.

```python
async def stream(
    path: Union[str, Path],
) -> AsyncIterator[StreamingProgress]
```

#### Returns

`StreamingProgress` with:
- `chunks_processed`: int
- `total_chunks`: Optional[int]
- `issues_found`: int
- `current_score`: float
- `status`: "processing", "complete", "error"

## Example: Basic Streaming

```python
import asyncio
from clean.streaming import StreamingCleaner

async def main():
    cleaner = StreamingCleaner(
        label_column="label",
        chunk_size=50000,
    )
    
    report = await cleaner.analyze_file("large_dataset.csv")
    print(report.summary())

asyncio.run(main())
```

## Example: Progress Updates

```python
import asyncio
from clean.streaming import StreamingCleaner

async def main():
    cleaner = StreamingCleaner(
        label_column="label",
        chunk_size=10000,
    )
    
    async for progress in cleaner.stream("large_dataset.csv"):
        pct = (progress.chunks_processed / progress.total_chunks * 100
               if progress.total_chunks else 0)
        print(f"Progress: {pct:.1f}% - Issues: {progress.issues_found}")
        
        if progress.status == "complete":
            print(f"Final score: {progress.current_score}")

asyncio.run(main())
```

## Example: Custom Chunks

```python
import asyncio
import pandas as pd
from clean.streaming import StreamingCleaner

def read_in_chunks(path, chunk_size=10000):
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        yield chunk

async def main():
    cleaner = StreamingCleaner(label_column="label")
    
    chunks = read_in_chunks("data.csv", chunk_size=50000)
    report = await cleaner.analyze_chunks(chunks)
    
    print(report.summary())

asyncio.run(main())
```

## Example: With Callback

```python
import asyncio
from clean.streaming import StreamingCleaner

def on_progress(processed: int, total: int):
    print(f"Processed {processed}/{total} chunks")

async def main():
    cleaner = StreamingCleaner(
        label_column="label",
        chunk_size=25000,
    )
    
    report = await cleaner.analyze_file(
        "data.csv",
        progress_callback=on_progress,
    )
    
    print(report.summary())

asyncio.run(main())
```

## Synchronous Wrapper

For non-async contexts:

```python
from clean.streaming import StreamingCleaner

cleaner = StreamingCleaner(label_column="label")
report = cleaner.analyze_file_sync("data.csv")
print(report.summary())
```

## Memory Considerations

| Chunk Size | Memory Usage | Trade-off |
|------------|--------------|-----------|
| 1,000 | ~5 MB | More overhead |
| 10,000 | ~50 MB | Balanced |
| 100,000 | ~500 MB | Faster |

Choose based on:
- Available memory
- Row size (columns × data types)
- Processing time requirements

## Limitations

- Global duplicate detection uses sampling (may miss some)
- Label error detection accuracy may be lower than batch mode
- Outlier detection uses approximate methods

For best accuracy on critical data, use batch mode when possible:

```python
# Batch mode (if data fits)
from clean import DatasetCleaner
cleaner = DatasetCleaner(df, labels="label")
report = cleaner.analyze()

# Streaming mode (large data)
from clean.streaming import StreamingCleaner
cleaner = StreamingCleaner(label_column="label")
report = await cleaner.analyze_file("large.csv")
```
