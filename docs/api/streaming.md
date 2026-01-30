# Streaming API

API reference for Clean's streaming analysis capabilities.

## StreamingCleaner

The main class for streaming analysis.

```python
from clean import StreamingCleaner

cleaner = StreamingCleaner(
    label_column="label",
    chunk_size=10000,
    detectors=["duplicates", "outliers"],
)
```

### Constructor

```python
StreamingCleaner(
    label_column: str = "label",
    chunk_size: int = 10000,
    detectors: list[str] | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_column` | str | "label" | Label column name |
| `chunk_size` | int | 10000 | Rows per chunk |
| `detectors` | list | ["duplicates", "outliers", "imbalance"] | Active detectors |

### Methods

#### analyze_file

Analyze a CSV file in streaming fashion.

```python
async for result in cleaner.analyze_file("data.csv"):
    print(result.chunk_id, result.total_issues)
```

#### analyze_dataframe

Analyze a DataFrame in streaming fashion.

```python
async for result in cleaner.analyze_dataframe(df):
    print(result.chunk_id, result.quality_score)
```

#### get_summary

Get summary after processing all chunks.

```python
summary = cleaner.get_summary()
```

#### reset

Reset state for new analysis.

```python
cleaner.reset()
```

## ChunkResult

Result from processing a single chunk.

```python
@dataclass
class ChunkResult:
    chunk_id: int                      # Sequential chunk number
    start_row: int                     # First row index
    end_row: int                       # Last row index (exclusive)
    issues: dict[str, list[int]]       # Issues by type -> affected indices
    quality_score: float               # Chunk quality (0-100)
    stats: dict[str, Any]              # Additional statistics
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_issues` | int | Sum of all issue counts |

### Example

```python
result = ChunkResult(
    chunk_id=0,
    start_row=0,
    end_row=1000,
    issues={"outlier": [5, 42, 99], "duplicate": [100, 101]},
    quality_score=99.5,
    stats={"rows": 1000, "unique_labels": 3}
)

print(result.total_issues)  # 5
```

## StreamingSummary

Summary of streaming analysis across all chunks.

```python
@dataclass
class StreamingSummary:
    total_rows: int                    # Total rows processed
    total_chunks: int                  # Number of chunks
    total_issues: int                  # Total issues found
    issue_breakdown: dict[str, int]    # Issues by type
    average_quality_score: float       # Mean quality score
    processing_time_seconds: float     # Total time
```

### Methods

#### to_dict

Convert summary to dictionary.

```python
summary.to_dict()
# {
#     "total_rows": 100000,
#     "total_chunks": 10,
#     "total_issues": 500,
#     ...
# }
```

## stream_analyze

Synchronous helper function for simpler use cases.

```python
from clean import stream_analyze

for result in stream_analyze(df, label_column="label", chunk_size=5000):
    print(f"Chunk {result.chunk_id}: {result.quality_score:.1f}")
```

### Signature

```python
def stream_analyze(
    df: pd.DataFrame,
    label_column: str = "label",
    chunk_size: int = 10000,
) -> Iterator[ChunkResult]:
    ...
```

## Full Example

```python
import asyncio
from clean import StreamingCleaner

async def analyze_large_dataset(filepath: str):
    cleaner = StreamingCleaner(
        label_column="target",
        chunk_size=50000,
        detectors=["duplicates", "outliers"],
    )

    chunk_count = 0
    async for result in cleaner.analyze_file(filepath):
        chunk_count += 1
        print(f"[Chunk {result.chunk_id}] "
              f"Rows {result.start_row}-{result.end_row}: "
              f"{result.total_issues} issues, "
              f"quality: {result.quality_score:.1f}")

    summary = cleaner.get_summary()
    print(f"\n=== Summary ===")
    print(f"Processed {summary.total_rows:,} rows in {summary.total_chunks} chunks")
    print(f"Found {summary.total_issues:,} total issues")
    print(f"Average quality: {summary.average_quality_score:.1f}")

    return summary

# Run
summary = asyncio.run(analyze_large_dataset("big_data.csv"))
```
