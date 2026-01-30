# ADR-0007: Async Streaming for Large Dataset Processing

## Status

Accepted

## Context

The core `DatasetCleaner` loads entire datasets into memory, which works well for datasets up to ~10M rows. However, production ML pipelines often have:

- Training datasets with 100M+ rows
- Continuous data streams from logging systems
- Memory-constrained environments (notebooks, containers)
- Need for progress feedback during long-running analysis

Requirements:
1. Process datasets larger than available RAM
2. Provide incremental results during processing
3. Support both file-based and streaming sources
4. Maintain compatibility with core detector logic

Approaches considered:

1. **Dask/Spark**: Full distributed computing - heavyweight for single-machine use
2. **Memory-mapped files**: Complex and OS-dependent
3. **Chunked synchronous**: Simple but blocks the event loop
4. **Async generators**: Chosen approach - native Python, composable

## Decision

We implemented an **async streaming architecture** using Python's `async/await` and `AsyncIterator`.

```python
# streaming.py
class StreamingCleaner:
    def __init__(self, label_column: str = "label", chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self._chunk_results: list[ChunkResult] = []

    async def analyze_file(self, file_path: str) -> AsyncIterator[ChunkResult]:
        """Analyze a CSV file in streaming fashion."""
        chunk_id = 0
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            result = await self._analyze_chunk(chunk, chunk_id)
            self._chunk_results.append(result)
            yield result
            chunk_id += 1
            await asyncio.sleep(0)  # Yield control to event loop

    async def _analyze_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> ChunkResult:
        """Analyze a single chunk."""
        issues = {}
        if "outliers" in self.detectors:
            issues["outlier"] = self._detect_outliers_chunk(chunk)
        if "duplicates" in self.detectors:
            issues["duplicate"] = self._detect_duplicates_chunk(chunk)
        
        return ChunkResult(
            chunk_id=chunk_id,
            issues=issues,
            quality_score=self._calculate_chunk_score(issues, len(chunk)),
        )
```

Usage provides incremental feedback:

```python
async def process_large_dataset():
    cleaner = StreamingCleaner(label_column="label", chunk_size=50000)
    
    async for result in cleaner.analyze_file("100GB_dataset.csv"):
        print(f"Chunk {result.chunk_id}: {result.total_issues} issues found")
        # Could update progress bar, send to dashboard, etc.
    
    summary = cleaner.get_summary()
    print(f"Total: {summary.total_issues} issues in {summary.total_rows:,} rows")

asyncio.run(process_large_dataset())
```

A synchronous wrapper exists for simpler use cases:

```python
def stream_analyze(df: pd.DataFrame, chunk_size: int = 10000) -> Iterator[ChunkResult]:
    """Synchronous streaming analysis helper."""
    cleaner = StreamingCleaner(chunk_size=chunk_size)
    # Runs async code in a new event loop
    for result in _run_async_generator(cleaner.analyze_dataframe(df)):
        yield result
```

## Consequences

### Positive

- **Bounded memory**: Only one chunk in memory at a time
- **Progress visibility**: Users see results as they're computed
- **Cancellation**: Async allows graceful cancellation mid-analysis
- **Composability**: Async generators compose with other async code
- **Event loop integration**: Works in Jupyter, FastAPI, and async frameworks

### Negative

- **Async complexity**: Users unfamiliar with async may struggle
- **Cross-chunk issues**: Some detectors (duplicates across chunks) are harder
- **Aggregation needed**: Final summary requires collecting all chunk results
- **Two APIs**: Sync and async versions to maintain

### Neutral

- **Chunk size tuning**: Users must choose appropriate chunk sizes
- **Not distributed**: Single-machine only (Dask integration in ADR-0011 addresses this)

## Chunk-Level Detection Limitations

Some detections are inherently limited in streaming mode:

| Detector | Streaming Capability |
|----------|---------------------|
| Outliers | ✅ Per-chunk IQR works well |
| Duplicates | ⚠️ Only within-chunk (cross-chunk requires hash index) |
| Label errors | ❌ Requires full dataset for cross-validation |
| Imbalance | ⚠️ Running totals, final ratio at end |
| Bias | ⚠️ Running statistics, final metrics at end |

## Related Decisions

- ADR-0003 (Pandas Interface): Chunks are DataFrames
- ADR-0008 (FastAPI): API has `/analyze/streaming` endpoint
