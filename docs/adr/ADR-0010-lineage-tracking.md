# ADR-0010: Lineage Tracking with Append-Only Audit Log

## Status

Accepted

## Context

Data quality analysis in production environments requires:

1. **Reproducibility**: "What issues were found last Tuesday?"
2. **Audit compliance**: "Who reviewed sample #42 and what did they decide?"
3. **Progress tracking**: "How many of the 500 flagged items have been reviewed?"
4. **Historical comparison**: "Is data quality improving over time?"

Regulatory frameworks (EU AI Act, NIST AI RMF) increasingly require documentation of data quality processes for ML systems.

Requirements:
- Track analysis runs with parameters and results
- Record human review decisions
- Query history for specific samples
- Support both file and database backends
- Immutable records (append-only)

Approaches considered:

1. **Database (PostgreSQL)**: Full-featured but heavy dependency
2. **SQLite**: Good balance but requires schema management
3. **JSON files**: Simple, portable, human-readable - chosen for MVP
4. **Event sourcing**: Powerful but complex

## Decision

We implemented a **LineageTracker** with append-only JSON storage and immutable record types.

```python
# lineage.py
@dataclass
class AnalysisRun:
    """Record of a single analysis run."""
    run_id: str
    timestamp: str
    project_name: str
    n_samples: int
    n_label_errors: int
    n_duplicates: int
    n_outliers: int
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ReviewRecord:
    """Record of a review decision."""
    run_id: str
    sample_id: int | tuple[int, int]  # Single index or pair for duplicates
    issue_type: str
    decision: ReviewDecision  # "keep" | "remove" | "relabel" | "skip" | "defer"
    reviewer: str
    timestamp: str
    notes: str = ""
    old_value: Any = None
    new_value: Any = None

ReviewDecision = Literal["keep", "remove", "relabel", "skip", "defer"]
```

The `LineageTracker` manages persistence:

```python
class LineageTracker:
    def __init__(self, project_name: str, storage_path: Path = None):
        self.project_name = project_name
        self.storage_path = storage_path or Path.home() / ".clean" / "lineage"
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def log_analysis(self, report: QualityReport, metadata: dict = None) -> str:
        """Log an analysis run, returns run_id."""
        run = AnalysisRun(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            project_name=self.project_name,
            n_samples=report.info.n_samples,
            n_label_errors=len(report.label_errors()),
            # ...
        )
        self._append_run(run)
        return run.run_id
    
    def log_review(self, run_id: str, sample_id: int, decision: ReviewDecision,
                   reviewer: str, notes: str = "", new_value: Any = None):
        """Log a review decision."""
        record = ReviewRecord(
            run_id=run_id,
            sample_id=sample_id,
            decision=decision,
            reviewer=reviewer,
            timestamp=datetime.now().isoformat(),
            notes=notes,
            new_value=new_value,
        )
        self._append_review(record)
    
    def get_sample_history(self, sample_id: int) -> SampleHistory:
        """Get complete history for a sample across all runs."""
        reviews = [r for r in self._load_reviews() if r.sample_id == sample_id]
        detections = [...]  # Find all runs where this sample was flagged
        return SampleHistory(sample_id=sample_id, reviews=reviews, ...)
```

Storage is simple JSON files:

```
~/.clean/lineage/
├── my_project/
│   ├── runs.jsonl       # Append-only analysis runs
│   ├── reviews.jsonl    # Append-only review decisions
│   └── metadata.json    # Project configuration
```

## Consequences

### Positive

- **Simple storage**: JSON files are portable and human-readable
- **Append-only**: Immutable history, no accidental deletions
- **No database required**: Works out of the box, no setup
- **Query flexibility**: Load into Pandas for complex analysis
- **Compliance-ready**: Timestamped, attributed decisions

### Negative

- **Limited query performance**: Full file scan for queries (OK for <100K records)
- **No concurrent writes**: File locking needed for multi-process
- **No relationships**: Flat structure, joins done in application
- **Storage growth**: No automatic archival or cleanup

### Neutral

- **JSONL format**: One JSON object per line, easy to append
- **Local storage default**: Could add cloud backends later
- **No encryption**: Sensitive data should use encrypted storage

## Integration with API

The REST API integrates lineage tracking:

```python
@app.post("/analyze")
async def analyze_dataset(file: UploadFile, label_column: str):
    report = cleaner.analyze()
    
    # Automatically log to lineage
    tracker = get_lineage_tracker()
    tracker.log_analysis(report, metadata={"filename": file.filename})
    
    return JSONResponse(content={...})

@app.post("/review")
async def submit_review(decision: ReviewDecision):
    tracker.log_review(
        run_id="latest",
        sample_id=decision.sample_index,
        decision=decision.decision,
        reviewer="api_user",
    )
```

## Related Decisions

- ADR-0009 (Fix Strategy): Fixes are logged as review decisions
- ADR-0008 (FastAPI): `/lineage/runs` endpoint exposes history
