"""FastAPI REST API for Clean data quality platform.

This module provides HTTP endpoints for remote data quality analysis,
enabling dashboard integration and web-based access to Clean features.

Usage:
    # Run the server
    uvicorn clean.api:app --host 0.0.0.0 --port 8000

    # Or programmatically
    from clean.api import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from clean.exceptions import DependencyError

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise DependencyError("fastapi", "api", feature="REST API") from e

import pandas as pd

from clean import DatasetCleaner, FixConfig, FixEngine, FixStrategy
from clean.lineage import LineageTracker
from clean.llm import LLMDataCleaner
from clean.streaming import StreamingCleaner

# Create FastAPI app
app = FastAPI(
    title="Clean Data Quality API",
    description="AI-powered data quality analysis for ML datasets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lineage tracker
_lineage_tracker: LineageTracker | None = None


def get_lineage_tracker() -> LineageTracker:
    """Get or create the global lineage tracker."""
    global _lineage_tracker
    if _lineage_tracker is None:
        storage_path = Path(tempfile.gettempdir()) / "clean_lineage"
        _lineage_tracker = LineageTracker(project_name="api", storage_path=storage_path)
    return _lineage_tracker


# --- Request/Response Models ---


class AnalysisRequest(BaseModel):
    """Request model for dataset analysis."""

    label_column: str = Field(default="label", description="Name of the label column")
    task_type: str | None = Field(
        default=None, description="Task type: classification, regression, etc."
    )
    detectors: list[str] | None = Field(
        default=None,
        description="Detectors to run: label_errors, duplicates, outliers, imbalance, bias",
    )


class FixRequest(BaseModel):
    """Request model for applying fixes."""

    strategy: str = Field(
        default="conservative", description="Fix strategy: conservative, aggressive, custom"
    )
    remove_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    remove_outliers: bool = Field(default=False, description="Remove outlier rows")
    relabel_errors: bool = Field(default=False, description="Auto-relabel detected errors")
    min_confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum confidence for fixes"
    )


class LLMAnalysisRequest(BaseModel):
    """Request model for LLM data analysis."""

    instruction_column: str = Field(
        default="instruction", description="Column containing instructions/prompts"
    )
    response_column: str = Field(
        default="response", description="Column containing model responses"
    )
    mode: str = Field(
        default="instruction", description="Mode: instruction or rag"
    )
    min_response_length: int = Field(
        default=10, description="Minimum acceptable response length"
    )


class ReviewDecision(BaseModel):
    """Model for review decisions."""

    sample_index: int = Field(..., description="Index of the sample")
    decision: str = Field(..., description="Decision: correct, incorrect, skip")
    new_label: str | None = Field(default=None, description="New label if correcting")
    notes: str | None = Field(default=None, description="Optional notes")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""

    dataset_id: str
    total_samples: int
    quality_score: float
    issues: dict[str, Any]
    summary: str


# --- Endpoints ---


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Health check endpoint."""
    from clean import __version__

    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return await root()


@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    label_column: str = Query(default="label"),
    task_type: str | None = Query(default=None),
) -> JSONResponse:
    """Analyze a CSV dataset for quality issues.

    Upload a CSV file and receive a quality analysis report.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if label_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Label column '{label_column}' not found. Available: {list(df.columns)}",
        )

    try:
        cleaner = DatasetCleaner(data=df, label_column=label_column)
        report = cleaner.analyze()

        # Log to lineage tracker
        tracker = get_lineage_tracker()
        tracker.log_analysis(
            report=report,
            metadata={"label_column": label_column, "task_type": task_type, "filename": file.filename},
        )

        return JSONResponse(
            content={
                "dataset_name": file.filename,
                "total_samples": len(df),
                "quality_score": report.quality_score.overall,
                "issues": {
                    "label_errors": len(report.label_errors()),
                    "duplicates": len(report.duplicates()),
                    "outliers": len(report.outliers()),
                },
                "summary": report.summary(),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/analyze/streaming")
async def analyze_streaming(
    file: UploadFile = File(...),
    label_column: str = Query(default="label"),
    chunk_size: int = Query(default=10000, ge=100, le=1000000),
) -> JSONResponse:
    """Analyze a large CSV dataset using streaming.

    Processes the file in chunks for memory efficiency.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Save to temp file for streaming
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        cleaner = StreamingCleaner(label_column=label_column, chunk_size=chunk_size)

        chunk_results = []
        async for result in cleaner.analyze_file(tmp_path):
            chunk_results.append({
                "chunk_id": result.chunk_id,
                "rows": result.end_row - result.start_row,
                "issues": result.total_issues,
                "quality_score": result.quality_score,
            })

        summary = cleaner.get_summary()

        # Clean up temp file
        Path(tmp_path).unlink()

        return JSONResponse(
            content={
                "dataset_name": file.filename,
                "streaming_summary": summary.to_dict(),
                "chunk_results": chunk_results,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming analysis failed: {e}")


@app.post("/analyze/llm")
async def analyze_llm_data(
    file: UploadFile = File(...),
    instruction_column: str = Query(default="instruction"),
    response_column: str = Query(default="response"),
    mode: str = Query(default="instruction"),
    min_response_length: int = Query(default=10),
) -> JSONResponse:
    """Analyze LLM instruction-tuning or RAG data quality."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        if mode == "instruction":
            cleaner = LLMDataCleaner(
                data=df,
                instruction_column=instruction_column,
                response_column=response_column,
                min_response_length=min_response_length,
            )
        else:
            cleaner = LLMDataCleaner(
                data=df,
                text_column=instruction_column,
            )

        report = cleaner.analyze()

        return JSONResponse(
            content={
                "dataset_name": file.filename,
                "mode": mode,
                "summary": report.summary(),
                "issues": report.to_dataframe().to_dict(orient="records")
                if len(report.issues) > 0
                else [],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")


@app.post("/fix")
async def apply_fixes(
    file: UploadFile = File(...),
    label_column: str = Query(default="label"),
    strategy: str = Query(default="conservative"),
    remove_duplicates: bool = Query(default=True),
    remove_outliers: bool = Query(default=False),
    min_confidence: float = Query(default=0.8, ge=0.0, le=1.0),
) -> JSONResponse:
    """Apply automatic fixes to a dataset.

    Returns the cleaned dataset and a summary of fixes applied.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        # Analyze first
        cleaner = DatasetCleaner(data=df, label_column=label_column)
        report = cleaner.analyze()

        # Configure fixes using strategy
        fix_strategy = FixStrategy[strategy.upper()]
        config = FixConfig.from_strategy(fix_strategy)
        config.label_error_threshold = min_confidence

        # Extract labels if present
        labels = df[label_column].values if label_column in df.columns else None
        features = df.drop(columns=[label_column]) if label_column in df.columns else df

        # Apply fixes
        engine = FixEngine(report=report, features=features, labels=labels, config=config)
        fixes = engine.suggest_fixes(
            include_duplicates=remove_duplicates,
            include_outliers=remove_outliers,
        )

        # Filter by confidence
        confident_fixes = [f for f in fixes if f.confidence >= min_confidence]
        result = engine.apply_fixes(confident_fixes, dry_run=False)

        # Reconstruct cleaned DataFrame
        cleaned_df = result.features.copy()
        if result.labels is not None and label_column:
            cleaned_df[label_column] = result.labels

        return JSONResponse(
            content={
                "original_rows": len(df),
                "final_rows": len(result.features),
                "rows_removed": len(df) - len(result.features),
                "fixes_applied": result.n_applied,
                "fixes_skipped": result.n_skipped,
                "fixes_errors": result.n_errors,
                "cleaned_data_preview": cleaned_df.head(10).to_dict(orient="records"),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fix application failed: {e}")


@app.get("/lineage/runs")
async def list_analysis_runs(
    limit: int = Query(default=50, ge=1, le=1000),
) -> JSONResponse:
    """List recent analysis runs."""
    tracker = get_lineage_tracker()
    runs = tracker.list_runs()[:limit]

    return JSONResponse(
        content={
            "runs": [
                {
                    "run_id": r.run_id,
                    "project_name": r.project_name,
                    "timestamp": r.timestamp,
                    "n_samples": r.n_samples,
                    "quality_score": r.quality_score,
                }
                for r in runs
            ]
        }
    )


@app.get("/lineage/runs/{run_id}")
async def get_analysis_run(run_id: str) -> JSONResponse:
    """Get details of a specific analysis run."""
    tracker = get_lineage_tracker()
    run = tracker.get_run(run_id)

    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return JSONResponse(
        content={
            "run_id": run.run_id,
            "project_name": run.project_name,
            "timestamp": run.timestamp,
            "n_samples": run.n_samples,
            "n_label_errors": run.n_label_errors,
            "n_duplicates": run.n_duplicates,
            "n_outliers": run.n_outliers,
            "quality_score": run.quality_score,
            "metadata": run.metadata,
        }
    )


@app.post("/review")
async def submit_review(decision: ReviewDecision) -> JSONResponse:
    """Submit a review decision for a sample."""
    from clean.lineage import ReviewDecision as ReviewDecisionType

    tracker = get_lineage_tracker()

    # Map API decision to lineage decision type
    decision_map = {
        "correct": "keep",
        "incorrect": "remove",
        "skip": "skip",
        "keep": "keep",
        "remove": "remove",
        "relabel": "relabel",
        "defer": "defer",
    }
    mapped_decision: ReviewDecisionType = decision_map.get(decision.decision, "skip")  # type: ignore[assignment]

    tracker.log_review(
        run_id="latest",  # Would need to track current run
        sample_id=decision.sample_index,
        issue_type="manual_review",
        decision=mapped_decision,
        reviewer="api_user",
        notes=decision.notes or "",
        new_value=decision.new_label,
    )

    return JSONResponse(
        content={
            "status": "recorded",
            "sample_index": decision.sample_index,
            "decision": decision.decision,
        }
    )


@app.get("/detectors")
async def list_detectors() -> JSONResponse:
    """List available detectors."""
    return JSONResponse(
        content={
            "detectors": [
                {
                    "name": "label_errors",
                    "description": "Detect mislabeled samples using confident learning",
                },
                {
                    "name": "duplicates",
                    "description": "Find exact and near-duplicate samples",
                },
                {
                    "name": "outliers",
                    "description": "Identify statistical outliers",
                },
                {
                    "name": "imbalance",
                    "description": "Analyze class distribution imbalance",
                },
                {
                    "name": "bias",
                    "description": "Detect demographic bias in data",
                },
            ]
        }
    )


@app.get("/plugins")
async def list_plugins() -> JSONResponse:
    """List registered plugins."""
    from clean.plugins import registry

    return JSONResponse(
        content={
            "detectors": list(registry.list_detectors()),
            "loaders": list(registry.list_loaders()),
            "exporters": list(registry.list_exporters()),
            "fixers": list(registry.list_fixers()),
        }
    )


# Entry point for direct execution
def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API server.

    Args:
        host: Host to bind to (default: 127.0.0.1 for local-only access)
        port: Port to bind to
    """
    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        raise ImportError("uvicorn not installed. Install with: pip install uvicorn")


if __name__ == "__main__":
    run_server()
