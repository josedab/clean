"""Visual Data Quality Studio.

Desktop/web IDE for interactive data exploration with point-and-click
issue discovery, annotation, and fix application.

Example:
    >>> from clean.studio import DataStudio, launch_studio
    >>>
    >>> # Launch interactive studio
    >>> launch_studio(data=df, label_column="target", port=8080)
    >>>
    >>> # Or use programmatically
    >>> studio = DataStudio(data=df, label_column="target")
    >>> studio.run(port=8080)
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ViewType(Enum):
    """Types of views in the studio."""

    OVERVIEW = "overview"
    LABEL_ERRORS = "label_errors"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    DISTRIBUTIONS = "distributions"
    SAMPLES = "samples"
    HISTORY = "history"


class AnnotationType(Enum):
    """Types of annotations."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    FLAG_FOR_REVIEW = "flag_for_review"
    RELABEL = "relabel"
    REMOVE = "remove"


@dataclass
class Annotation:
    """User annotation on a data sample."""

    annotation_id: str
    sample_index: int
    annotation_type: AnnotationType
    old_value: Any | None = None
    new_value: Any | None = None
    note: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "user"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "sample_index": self.sample_index,
            "annotation_type": self.annotation_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "note": self.note,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


@dataclass
class StudioSession:
    """Session state for the studio."""

    session_id: str
    data: pd.DataFrame
    label_column: str | None
    annotations: list[Annotation] = field(default_factory=list)
    quality_report: Any = None
    current_view: ViewType = ViewType.OVERVIEW
    filters: dict[str, Any] = field(default_factory=dict)
    selected_indices: list[int] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation."""
        self.annotations.append(annotation)
        self.history.append({
            "action": "annotate",
            "annotation_id": annotation.annotation_id,
            "timestamp": datetime.now().isoformat(),
        })

    def get_annotations_for_sample(self, sample_index: int) -> list[Annotation]:
        """Get all annotations for a sample."""
        return [a for a in self.annotations if a.sample_index == sample_index]

    def export_annotations(self) -> list[dict[str, Any]]:
        """Export all annotations."""
        return [a.to_dict() for a in self.annotations]


class StudioAPI:
    """API for the studio backend."""

    def __init__(self, session: StudioSession):
        self.session = session
        self._analysis_cache: dict[str, Any] = {}

    def analyze(self) -> dict[str, Any]:
        """Run quality analysis."""
        from clean import DatasetCleaner

        if "report" not in self._analysis_cache:
            cleaner = DatasetCleaner(
                data=self.session.data,
                label_column=self.session.label_column,
            )
            report = cleaner.analyze(show_progress=False)
            self.session.quality_report = report
            self._analysis_cache["report"] = report

        report = self._analysis_cache["report"]

        return {
            "quality_score": report.quality_score.overall,
            "n_samples": len(self.session.data),
            "n_label_errors": len(report.label_errors()) if report.label_errors_result else 0,
            "n_duplicates": len(report.duplicates()) if report.duplicates_result else 0,
            "n_outliers": len(report.outliers()) if report.outliers_result else 0,
            "summary": report.summary(),
        }

    def get_overview(self) -> dict[str, Any]:
        """Get dataset overview."""
        data = self.session.data

        # Basic stats
        overview = {
            "n_samples": len(data),
            "n_features": len(data.columns),
            "n_numeric": len(data.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(data.select_dtypes(include=["object", "category"]).columns),
            "missing_rate": float(data.isna().mean().mean()),
            "duplicate_rate": float(data.duplicated().mean()),
        }

        # Column info
        columns = []
        for col in data.columns:
            col_info = {
                "name": col,
                "dtype": str(data[col].dtype),
                "missing": int(data[col].isna().sum()),
                "missing_pct": float(data[col].isna().mean() * 100),
                "unique": int(data[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(data[col]):
                col_info["mean"] = float(data[col].mean()) if not data[col].isna().all() else None
                col_info["std"] = float(data[col].std()) if not data[col].isna().all() else None
                col_info["min"] = float(data[col].min()) if not data[col].isna().all() else None
                col_info["max"] = float(data[col].max()) if not data[col].isna().all() else None

            columns.append(col_info)

        overview["columns"] = columns

        # Quality score if available
        if self.session.quality_report:
            overview["quality_score"] = self.session.quality_report.quality_score.overall

        return overview

    def get_label_errors(
        self,
        page: int = 1,
        page_size: int = 20,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Get label errors with pagination."""
        if not self.session.quality_report:
            self.analyze()

        report = self.session.quality_report
        if not report.label_errors_result:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        errors = report.label_errors_result.issues
        errors = [e for e in errors if e.confidence >= min_confidence]
        errors = sorted(errors, key=lambda x: -x.confidence)

        total = len(errors)
        start = (page - 1) * page_size
        end = start + page_size
        page_errors = errors[start:end]

        items = []
        for error in page_errors:
            sample_data = self.session.data.iloc[error.index].to_dict()
            annotations = self.session.get_annotations_for_sample(error.index)

            items.append({
                "index": error.index,
                "given_label": error.given_label,
                "predicted_label": error.predicted_label,
                "confidence": error.confidence,
                "sample_data": sample_data,
                "annotations": [a.to_dict() for a in annotations],
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_duplicates(
        self,
        page: int = 1,
        page_size: int = 20,
        min_similarity: float = 0.0,
    ) -> dict[str, Any]:
        """Get duplicates with pagination."""
        if not self.session.quality_report:
            self.analyze()

        report = self.session.quality_report
        if not report.duplicates_result:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        duplicates = report.duplicates_result.issues
        duplicates = [d for d in duplicates if d.similarity >= min_similarity]
        duplicates = sorted(duplicates, key=lambda x: -x.similarity)

        total = len(duplicates)
        start = (page - 1) * page_size
        end = start + page_size
        page_dups = duplicates[start:end]

        items = []
        for dup in page_dups:
            items.append({
                "index1": dup.index1,
                "index2": dup.index2,
                "similarity": dup.similarity,
                "is_exact": dup.is_exact,
                "sample1": self.session.data.iloc[dup.index1].to_dict(),
                "sample2": self.session.data.iloc[dup.index2].to_dict(),
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_outliers(
        self,
        page: int = 1,
        page_size: int = 20,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Get outliers with pagination."""
        if not self.session.quality_report:
            self.analyze()

        report = self.session.quality_report
        if not report.outliers_result:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        outliers = report.outliers_result.issues
        outliers = [o for o in outliers if o.score >= min_score]
        outliers = sorted(outliers, key=lambda x: -x.score)

        total = len(outliers)
        start = (page - 1) * page_size
        end = start + page_size
        page_outliers = outliers[start:end]

        items = []
        for outlier in page_outliers:
            items.append({
                "index": outlier.index,
                "score": outlier.score,
                "method": outlier.method,
                "sample_data": self.session.data.iloc[outlier.index].to_dict(),
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_sample(self, index: int) -> dict[str, Any]:
        """Get a specific sample."""
        if index < 0 or index >= len(self.session.data):
            return {"error": "Index out of range"}

        sample = self.session.data.iloc[index]
        annotations = self.session.get_annotations_for_sample(index)

        issues = []
        if self.session.quality_report:
            report = self.session.quality_report

            # Check label errors
            if report.label_errors_result:
                for error in report.label_errors_result.issues:
                    if error.index == index:
                        issues.append({
                            "type": "label_error",
                            "confidence": error.confidence,
                            "given_label": error.given_label,
                            "predicted_label": error.predicted_label,
                        })

            # Check outliers
            if report.outliers_result:
                for outlier in report.outliers_result.issues:
                    if outlier.index == index:
                        issues.append({
                            "type": "outlier",
                            "score": outlier.score,
                            "method": outlier.method,
                        })

        return {
            "index": index,
            "data": sample.to_dict(),
            "annotations": [a.to_dict() for a in annotations],
            "issues": issues,
        }

    def annotate_sample(
        self,
        sample_index: int,
        annotation_type: str,
        new_value: Any | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Add annotation to a sample."""
        try:
            ann_type = AnnotationType(annotation_type)
        except ValueError:
            return {"error": f"Invalid annotation type: {annotation_type}"}

        old_value = None
        if self.session.label_column:
            old_value = self.session.data.iloc[sample_index][self.session.label_column]

        annotation = Annotation(
            annotation_id=str(uuid4()),
            sample_index=sample_index,
            annotation_type=ann_type,
            old_value=old_value,
            new_value=new_value,
            note=note,
        )

        self.session.add_annotation(annotation)

        return {
            "success": True,
            "annotation": annotation.to_dict(),
        }

    def batch_annotate(
        self,
        sample_indices: list[int],
        annotation_type: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Batch annotate multiple samples."""
        results = []
        for idx in sample_indices:
            result = self.annotate_sample(idx, annotation_type, note=note)
            results.append(result)

        success_count = sum(1 for r in results if r.get("success"))
        return {
            "success": True,
            "annotated": success_count,
            "total": len(sample_indices),
        }

    def apply_fixes(self) -> dict[str, Any]:
        """Apply annotations as fixes to the data."""
        modifications = 0
        removals = 0
        data = self.session.data.copy()
        indices_to_remove = set()

        for annotation in self.session.annotations:
            if annotation.annotation_type == AnnotationType.RELABEL:
                if (
                    self.session.label_column
                    and annotation.new_value is not None
                    and annotation.sample_index in data.index
                ):
                    data.loc[annotation.sample_index, self.session.label_column] = annotation.new_value
                    modifications += 1

            elif annotation.annotation_type == AnnotationType.REMOVE:
                indices_to_remove.add(annotation.sample_index)
                removals += 1

        # Remove marked samples
        if indices_to_remove:
            data = data.drop(index=list(indices_to_remove))

        # Update session data
        self.session.data = data.reset_index(drop=True)
        self._analysis_cache.clear()  # Clear cache to re-analyze

        self.session.history.append({
            "action": "apply_fixes",
            "modifications": modifications,
            "removals": removals,
            "timestamp": datetime.now().isoformat(),
        })

        return {
            "success": True,
            "modifications": modifications,
            "removals": removals,
            "new_row_count": len(data),
        }

    def export_data(self, format: str = "csv") -> dict[str, Any]:
        """Export the current data."""
        data = self.session.data

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{format}", delete=False
        ) as f:
            if format == "csv":
                data.to_csv(f.name, index=False)
            elif format == "json":
                data.to_json(f.name, orient="records", indent=2)
            elif format == "parquet":
                data.to_parquet(f.name.replace(".parquet", ".parquet"), index=False)

            return {
                "success": True,
                "path": f.name,
                "format": format,
                "rows": len(data),
            }

    def get_distribution(self, column: str) -> dict[str, Any]:
        """Get distribution data for a column."""
        if column not in self.session.data.columns:
            return {"error": f"Column not found: {column}"}

        col_data = self.session.data[column]

        if pd.api.types.is_numeric_dtype(col_data):
            # Histogram for numeric
            hist, bin_edges = np.histogram(col_data.dropna(), bins=30)
            return {
                "type": "histogram",
                "column": column,
                "bins": [float(b) for b in bin_edges],
                "counts": [int(c) for c in hist],
                "stats": {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                },
            }
        else:
            # Value counts for categorical
            value_counts = col_data.value_counts().head(20)
            return {
                "type": "bar",
                "column": column,
                "labels": [str(l) for l in value_counts.index.tolist()],
                "counts": [int(c) for c in value_counts.values.tolist()],
                "unique_values": int(col_data.nunique()),
            }


class DataStudio:
    """Main class for the Visual Data Quality Studio."""

    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize the studio.

        Args:
            data: DataFrame to analyze
            label_column: Name of the label column
            session_id: Optional session ID
        """
        self.session = StudioSession(
            session_id=session_id or str(uuid4()),
            data=data.copy(),
            label_column=label_column,
        )
        self.api = StudioAPI(self.session)
        self._app = None

    def create_app(self) -> Any:
        """Create the FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException, Query
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import HTMLResponse, JSONResponse
        except ImportError as e:
            raise ImportError(
                "FastAPI is required for the studio. Install with: pip install fastapi uvicorn"
            ) from e

        app = FastAPI(
            title="Clean Data Quality Studio",
            description="Interactive data quality exploration and annotation",
            version="1.0.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return self._get_html_template()

        @app.get("/api/analyze")
        async def analyze():
            return self.api.analyze()

        @app.get("/api/overview")
        async def overview():
            return self.api.get_overview()

        @app.get("/api/label-errors")
        async def label_errors(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
            min_confidence: float = Query(0.0, ge=0, le=1),
        ):
            return self.api.get_label_errors(page, page_size, min_confidence)

        @app.get("/api/duplicates")
        async def duplicates(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
            min_similarity: float = Query(0.0, ge=0, le=1),
        ):
            return self.api.get_duplicates(page, page_size, min_similarity)

        @app.get("/api/outliers")
        async def outliers(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
            min_score: float = Query(0.0, ge=0, le=1),
        ):
            return self.api.get_outliers(page, page_size, min_score)

        @app.get("/api/sample/{index}")
        async def get_sample(index: int):
            result = self.api.get_sample(index)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result

        @app.post("/api/annotate")
        async def annotate(body: dict):
            return self.api.annotate_sample(
                sample_index=body["sample_index"],
                annotation_type=body["annotation_type"],
                new_value=body.get("new_value"),
                note=body.get("note", ""),
            )

        @app.post("/api/batch-annotate")
        async def batch_annotate(body: dict):
            return self.api.batch_annotate(
                sample_indices=body["sample_indices"],
                annotation_type=body["annotation_type"],
                note=body.get("note", ""),
            )

        @app.post("/api/apply-fixes")
        async def apply_fixes():
            return self.api.apply_fixes()

        @app.get("/api/export")
        async def export_data(format: str = Query("csv")):
            return self.api.export_data(format)

        @app.get("/api/distribution/{column}")
        async def distribution(column: str):
            return self.api.get_distribution(column)

        @app.get("/api/annotations")
        async def get_annotations():
            return {"annotations": self.session.export_annotations()}

        self._app = app
        return app

    def _get_html_template(self) -> str:
        """Get the HTML template for the studio UI."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clean Data Quality Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .stat-card h3 { color: #64748b; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
        .stat-card .value { font-size: 32px; font-weight: 700; color: #1e293b; }
        .stat-card .value.good { color: #22c55e; }
        .stat-card .value.warning { color: #f59e0b; }
        .stat-card .value.bad { color: #ef4444; }
        .tabs { display: flex; gap: 8px; margin-bottom: 20px; }
        .tab { padding: 12px 24px; background: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 500; transition: all 0.2s; }
        .tab.active { background: #667eea; color: white; }
        .tab:hover:not(.active) { background: #e2e8f0; }
        .panel { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 24px; }
        .table { width: 100%; border-collapse: collapse; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        .table th { background: #f8fafc; font-weight: 600; color: #64748b; font-size: 12px; text-transform: uppercase; }
        .table tr:hover { background: #f8fafc; }
        .badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .badge.high { background: #fef2f2; color: #dc2626; }
        .badge.medium { background: #fff7ed; color: #ea580c; }
        .badge.low { background: #f0fdf4; color: #16a34a; }
        .btn { padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer; font-weight: 500; transition: all 0.2s; }
        .btn-primary { background: #667eea; color: white; }
        .btn-primary:hover { background: #5a67d8; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-success { background: #22c55e; color: white; }
        .loading { text-align: center; padding: 40px; color: #64748b; }
        .chart-container { height: 300px; margin-top: 20px; }
        .actions { display: flex; gap: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧹 Clean Data Quality Studio</h1>
            <p>Interactive data quality exploration and annotation</p>
        </div>

        <div id="stats" class="stats-grid">
            <div class="stat-card">
                <h3>Quality Score</h3>
                <div id="quality-score" class="value">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Samples</h3>
                <div id="total-samples" class="value">-</div>
            </div>
            <div class="stat-card">
                <h3>Label Errors</h3>
                <div id="label-errors" class="value">-</div>
            </div>
            <div class="stat-card">
                <h3>Duplicates</h3>
                <div id="duplicates" class="value">-</div>
            </div>
            <div class="stat-card">
                <h3>Outliers</h3>
                <div id="outliers" class="value">-</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" data-view="overview">Overview</button>
            <button class="tab" data-view="label-errors">Label Errors</button>
            <button class="tab" data-view="duplicates">Duplicates</button>
            <button class="tab" data-view="outliers">Outliers</button>
        </div>

        <div id="content" class="panel">
            <div class="loading">Loading data...</div>
        </div>

        <div class="actions">
            <button class="btn btn-primary" onclick="applyFixes()">Apply Fixes</button>
            <button class="btn btn-success" onclick="exportData('csv')">Export CSV</button>
        </div>
    </div>

    <script>
        let currentView = 'overview';

        async function loadAnalysis() {
            const resp = await fetch('/api/analyze');
            const data = await resp.json();

            document.getElementById('quality-score').textContent = data.quality_score.toFixed(1);
            document.getElementById('quality-score').className = 'value ' +
                (data.quality_score >= 80 ? 'good' : data.quality_score >= 60 ? 'warning' : 'bad');
            document.getElementById('total-samples').textContent = data.n_samples.toLocaleString();
            document.getElementById('label-errors').textContent = data.n_label_errors.toLocaleString();
            document.getElementById('duplicates').textContent = data.n_duplicates.toLocaleString();
            document.getElementById('outliers').textContent = data.n_outliers.toLocaleString();

            loadView(currentView);
        }

        async function loadView(view) {
            currentView = view;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`[data-view="${view}"]`).classList.add('active');

            const content = document.getElementById('content');
            content.innerHTML = '<div class="loading">Loading...</div>';

            if (view === 'overview') {
                const resp = await fetch('/api/overview');
                const data = await resp.json();
                content.innerHTML = renderOverview(data);
            } else if (view === 'label-errors') {
                const resp = await fetch('/api/label-errors');
                const data = await resp.json();
                content.innerHTML = renderLabelErrors(data);
            } else if (view === 'duplicates') {
                const resp = await fetch('/api/duplicates');
                const data = await resp.json();
                content.innerHTML = renderDuplicates(data);
            } else if (view === 'outliers') {
                const resp = await fetch('/api/outliers');
                const data = await resp.json();
                content.innerHTML = renderOutliers(data);
            }
        }

        function renderOverview(data) {
            let html = '<h2>Dataset Overview</h2><table class="table"><thead><tr>';
            html += '<th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>';
            for (const col of data.columns) {
                html += `<tr><td>${col.name}</td><td>${col.dtype}</td>`;
                html += `<td>${col.missing_pct.toFixed(1)}%</td><td>${col.unique}</td></tr>`;
            }
            html += '</tbody></table>';
            return html;
        }

        function renderLabelErrors(data) {
            if (data.items.length === 0) return '<p>No label errors detected.</p>';
            let html = `<h2>Label Errors (${data.total})</h2><table class="table"><thead><tr>`;
            html += '<th>Index</th><th>Given</th><th>Predicted</th><th>Confidence</th><th>Action</th></tr></thead><tbody>';
            for (const item of data.items) {
                const badge = item.confidence > 0.9 ? 'high' : item.confidence > 0.7 ? 'medium' : 'low';
                html += `<tr><td>${item.index}</td><td>${item.given_label}</td><td>${item.predicted_label}</td>`;
                html += `<td><span class="badge ${badge}">${(item.confidence * 100).toFixed(0)}%</span></td>`;
                html += `<td><button class="btn btn-primary" onclick="annotate(${item.index}, 'relabel', '${item.predicted_label}')">Accept</button></td></tr>`;
            }
            html += '</tbody></table>';
            return html;
        }

        function renderDuplicates(data) {
            if (data.items.length === 0) return '<p>No duplicates detected.</p>';
            let html = `<h2>Duplicates (${data.total})</h2><table class="table"><thead><tr>`;
            html += '<th>Index 1</th><th>Index 2</th><th>Similarity</th><th>Action</th></tr></thead><tbody>';
            for (const item of data.items) {
                html += `<tr><td>${item.index1}</td><td>${item.index2}</td>`;
                html += `<td>${(item.similarity * 100).toFixed(1)}%</td>`;
                html += `<td><button class="btn btn-danger" onclick="annotate(${item.index2}, 'remove')">Remove</button></td></tr>`;
            }
            html += '</tbody></table>';
            return html;
        }

        function renderOutliers(data) {
            if (data.items.length === 0) return '<p>No outliers detected.</p>';
            let html = `<h2>Outliers (${data.total})</h2><table class="table"><thead><tr>`;
            html += '<th>Index</th><th>Score</th><th>Method</th><th>Action</th></tr></thead><tbody>';
            for (const item of data.items) {
                html += `<tr><td>${item.index}</td><td>${item.score.toFixed(3)}</td><td>${item.method}</td>`;
                html += `<td><button class="btn btn-danger" onclick="annotate(${item.index}, 'remove')">Remove</button></td></tr>`;
            }
            html += '</tbody></table>';
            return html;
        }

        async function annotate(index, type, newValue = null) {
            await fetch('/api/annotate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sample_index: index, annotation_type: type, new_value: newValue})
            });
            loadView(currentView);
        }

        async function applyFixes() {
            const resp = await fetch('/api/apply-fixes', {method: 'POST'});
            const data = await resp.json();
            alert(`Applied fixes: ${data.modifications} modifications, ${data.removals} removals`);
            loadAnalysis();
        }

        async function exportData(format) {
            const resp = await fetch(`/api/export?format=${format}`);
            const data = await resp.json();
            alert(`Data exported to: ${data.path}`);
        }

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => loadView(tab.dataset.view));
        });

        loadAnalysis();
    </script>
</body>
</html>"""

    def run(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Run the studio server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "uvicorn is required. Install with: pip install uvicorn"
            ) from e

        app = self.create_app()
        logger.info("Starting Data Quality Studio at http://%s:%d", host, port)
        uvicorn.run(app, host=host, port=port)


def launch_studio(
    data: pd.DataFrame,
    label_column: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Launch the interactive Data Quality Studio.

    Args:
        data: DataFrame to analyze
        label_column: Name of the label column
        host: Host to bind to
        port: Port to bind to
    """
    studio = DataStudio(data=data, label_column=label_column)
    studio.run(host=host, port=port)


def create_studio(
    data: pd.DataFrame,
    label_column: str | None = None,
) -> DataStudio:
    """Create a DataStudio instance without running it.

    Args:
        data: DataFrame to analyze
        label_column: Name of the label column

    Returns:
        DataStudio instance
    """
    return DataStudio(data=data, label_column=label_column)
