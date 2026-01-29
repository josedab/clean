"""Data lineage tracking for Clean.

This module provides audit trail capabilities for tracking
which samples were flagged, reviewed, and fixed across multiple
analysis runs.

Example:
    >>> from clean import DatasetCleaner
    >>> from clean.lineage import LineageTracker
    >>>
    >>> tracker = LineageTracker(project_name="my_dataset")
    >>>
    >>> # Run analysis
    >>> cleaner = DatasetCleaner(data=df, label_column='label')
    >>> report = cleaner.analyze()
    >>>
    >>> # Log the analysis run
    >>> run_id = tracker.log_analysis(report)
    >>>
    >>> # Log review decisions
    >>> tracker.log_review(run_id, sample_id=42, decision="keep", reviewer="alice")
    >>> tracker.log_review(run_id, sample_id=187, decision="remove", reviewer="alice")
    >>>
    >>> # Get history for a sample
    >>> history = tracker.get_sample_history(42)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

ReviewDecision = Literal["keep", "remove", "relabel", "skip", "defer"]


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
    sample_id: int | tuple[int, int]
    issue_type: str
    decision: ReviewDecision
    reviewer: str
    timestamp: str
    notes: str = ""
    old_value: Any = None
    new_value: Any = None


@dataclass
class SampleHistory:
    """Complete history for a sample."""

    sample_id: int
    issue_detections: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[ReviewRecord] = field(default_factory=list)
    current_status: str = "unreviewed"


class LineageTracker:
    """Track data quality lineage across analysis runs.

    The LineageTracker maintains a persistent audit trail of:
    - Analysis runs and their results
    - Review decisions for individual samples
    - Changes applied to the dataset

    This enables:
    - Compliance with data governance requirements
    - Team collaboration on data cleaning
    - Tracking improvements over time
    - Reproducibility of data cleaning decisions

    Example:
        >>> tracker = LineageTracker("my_project", storage_path="./lineage")
        >>>
        >>> # Log an analysis run
        >>> run_id = tracker.log_analysis(report, metadata={"version": "v1"})
        >>>
        >>> # Log review decisions
        >>> tracker.log_review(
        ...     run_id=run_id,
        ...     sample_id=42,
        ...     issue_type="label_error",
        ...     decision="relabel",
        ...     reviewer="alice",
        ...     notes="Clearly a dog, not a cat",
        ...     old_value="cat",
        ...     new_value="dog"
        ... )
        >>>
        >>> # Get history
        >>> history = tracker.get_sample_history(42)
        >>> print(f"Sample 42 reviewed {len(history.reviews)} times")
    """

    def __init__(
        self,
        project_name: str,
        storage_path: str | Path | None = None,
        auto_persist: bool = True,
    ):
        """Initialize the lineage tracker.

        Args:
            project_name: Name for this project/dataset
            storage_path: Directory for persistent storage (optional)
            auto_persist: Automatically save after each operation
        """
        self.project_name = project_name
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_persist = auto_persist

        self._runs: dict[str, AnalysisRun] = {}
        self._reviews: list[ReviewRecord] = []
        self._sample_detections: dict[int, list[dict[str, Any]]] = {}

        if self.storage_path:
            self._load()

    def log_analysis(
        self,
        report: Any,  # QualityReport
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Log an analysis run.

        Args:
            report: QualityReport from DatasetCleaner.analyze()
            metadata: Additional metadata to store

        Returns:
            Run ID for this analysis
        """
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Extract counts from report
        n_label_errors = 0
        n_duplicates = 0
        n_outliers = 0

        if report.label_errors_result:
            n_label_errors = len(report.label_errors_result.issues)
            for issue in report.label_errors_result.issues:
                self._add_detection(issue.index, run_id, "label_error", {
                    "given_label": issue.given_label,
                    "predicted_label": issue.predicted_label,
                    "confidence": issue.confidence,
                })

        if report.duplicates_result:
            n_duplicates = len(report.duplicates_result.issues)
            for issue in report.duplicates_result.issues:
                self._add_detection(issue.index1, run_id, "duplicate", {
                    "paired_with": issue.index2,
                    "similarity": issue.similarity,
                    "is_exact": issue.is_exact,
                })

        if report.outliers_result:
            n_outliers = len(report.outliers_result.issues)
            for issue in report.outliers_result.issues:
                self._add_detection(issue.index, run_id, "outlier", {
                    "method": issue.method,
                    "score": issue.score,
                })

        run = AnalysisRun(
            run_id=run_id,
            timestamp=timestamp,
            project_name=self.project_name,
            n_samples=report.dataset_info.n_samples,
            n_label_errors=n_label_errors,
            n_duplicates=n_duplicates,
            n_outliers=n_outliers,
            quality_score=report.quality_score.overall,
            metadata=metadata or {},
        )

        self._runs[run_id] = run

        if self.auto_persist:
            self._save()

        return run_id

    def _add_detection(
        self,
        sample_id: int,
        run_id: str,
        issue_type: str,
        details: dict[str, Any],
    ) -> None:
        """Add a detection record for a sample."""
        if sample_id not in self._sample_detections:
            self._sample_detections[sample_id] = []

        self._sample_detections[sample_id].append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "issue_type": issue_type,
            **details,
        })

    def log_review(
        self,
        run_id: str,
        sample_id: int | tuple[int, int],
        issue_type: str,
        decision: ReviewDecision,
        reviewer: str,
        notes: str = "",
        old_value: Any = None,
        new_value: Any = None,
    ) -> None:
        """Log a review decision for a sample.

        Args:
            run_id: ID of the analysis run
            sample_id: Sample index (or tuple for duplicates)
            issue_type: Type of issue being reviewed
            decision: Review decision (keep, remove, relabel, skip, defer)
            reviewer: Name/ID of the reviewer
            notes: Optional notes about the decision
            old_value: Original value (for relabeling)
            new_value: New value (for relabeling)
        """
        if run_id not in self._runs:
            raise ValueError(f"Unknown run_id: {run_id}")

        record = ReviewRecord(
            run_id=run_id,
            sample_id=sample_id,
            issue_type=issue_type,
            decision=decision,
            reviewer=reviewer,
            timestamp=datetime.now().isoformat(),
            notes=notes,
            old_value=old_value,
            new_value=new_value,
        )

        self._reviews.append(record)

        if self.auto_persist:
            self._save()

    def get_sample_history(self, sample_id: int) -> SampleHistory:
        """Get complete history for a sample.

        Args:
            sample_id: Sample index

        Returns:
            SampleHistory with all detections and reviews
        """
        detections = self._sample_detections.get(sample_id, [])
        reviews = [r for r in self._reviews if r.sample_id == sample_id]

        # Determine current status
        if reviews:
            last_decision = reviews[-1].decision
            if last_decision == "keep":
                status = "kept"
            elif last_decision == "remove":
                status = "removed"
            elif last_decision == "relabel":
                status = "relabeled"
            elif last_decision == "defer":
                status = "deferred"
            else:
                status = "reviewed"
        elif detections:
            status = "flagged"
        else:
            status = "clean"

        return SampleHistory(
            sample_id=sample_id,
            issue_detections=detections,
            reviews=reviews,
            current_status=status,
        )

    def get_run(self, run_id: str) -> AnalysisRun | None:
        """Get an analysis run by ID."""
        return self._runs.get(run_id)

    def list_runs(self) -> list[AnalysisRun]:
        """List all analysis runs."""
        return sorted(self._runs.values(), key=lambda r: r.timestamp, reverse=True)

    def get_reviews_for_run(self, run_id: str) -> list[ReviewRecord]:
        """Get all reviews for a specific run."""
        return [r for r in self._reviews if r.run_id == run_id]

    def get_pending_reviews(self, run_id: str) -> list[int]:
        """Get sample IDs that have been flagged but not reviewed.

        Args:
            run_id: Analysis run to check

        Returns:
            List of sample IDs pending review
        """
        reviewed = {r.sample_id for r in self._reviews if r.run_id == run_id}

        pending = []
        for sample_id, detections in self._sample_detections.items():
            if any(d["run_id"] == run_id for d in detections):
                if sample_id not in reviewed:
                    pending.append(sample_id)

        return sorted(pending)

    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics.

        Returns:
            Dict with lineage statistics
        """
        total_reviews = len(self._reviews)
        decision_counts = {}
        reviewer_counts = {}

        for r in self._reviews:
            decision_counts[r.decision] = decision_counts.get(r.decision, 0) + 1
            reviewer_counts[r.reviewer] = reviewer_counts.get(r.reviewer, 0) + 1

        return {
            "project_name": self.project_name,
            "n_runs": len(self._runs),
            "n_reviews": total_reviews,
            "n_samples_flagged": len(self._sample_detections),
            "decision_distribution": decision_counts,
            "reviewer_activity": reviewer_counts,
        }

    def export(self, path: str | Path) -> None:
        """Export lineage data to JSON.

        Args:
            path: Output file path
        """
        data = {
            "project_name": self.project_name,
            "exported_at": datetime.now().isoformat(),
            "runs": [asdict(r) for r in self._runs.values()],
            "reviews": [asdict(r) for r in self._reviews],
            "sample_detections": self._sample_detections,
        }

        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def _save(self) -> None:
        """Save lineage data to storage."""
        if self.storage_path is None:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        data_path = self.storage_path / f"{self.project_name}_lineage.json"

        data = {
            "project_name": self.project_name,
            "runs": {k: asdict(v) for k, v in self._runs.items()},
            "reviews": [asdict(r) for r in self._reviews],
            "sample_detections": self._sample_detections,
        }

        data_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        """Load lineage data from storage."""
        if self.storage_path is None:
            return

        data_path = self.storage_path / f"{self.project_name}_lineage.json"
        if not data_path.exists():
            return

        data = json.loads(data_path.read_text())

        self._runs = {
            k: AnalysisRun(**v) for k, v in data.get("runs", {}).items()
        }

        self._reviews = [
            ReviewRecord(**r) for r in data.get("reviews", [])
        ]

        self._sample_detections = {
            int(k): v for k, v in data.get("sample_detections", {}).items()
        }


__all__ = [
    "LineageTracker",
    "AnalysisRun",
    "ReviewRecord",
    "SampleHistory",
    "ReviewDecision",
]
