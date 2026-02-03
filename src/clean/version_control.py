"""Version Control for Data Quality - Git-like History for Quality Reports.

This module provides version control capabilities for data quality reports,
enabling tracking of changes over time, diffing between versions, and
rollback to previous states.

Example:
    >>> from clean.version_control import QualityVersionControl
    >>>
    >>> vc = QualityVersionControl("./quality_history")
    >>> vc.commit(report, message="Initial quality baseline")
    >>>
    >>> # Later, after data changes
    >>> vc.commit(new_report, message="After deduplication")
    >>>
    >>> # Compare versions
    >>> diff = vc.diff("v1", "v2")
    >>> print(diff.summary())
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.exceptions import CleanError

if TYPE_CHECKING:
    pass

import logging

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of change between versions."""

    IMPROVED = "improved"
    DEGRADED = "degraded"
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class QualitySnapshot:
    """Snapshot of quality metrics at a point in time."""

    version: str
    timestamp: datetime
    message: str
    overall_score: float
    metrics: dict[str, float]
    issue_counts: dict[str, int]
    sample_count: int
    column_stats: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "overall_score": self.overall_score,
            "metrics": self.metrics,
            "issue_counts": self.issue_counts,
            "sample_count": self.sample_count,
            "column_stats": self.column_stats,
            "metadata": self.metadata,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualitySnapshot":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message=data["message"],
            overall_score=data["overall_score"],
            metrics=data["metrics"],
            issue_counts=data["issue_counts"],
            sample_count=data["sample_count"],
            column_stats=data["column_stats"],
            metadata=data.get("metadata", {}),
            parent_version=data.get("parent_version"),
        )


@dataclass
class MetricChange:
    """Change in a single metric."""

    name: str
    old_value: float | None
    new_value: float | None
    change_type: ChangeType
    delta: float | None = None
    percent_change: float | None = None

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if change is significant."""
        if self.percent_change is None:
            return self.change_type in (ChangeType.ADDED, ChangeType.REMOVED)
        return abs(self.percent_change) > threshold


@dataclass
class QualityDiff:
    """Diff between two quality snapshots."""

    old_version: str
    new_version: str
    old_timestamp: datetime
    new_timestamp: datetime
    score_change: MetricChange
    metric_changes: list[MetricChange]
    issue_changes: dict[str, MetricChange]
    sample_count_change: MetricChange
    column_changes: dict[str, dict[str, MetricChange]]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Quality Diff: {self.old_version} → {self.new_version}",
            f"Time: {self.old_timestamp.isoformat()} → {self.new_timestamp.isoformat()}",
            "",
            "## Overall Score",
            f"  {self.score_change.old_value:.2f} → {self.score_change.new_value:.2f} "
            f"({self._format_delta(self.score_change)})",
            "",
        ]

        # Metric changes
        significant = [m for m in self.metric_changes if m.is_significant()]
        if significant:
            lines.append("## Significant Metric Changes")
            for m in significant:
                lines.append(f"  {m.name}: {self._format_delta(m)}")
            lines.append("")

        # Issue changes
        issue_changes = [c for c in self.issue_changes.values() if c.is_significant()]
        if issue_changes:
            lines.append("## Issue Changes")
            for c in issue_changes:
                lines.append(
                    f"  {c.name}: {c.old_value} → {c.new_value} ({self._format_delta(c)})"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_delta(self, change: MetricChange) -> str:
        """Format delta for display."""
        if change.delta is None:
            return change.change_type.value
        sign = "+" if change.delta > 0 else ""
        if change.percent_change is not None:
            return f"{sign}{change.delta:.2f} ({sign}{change.percent_change*100:.1f}%)"
        return f"{sign}{change.delta:.2f}"

    def has_degradation(self) -> bool:
        """Check if any metrics degraded."""
        if self.score_change.change_type == ChangeType.DEGRADED:
            return True
        return any(m.change_type == ChangeType.DEGRADED for m in self.metric_changes)

    def has_improvement(self) -> bool:
        """Check if any metrics improved."""
        if self.score_change.change_type == ChangeType.IMPROVED:
            return True
        return any(m.change_type == ChangeType.IMPROVED for m in self.metric_changes)


@dataclass
class BranchInfo:
    """Information about a branch."""

    name: str
    head_version: str
    created_at: datetime
    description: str = ""


class QualityVersionControl:
    """Version control system for data quality reports.

    Provides Git-like version control for quality metrics, enabling
    tracking of quality evolution over time.

    Example:
        >>> vc = QualityVersionControl("./quality_history")
        >>> vc.commit(report, message="Initial baseline")
        >>> print(vc.log())
    """

    def __init__(
        self,
        storage_path: str | Path,
        auto_create: bool = True,
    ):
        """Initialize version control.

        Args:
            storage_path: Path to store version history
            auto_create: Automatically create storage directory
        """
        self.storage_path = Path(storage_path)

        if auto_create and not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)

        self._snapshots_dir = self.storage_path / "snapshots"
        self._refs_dir = self.storage_path / "refs"
        self._branches_dir = self._refs_dir / "branches"

        # Create directories
        for d in [self._snapshots_dir, self._refs_dir, self._branches_dir]:
            d.mkdir(exist_ok=True)

        # Initialize main branch if not exists
        self._current_branch = "main"
        main_ref = self._branches_dir / "main"
        if not main_ref.exists():
            main_ref.write_text("")

    def commit(
        self,
        report: Any,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> QualitySnapshot:
        """Commit a quality report as a new version.

        Args:
            report: Quality report from DatasetCleaner.analyze()
            message: Commit message
            metadata: Additional metadata

        Returns:
            QualitySnapshot of the committed version
        """
        # Extract metrics from report
        snapshot = self._create_snapshot(report, message, metadata)

        # Save snapshot
        self._save_snapshot(snapshot)

        # Update branch head
        self._update_branch_head(self._current_branch, snapshot.version)

        logger.info(f"Committed version {snapshot.version}: {message}")
        return snapshot

    def _create_snapshot(
        self,
        report: Any,
        message: str,
        metadata: dict[str, Any] | None,
    ) -> QualitySnapshot:
        """Create snapshot from report."""
        # Get parent version
        parent = self._get_head_version()

        # Extract metrics
        metrics = {}
        issue_counts = {}
        overall_score = 0.0
        sample_count = 0
        column_stats = {}

        if hasattr(report, "score"):
            score = report.score
            if hasattr(score, "overall"):
                overall_score = score.overall
            if hasattr(score, "completeness"):
                metrics["completeness"] = score.completeness
            if hasattr(score, "consistency"):
                metrics["consistency"] = score.consistency
            if hasattr(score, "uniqueness"):
                metrics["uniqueness"] = score.uniqueness
            if hasattr(score, "validity"):
                metrics["validity"] = score.validity

        if hasattr(report, "label_errors"):
            issue_counts["label_errors"] = len(report.label_errors())
        if hasattr(report, "duplicates"):
            issue_counts["duplicates"] = len(report.duplicates())
        if hasattr(report, "outliers"):
            issue_counts["outliers"] = len(report.outliers())

        if hasattr(report, "n_samples"):
            sample_count = report.n_samples

        # Generate version hash
        timestamp = datetime.now()
        version_content = f"{timestamp.isoformat()}|{message}|{overall_score}|{parent}"
        version = hashlib.sha256(version_content.encode()).hexdigest()[:12]

        return QualitySnapshot(
            version=version,
            timestamp=timestamp,
            message=message,
            overall_score=overall_score,
            metrics=metrics,
            issue_counts=issue_counts,
            sample_count=sample_count,
            column_stats=column_stats,
            metadata=metadata or {},
            parent_version=parent,
        )

    def _save_snapshot(self, snapshot: QualitySnapshot) -> None:
        """Save snapshot to storage."""
        path = self._snapshots_dir / f"{snapshot.version}.json"
        path.write_text(json.dumps(snapshot.to_dict(), indent=2))

    def _load_snapshot(self, version: str) -> QualitySnapshot:
        """Load snapshot from storage."""
        path = self._snapshots_dir / f"{version}.json"
        if not path.exists():
            raise CleanError(f"Version not found: {version}")
        data = json.loads(path.read_text())
        return QualitySnapshot.from_dict(data)

    def _get_head_version(self) -> str | None:
        """Get current head version."""
        ref_file = self._branches_dir / self._current_branch
        if ref_file.exists():
            content = ref_file.read_text().strip()
            return content if content else None
        return None

    def _update_branch_head(self, branch: str, version: str) -> None:
        """Update branch head reference."""
        ref_file = self._branches_dir / branch
        ref_file.write_text(version)

    def diff(
        self,
        old_version: str | None = None,
        new_version: str | None = None,
    ) -> QualityDiff:
        """Compute diff between two versions.

        Args:
            old_version: Old version (default: parent of HEAD)
            new_version: New version (default: HEAD)

        Returns:
            QualityDiff with all changes
        """
        # Resolve versions
        if new_version is None:
            new_version = self._get_head_version()
            if new_version is None:
                raise CleanError("No commits yet")

        new_snapshot = self._load_snapshot(new_version)

        if old_version is None:
            old_version = new_snapshot.parent_version
            if old_version is None:
                raise CleanError("No parent version to diff against")

        old_snapshot = self._load_snapshot(old_version)

        return self._compute_diff(old_snapshot, new_snapshot)

    def _compute_diff(
        self,
        old: QualitySnapshot,
        new: QualitySnapshot,
    ) -> QualityDiff:
        """Compute diff between snapshots."""
        # Score change
        score_change = self._compute_metric_change(
            "overall_score", old.overall_score, new.overall_score
        )

        # Metric changes
        all_metrics = set(old.metrics.keys()) | set(new.metrics.keys())
        metric_changes = [
            self._compute_metric_change(
                name,
                old.metrics.get(name),
                new.metrics.get(name),
            )
            for name in all_metrics
        ]

        # Issue changes
        all_issues = set(old.issue_counts.keys()) | set(new.issue_counts.keys())
        issue_changes = {
            name: self._compute_metric_change(
                name,
                old.issue_counts.get(name),
                new.issue_counts.get(name),
                invert=True,  # Lower issues = better
            )
            for name in all_issues
        }

        # Sample count change
        sample_change = self._compute_metric_change(
            "sample_count", old.sample_count, new.sample_count
        )

        return QualityDiff(
            old_version=old.version,
            new_version=new.version,
            old_timestamp=old.timestamp,
            new_timestamp=new.timestamp,
            score_change=score_change,
            metric_changes=metric_changes,
            issue_changes=issue_changes,
            sample_count_change=sample_change,
            column_changes={},
        )

    def _compute_metric_change(
        self,
        name: str,
        old_value: float | int | None,
        new_value: float | int | None,
        invert: bool = False,
    ) -> MetricChange:
        """Compute change for a single metric."""
        if old_value is None and new_value is None:
            return MetricChange(name, None, None, ChangeType.UNCHANGED)

        if old_value is None:
            return MetricChange(name, None, new_value, ChangeType.ADDED)

        if new_value is None:
            return MetricChange(name, old_value, None, ChangeType.REMOVED)

        delta = new_value - old_value
        if old_value != 0:
            percent_change = delta / old_value
        else:
            percent_change = float("inf") if delta != 0 else 0.0

        # Determine change type
        if abs(delta) < 0.001:
            change_type = ChangeType.UNCHANGED
        elif (delta > 0) != invert:
            change_type = ChangeType.IMPROVED
        else:
            change_type = ChangeType.DEGRADED

        return MetricChange(
            name=name,
            old_value=old_value,
            new_value=new_value,
            change_type=change_type,
            delta=delta,
            percent_change=percent_change,
        )

    def log(
        self,
        max_entries: int = 10,
        branch: str | None = None,
    ) -> list[QualitySnapshot]:
        """Get commit history.

        Args:
            max_entries: Maximum entries to return
            branch: Branch to show (default: current)

        Returns:
            List of snapshots, newest first
        """
        branch = branch or self._current_branch
        ref_file = self._branches_dir / branch
        if not ref_file.exists():
            return []

        version = ref_file.read_text().strip()
        if not version:
            return []

        history = []
        while version and len(history) < max_entries:
            try:
                snapshot = self._load_snapshot(version)
                history.append(snapshot)
                version = snapshot.parent_version
            except CleanError:
                break

        return history

    def checkout(self, version: str) -> QualitySnapshot:
        """Checkout a specific version.

        Args:
            version: Version to checkout

        Returns:
            The checked-out snapshot
        """
        snapshot = self._load_snapshot(version)
        logger.info(f"Checked out version {version}")
        return snapshot

    def branch(
        self,
        name: str,
        from_version: str | None = None,
        description: str = "",
    ) -> BranchInfo:
        """Create a new branch.

        Args:
            name: Branch name
            from_version: Version to branch from (default: HEAD)
            description: Branch description

        Returns:
            BranchInfo for the new branch
        """
        if (self._branches_dir / name).exists():
            raise CleanError(f"Branch already exists: {name}")

        version = from_version or self._get_head_version() or ""

        ref_file = self._branches_dir / name
        ref_file.write_text(version)

        info = BranchInfo(
            name=name,
            head_version=version,
            created_at=datetime.now(),
            description=description,
        )

        logger.info(f"Created branch {name} at {version}")
        return info

    def switch_branch(self, name: str) -> None:
        """Switch to a different branch.

        Args:
            name: Branch name
        """
        if not (self._branches_dir / name).exists():
            raise CleanError(f"Branch not found: {name}")

        self._current_branch = name
        logger.info(f"Switched to branch {name}")

    def list_branches(self) -> list[BranchInfo]:
        """List all branches.

        Returns:
            List of BranchInfo
        """
        branches = []
        for ref_file in self._branches_dir.iterdir():
            version = ref_file.read_text().strip()
            branches.append(
                BranchInfo(
                    name=ref_file.name,
                    head_version=version,
                    created_at=datetime.fromtimestamp(ref_file.stat().st_mtime),
                )
            )
        return branches

    def tag(
        self,
        name: str,
        version: str | None = None,
        message: str = "",
    ) -> None:
        """Create a tag for a version.

        Args:
            name: Tag name
            version: Version to tag (default: HEAD)
            message: Tag message
        """
        tags_dir = self._refs_dir / "tags"
        tags_dir.mkdir(exist_ok=True)

        version = version or self._get_head_version()
        if version is None:
            raise CleanError("No version to tag")

        tag_file = tags_dir / name
        tag_file.write_text(json.dumps({"version": version, "message": message}))
        logger.info(f"Created tag {name} at {version}")

    def list_tags(self) -> dict[str, str]:
        """List all tags.

        Returns:
            Dict of tag name to version
        """
        tags_dir = self._refs_dir / "tags"
        if not tags_dir.exists():
            return {}

        tags = {}
        for tag_file in tags_dir.iterdir():
            data = json.loads(tag_file.read_text())
            tags[tag_file.name] = data["version"]
        return tags

    def export_history(
        self,
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """Export version history.

        Args:
            output_path: Output file path
            format: Export format ("json" or "csv")
        """
        history = self.log(max_entries=1000)

        if format == "json":
            data = [s.to_dict() for s in history]
            Path(output_path).write_text(json.dumps(data, indent=2))
        elif format == "csv":
            rows = []
            for s in history:
                row = {
                    "version": s.version,
                    "timestamp": s.timestamp.isoformat(),
                    "message": s.message,
                    "overall_score": s.overall_score,
                    "sample_count": s.sample_count,
                }
                row.update(s.metrics)
                row.update({f"issues_{k}": v for k, v in s.issue_counts.items()})
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_path, index=False)
        else:
            raise CleanError(f"Unknown format: {format}")

        logger.info(f"Exported history to {output_path}")

    def generate_trend_report(
        self,
        n_versions: int = 10,
    ) -> dict[str, Any]:
        """Generate trend analysis report.

        Args:
            n_versions: Number of versions to analyze

        Returns:
            Trend analysis data
        """
        history = self.log(max_entries=n_versions)
        if len(history) < 2:
            return {"error": "Not enough history for trend analysis"}

        # Reverse to chronological order
        history = list(reversed(history))

        # Calculate trends
        scores = [s.overall_score for s in history]
        timestamps = [s.timestamp for s in history]

        trend = {
            "score_trend": {
                "current": scores[-1],
                "previous": scores[-2],
                "min": min(scores),
                "max": max(scores),
                "mean": sum(scores) / len(scores),
                "direction": "improving" if scores[-1] > scores[0] else "declining",
            },
            "versions_analyzed": len(history),
            "time_span": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
            },
        }

        # Metric trends
        metric_trends = {}
        for metric in history[0].metrics:
            values = [s.metrics.get(metric, 0) for s in history]
            metric_trends[metric] = {
                "current": values[-1],
                "trend": "improving" if values[-1] > values[0] else "declining",
            }
        trend["metrics"] = metric_trends

        return trend


def create_version_control(
    storage_path: str | Path = ".quality_history",
) -> QualityVersionControl:
    """Create version control instance.

    Args:
        storage_path: Path for version storage

    Returns:
        Configured QualityVersionControl
    """
    return QualityVersionControl(storage_path)


__all__ = [
    # Core classes
    "QualityVersionControl",
    "QualitySnapshot",
    "QualityDiff",
    "MetricChange",
    "BranchInfo",
    # Enums
    "ChangeType",
    # Functions
    "create_version_control",
]
