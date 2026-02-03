"""Version-Aware Quality Tracking.

This module provides Git-like quality history tracking, enabling comparison
of data quality across different versions, datasets, or time periods.

Example:
    >>> from clean.quality_tracker import QualityTracker, track_quality
    >>>
    >>> # Initialize tracker
    >>> tracker = QualityTracker(storage_path=".clean_history")
    >>>
    >>> # Track quality over time
    >>> report = analyze(df, labels=labels)
    >>> tracker.record(report, version="v1.0", tags=["production"])
    >>>
    >>> # Compare versions
    >>> diff = tracker.compare("v1.0", "v2.0")
    >>> print(diff.summary())
    >>>
    >>> # View history
    >>> history = tracker.history(limit=10)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd

from clean.core.report import QualityReport
from clean.exceptions import CleanError

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of quality change."""

    IMPROVED = "improved"
    DEGRADED = "degraded"
    UNCHANGED = "unchanged"
    NEW = "new"
    REMOVED = "removed"


@dataclass
class QualitySnapshot:
    """A point-in-time quality measurement."""

    id: str
    timestamp: str
    version: str | None
    data_hash: str
    quality_score: float
    n_samples: int
    n_label_errors: int
    n_duplicates: int
    n_outliers: int
    label_quality: float
    uniqueness: float
    consistency: float
    completeness: float
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "version": self.version,
            "data_hash": self.data_hash,
            "quality_score": self.quality_score,
            "n_samples": self.n_samples,
            "n_label_errors": self.n_label_errors,
            "n_duplicates": self.n_duplicates,
            "n_outliers": self.n_outliers,
            "label_quality": self.label_quality,
            "uniqueness": self.uniqueness,
            "consistency": self.consistency,
            "completeness": self.completeness,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualitySnapshot:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            version=data.get("version"),
            data_hash=data["data_hash"],
            quality_score=data["quality_score"],
            n_samples=data["n_samples"],
            n_label_errors=data.get("n_label_errors", 0),
            n_duplicates=data.get("n_duplicates", 0),
            n_outliers=data.get("n_outliers", 0),
            label_quality=data.get("label_quality", 0),
            uniqueness=data.get("uniqueness", 0),
            consistency=data.get("consistency", 0),
            completeness=data.get("completeness", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_report(
        cls,
        report: QualityReport,
        version: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QualitySnapshot:
        """Create snapshot from a QualityReport."""
        # Generate unique ID
        timestamp = datetime.now().isoformat()
        id_content = f"{timestamp}_{report.dataset_info.n_samples}_{version or ''}"
        snapshot_id = hashlib.md5(id_content.encode()).hexdigest()[:12]

        # Calculate data hash from report info
        data_hash = hashlib.md5(
            f"{report.dataset_info.n_samples}_{report.dataset_info.n_features}".encode()
        ).hexdigest()[:12]

        # Extract issue counts
        def get_count(result: Any) -> int:
            if result is None:
                return 0
            if hasattr(result, "issues"):
                return len(result.issues)
            return 0

        # Extract quality scores
        score = report.quality_score
        overall = score.overall if hasattr(score, "overall") else float(score)
        label_quality = getattr(score, "label_quality", overall)
        uniqueness = getattr(score, "uniqueness", overall)
        consistency = getattr(score, "consistency", overall)
        completeness = getattr(score, "completeness", overall)

        return cls(
            id=snapshot_id,
            timestamp=timestamp,
            version=version,
            data_hash=data_hash,
            quality_score=overall,
            n_samples=report.dataset_info.n_samples,
            n_label_errors=get_count(report.label_errors_result),
            n_duplicates=get_count(report.duplicates_result),
            n_outliers=get_count(report.outliers_result),
            label_quality=label_quality,
            uniqueness=uniqueness,
            consistency=consistency,
            completeness=completeness,
            tags=tags or [],
            metadata=metadata or {},
        )


@dataclass
class MetricChange:
    """Change in a single metric between versions."""

    metric: str
    old_value: float
    new_value: float
    change: float
    change_pct: float
    change_type: ChangeType

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change": self.change,
            "change_pct": self.change_pct,
            "change_type": self.change_type.value,
        }


@dataclass
class QualityDiff:
    """Difference between two quality snapshots."""

    from_snapshot: QualitySnapshot
    to_snapshot: QualitySnapshot
    changes: list[MetricChange]
    overall_change: ChangeType

    def summary(self) -> str:
        """Generate text summary of changes."""
        lines = [
            f"Quality Comparison: {self.from_snapshot.version or self.from_snapshot.id[:8]} → "
            f"{self.to_snapshot.version or self.to_snapshot.id[:8]}",
            "=" * 60,
            "",
        ]

        # Overall status
        emoji = {"improved": "✅", "degraded": "⚠️", "unchanged": "➖"}
        lines.append(
            f"Overall: {emoji.get(self.overall_change.value, '•')} {self.overall_change.value.upper()}"
        )
        lines.append("")

        # Score change
        score_change = self.to_snapshot.quality_score - self.from_snapshot.quality_score
        lines.append(
            f"Quality Score: {self.from_snapshot.quality_score:.1f} → "
            f"{self.to_snapshot.quality_score:.1f} ({score_change:+.1f})"
        )
        lines.append("")

        # Significant changes
        significant = [c for c in self.changes if abs(c.change_pct) > 5]
        if significant:
            lines.append("Significant Changes:")
            for change in sorted(significant, key=lambda x: abs(x.change_pct), reverse=True):
                direction = "↑" if change.change > 0 else "↓"
                lines.append(
                    f"  {direction} {change.metric}: {change.old_value:.1f} → "
                    f"{change.new_value:.1f} ({change.change_pct:+.1f}%)"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from": self.from_snapshot.to_dict(),
            "to": self.to_snapshot.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
            "overall_change": self.overall_change.value,
        }


@dataclass
class QualityTrend:
    """Quality trend over time."""

    metric: str
    timestamps: list[str]
    values: list[float]
    trend_direction: Literal["up", "down", "stable"]
    trend_strength: float  # 0-1, how consistent the trend is

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "timestamp": pd.to_datetime(self.timestamps),
            self.metric: self.values,
        })


class QualityTracker:
    """Track quality metrics over time with version support.

    Example:
        >>> tracker = QualityTracker(".clean_history")
        >>> tracker.record(report, version="v1.0")
        >>> history = tracker.history(limit=10)
    """

    def __init__(
        self,
        storage_path: str | Path = ".clean_history",
        dataset_name: str = "default",
    ):
        """Initialize tracker.

        Args:
            storage_path: Path to store tracking database
            dataset_name: Name of dataset being tracked
        """
        self.storage_path = Path(storage_path)
        self.dataset_name = dataset_name
        self._db_path = self.storage_path / f"{dataset_name}_quality.db"
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize SQLite storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    version TEXT,
                    data_hash TEXT,
                    quality_score REAL,
                    n_samples INTEGER,
                    n_label_errors INTEGER,
                    n_duplicates INTEGER,
                    n_outliers INTEGER,
                    label_quality REAL,
                    uniqueness REAL,
                    consistency REAL,
                    completeness REAL,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON snapshots(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_version ON snapshots(version)
            """)
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        report: QualityReport,
        version: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QualitySnapshot:
        """Record a quality report snapshot.

        Args:
            report: Quality report to record
            version: Optional version identifier
            tags: Optional tags for filtering
            metadata: Optional metadata

        Returns:
            The created snapshot
        """
        snapshot = QualitySnapshot.from_report(
            report, version=version, tags=tags, metadata=metadata
        )

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                INSERT INTO snapshots (
                    id, timestamp, version, data_hash, quality_score,
                    n_samples, n_label_errors, n_duplicates, n_outliers,
                    label_quality, uniqueness, consistency, completeness,
                    tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.id,
                    snapshot.timestamp,
                    snapshot.version,
                    snapshot.data_hash,
                    snapshot.quality_score,
                    snapshot.n_samples,
                    snapshot.n_label_errors,
                    snapshot.n_duplicates,
                    snapshot.n_outliers,
                    snapshot.label_quality,
                    snapshot.uniqueness,
                    snapshot.consistency,
                    snapshot.completeness,
                    json.dumps(snapshot.tags),
                    json.dumps(snapshot.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Recorded quality snapshot {snapshot.id} (version={version})")
        return snapshot

    def get(self, id_or_version: str) -> QualitySnapshot | None:
        """Get a snapshot by ID or version.

        Args:
            id_or_version: Snapshot ID or version string

        Returns:
            Snapshot if found, None otherwise
        """
        conn = sqlite3.connect(self._db_path)
        try:
            # Try by ID first
            cursor = conn.execute(
                "SELECT * FROM snapshots WHERE id = ?", (id_or_version,)
            )
            row = cursor.fetchone()

            # Try by version
            if row is None:
                cursor = conn.execute(
                    "SELECT * FROM snapshots WHERE version = ? ORDER BY timestamp DESC LIMIT 1",
                    (id_or_version,),
                )
                row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_snapshot(row)
        finally:
            conn.close()

    def history(
        self,
        limit: int = 100,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        tags: list[str] | None = None,
    ) -> list[QualitySnapshot]:
        """Get quality history.

        Args:
            limit: Maximum snapshots to return
            since: Start timestamp
            until: End timestamp
            tags: Filter by tags

        Returns:
            List of snapshots, newest first
        """
        conn = sqlite3.connect(self._db_path)
        try:
            query = "SELECT * FROM snapshots WHERE 1=1"
            params: list[Any] = []

            if since:
                if isinstance(since, datetime):
                    since = since.isoformat()
                query += " AND timestamp >= ?"
                params.append(since)

            if until:
                if isinstance(until, datetime):
                    until = until.isoformat()
                query += " AND timestamp <= ?"
                params.append(until)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            snapshots = [self._row_to_snapshot(row) for row in cursor.fetchall()]

            # Filter by tags if specified
            if tags:
                snapshots = [
                    s for s in snapshots if any(t in s.tags for t in tags)
                ]

            return snapshots
        finally:
            conn.close()

    def compare(
        self,
        from_id: str,
        to_id: str,
    ) -> QualityDiff:
        """Compare two snapshots.

        Args:
            from_id: ID or version of first snapshot
            to_id: ID or version of second snapshot

        Returns:
            QualityDiff with changes
        """
        from_snap = self.get(from_id)
        to_snap = self.get(to_id)

        if from_snap is None:
            raise CleanError(f"Snapshot not found: {from_id}")
        if to_snap is None:
            raise CleanError(f"Snapshot not found: {to_id}")

        return self._compare_snapshots(from_snap, to_snap)

    def _compare_snapshots(
        self, from_snap: QualitySnapshot, to_snap: QualitySnapshot
    ) -> QualityDiff:
        """Compare two snapshots and compute changes."""
        metrics = [
            ("quality_score", from_snap.quality_score, to_snap.quality_score),
            ("label_quality", from_snap.label_quality, to_snap.label_quality),
            ("uniqueness", from_snap.uniqueness, to_snap.uniqueness),
            ("consistency", from_snap.consistency, to_snap.consistency),
            ("completeness", from_snap.completeness, to_snap.completeness),
            ("n_label_errors", from_snap.n_label_errors, to_snap.n_label_errors),
            ("n_duplicates", from_snap.n_duplicates, to_snap.n_duplicates),
            ("n_outliers", from_snap.n_outliers, to_snap.n_outliers),
        ]

        changes = []
        for metric, old_val, new_val in metrics:
            change = new_val - old_val
            change_pct = (change / old_val * 100) if old_val != 0 else 0

            # Determine change type
            if abs(change_pct) < 1:
                change_type = ChangeType.UNCHANGED
            elif metric in ("n_label_errors", "n_duplicates", "n_outliers"):
                # Lower is better for issue counts
                change_type = ChangeType.IMPROVED if change < 0 else ChangeType.DEGRADED
            else:
                # Higher is better for scores
                change_type = ChangeType.IMPROVED if change > 0 else ChangeType.DEGRADED

            changes.append(MetricChange(
                metric=metric,
                old_value=old_val,
                new_value=new_val,
                change=change,
                change_pct=change_pct,
                change_type=change_type,
            ))

        # Determine overall change
        score_change = to_snap.quality_score - from_snap.quality_score
        if score_change > 1:
            overall = ChangeType.IMPROVED
        elif score_change < -1:
            overall = ChangeType.DEGRADED
        else:
            overall = ChangeType.UNCHANGED

        return QualityDiff(
            from_snapshot=from_snap,
            to_snapshot=to_snap,
            changes=changes,
            overall_change=overall,
        )

    def trend(
        self,
        metric: str = "quality_score",
        limit: int = 30,
    ) -> QualityTrend:
        """Analyze trend for a metric over time.

        Args:
            metric: Metric to analyze
            limit: Number of snapshots to consider

        Returns:
            QualityTrend analysis
        """
        history = self.history(limit=limit)
        if len(history) < 2:
            return QualityTrend(
                metric=metric,
                timestamps=[],
                values=[],
                trend_direction="stable",
                trend_strength=0.0,
            )

        timestamps = [s.timestamp for s in reversed(history)]
        values = [getattr(s, metric, 0) for s in reversed(history)]

        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        # Normalize slope to determine direction and strength
        value_range = max(values) - min(values) if max(values) != min(values) else 1
        normalized_slope = slope / value_range * len(values)

        if normalized_slope > 0.1:
            direction = "up"
        elif normalized_slope < -0.1:
            direction = "down"
        else:
            direction = "stable"

        # Calculate R² as trend strength
        y_pred = np.polyval([slope, np.mean(values) - slope * np.mean(x)], x)
        ss_res = np.sum((np.array(values) - y_pred) ** 2)
        ss_tot = np.sum((np.array(values) - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return QualityTrend(
            metric=metric,
            timestamps=timestamps,
            values=values,
            trend_direction=direction,
            trend_strength=max(0, min(1, r_squared)),
        )

    def versions(self) -> list[str]:
        """Get all recorded versions.

        Returns:
            List of version strings
        """
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT DISTINCT version FROM snapshots WHERE version IS NOT NULL ORDER BY timestamp DESC"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def latest(self) -> QualitySnapshot | None:
        """Get the most recent snapshot.

        Returns:
            Latest snapshot or None
        """
        history = self.history(limit=1)
        return history[0] if history else None

    def delete(self, id_or_version: str) -> bool:
        """Delete a snapshot.

        Args:
            id_or_version: ID or version to delete

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM snapshots WHERE id = ? OR version = ?",
                (id_or_version, id_or_version),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def clear(self) -> int:
        """Clear all snapshots.

        Returns:
            Number of snapshots deleted
        """
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("DELETE FROM snapshots")
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Cleared {count} snapshots")
            return count
        finally:
            conn.close()

    def export(self, path: str | Path, format: str = "json") -> None:
        """Export tracking data.

        Args:
            path: Output path
            format: 'json' or 'csv'
        """
        history = self.history(limit=10000)
        path = Path(path)

        if format == "json":
            data = [s.to_dict() for s in history]
            path.write_text(json.dumps(data, indent=2))
        elif format == "csv":
            df = pd.DataFrame([s.to_dict() for s in history])
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported {len(history)} snapshots to {path}")

    def _row_to_snapshot(self, row: tuple) -> QualitySnapshot:
        """Convert database row to snapshot."""
        return QualitySnapshot(
            id=row[0],
            timestamp=row[1],
            version=row[2],
            data_hash=row[3],
            quality_score=row[4],
            n_samples=row[5],
            n_label_errors=row[6],
            n_duplicates=row[7],
            n_outliers=row[8],
            label_quality=row[9],
            uniqueness=row[10],
            consistency=row[11],
            completeness=row[12],
            tags=json.loads(row[13]) if row[13] else [],
            metadata=json.loads(row[14]) if row[14] else {},
        )


def track_quality(
    report: QualityReport,
    version: str | None = None,
    storage_path: str = ".clean_history",
    dataset_name: str = "default",
    tags: list[str] | None = None,
) -> QualitySnapshot:
    """Convenience function to track a quality report.

    Args:
        report: Quality report to track
        version: Optional version identifier
        storage_path: Storage location
        dataset_name: Dataset name
        tags: Optional tags

    Returns:
        Created snapshot

    Example:
        >>> report = analyze(df, labels=labels)
        >>> snapshot = track_quality(report, version="v1.0")
    """
    tracker = QualityTracker(storage_path=storage_path, dataset_name=dataset_name)
    return tracker.record(report, version=version, tags=tags)


__all__ = [
    "QualityTracker",
    "QualitySnapshot",
    "QualityDiff",
    "QualityTrend",
    "MetricChange",
    "ChangeType",
    "track_quality",
]
