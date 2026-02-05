"""Quality Regression Testing.

This module provides tools for tracking quality metrics over time
and detecting regressions, like unit tests for data quality.

Example:
    >>> from clean.quality_regression import QualityRegressionTester
    >>>
    >>> tester = QualityRegressionTester(baseline_report)
    >>> result = tester.test(current_report)
    >>> if result.has_regression:
    ...     print("Quality regression detected!")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from clean.core.report import QualityReport

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity of quality regression."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    NONE = "none"


class MetricType(Enum):
    """Types of quality metrics to track."""

    QUALITY_SCORE = "quality_score"
    LABEL_ERROR_RATE = "label_error_rate"
    DUPLICATE_RATE = "duplicate_rate"
    OUTLIER_RATE = "outlier_rate"
    MISSING_RATE = "missing_rate"
    CLASS_IMBALANCE = "class_imbalance"
    CUSTOM = "custom"


@dataclass
class MetricThreshold:
    """Threshold configuration for a metric."""

    metric: MetricType
    name: str
    warning_threshold: float
    critical_threshold: float
    direction: str = "lower_is_worse"  # or "higher_is_worse"

    def check(
        self,
        current: float,
        baseline: float,
    ) -> tuple[RegressionSeverity, float]:
        """Check if metric crossed threshold.

        Args:
            current: Current metric value
            baseline: Baseline metric value

        Returns:
            Tuple of (severity, change_amount)
        """
        if self.direction == "lower_is_worse":
            change = baseline - current
        else:
            change = current - baseline

        if change >= self.critical_threshold:
            return RegressionSeverity.CRITICAL, change
        elif change >= self.warning_threshold:
            return RegressionSeverity.WARNING, change
        elif change > 0:
            return RegressionSeverity.INFO, change
        else:
            return RegressionSeverity.NONE, change


@dataclass
class RegressionResult:
    """Result of a single metric regression check."""

    metric_name: str
    baseline_value: float
    current_value: float
    change: float
    change_percent: float
    severity: RegressionSeverity
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change": self.change,
            "change_percent": self.change_percent,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass
class QualityTestResult:
    """Complete result of quality regression testing."""

    timestamp: datetime
    baseline_id: str
    current_id: str

    has_regression: bool
    overall_passed: bool
    exit_code: int  # 0 = pass, 1 = warning, 2 = critical

    results: list[RegressionResult]
    critical_regressions: list[str]
    warnings: list[str]

    baseline_summary: dict[str, float]
    current_summary: dict[str, float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "❌ FAILED" if not self.overall_passed else "✅ PASSED"

        lines = [
            "Quality Regression Test Result",
            "=" * 50,
            "",
            f"Status: {status}",
            f"Baseline: {self.baseline_id}",
            f"Current: {self.current_id}",
            "",
        ]

        # Show regressions
        if self.critical_regressions:
            lines.append("Critical Regressions:")
            for msg in self.critical_regressions:
                lines.append(f"  🔴 {msg}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for msg in self.warnings:
                lines.append(f"  🟡 {msg}")
            lines.append("")

        # Show all metrics
        lines.append("Metrics:")
        for result in self.results:
            icon = "✓" if result.passed else "✗"
            lines.append(
                f"  {icon} {result.metric_name}: "
                f"{result.baseline_value:.2f} → {result.current_value:.2f} "
                f"({result.change_percent:+.1f}%)"
            )

        return "\n".join(lines)

    def to_github_output(self) -> str:
        """Generate GitHub Actions output format."""
        lines = []

        for result in self.results:
            if result.severity == RegressionSeverity.CRITICAL:
                lines.append(f"::error::{result.message}")
            elif result.severity == RegressionSeverity.WARNING:
                lines.append(f"::warning::{result.message}")

        lines.append(f"::set-output name=passed::{str(self.overall_passed).lower()}")
        lines.append(f"::set-output name=exit_code::{self.exit_code}")

        return "\n".join(lines)


@dataclass
class QualitySnapshot:
    """Snapshot of quality metrics at a point in time."""

    id: str
    timestamp: datetime
    dataset_name: str
    n_samples: int
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualitySnapshot:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            dataset_name=data["dataset_name"],
            n_samples=data["n_samples"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_report(
        cls,
        report: QualityReport,
        dataset_name: str = "default",
        snapshot_id: str | None = None,
    ) -> QualitySnapshot:
        """Create snapshot from quality report."""
        import hashlib

        # Extract metrics from report
        metrics = {
            "quality_score": report.quality_score,
            "n_issues": report.total_issues,
        }

        # Add specific issue rates
        if hasattr(report, "label_errors") and report.label_errors() is not None:
            metrics["label_error_count"] = len(report.label_errors())
            metrics["label_error_rate"] = len(report.label_errors()) / report.n_samples

        if hasattr(report, "duplicates") and report.duplicates() is not None:
            metrics["duplicate_count"] = len(report.duplicates())
            metrics["duplicate_rate"] = len(report.duplicates()) / report.n_samples

        if hasattr(report, "outliers") and report.outliers() is not None:
            metrics["outlier_count"] = len(report.outliers())
            metrics["outlier_rate"] = len(report.outliers()) / report.n_samples

        # Generate ID if not provided
        if snapshot_id is None:
            snapshot_id = hashlib.sha256(
                f"{dataset_name}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

        return cls(
            id=snapshot_id,
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            n_samples=report.n_samples,
            metrics=metrics,
        )


class QualityHistoryStore:
    """Store for quality history and snapshots."""

    def __init__(self, path: str | Path | None = None):
        """Initialize history store.

        Args:
            path: Path to SQLite database (None for in-memory)
        """
        self.path = Path(path) if path else None

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
        else:
            self._conn = sqlite3.connect(":memory:")

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                n_samples INTEGER NOT NULL,
                metrics TEXT NOT NULL,
                metadata TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                baseline_id TEXT NOT NULL,
                current_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                result_json TEXT NOT NULL
            )
        """)

        self._conn.commit()

    def save_snapshot(self, snapshot: QualitySnapshot) -> None:
        """Save a quality snapshot.

        Args:
            snapshot: Snapshot to save
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO snapshots
            (id, timestamp, dataset_name, n_samples, metrics, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.id,
                snapshot.timestamp.isoformat(),
                snapshot.dataset_name,
                snapshot.n_samples,
                json.dumps(snapshot.metrics),
                json.dumps(snapshot.metadata),
            ),
        )
        self._conn.commit()

    def get_snapshot(self, snapshot_id: str) -> QualitySnapshot | None:
        """Get a snapshot by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            QualitySnapshot or None
        """
        cursor = self._conn.execute(
            "SELECT * FROM snapshots WHERE id = ?",
            (snapshot_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return QualitySnapshot(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            dataset_name=row[2],
            n_samples=row[3],
            metrics=json.loads(row[4]),
            metadata=json.loads(row[5]) if row[5] else {},
        )

    def get_latest_snapshot(self, dataset_name: str) -> QualitySnapshot | None:
        """Get the latest snapshot for a dataset.

        Args:
            dataset_name: Dataset name

        Returns:
            Latest QualitySnapshot or None
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM snapshots
            WHERE dataset_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (dataset_name,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return QualitySnapshot(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            dataset_name=row[2],
            n_samples=row[3],
            metrics=json.loads(row[4]),
            metadata=json.loads(row[5]) if row[5] else {},
        )

    def get_history(
        self,
        dataset_name: str,
        limit: int = 100,
    ) -> list[QualitySnapshot]:
        """Get snapshot history for a dataset.

        Args:
            dataset_name: Dataset name
            limit: Maximum snapshots to return

        Returns:
            List of QualitySnapshot objects
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM snapshots
            WHERE dataset_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (dataset_name, limit),
        )

        snapshots = []
        for row in cursor.fetchall():
            snapshots.append(QualitySnapshot(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                dataset_name=row[2],
                n_samples=row[3],
                metrics=json.loads(row[4]),
                metadata=json.loads(row[5]) if row[5] else {},
            ))

        return snapshots

    def save_test_result(self, result: QualityTestResult) -> None:
        """Save a test result.

        Args:
            result: Test result to save
        """
        result_dict = {
            "baseline_id": result.baseline_id,
            "current_id": result.current_id,
            "has_regression": result.has_regression,
            "overall_passed": result.overall_passed,
            "exit_code": result.exit_code,
            "results": [r.to_dict() for r in result.results],
            "critical_regressions": result.critical_regressions,
            "warnings": result.warnings,
        }

        self._conn.execute(
            """
            INSERT INTO test_results
            (timestamp, baseline_id, current_id, passed, result_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result.timestamp.isoformat(),
                result.baseline_id,
                result.current_id,
                1 if result.overall_passed else 0,
                json.dumps(result_dict),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


@dataclass
class RegressionTestConfig:
    """Configuration for quality regression testing."""

    # Thresholds
    quality_score_warning: float = 5.0  # Warn if drops by 5 points
    quality_score_critical: float = 10.0  # Critical if drops by 10 points

    label_error_rate_warning: float = 0.01  # Warn if increases by 1%
    label_error_rate_critical: float = 0.05  # Critical if increases by 5%

    duplicate_rate_warning: float = 0.02
    duplicate_rate_critical: float = 0.05

    outlier_rate_warning: float = 0.02
    outlier_rate_critical: float = 0.05

    # Behavior
    fail_on_warning: bool = False
    fail_on_critical: bool = True

    # Statistical
    use_rolling_baseline: bool = False
    rolling_window: int = 5

    def get_thresholds(self) -> list[MetricThreshold]:
        """Get metric thresholds."""
        return [
            MetricThreshold(
                metric=MetricType.QUALITY_SCORE,
                name="quality_score",
                warning_threshold=self.quality_score_warning,
                critical_threshold=self.quality_score_critical,
                direction="lower_is_worse",
            ),
            MetricThreshold(
                metric=MetricType.LABEL_ERROR_RATE,
                name="label_error_rate",
                warning_threshold=self.label_error_rate_warning,
                critical_threshold=self.label_error_rate_critical,
                direction="higher_is_worse",
            ),
            MetricThreshold(
                metric=MetricType.DUPLICATE_RATE,
                name="duplicate_rate",
                warning_threshold=self.duplicate_rate_warning,
                critical_threshold=self.duplicate_rate_critical,
                direction="higher_is_worse",
            ),
            MetricThreshold(
                metric=MetricType.OUTLIER_RATE,
                name="outlier_rate",
                warning_threshold=self.outlier_rate_warning,
                critical_threshold=self.outlier_rate_critical,
                direction="higher_is_worse",
            ),
        ]


class QualityRegressionTester:
    """Test for quality regressions between snapshots.

    Like unit tests for data quality - detects when metrics degrade.
    """

    def __init__(
        self,
        baseline: QualitySnapshot | QualityReport | None = None,
        config: RegressionTestConfig | None = None,
        store: QualityHistoryStore | None = None,
    ):
        """Initialize regression tester.

        Args:
            baseline: Baseline snapshot or report
            config: Test configuration
            store: History store for persistence
        """
        self.config = config or RegressionTestConfig()
        self.store = store
        self.thresholds = self.config.get_thresholds()

        if baseline is not None:
            if hasattr(baseline, "quality_score"):
                self._baseline = QualitySnapshot.from_report(baseline)
            else:
                self._baseline = baseline
        else:
            self._baseline = None

    def set_baseline(
        self,
        baseline: QualitySnapshot | QualityReport,
    ) -> None:
        """Set baseline for comparison.

        Args:
            baseline: Baseline snapshot or report
        """
        if hasattr(baseline, "quality_score"):
            self._baseline = QualitySnapshot.from_report(baseline)
        else:
            self._baseline = baseline

        if self.store:
            self.store.save_snapshot(self._baseline)

    def test(
        self,
        current: QualitySnapshot | QualityReport,
        dataset_name: str = "default",
    ) -> QualityTestResult:
        """Test for quality regressions.

        Args:
            current: Current snapshot or report
            dataset_name: Dataset name (for baseline lookup)

        Returns:
            QualityTestResult
        """
        # Convert to snapshot if needed
        if hasattr(current, "quality_score"):
            current_snapshot = QualitySnapshot.from_report(current, dataset_name)
        else:
            current_snapshot = current

        # Get baseline
        baseline = self._baseline
        if baseline is None and self.store:
            baseline = self.store.get_latest_snapshot(dataset_name)

        if baseline is None:
            # No baseline - this becomes the baseline
            if self.store:
                self.store.save_snapshot(current_snapshot)

            return QualityTestResult(
                timestamp=datetime.now(),
                baseline_id="none",
                current_id=current_snapshot.id,
                has_regression=False,
                overall_passed=True,
                exit_code=0,
                results=[],
                critical_regressions=[],
                warnings=["No baseline found - using current as baseline"],
                baseline_summary={},
                current_summary=current_snapshot.metrics,
            )

        # Run regression tests
        results = []
        critical_regressions = []
        warnings = []

        for threshold in self.thresholds:
            metric_name = threshold.name

            baseline_value = baseline.metrics.get(metric_name, 0)
            current_value = current_snapshot.metrics.get(metric_name, 0)

            severity, change = threshold.check(current_value, baseline_value)

            # Calculate percentage change
            if baseline_value != 0:
                change_percent = (current_value - baseline_value) / abs(baseline_value) * 100
            else:
                change_percent = 0 if current_value == 0 else 100

            # Determine if passed
            if severity == RegressionSeverity.CRITICAL:
                passed = not self.config.fail_on_critical
                message = f"CRITICAL: {metric_name} regressed from {baseline_value:.3f} to {current_value:.3f}"
                critical_regressions.append(message)
            elif severity == RegressionSeverity.WARNING:
                passed = not self.config.fail_on_warning
                message = f"WARNING: {metric_name} regressed from {baseline_value:.3f} to {current_value:.3f}"
                warnings.append(message)
            else:
                passed = True
                message = f"{metric_name}: {baseline_value:.3f} → {current_value:.3f}"

            results.append(RegressionResult(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                change=change,
                change_percent=change_percent,
                severity=severity,
                passed=passed,
                message=message,
            ))

        # Overall result
        has_regression = any(
            r.severity in (RegressionSeverity.CRITICAL, RegressionSeverity.WARNING)
            for r in results
        )

        overall_passed = all(r.passed for r in results)

        if any(r.severity == RegressionSeverity.CRITICAL for r in results):
            exit_code = 2
        elif any(r.severity == RegressionSeverity.WARNING for r in results):
            exit_code = 1 if self.config.fail_on_warning else 0
        else:
            exit_code = 0

        result = QualityTestResult(
            timestamp=datetime.now(),
            baseline_id=baseline.id,
            current_id=current_snapshot.id,
            has_regression=has_regression,
            overall_passed=overall_passed,
            exit_code=exit_code,
            results=results,
            critical_regressions=critical_regressions,
            warnings=warnings,
            baseline_summary=baseline.metrics,
            current_summary=current_snapshot.metrics,
        )

        # Save results
        if self.store:
            self.store.save_snapshot(current_snapshot)
            self.store.save_test_result(result)

        return result

    def add_threshold(self, threshold: MetricThreshold) -> None:
        """Add custom metric threshold.

        Args:
            threshold: Threshold to add
        """
        self.thresholds.append(threshold)


def run_quality_test(
    current: QualityReport,
    baseline: QualityReport | QualitySnapshot | None = None,
    config: RegressionTestConfig | None = None,
) -> QualityTestResult:
    """Convenience function to run quality regression test.

    Args:
        current: Current quality report
        baseline: Baseline for comparison
        config: Test configuration

    Returns:
        QualityTestResult
    """
    tester = QualityRegressionTester(baseline=baseline, config=config)
    return tester.test(current)


def create_quality_tester(
    store_path: str | Path | None = None,
    config: RegressionTestConfig | None = None,
) -> QualityRegressionTester:
    """Create quality regression tester with persistence.

    Args:
        store_path: Path to store quality history
        config: Test configuration

    Returns:
        QualityRegressionTester
    """
    store = QualityHistoryStore(store_path) if store_path else None
    return QualityRegressionTester(config=config, store=store)


class QualityGate:
    """Quality gate for CI/CD pipelines.

    Fails builds when quality regresses beyond thresholds.
    """

    def __init__(
        self,
        tester: QualityRegressionTester,
        output_format: str = "console",  # console, github, json
    ):
        """Initialize quality gate.

        Args:
            tester: Regression tester
            output_format: Output format
        """
        self.tester = tester
        self.output_format = output_format

    def check(
        self,
        report: QualityReport,
        dataset_name: str = "default",
    ) -> int:
        """Check quality gate.

        Args:
            report: Quality report to check
            dataset_name: Dataset name

        Returns:
            Exit code (0=pass, 1=warning, 2=critical)
        """
        result = self.tester.test(report, dataset_name)

        # Output result
        if self.output_format == "github":
            print(result.to_github_output())
        elif self.output_format == "json":
            print(json.dumps(result.to_dict() if hasattr(result, "to_dict") else {
                "passed": result.overall_passed,
                "exit_code": result.exit_code,
            }))
        else:
            print(result.summary())

        return result.exit_code
