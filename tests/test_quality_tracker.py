"""Tests for quality tracking module."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from clean.quality_tracker import (
    ChangeType,
    MetricChange,
    QualityDiff,
    QualitySnapshot,
    QualityTracker,
    QualityTrend,
    track_quality,
)


class TestQualitySnapshot:
    """Tests for QualitySnapshot dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        snapshot = QualitySnapshot(
            id="abc123",
            timestamp="2024-01-01T00:00:00",
            version="v1.0",
            data_hash="hash123",
            quality_score=85.0,
            n_samples=1000,
            n_label_errors=50,
            n_duplicates=20,
            n_outliers=10,
            label_quality=90.0,
            uniqueness=95.0,
            consistency=80.0,
            completeness=85.0,
            tags=["production"],
        )

        result = snapshot.to_dict()

        assert result["id"] == "abc123"
        assert result["quality_score"] == 85.0
        assert result["tags"] == ["production"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "abc123",
            "timestamp": "2024-01-01T00:00:00",
            "version": "v1.0",
            "data_hash": "hash123",
            "quality_score": 85.0,
            "n_samples": 1000,
            "n_label_errors": 50,
            "n_duplicates": 20,
            "n_outliers": 10,
            "label_quality": 90.0,
            "uniqueness": 95.0,
            "consistency": 80.0,
            "completeness": 85.0,
        }

        snapshot = QualitySnapshot.from_dict(data)

        assert snapshot.id == "abc123"
        assert snapshot.quality_score == 85.0

    def test_from_report(self):
        """Test creation from QualityReport."""
        # Create mock report
        report = MagicMock()
        report.dataset_info.n_samples = 1000
        report.dataset_info.n_features = 10
        report.quality_score.overall = 85.0
        report.quality_score.label_quality = 90.0
        report.quality_score.uniqueness = 95.0
        report.quality_score.consistency = 80.0
        report.quality_score.completeness = 85.0
        report.label_errors_result.issues = [MagicMock()] * 50
        report.duplicates_result.issues = [MagicMock()] * 20
        report.outliers_result.issues = [MagicMock()] * 10

        snapshot = QualitySnapshot.from_report(
            report, version="v1.0", tags=["test"]
        )

        assert snapshot.version == "v1.0"
        assert snapshot.quality_score == 85.0
        assert snapshot.n_label_errors == 50
        assert snapshot.tags == ["test"]


class TestQualityDiff:
    """Tests for QualityDiff dataclass."""

    @pytest.fixture
    def snapshots(self):
        """Create two snapshots for comparison."""
        old = QualitySnapshot(
            id="old123",
            timestamp="2024-01-01T00:00:00",
            version="v1.0",
            data_hash="hash1",
            quality_score=80.0,
            n_samples=1000,
            n_label_errors=100,
            n_duplicates=50,
            n_outliers=20,
            label_quality=85.0,
            uniqueness=90.0,
            consistency=75.0,
            completeness=80.0,
        )
        new = QualitySnapshot(
            id="new456",
            timestamp="2024-01-02T00:00:00",
            version="v2.0",
            data_hash="hash2",
            quality_score=90.0,
            n_samples=1000,
            n_label_errors=50,
            n_duplicates=30,
            n_outliers=10,
            label_quality=92.0,
            uniqueness=95.0,
            consistency=85.0,
            completeness=90.0,
        )
        return old, new

    def test_summary(self, snapshots):
        """Test summary generation."""
        old, new = snapshots
        diff = QualityDiff(
            from_snapshot=old,
            to_snapshot=new,
            changes=[
                MetricChange(
                    metric="quality_score",
                    old_value=80.0,
                    new_value=90.0,
                    change=10.0,
                    change_pct=12.5,
                    change_type=ChangeType.IMPROVED,
                ),
            ],
            overall_change=ChangeType.IMPROVED,
        )

        summary = diff.summary()

        assert "v1.0" in summary
        assert "v2.0" in summary
        assert "IMPROVED" in summary

    def test_to_dict(self, snapshots):
        """Test conversion to dictionary."""
        old, new = snapshots
        diff = QualityDiff(
            from_snapshot=old,
            to_snapshot=new,
            changes=[],
            overall_change=ChangeType.IMPROVED,
        )

        result = diff.to_dict()

        assert "from" in result
        assert "to" in result
        assert result["overall_change"] == "improved"


class TestQualityTrend:
    """Tests for QualityTrend dataclass."""

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        trend = QualityTrend(
            metric="quality_score",
            timestamps=["2024-01-01", "2024-01-02", "2024-01-03"],
            values=[80.0, 85.0, 90.0],
            trend_direction="up",
            trend_strength=0.95,
        )

        df = trend.to_dataframe()

        assert len(df) == 3
        assert "quality_score" in df.columns


class TestQualityTracker:
    """Tests for QualityTracker class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tracker(self, temp_storage):
        """Create tracker with temp storage."""
        return QualityTracker(storage_path=temp_storage)

    @pytest.fixture
    def mock_report(self):
        """Create mock quality report."""
        report = MagicMock()
        report.dataset_info.n_samples = 1000
        report.dataset_info.n_features = 10
        report.quality_score.overall = 85.0
        report.quality_score.label_quality = 90.0
        report.quality_score.uniqueness = 95.0
        report.quality_score.consistency = 80.0
        report.quality_score.completeness = 85.0
        report.label_errors_result.issues = [MagicMock()] * 50
        report.duplicates_result.issues = [MagicMock()] * 20
        report.outliers_result.issues = [MagicMock()] * 10
        return report

    def test_record_creates_snapshot(self, tracker, mock_report):
        """Test recording a snapshot."""
        snapshot = tracker.record(mock_report, version="v1.0")

        assert snapshot.version == "v1.0"
        assert snapshot.quality_score == 85.0

    def test_get_by_id(self, tracker, mock_report):
        """Test getting snapshot by ID."""
        original = tracker.record(mock_report)

        retrieved = tracker.get(original.id)

        assert retrieved is not None
        assert retrieved.id == original.id

    def test_get_by_version(self, tracker, mock_report):
        """Test getting snapshot by version."""
        tracker.record(mock_report, version="v1.0")

        retrieved = tracker.get("v1.0")

        assert retrieved is not None
        assert retrieved.version == "v1.0"

    def test_get_nonexistent(self, tracker):
        """Test getting nonexistent snapshot."""
        result = tracker.get("nonexistent")
        assert result is None

    def test_history_returns_snapshots(self, tracker, mock_report):
        """Test getting history."""
        tracker.record(mock_report, version="v1.0")
        tracker.record(mock_report, version="v2.0")
        tracker.record(mock_report, version="v3.0")

        history = tracker.history(limit=10)

        assert len(history) == 3
        # Should be newest first
        assert history[0].version == "v3.0"

    def test_history_with_limit(self, tracker, mock_report):
        """Test history with limit."""
        for i in range(5):
            tracker.record(mock_report, version=f"v{i}")

        history = tracker.history(limit=3)

        assert len(history) == 3

    def test_compare_versions(self, tracker, mock_report):
        """Test comparing versions."""
        tracker.record(mock_report, version="v1.0")

        # Modify mock for v2
        mock_report.quality_score.overall = 90.0
        mock_report.label_errors_result.issues = [MagicMock()] * 30
        tracker.record(mock_report, version="v2.0")

        diff = tracker.compare("v1.0", "v2.0")

        assert diff.from_snapshot.version == "v1.0"
        assert diff.to_snapshot.version == "v2.0"
        assert len(diff.changes) > 0

    def test_trend_analysis(self, tracker, mock_report):
        """Test trend analysis."""
        # Record multiple snapshots with increasing quality
        for i in range(5):
            mock_report.quality_score.overall = 80.0 + i * 2
            tracker.record(mock_report)

        trend = tracker.trend(metric="quality_score")

        assert trend.metric == "quality_score"
        assert len(trend.values) == 5
        assert trend.trend_direction == "up"

    def test_versions_list(self, tracker, mock_report):
        """Test listing versions."""
        tracker.record(mock_report, version="v1.0")
        tracker.record(mock_report, version="v2.0")
        tracker.record(mock_report)  # No version

        versions = tracker.versions()

        assert "v1.0" in versions
        assert "v2.0" in versions
        assert len(versions) == 2

    def test_latest(self, tracker, mock_report):
        """Test getting latest snapshot."""
        tracker.record(mock_report, version="v1.0")
        tracker.record(mock_report, version="v2.0")

        latest = tracker.latest()

        assert latest is not None
        assert latest.version == "v2.0"

    def test_delete(self, tracker, mock_report):
        """Test deleting snapshot."""
        tracker.record(mock_report, version="v1.0")

        result = tracker.delete("v1.0")

        assert result is True
        assert tracker.get("v1.0") is None

    def test_clear(self, tracker, mock_report):
        """Test clearing all snapshots."""
        tracker.record(mock_report, version="v1.0")
        tracker.record(mock_report, version="v2.0")

        count = tracker.clear()

        assert count == 2
        assert len(tracker.history()) == 0

    def test_export_json(self, tracker, mock_report, temp_storage):
        """Test exporting to JSON."""
        tracker.record(mock_report, version="v1.0")

        export_path = temp_storage / "export.json"
        tracker.export(export_path, format="json")

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert len(data) == 1

    def test_export_csv(self, tracker, mock_report, temp_storage):
        """Test exporting to CSV."""
        tracker.record(mock_report, version="v1.0")

        export_path = temp_storage / "export.csv"
        tracker.export(export_path, format="csv")

        assert export_path.exists()
        df = pd.read_csv(export_path)
        assert len(df) == 1


class TestTrackQualityFunction:
    """Tests for track_quality convenience function."""

    def test_track_quality_basic(self):
        """Test basic tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock report
            report = MagicMock()
            report.dataset_info.n_samples = 100
            report.dataset_info.n_features = 5
            report.quality_score.overall = 85.0
            report.quality_score.label_quality = 90.0
            report.quality_score.uniqueness = 95.0
            report.quality_score.consistency = 80.0
            report.quality_score.completeness = 85.0
            report.label_errors_result.issues = []
            report.duplicates_result.issues = []
            report.outliers_result.issues = []

            snapshot = track_quality(
                report,
                version="v1.0",
                storage_path=tmpdir,
            )

            assert snapshot.version == "v1.0"
            assert snapshot.quality_score == 85.0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_empty_history(self, temp_storage):
        """Test with no snapshots."""
        tracker = QualityTracker(storage_path=temp_storage)

        history = tracker.history()

        assert history == []

    def test_latest_empty(self, temp_storage):
        """Test latest with no snapshots."""
        tracker = QualityTracker(storage_path=temp_storage)

        latest = tracker.latest()

        assert latest is None

    def test_trend_insufficient_data(self, temp_storage):
        """Test trend with insufficient data."""
        tracker = QualityTracker(storage_path=temp_storage)

        trend = tracker.trend()

        assert trend.trend_direction == "stable"
        assert trend.trend_strength == 0.0

    def test_compare_nonexistent(self, temp_storage):
        """Test comparing nonexistent versions."""
        tracker = QualityTracker(storage_path=temp_storage)

        with pytest.raises(Exception):  # CleanError
            tracker.compare("v1.0", "v2.0")

    def test_multiple_datasets(self, temp_storage):
        """Test tracking multiple datasets."""
        tracker1 = QualityTracker(storage_path=temp_storage, dataset_name="dataset1")
        tracker2 = QualityTracker(storage_path=temp_storage, dataset_name="dataset2")

        # Create mock report
        report = MagicMock()
        report.dataset_info.n_samples = 100
        report.dataset_info.n_features = 5
        report.quality_score.overall = 85.0
        report.quality_score.label_quality = 90.0
        report.quality_score.uniqueness = 95.0
        report.quality_score.consistency = 80.0
        report.quality_score.completeness = 85.0
        report.label_errors_result.issues = []
        report.duplicates_result.issues = []
        report.outliers_result.issues = []

        tracker1.record(report, version="v1.0")
        tracker2.record(report, version="v1.0")

        assert len(tracker1.history()) == 1
        assert len(tracker2.history()) == 1
