"""Tests for Version Control for Data Quality."""

from __future__ import annotations

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from clean.version_control import (
    QualityVersionControl,
    QualitySnapshot,
    QualityDiff,
    MetricChange,
    BranchInfo,
    ChangeType,
    create_version_control,
)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_exist(self) -> None:
        assert ChangeType.IMPROVED is not None
        assert ChangeType.DEGRADED is not None
        assert ChangeType.UNCHANGED is not None
        assert ChangeType.ADDED is not None
        assert ChangeType.REMOVED is not None


class TestQualitySnapshot:
    """Tests for QualitySnapshot dataclass."""

    def test_snapshot_creation(self) -> None:
        snapshot = QualitySnapshot(
            version="abc123",
            timestamp=datetime.now(),
            message="Initial commit",
            overall_score=0.85,
            metrics={"completeness": 0.9, "consistency": 0.8},
            issue_counts={"label_errors": 10, "duplicates": 5},
            sample_count=1000,
            column_stats={},
        )
        assert snapshot.version == "abc123"
        assert snapshot.overall_score == 0.85
        assert snapshot.metrics["completeness"] == 0.9

    def test_snapshot_to_dict(self) -> None:
        snapshot = QualitySnapshot(
            version="v1",
            timestamp=datetime.now(),
            message="Test",
            overall_score=0.9,
            metrics={},
            issue_counts={},
            sample_count=100,
            column_stats={},
        )
        d = snapshot.to_dict()
        assert "version" in d
        assert "overall_score" in d
        assert d["version"] == "v1"

    def test_snapshot_from_dict(self) -> None:
        data = {
            "version": "xyz",
            "timestamp": datetime.now().isoformat(),
            "message": "Test snapshot",
            "overall_score": 0.75,
            "metrics": {"m1": 0.8},
            "issue_counts": {"errors": 5},
            "sample_count": 50,
            "column_stats": {},
        }
        snapshot = QualitySnapshot.from_dict(data)
        assert snapshot.version == "xyz"
        assert snapshot.overall_score == 0.75


class TestMetricChange:
    """Tests for MetricChange dataclass."""

    def test_change_creation(self) -> None:
        change = MetricChange(
            name="accuracy",
            old_value=0.8,
            new_value=0.85,
            change_type=ChangeType.IMPROVED,
            delta=0.05,
            percent_change=0.0625,
        )
        assert change.name == "accuracy"
        assert change.delta == 0.05

    def test_is_significant(self) -> None:
        significant = MetricChange(
            name="m1", old_value=0.8, new_value=0.9,
            change_type=ChangeType.IMPROVED,
            delta=0.1, percent_change=0.125,
        )
        not_significant = MetricChange(
            name="m2", old_value=0.8, new_value=0.81,
            change_type=ChangeType.IMPROVED,
            delta=0.01, percent_change=0.0125,
        )
        assert significant.is_significant(threshold=0.05)
        assert not not_significant.is_significant(threshold=0.05)


class TestQualityDiff:
    """Tests for QualityDiff dataclass."""

    def test_diff_creation(self) -> None:
        diff = QualityDiff(
            old_version="v1",
            new_version="v2",
            old_timestamp=datetime.now(),
            new_timestamp=datetime.now(),
            score_change=MetricChange("score", 0.8, 0.85, ChangeType.IMPROVED, 0.05, 0.0625),
            metric_changes=[],
            issue_changes={},
            sample_count_change=MetricChange("count", 100, 150, ChangeType.IMPROVED, 50, 0.5),
            column_changes={},
        )
        assert diff.old_version == "v1"
        assert diff.new_version == "v2"

    def test_summary(self) -> None:
        diff = QualityDiff(
            old_version="v1",
            new_version="v2",
            old_timestamp=datetime.now(),
            new_timestamp=datetime.now(),
            score_change=MetricChange("score", 0.8, 0.9, ChangeType.IMPROVED, 0.1, 0.125),
            metric_changes=[],
            issue_changes={},
            sample_count_change=MetricChange("count", 100, 100, ChangeType.UNCHANGED, 0, 0),
            column_changes={},
        )
        summary = diff.summary()
        assert "v1" in summary
        assert "v2" in summary

    def test_has_degradation(self) -> None:
        degraded_diff = QualityDiff(
            old_version="v1",
            new_version="v2",
            old_timestamp=datetime.now(),
            new_timestamp=datetime.now(),
            score_change=MetricChange("score", 0.9, 0.8, ChangeType.DEGRADED, -0.1, -0.11),
            metric_changes=[],
            issue_changes={},
            sample_count_change=MetricChange("count", 100, 100, ChangeType.UNCHANGED, 0, 0),
            column_changes={},
        )
        assert degraded_diff.has_degradation()

    def test_has_improvement(self) -> None:
        improved_diff = QualityDiff(
            old_version="v1",
            new_version="v2",
            old_timestamp=datetime.now(),
            new_timestamp=datetime.now(),
            score_change=MetricChange("score", 0.8, 0.9, ChangeType.IMPROVED, 0.1, 0.125),
            metric_changes=[],
            issue_changes={},
            sample_count_change=MetricChange("count", 100, 100, ChangeType.UNCHANGED, 0, 0),
            column_changes={},
        )
        assert improved_diff.has_improvement()


class TestBranchInfo:
    """Tests for BranchInfo dataclass."""

    def test_branch_creation(self) -> None:
        branch = BranchInfo(
            name="feature-branch",
            head_version="abc123",
            created_at=datetime.now(),
            description="Feature work",
        )
        assert branch.name == "feature-branch"
        assert branch.head_version == "abc123"


class TestQualityVersionControl:
    """Tests for QualityVersionControl class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def mock_report(self):
        """Create mock quality report."""
        report = Mock()
        report.score = Mock()
        report.score.overall = 0.85
        report.score.completeness = 0.9
        report.score.consistency = 0.8
        report.score.uniqueness = 0.95
        report.score.validity = 0.88
        report.label_errors = Mock(return_value=Mock(__len__=Mock(return_value=10)))
        report.duplicates = Mock(return_value=Mock(__len__=Mock(return_value=5)))
        report.outliers = Mock(return_value=Mock(__len__=Mock(return_value=3)))
        report.n_samples = 1000
        return report

    def test_vc_init(self, temp_storage: str) -> None:
        vc = QualityVersionControl(temp_storage)
        assert vc is not None
        assert vc.storage_path == Path(temp_storage)

    def test_vc_auto_create_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            new_path = Path(tmp) / "new_dir"
            vc = QualityVersionControl(new_path, auto_create=True)
            assert new_path.exists()

    def test_commit(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        snapshot = vc.commit(mock_report, message="Initial commit")
        assert isinstance(snapshot, QualitySnapshot)
        assert snapshot.message == "Initial commit"
        assert snapshot.version is not None

    def test_multiple_commits(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        s1 = vc.commit(mock_report, message="First")
        
        mock_report.score.overall = 0.9
        s2 = vc.commit(mock_report, message="Second")
        
        assert s1.version != s2.version
        assert s2.parent_version == s1.version

    def test_log(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        vc.commit(mock_report, message="Commit 1")
        vc.commit(mock_report, message="Commit 2")
        
        history = vc.log(max_entries=10)
        assert len(history) == 2
        assert history[0].message == "Commit 2"  # Most recent first

    def test_checkout(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        s1 = vc.commit(mock_report, message="First")
        
        checked_out = vc.checkout(s1.version)
        assert checked_out.version == s1.version

    def test_diff(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        vc.commit(mock_report, message="Before")
        
        mock_report.score.overall = 0.9
        vc.commit(mock_report, message="After")
        
        diff = vc.diff()
        assert isinstance(diff, QualityDiff)

    def test_branch(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        vc.commit(mock_report, message="Main commit")
        
        branch = vc.branch("feature", description="Feature branch")
        assert branch.name == "feature"

    def test_switch_branch(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        vc.commit(mock_report, message="Main commit")
        vc.branch("feature")
        
        vc.switch_branch("feature")
        assert vc._current_branch == "feature"

    def test_list_branches(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        vc.commit(mock_report, message="Main")
        vc.branch("feature1")
        vc.branch("feature2")
        
        branches = vc.list_branches()
        branch_names = [b.name for b in branches]
        assert "main" in branch_names
        assert "feature1" in branch_names
        assert "feature2" in branch_names

    def test_tag(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        s = vc.commit(mock_report, message="Release")
        vc.tag("v1.0", message="First release")
        
        tags = vc.list_tags()
        assert "v1.0" in tags

    def test_generate_trend_report(self, temp_storage: str, mock_report: Mock) -> None:
        vc = QualityVersionControl(temp_storage)
        
        # Create history
        for i in range(5):
            mock_report.score.overall = 0.8 + i * 0.02
            vc.commit(mock_report, message=f"Commit {i}")
        
        trend = vc.generate_trend_report(n_versions=5)
        assert "score_trend" in trend
        assert trend["score_trend"]["direction"] == "improving"


class TestCreateVersionControl:
    """Tests for create_version_control factory function."""

    def test_create_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vc = create_version_control(tmp)
            assert isinstance(vc, QualityVersionControl)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_diff_no_commits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vc = QualityVersionControl(tmp)
            with pytest.raises(Exception):
                vc.diff()

    def test_checkout_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vc = QualityVersionControl(tmp)
            with pytest.raises(Exception):
                vc.checkout("nonexistent")

    def test_switch_nonexistent_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vc = QualityVersionControl(tmp)
            with pytest.raises(Exception):
                vc.switch_branch("nonexistent")
