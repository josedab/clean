"""Tests for the lineage tracking system."""

import tempfile
from pathlib import Path

import pytest

from clean.core.report import QualityReport
from clean.core.types import (
    ClassDistribution,
    DatasetInfo,
    DataType,
    LabelError,
    Outlier,
    QualityScore,
    TaskType,
)
from clean.detection.base import DetectorResult
from clean.lineage import (
    LineageTracker,
)


@pytest.fixture
def sample_report():
    """Create a sample QualityReport for testing."""
    label_errors = [
        LabelError(index=5, given_label="cat", predicted_label="dog", confidence=0.95, self_confidence=0.1),
        LabelError(index=10, given_label="bird", predicted_label="cat", confidence=0.85, self_confidence=0.2),
    ]

    outliers = [
        Outlier(index=3, score=0.9, method="isolation_forest"),
    ]

    return QualityReport(
        quality_score=QualityScore(
            overall=0.85,
            label_quality=0.9,
            duplicate_quality=1.0,
            outlier_quality=0.95,
            imbalance_quality=0.8,
            bias_quality=0.9,
        ),
        dataset_info=DatasetInfo(
            n_samples=100,
            n_features=10,
            n_classes=3,
            feature_names=[f"f{i}" for i in range(10)],
            label_column="label",
            data_type=DataType.TABULAR,
            task_type=TaskType.CLASSIFICATION,
        ),
        class_distribution=ClassDistribution(
            class_counts={"cat": 40, "dog": 40, "bird": 20},
            class_ratios={"cat": 0.4, "dog": 0.4, "bird": 0.2},
            imbalance_ratio=2.0,
            majority_class="cat",
            minority_class="bird",
        ),
        label_errors_result=DetectorResult(issues=label_errors, metadata={}),
        duplicates_result=None,
        outliers_result=DetectorResult(issues=outliers, metadata={}),
        imbalance_result=None,
        bias_result=None,
    )


class TestLineageTracker:
    """Tests for LineageTracker."""

    def test_init(self):
        """Test tracker initialization."""
        tracker = LineageTracker("test_project")
        assert tracker.project_name == "test_project"
        assert len(tracker.list_runs()) == 0

    def test_log_analysis(self, sample_report):
        """Test logging an analysis run."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report)

        assert run_id is not None
        assert len(tracker.list_runs()) == 1

        run = tracker.get_run(run_id)
        assert run is not None
        assert run.n_samples == 100
        assert run.n_label_errors == 2
        assert run.n_outliers == 1
        assert run.quality_score == 0.85

    def test_log_analysis_with_metadata(self, sample_report):
        """Test logging with metadata."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report, metadata={"version": "v1", "source": "test"})

        run = tracker.get_run(run_id)
        assert run.metadata["version"] == "v1"
        assert run.metadata["source"] == "test"

    def test_log_review(self, sample_report):
        """Test logging a review decision."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report)

        tracker.log_review(
            run_id=run_id,
            sample_id=5,
            issue_type="label_error",
            decision="relabel",
            reviewer="alice",
            notes="Clearly a dog",
            old_value="cat",
            new_value="dog",
        )

        reviews = tracker.get_reviews_for_run(run_id)
        assert len(reviews) == 1
        assert reviews[0].sample_id == 5
        assert reviews[0].decision == "relabel"
        assert reviews[0].reviewer == "alice"

    def test_log_review_invalid_run(self, sample_report):
        """Test logging review with invalid run ID."""
        tracker = LineageTracker("test_project")

        with pytest.raises(ValueError):
            tracker.log_review(
                run_id="invalid",
                sample_id=5,
                issue_type="label_error",
                decision="keep",
                reviewer="alice",
            )

    def test_get_sample_history(self, sample_report):
        """Test getting sample history."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report)

        # Sample 5 was flagged as label error
        history = tracker.get_sample_history(5)
        assert history.sample_id == 5
        assert len(history.issue_detections) == 1
        assert history.issue_detections[0]["issue_type"] == "label_error"
        assert history.current_status == "flagged"

        # Log a review
        tracker.log_review(
            run_id=run_id,
            sample_id=5,
            issue_type="label_error",
            decision="keep",
            reviewer="alice",
        )

        history = tracker.get_sample_history(5)
        assert len(history.reviews) == 1
        assert history.current_status == "kept"

    def test_get_sample_history_clean(self, sample_report):
        """Test history for a clean sample."""
        tracker = LineageTracker("test_project")
        tracker.log_analysis(sample_report)

        # Sample 99 was not flagged
        history = tracker.get_sample_history(99)
        assert len(history.issue_detections) == 0
        assert len(history.reviews) == 0
        assert history.current_status == "clean"

    def test_get_pending_reviews(self, sample_report):
        """Test getting pending reviews."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report)

        pending = tracker.get_pending_reviews(run_id)
        # Should have 3 flagged samples: 5, 10 (label errors), 3 (outlier)
        assert set(pending) == {3, 5, 10}

        # Review one
        tracker.log_review(
            run_id=run_id,
            sample_id=5,
            issue_type="label_error",
            decision="keep",
            reviewer="alice",
        )

        pending = tracker.get_pending_reviews(run_id)
        assert set(pending) == {3, 10}

    def test_get_statistics(self, sample_report):
        """Test getting statistics."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report)

        tracker.log_review(run_id, 5, "label_error", "keep", "alice")
        tracker.log_review(run_id, 10, "label_error", "relabel", "bob")
        tracker.log_review(run_id, 3, "outlier", "remove", "alice")

        stats = tracker.get_statistics()
        assert stats["n_runs"] == 1
        assert stats["n_reviews"] == 3
        assert stats["n_samples_flagged"] == 3
        assert stats["decision_distribution"]["keep"] == 1
        assert stats["decision_distribution"]["relabel"] == 1
        assert stats["decision_distribution"]["remove"] == 1
        assert stats["reviewer_activity"]["alice"] == 2
        assert stats["reviewer_activity"]["bob"] == 1

    def test_multiple_runs(self, sample_report):
        """Test tracking multiple analysis runs."""
        tracker = LineageTracker("test_project")

        run1 = tracker.log_analysis(sample_report)
        run2 = tracker.log_analysis(sample_report, metadata={"version": "v2"})

        runs = tracker.list_runs()
        assert len(runs) == 2
        # Should be sorted by timestamp (most recent first)
        assert runs[0].run_id == run2
        assert runs[1].run_id == run1


class TestPersistence:
    """Tests for lineage persistence."""

    def test_save_and_load(self, sample_report):
        """Test saving and loading lineage data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate tracker
            tracker1 = LineageTracker("test_project", storage_path=tmpdir)
            run_id = tracker1.log_analysis(sample_report)
            tracker1.log_review(run_id, 5, "label_error", "keep", "alice")

            # Create new tracker from same storage
            tracker2 = LineageTracker("test_project", storage_path=tmpdir)

            # Should have same data
            assert len(tracker2.list_runs()) == 1
            reviews = tracker2.get_reviews_for_run(run_id)
            assert len(reviews) == 1
            assert reviews[0].reviewer == "alice"

    def test_export(self, sample_report):
        """Test exporting lineage data."""
        tracker = LineageTracker("test_project")
        run_id = tracker.log_analysis(sample_report)
        tracker.log_review(run_id, 5, "label_error", "keep", "alice")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tracker.export(f.name)

            # Verify file exists and has content
            content = Path(f.name).read_text()
            assert "test_project" in content
            assert "alice" in content
            assert run_id in content

            Path(f.name).unlink()
