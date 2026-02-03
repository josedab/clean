"""Tests for quality advisor module."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from clean.advisor import (
    ActionCategory,
    AdvisorReport,
    Priority,
    QualityAdvisor,
    Recommendation,
    get_recommendations,
)


class TestRecommendation:
    """Tests for Recommendation dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = Recommendation(
            id="REC-001",
            title="Test recommendation",
            description="Test description",
            priority=Priority.HIGH,
            category=ActionCategory.RELABEL,
            impact_score=50.0,
            effort_score=25.0,
        )

        result = rec.to_dict()

        assert result["id"] == "REC-001"
        assert result["priority"] == "high"
        assert result["category"] == "relabel"

    def test_roi_score(self):
        """Test ROI calculation."""
        rec = Recommendation(
            id="REC-001",
            title="Test",
            description="Test",
            priority=Priority.MEDIUM,
            category=ActionCategory.CLEAN,
            impact_score=50.0,
            effort_score=25.0,
        )

        # ROI = impact / effort * 100
        assert rec.roi_score == 200.0

    def test_roi_score_zero_effort(self):
        """Test ROI with zero effort."""
        rec = Recommendation(
            id="REC-001",
            title="Test",
            description="Test",
            priority=Priority.LOW,
            category=ActionCategory.MONITOR,
            impact_score=50.0,
            effort_score=0.0,
        )

        assert rec.roi_score == 50.0


class TestAdvisorReport:
    """Tests for AdvisorReport dataclass."""

    @pytest.fixture
    def sample_recommendations(self):
        """Create sample recommendations."""
        return [
            Recommendation(
                id="REC-001",
                title="Critical fix",
                description="Fix this now",
                priority=Priority.CRITICAL,
                category=ActionCategory.RELABEL,
                impact_score=80.0,
                effort_score=40.0,
            ),
            Recommendation(
                id="REC-002",
                title="High priority fix",
                description="Fix soon",
                priority=Priority.HIGH,
                category=ActionCategory.DEDUPLICATE,
                impact_score=50.0,
                effort_score=20.0,
            ),
            Recommendation(
                id="REC-003",
                title="Medium priority fix",
                description="Consider fixing",
                priority=Priority.MEDIUM,
                category=ActionCategory.CLEAN,
                impact_score=30.0,
                effort_score=30.0,
            ),
        ]

    def test_top_recommendations(self, sample_recommendations):
        """Test getting top recommendations by ROI."""
        report = AdvisorReport(
            recommendations=sample_recommendations,
            quality_score=70.0,
            projected_score=85.0,
            analysis_timestamp="2024-01-01T00:00:00",
        )

        top = report.top(2)

        assert len(top) == 2
        # Should be sorted by ROI (impact/effort)
        assert top[0].id == "REC-002"  # ROI = 250
        assert top[1].id == "REC-001"  # ROI = 200

    def test_by_priority(self, sample_recommendations):
        """Test filtering by priority."""
        report = AdvisorReport(
            recommendations=sample_recommendations,
            quality_score=70.0,
            projected_score=85.0,
            analysis_timestamp="2024-01-01T00:00:00",
        )

        critical = report.by_priority(Priority.CRITICAL)

        assert len(critical) == 1
        assert critical[0].id == "REC-001"

    def test_by_category(self, sample_recommendations):
        """Test filtering by category."""
        report = AdvisorReport(
            recommendations=sample_recommendations,
            quality_score=70.0,
            projected_score=85.0,
            analysis_timestamp="2024-01-01T00:00:00",
        )

        relabel = report.by_category(ActionCategory.RELABEL)

        assert len(relabel) == 1
        assert relabel[0].id == "REC-001"

    def test_summary(self, sample_recommendations):
        """Test summary generation."""
        report = AdvisorReport(
            recommendations=sample_recommendations,
            quality_score=70.0,
            projected_score=85.0,
            analysis_timestamp="2024-01-01T00:00:00",
        )

        summary = report.summary()

        assert "Quality Advisor Report" in summary
        assert "70.0" in summary
        assert "85.0" in summary
        assert "Critical fix" in summary

    def test_to_dict(self, sample_recommendations):
        """Test conversion to dictionary."""
        report = AdvisorReport(
            recommendations=sample_recommendations,
            quality_score=70.0,
            projected_score=85.0,
            analysis_timestamp="2024-01-01T00:00:00",
        )

        result = report.to_dict()

        assert result["quality_score"] == 70.0
        assert len(result["recommendations"]) == 3

    def test_to_markdown(self, sample_recommendations):
        """Test markdown generation."""
        report = AdvisorReport(
            recommendations=sample_recommendations,
            quality_score=70.0,
            projected_score=85.0,
            analysis_timestamp="2024-01-01T00:00:00",
        )

        md = report.to_markdown()

        assert "# Quality Advisor Report" in md
        assert "Critical Priority" in md
        assert "High Priority" in md


class TestQualityAdvisor:
    """Tests for QualityAdvisor class."""

    @pytest.fixture
    def mock_report(self):
        """Create mock quality report."""
        report = MagicMock()
        report.dataset_info.n_samples = 1000
        report.dataset_info.n_features = 10
        report.quality_score.overall = 75.0
        report.quality_score.completeness = 85.0
        report.label_errors_result.issues = [MagicMock()] * 50
        report.duplicates_result.issues = [MagicMock()] * 30
        report.outliers_result.issues = [MagicMock()] * 20
        report.imbalance_result = None
        report.class_distribution = None
        return report

    def test_analyze_generates_recommendations(self, mock_report):
        """Test that analyze generates recommendations."""
        advisor = QualityAdvisor()
        result = advisor.analyze(mock_report)

        assert isinstance(result, AdvisorReport)
        assert len(result.recommendations) > 0

    def test_analyze_label_errors(self, mock_report):
        """Test label error analysis."""
        advisor = QualityAdvisor()
        result = advisor.analyze(mock_report)

        # Should have recommendation for label errors
        label_recs = [
            r for r in result.recommendations
            if r.category == ActionCategory.RELABEL
        ]
        assert len(label_recs) >= 1

    def test_analyze_duplicates(self, mock_report):
        """Test duplicate analysis."""
        advisor = QualityAdvisor()
        result = advisor.analyze(mock_report)

        # Should have recommendation for duplicates
        dup_recs = [
            r for r in result.recommendations
            if r.category == ActionCategory.DEDUPLICATE
        ]
        assert len(dup_recs) >= 1

    def test_analyze_outliers(self, mock_report):
        """Test outlier analysis."""
        advisor = QualityAdvisor()
        result = advisor.analyze(mock_report)

        # Should have recommendation for outliers
        outlier_recs = [
            r for r in result.recommendations
            if r.category == ActionCategory.CLEAN
        ]
        assert len(outlier_recs) >= 1

    def test_analyze_no_issues(self):
        """Test analysis with no issues."""
        report = MagicMock()
        report.dataset_info.n_samples = 1000
        report.dataset_info.n_features = 10
        report.quality_score.overall = 95.0
        report.quality_score.completeness = 100.0
        report.label_errors_result.issues = []
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = None
        report.class_distribution = None

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        # Should still return a report, possibly with no recommendations
        assert isinstance(result, AdvisorReport)

    def test_analyze_with_imbalance(self):
        """Test analysis with class imbalance."""
        report = MagicMock()
        report.dataset_info.n_samples = 1000
        report.dataset_info.n_features = 10
        report.quality_score.overall = 70.0
        report.quality_score.completeness = 100.0
        report.label_errors_result.issues = []
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = MagicMock()
        report.class_distribution = MagicMock()
        report.class_distribution.class_counts = {"A": 900, "B": 100}

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        # Should have recommendation for imbalance
        balance_recs = [
            r for r in result.recommendations
            if r.category == ActionCategory.BALANCE
        ]
        assert len(balance_recs) >= 1

    def test_projected_score(self, mock_report):
        """Test projected score calculation."""
        advisor = QualityAdvisor()
        result = advisor.analyze(mock_report)

        # Projected score should be higher than current
        assert result.projected_score >= result.quality_score

    def test_custom_rules(self, mock_report):
        """Test custom rule execution."""
        def custom_rule(report, data):
            return [
                Recommendation(
                    id="CUSTOM-001",
                    title="Custom recommendation",
                    description="From custom rule",
                    priority=Priority.MEDIUM,
                    category=ActionCategory.VALIDATE,
                    impact_score=20.0,
                    effort_score=10.0,
                )
            ]

        advisor = QualityAdvisor(custom_rules=[custom_rule])
        result = advisor.analyze(mock_report)

        # Should include custom recommendation
        custom_recs = [r for r in result.recommendations if r.id == "CUSTOM-001"]
        assert len(custom_recs) == 1


class TestGetRecommendations:
    """Tests for get_recommendations function."""

    def test_get_recommendations_basic(self):
        """Test basic recommendation generation."""
        report = MagicMock()
        report.dataset_info.n_samples = 100
        report.dataset_info.n_features = 5
        report.quality_score.overall = 80.0
        report.quality_score.completeness = 90.0
        report.label_errors_result.issues = [MagicMock()] * 10
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = None
        report.class_distribution = None

        result = get_recommendations(report)

        assert isinstance(result, AdvisorReport)
        assert len(result.recommendations) > 0


class TestPriorityDetermination:
    """Tests for priority determination logic."""

    def test_high_label_error_rate_is_critical(self):
        """Test that high label error rate gets critical priority."""
        report = MagicMock()
        report.dataset_info.n_samples = 100
        report.dataset_info.n_features = 5
        report.quality_score.overall = 60.0
        report.quality_score.completeness = 100.0
        # 15% error rate should be critical
        report.label_errors_result.issues = [MagicMock()] * 15
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = None
        report.class_distribution = None

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        label_recs = [r for r in result.recommendations if r.category == ActionCategory.RELABEL]
        assert any(r.priority == Priority.CRITICAL for r in label_recs)

    def test_low_label_error_rate_is_low_priority(self):
        """Test that low label error rate gets low priority."""
        report = MagicMock()
        report.dataset_info.n_samples = 1000
        report.dataset_info.n_features = 5
        report.quality_score.overall = 90.0
        report.quality_score.completeness = 100.0
        # 1% error rate should be low priority
        report.label_errors_result.issues = [MagicMock()] * 10
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = None
        report.class_distribution = None

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        label_recs = [r for r in result.recommendations if r.category == ActionCategory.RELABEL]
        assert any(r.priority == Priority.LOW for r in label_recs)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_none_results(self):
        """Test with None result objects."""
        report = MagicMock()
        report.dataset_info.n_samples = 100
        report.dataset_info.n_features = 5
        report.quality_score.overall = 80.0
        report.quality_score.completeness = 100.0
        report.label_errors_result = None
        report.duplicates_result = None
        report.outliers_result = None
        report.imbalance_result = None
        report.class_distribution = None

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        assert isinstance(result, AdvisorReport)

    def test_empty_issues(self):
        """Test with empty issue lists."""
        report = MagicMock()
        report.dataset_info.n_samples = 100
        report.dataset_info.n_features = 5
        report.quality_score.overall = 95.0
        report.quality_score.completeness = 100.0
        report.label_errors_result.issues = []
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = None
        report.class_distribution = None

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        # Should return report with possibly no recommendations
        assert isinstance(result, AdvisorReport)

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        report = MagicMock()
        report.dataset_info.n_samples = 10
        report.dataset_info.n_features = 2
        report.quality_score.overall = 70.0
        report.quality_score.completeness = 80.0
        report.label_errors_result.issues = [MagicMock()] * 5
        report.duplicates_result.issues = []
        report.outliers_result.issues = []
        report.imbalance_result = None
        report.class_distribution = None

        advisor = QualityAdvisor()
        result = advisor.analyze(report)

        assert isinstance(result, AdvisorReport)
