"""Tests for cost-impact estimator module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from clean.cost_impact import (
    ActionCost,
    CleaningAction,
    CostConfig,
    CostImpactEstimator,
    ImpactReport,
    estimate_impact,
)
from clean.core.types import IssueType


class TestActionCost:
    """Tests for ActionCost dataclass."""

    def test_total_cost(self):
        """Test total cost calculation."""
        cost = ActionCost(
            action=CleaningAction.RELABEL,
            issue_type=IssueType.LABEL_ERROR,
            n_samples=100,
            human_cost_usd=50.0,
            compute_cost_usd=10.0,
            time_hours=5.0,
            expected_accuracy_gain=0.05,
            confidence=0.8,
        )
        assert cost.total_cost_usd == 60.0

    def test_roi_calculation(self):
        """Test ROI calculation."""
        cost = ActionCost(
            action=CleaningAction.RELABEL,
            issue_type=IssueType.LABEL_ERROR,
            n_samples=100,
            human_cost_usd=50.0,
            compute_cost_usd=10.0,
            time_hours=5.0,
            expected_accuracy_gain=0.05,
            confidence=0.8,
        )
        # ROI = gain / cost = 0.05 / 60 = 0.000833...
        assert cost.roi == pytest.approx(0.05 / 60.0)

    def test_roi_zero_cost(self):
        """Test ROI with zero cost returns infinity."""
        cost = ActionCost(
            action=CleaningAction.DEDUPLICATE,
            issue_type=IssueType.DUPLICATE,
            n_samples=10,
            human_cost_usd=0.0,
            compute_cost_usd=0.0,
            time_hours=0.0,
            expected_accuracy_gain=0.01,
            confidence=0.9,
        )
        assert cost.roi == float("inf")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cost = ActionCost(
            action=CleaningAction.RELABEL,
            issue_type=IssueType.LABEL_ERROR,
            n_samples=100,
            human_cost_usd=50.0,
            compute_cost_usd=10.0,
            time_hours=5.0,
            expected_accuracy_gain=0.05,
            confidence=0.8,
        )
        result = cost.to_dict()

        assert result["action"] == "relabel"
        assert result["issue_type"] == "label_error"
        assert result["n_samples"] == 100
        assert result["total_cost_usd"] == 60.0


class TestCostConfig:
    """Tests for CostConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CostConfig()
        assert config.labeling_cost_per_sample == 0.10
        assert config.compute_cost_per_hour == 1.00

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CostConfig(
            labeling_cost_per_sample=0.25,
            compute_cost_per_hour=2.00,
        )
        assert config.labeling_cost_per_sample == 0.25
        assert config.compute_cost_per_hour == 2.00


class TestCostImpactEstimator:
    """Tests for CostImpactEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create estimator with default config."""
        return CostImpactEstimator()

    @pytest.fixture
    def mock_report(self):
        """Create mock quality report."""
        report = MagicMock()
        report.n_samples = 10000
        report.quality_score = 75.0
        report.label_error_count = 350
        report.duplicate_count = 200
        report.outlier_count = 100
        report.class_distribution = {"A": 8000, "B": 2000}
        report.issue_counts = {
            IssueType.LABEL_ERROR: 350,
            IssueType.DUPLICATE: 200,
            IssueType.OUTLIER: 100,
        }
        return report

    def test_estimate_returns_impact_report(self, estimator, mock_report):
        """Test that estimate returns an ImpactReport."""
        impact = estimator.estimate(mock_report)
        assert isinstance(impact, ImpactReport)

    def test_estimate_with_label_errors(self, estimator, mock_report):
        """Test estimation with label errors."""
        impact = estimator.estimate(mock_report)

        # Should have a label error action
        label_actions = [a for a in impact.actions if a.issue_type == IssueType.LABEL_ERROR]
        assert len(label_actions) == 1
        assert label_actions[0].n_samples == 350

    def test_estimate_with_duplicates(self, estimator, mock_report):
        """Test estimation with duplicates."""
        impact = estimator.estimate(mock_report)

        # Should have a duplicate action
        dup_actions = [a for a in impact.actions if a.issue_type == IssueType.DUPLICATE]
        assert len(dup_actions) == 1
        assert dup_actions[0].n_samples == 200

    def test_estimate_with_imbalance(self, estimator, mock_report):
        """Test estimation with class imbalance."""
        impact = estimator.estimate(mock_report)

        # Should have a rebalance action (4:1 ratio > 3)
        rebalance_actions = [a for a in impact.actions if a.action == CleaningAction.REBALANCE]
        assert len(rebalance_actions) == 1

    def test_estimate_accuracy_projection(self, estimator, mock_report):
        """Test accuracy projection."""
        impact = estimator.estimate(mock_report, current_accuracy=0.85)

        assert impact.current_accuracy == 0.85
        assert impact.projected_accuracy is not None
        assert impact.projected_accuracy > impact.current_accuracy
        assert impact.projected_accuracy <= 1.0

    def test_estimate_total_cost(self, estimator, mock_report):
        """Test total cost calculation."""
        impact = estimator.estimate(mock_report)

        assert impact.total_cost_usd > 0
        assert impact.total_cost_usd == sum(a.total_cost_usd for a in impact.actions)

    def test_estimate_recommendations(self, estimator, mock_report):
        """Test that recommendations are generated."""
        impact = estimator.estimate(mock_report)
        assert len(impact.recommendations) > 0

    def test_estimate_roi_ranking(self, estimator, mock_report):
        """Test ROI ranking is populated."""
        impact = estimator.estimate(mock_report)
        assert len(impact.roi_ranking) > 0

    def test_custom_labeling_cost(self, mock_report):
        """Test with custom labeling cost."""
        estimator = CostImpactEstimator(labeling_cost_per_sample=0.50)
        impact = estimator.estimate(mock_report)

        # Cost should be higher than default
        default_estimator = CostImpactEstimator()
        default_impact = default_estimator.estimate(mock_report)

        assert impact.total_cost_usd > default_impact.total_cost_usd

    def test_no_issues_report(self, estimator):
        """Test with report that has no issues."""
        report = MagicMock()
        report.n_samples = 10000
        report.label_error_count = 0
        report.duplicate_count = 0
        report.outlier_count = 0
        report.class_distribution = {"A": 5000, "B": 5000}  # Balanced
        report.issue_counts = {}

        impact = estimator.estimate(report)
        assert impact.total_issues == 0
        assert impact.total_cost_usd == 0


class TestImpactReport:
    """Tests for ImpactReport."""

    @pytest.fixture
    def sample_report(self):
        """Create sample impact report."""
        actions = [
            ActionCost(
                action=CleaningAction.RELABEL,
                issue_type=IssueType.LABEL_ERROR,
                n_samples=100,
                human_cost_usd=10.0,
                compute_cost_usd=5.0,
                time_hours=1.0,
                expected_accuracy_gain=0.03,
                confidence=0.8,
            ),
            ActionCost(
                action=CleaningAction.DEDUPLICATE,
                issue_type=IssueType.DUPLICATE,
                n_samples=50,
                human_cost_usd=2.5,
                compute_cost_usd=2.5,
                time_hours=0.5,
                expected_accuracy_gain=0.01,
                confidence=0.9,
            ),
        ]
        return ImpactReport(
            total_issues=150,
            total_cost_usd=20.0,
            total_time_hours=1.5,
            expected_accuracy_gain=0.04,
            current_accuracy=0.85,
            projected_accuracy=0.89,
            actions=actions,
            recommendations=["Start with relabeling"],
            roi_ranking=["relabel", "deduplicate"],
        )

    def test_summary_generation(self, sample_report):
        """Test summary generation."""
        summary = sample_report.summary()

        assert "COST-IMPACT ASSESSMENT" in summary
        assert "Total Issues Found: 150" in summary
        assert "Total Cost: $20.00" in summary
        assert "Current Accuracy: 85.0%" in summary
        assert "Projected Accuracy: 89.0%" in summary

    def test_to_dict(self, sample_report):
        """Test conversion to dictionary."""
        result = sample_report.to_dict()

        assert result["total_issues"] == 150
        assert result["total_cost_usd"] == 20.0
        assert len(result["actions"]) == 2


class TestEstimateImpact:
    """Tests for estimate_impact convenience function."""

    def test_estimate_impact_returns_report(self):
        """Test that function returns ImpactReport."""
        report = MagicMock()
        report.n_samples = 1000
        report.label_error_count = 50
        report.duplicate_count = 20
        report.outlier_count = 10
        report.class_distribution = {"A": 500, "B": 500}
        report.issue_counts = {}

        impact = estimate_impact(report)
        assert isinstance(impact, ImpactReport)

    def test_estimate_impact_with_params(self):
        """Test with custom parameters."""
        report = MagicMock()
        report.n_samples = 1000
        report.label_error_count = 50
        report.duplicate_count = 20
        report.outlier_count = 10
        report.class_distribution = {"A": 500, "B": 500}
        report.issue_counts = {}

        impact = estimate_impact(
            report,
            current_accuracy=0.80,
            labeling_cost=0.25,
            compute_cost=2.00,
        )

        assert impact.current_accuracy == 0.80
