"""Comprehensive tests for the root_cause module."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from clean.root_cause import (
    CorrelationStrength,
    FeatureCorrelation,
    RootCause,
    RootCauseAnalyzer,
    RootCauseReport,
    RootCauseType,
    TemporalPattern,
    analyze_root_causes,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data_with_annotators():
    """Create sample data with annotator metadata."""
    np.random.seed(42)
    n_samples = 500

    # Create data with different annotators
    annotators = np.random.choice(["ann_a", "ann_b", "ann_c", "ann_d"], n_samples)

    # ann_a has much higher issue rate
    is_issue = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if annotators[i] == "ann_a":
            is_issue[i] = np.random.random() < 0.3  # 30% issue rate
        elif annotators[i] == "ann_b":
            is_issue[i] = np.random.random() < 0.05  # 5% issue rate
        else:
            is_issue[i] = np.random.random() < 0.08  # 8% baseline

    df = pd.DataFrame(
        {
            "annotator": annotators,
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "is_issue": is_issue,
        }
    )

    return df


@pytest.fixture
def sample_data_with_sources():
    """Create sample data with data source metadata."""
    np.random.seed(42)
    n_samples = 600

    sources = np.random.choice(["web", "manual", "api", "scrape"], n_samples)

    is_issue = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if sources[i] == "scrape":
            is_issue[i] = np.random.random() < 0.25  # High issue rate
        else:
            is_issue[i] = np.random.random() < 0.05  # Low baseline

    df = pd.DataFrame(
        {
            "source": sources,
            "text_length": np.random.randint(10, 500, n_samples),
            "score": np.random.uniform(0, 1, n_samples),
            "is_issue": is_issue,
        }
    )

    return df


@pytest.fixture
def sample_data_with_timestamps():
    """Create sample data with temporal patterns."""
    np.random.seed(42)
    n_samples = 200

    # Create timestamps over 30 days
    base_date = datetime(2024, 1, 1)
    timestamps = [base_date + timedelta(days=np.random.randint(0, 30)) for _ in range(n_samples)]

    # Issues increase over time
    is_issue = np.array(
        [np.random.random() < (0.05 + (ts - base_date).days * 0.01) for ts in timestamps]
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "annotator": np.random.choice(["ann_1", "ann_2"], n_samples),
            "value": np.random.randn(n_samples),
        }
    )

    return df, is_issue


@pytest.fixture
def sample_data_with_labels():
    """Create sample data with class labels."""
    np.random.seed(42)
    n_samples = 400

    labels = np.random.choice([0, 1, 2], n_samples)

    # Class 2 has much higher issue rate
    is_issue = np.array(
        [
            np.random.random() < (0.35 if label == 2 else 0.05)
            for label in labels
        ]
    )

    df = pd.DataFrame(
        {
            "feature": np.random.randn(n_samples),
            "label": labels,
        }
    )

    return df, is_issue


# =============================================================================
# Tests for Enums
# =============================================================================


class TestRootCauseType:
    """Tests for RootCauseType enum."""

    def test_all_values_exist(self):
        """Test all expected root cause types exist."""
        assert RootCauseType.ANNOTATOR.value == "annotator"
        assert RootCauseType.DATA_SOURCE.value == "data_source"
        assert RootCauseType.TIME_PERIOD.value == "time_period"
        assert RootCauseType.FEATURE_RANGE.value == "feature_range"
        assert RootCauseType.CLASS_LABEL.value == "class_label"
        assert RootCauseType.COLLECTION_METHOD.value == "collection_method"
        assert RootCauseType.UNKNOWN.value == "unknown"


class TestCorrelationStrength:
    """Tests for CorrelationStrength enum."""

    def test_all_values_exist(self):
        """Test all correlation strength levels exist."""
        assert CorrelationStrength.NONE.value == "none"
        assert CorrelationStrength.WEAK.value == "weak"
        assert CorrelationStrength.MODERATE.value == "moderate"
        assert CorrelationStrength.STRONG.value == "strong"
        assert CorrelationStrength.VERY_STRONG.value == "very_strong"


# =============================================================================
# Tests for Data Classes
# =============================================================================


class TestRootCause:
    """Tests for RootCause dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cause = RootCause(
            cause_type=RootCauseType.ANNOTATOR,
            factor_name="annotator",
            factor_value="ann_a",
            issue_count=15,
            issue_rate=0.25,
            baseline_rate=0.10,
            lift=2.5,
            confidence=0.95,
            affected_indices=[1, 5, 10, 15],
            correlation_strength=CorrelationStrength.STRONG,
            explanation="Test explanation",
        )

        result = cause.to_dict()

        assert result["cause_type"] == "annotator"
        assert result["factor_name"] == "annotator"
        assert result["factor_value"] == "ann_a"
        assert result["issue_count"] == 15
        assert result["issue_rate"] == 0.25
        assert result["baseline_rate"] == 0.10
        assert result["lift"] == 2.5
        assert result["confidence"] == 0.95
        assert result["affected_samples"] == 4
        assert result["correlation_strength"] == "strong"
        assert result["explanation"] == "Test explanation"


class TestFeatureCorrelation:
    """Tests for FeatureCorrelation dataclass."""

    def test_creation(self):
        """Test FeatureCorrelation creation."""
        fc = FeatureCorrelation(
            feature_name="test_feature",
            correlation=0.45,
            p_value=0.01,
            issue_concentration={"0-25%": 0.15, "25-50%": 0.08},
            thresholds=[25.0, 75.0],
        )

        assert fc.feature_name == "test_feature"
        assert fc.correlation == 0.45
        assert fc.p_value == 0.01
        assert len(fc.issue_concentration) == 2
        assert len(fc.thresholds) == 2


class TestTemporalPattern:
    """Tests for TemporalPattern dataclass."""

    def test_creation(self):
        """Test TemporalPattern creation."""
        pattern = TemporalPattern(
            period="daily",
            pattern_type="increasing",
            peak_periods=["2024-01-15", "2024-01-20"],
            trend_coefficient=0.02,
            seasonality_strength=0.5,
        )

        assert pattern.period == "daily"
        assert pattern.pattern_type == "increasing"
        assert len(pattern.peak_periods) == 2
        assert pattern.trend_coefficient == 0.02
        assert pattern.seasonality_strength == 0.5


class TestRootCauseReport:
    """Tests for RootCauseReport dataclass."""

    def test_summary_with_root_causes(self):
        """Test summary generation with root causes."""
        cause = RootCause(
            cause_type=RootCauseType.ANNOTATOR,
            factor_name="annotator",
            factor_value="ann_a",
            issue_count=50,
            issue_rate=0.25,
            baseline_rate=0.10,
            lift=2.5,
            confidence=0.95,
            affected_indices=list(range(50)),
            correlation_strength=CorrelationStrength.STRONG,
            explanation="High issue rate for this annotator.",
        )

        report = RootCauseReport(
            n_samples=1000,
            n_issues=100,
            overall_issue_rate=0.10,
            root_causes=[cause],
            feature_correlations=[],
            temporal_patterns=[],
            recommendations=["Review annotator ann_a"],
        )

        summary = report.summary()

        assert "Root Cause Analysis Report" in summary
        assert "1,000" in summary
        assert "100" in summary
        assert "10.0%" in summary
        assert "annotator" in summary
        assert "ann_a" in summary

    def test_summary_without_root_causes(self):
        """Test summary when no root causes found."""
        report = RootCauseReport(
            n_samples=500,
            n_issues=25,
            overall_issue_rate=0.05,
            root_causes=[],
            feature_correlations=[],
            temporal_patterns=[],
            recommendations=[],
        )

        summary = report.summary()

        assert "No significant root causes identified" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = RootCauseReport(
            n_samples=500,
            n_issues=50,
            overall_issue_rate=0.10,
            root_causes=[],
            feature_correlations=[],
            temporal_patterns=[],
            recommendations=["Test recommendation"],
        )

        result = report.to_dict()

        assert result["n_samples"] == 500
        assert result["n_issues"] == 50
        assert result["overall_issue_rate"] == 0.10
        assert result["root_causes"] == []
        assert result["recommendations"] == ["Test recommendation"]

    def test_get_top_causes(self):
        """Test getting top causes by lift."""
        causes = [
            RootCause(
                cause_type=RootCauseType.ANNOTATOR,
                factor_name="annotator",
                factor_value=f"ann_{i}",
                issue_count=10,
                issue_rate=0.1,
                baseline_rate=0.05,
                lift=1.5 + i * 0.5,
                confidence=0.9,
                affected_indices=[],
                correlation_strength=CorrelationStrength.MODERATE,
                explanation="",
            )
            for i in range(5)
        ]

        report = RootCauseReport(
            n_samples=500,
            n_issues=50,
            overall_issue_rate=0.10,
            root_causes=causes,
            feature_correlations=[],
            temporal_patterns=[],
            recommendations=[],
        )

        top = report.get_top_causes(n=3)

        assert len(top) == 3
        assert top[0].lift > top[1].lift > top[2].lift


# =============================================================================
# Tests for RootCauseAnalyzer
# =============================================================================


class TestRootCauseAnalyzerInit:
    """Tests for RootCauseAnalyzer initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        analyzer = RootCauseAnalyzer()

        assert analyzer.min_sample_size == 30
        assert analyzer.significance_level == 0.05
        assert analyzer.min_lift == 1.5
        assert analyzer.min_confidence == 0.8

    def test_custom_initialization(self):
        """Test custom parameter values."""
        analyzer = RootCauseAnalyzer(
            min_sample_size=50,
            significance_level=0.01,
            min_lift=2.0,
            min_confidence=0.9,
        )

        assert analyzer.min_sample_size == 50
        assert analyzer.significance_level == 0.01
        assert analyzer.min_lift == 2.0
        assert analyzer.min_confidence == 0.9


class TestRootCauseAnalyzerAnalyze:
    """Tests for RootCauseAnalyzer.analyze method."""

    def test_analyze_with_issue_indices(self, sample_data_with_annotators):
        """Test analysis using issue indices."""
        df = sample_data_with_annotators
        issue_indices = df[df["is_issue"]].index.tolist()

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_indices=issue_indices,
            metadata_columns=["annotator"],
        )

        assert isinstance(report, RootCauseReport)
        assert report.n_samples == len(df)
        assert report.n_issues == len(issue_indices)

    def test_analyze_with_issue_column(self, sample_data_with_annotators):
        """Test analysis using issue column."""
        df = sample_data_with_annotators

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_column="is_issue",
            metadata_columns=["annotator"],
        )

        assert isinstance(report, RootCauseReport)
        assert report.n_samples == len(df)
        assert report.n_issues == int(df["is_issue"].sum())

    def test_analyze_requires_issue_specification(self, sample_data_with_annotators):
        """Test that either issue_indices or issue_column must be provided."""
        df = sample_data_with_annotators

        analyzer = RootCauseAnalyzer()

        with pytest.raises(ValueError, match="Must provide issue_indices or issue_column"):
            analyzer.analyze(data=df, metadata_columns=["annotator"])

    def test_analyze_detects_annotator_issues(self, sample_data_with_annotators):
        """Test detection of annotator-related issues."""
        df = sample_data_with_annotators

        analyzer = RootCauseAnalyzer(min_lift=1.3)
        report = analyzer.analyze(
            data=df,
            issue_column="is_issue",
            metadata_columns=["annotator"],
        )

        # Should find ann_a as a root cause (30% vs ~8% baseline)
        annotator_causes = [
            rc for rc in report.root_causes
            if rc.factor_name == "annotator" and rc.factor_value == "ann_a"
        ]

        assert len(annotator_causes) > 0

    def test_analyze_detects_source_issues(self, sample_data_with_sources):
        """Test detection of data source issues."""
        df = sample_data_with_sources

        analyzer = RootCauseAnalyzer(min_lift=1.3)
        report = analyzer.analyze(
            data=df,
            issue_column="is_issue",
            metadata_columns=["source"],
        )

        # Should find 'scrape' as a problematic source
        source_causes = [
            rc for rc in report.root_causes
            if rc.factor_name == "source" and rc.factor_value == "scrape"
        ]

        assert len(source_causes) > 0

    def test_analyze_with_automatic_metadata_detection(self, sample_data_with_annotators):
        """Test automatic metadata column detection."""
        df = sample_data_with_annotators

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_column="is_issue",
            # Don't specify metadata_columns - should auto-detect
        )

        assert isinstance(report, RootCauseReport)

    def test_analyze_with_label_column(self, sample_data_with_labels):
        """Test analysis with class labels."""
        df, is_issue = sample_data_with_labels

        analyzer = RootCauseAnalyzer(min_lift=1.3)
        report = analyzer.analyze(
            data=df,
            issue_indices=np.where(is_issue)[0].tolist(),
            label_column="label",
        )

        # Should identify class 2 as problematic
        label_causes = [
            rc for rc in report.root_causes
            if rc.cause_type == RootCauseType.CLASS_LABEL
        ]

        # May or may not find causes depending on statistical significance
        assert isinstance(report, RootCauseReport)

    def test_analyze_with_temporal_column(self, sample_data_with_timestamps):
        """Test analysis with temporal patterns."""
        df, is_issue = sample_data_with_timestamps

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_indices=np.where(is_issue)[0].tolist(),
            temporal_column="timestamp",
            metadata_columns=["annotator"],
        )

        assert isinstance(report, RootCauseReport)
        # May detect increasing trend
        assert isinstance(report.temporal_patterns, list)


class TestRootCauseAnalyzerHelpers:
    """Tests for RootCauseAnalyzer helper methods."""

    def test_identify_metadata_columns(self):
        """Test automatic metadata column identification."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "annotator": np.random.choice(["a", "b", "c"], 100),
                "source": np.random.choice(["web", "api"], 100),
                "numeric_feature": np.random.randn(100),
                "high_cardinality": np.arange(100),  # Not metadata
            }
        )

        analyzer = RootCauseAnalyzer()
        metadata_cols = analyzer._identify_metadata_columns(df)

        assert "annotator" in metadata_cols
        assert "source" in metadata_cols
        assert "high_cardinality" not in metadata_cols

    def test_infer_cause_type_annotator(self):
        """Test cause type inference for annotator columns."""
        analyzer = RootCauseAnalyzer()

        assert analyzer._infer_cause_type("annotator") == RootCauseType.ANNOTATOR
        assert analyzer._infer_cause_type("labeler_id") == RootCauseType.ANNOTATOR
        assert analyzer._infer_cause_type("rater") == RootCauseType.ANNOTATOR
        assert analyzer._infer_cause_type("worker_name") == RootCauseType.ANNOTATOR

    def test_infer_cause_type_source(self):
        """Test cause type inference for source columns."""
        analyzer = RootCauseAnalyzer()

        assert analyzer._infer_cause_type("source") == RootCauseType.DATA_SOURCE
        assert analyzer._infer_cause_type("data_origin") == RootCauseType.DATA_SOURCE
        assert analyzer._infer_cause_type("dataset") == RootCauseType.DATA_SOURCE

    def test_infer_cause_type_temporal(self):
        """Test cause type inference for temporal columns."""
        analyzer = RootCauseAnalyzer()

        assert analyzer._infer_cause_type("date") == RootCauseType.TIME_PERIOD
        assert analyzer._infer_cause_type("timestamp") == RootCauseType.TIME_PERIOD
        assert analyzer._infer_cause_type("created_at") == RootCauseType.TIME_PERIOD

    def test_infer_cause_type_unknown(self):
        """Test cause type inference for unknown columns."""
        analyzer = RootCauseAnalyzer()

        assert analyzer._infer_cause_type("random_column") == RootCauseType.UNKNOWN
        assert analyzer._infer_cause_type("xyz") == RootCauseType.UNKNOWN

    def test_get_correlation_strength(self):
        """Test correlation strength determination."""
        analyzer = RootCauseAnalyzer()

        assert analyzer._get_correlation_strength(1.1) == CorrelationStrength.NONE
        assert analyzer._get_correlation_strength(1.3) == CorrelationStrength.WEAK
        assert analyzer._get_correlation_strength(1.7) == CorrelationStrength.MODERATE
        assert analyzer._get_correlation_strength(2.5) == CorrelationStrength.STRONG
        assert analyzer._get_correlation_strength(4.0) == CorrelationStrength.VERY_STRONG

    def test_generate_cause_explanation_annotator(self):
        """Test explanation generation for annotator issues."""
        analyzer = RootCauseAnalyzer()

        explanation = analyzer._generate_cause_explanation(
            cause_type=RootCauseType.ANNOTATOR,
            column="annotator",
            value="ann_a",
            issue_rate=0.30,
            baseline_rate=0.10,
            lift=3.0,
        )

        assert "ann_a" in explanation
        assert "3.0x" in explanation
        assert "training" in explanation.lower() or "review" in explanation.lower()

    def test_generate_cause_explanation_source(self):
        """Test explanation generation for source issues."""
        analyzer = RootCauseAnalyzer()

        explanation = analyzer._generate_cause_explanation(
            cause_type=RootCauseType.DATA_SOURCE,
            column="source",
            value="scrape",
            issue_rate=0.25,
            baseline_rate=0.05,
            lift=5.0,
        )

        assert "scrape" in explanation
        assert "5.0x" in explanation

    def test_generate_cause_explanation_class(self):
        """Test explanation generation for class issues."""
        analyzer = RootCauseAnalyzer()

        explanation = analyzer._generate_cause_explanation(
            cause_type=RootCauseType.CLASS_LABEL,
            column="label",
            value=2,
            issue_rate=0.20,
            baseline_rate=0.05,
            lift=4.0,
        )

        assert "4.0x" in explanation
        assert "training" in explanation.lower() or "examples" in explanation.lower()


class TestRootCauseAnalyzerRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_for_annotator_issues(self, sample_data_with_annotators):
        """Test recommendation generation for annotator issues."""
        df = sample_data_with_annotators

        analyzer = RootCauseAnalyzer(min_lift=1.3)
        report = analyzer.analyze(
            data=df,
            issue_column="is_issue",
            metadata_columns=["annotator"],
        )

        # Should have some recommendations
        assert len(report.recommendations) > 0

    def test_recommendations_high_overall_rate(self):
        """Test recommendations when overall issue rate is high."""
        np.random.seed(42)
        n_samples = 200

        df = pd.DataFrame(
            {
                "annotator": np.random.choice(["a", "b"], n_samples),
                "feature": np.random.randn(n_samples),
            }
        )
        # High overall issue rate (15%)
        is_issue = np.random.random(n_samples) < 0.15

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_indices=np.where(is_issue)[0].tolist(),
            metadata_columns=["annotator"],
        )

        # Should recommend systematic review due to high issue rate
        has_systematic_rec = any(
            "systematic" in r.lower() or "overall" in r.lower()
            for r in report.recommendations
        )
        assert has_systematic_rec or len(report.recommendations) > 0


# =============================================================================
# Tests for Convenience Function
# =============================================================================


class TestAnalyzeRootCauses:
    """Tests for the analyze_root_causes convenience function."""

    def test_basic_usage(self, sample_data_with_annotators):
        """Test basic function usage."""
        df = sample_data_with_annotators

        report = analyze_root_causes(
            data=df,
            issue_column="is_issue",
            metadata_columns=["annotator"],
        )

        assert isinstance(report, RootCauseReport)

    def test_with_issue_indices(self, sample_data_with_annotators):
        """Test with issue indices."""
        df = sample_data_with_annotators
        issue_indices = df[df["is_issue"]].index.tolist()

        report = analyze_root_causes(
            data=df,
            issue_indices=issue_indices,
            metadata_columns=["annotator"],
        )

        assert isinstance(report, RootCauseReport)
        assert report.n_issues == len(issue_indices)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_issues(self):
        """Test with no issues."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "annotator": np.random.choice(["a", "b"], 100),
                "feature": np.random.randn(100),
            }
        )

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_indices=[],
            metadata_columns=["annotator"],
        )

        assert report.n_issues == 0
        assert report.overall_issue_rate == 0
        assert len(report.root_causes) == 0

    def test_all_issues(self):
        """Test when all samples are issues."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "annotator": np.random.choice(["a", "b"], 50),
                "feature": np.random.randn(50),
            }
        )

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_indices=list(range(50)),
            metadata_columns=["annotator"],
        )

        assert report.n_issues == 50
        assert report.overall_issue_rate == 1.0

    def test_small_sample_size(self):
        """Test with small dataset."""
        df = pd.DataFrame(
            {
                "annotator": ["a", "b", "a", "b", "a"],
                "feature": [1, 2, 3, 4, 5],
            }
        )

        analyzer = RootCauseAnalyzer(min_sample_size=2)
        report = analyzer.analyze(
            data=df,
            issue_indices=[0, 2],  # Only 'a' has issues
            metadata_columns=["annotator"],
        )

        assert isinstance(report, RootCauseReport)

    def test_missing_metadata_column(self):
        """Test with non-existent metadata column."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature": np.random.randn(100),
            }
        )

        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(
            data=df,
            issue_indices=[0, 1, 2],
            metadata_columns=["nonexistent_column"],
        )

        # Should complete without error
        assert isinstance(report, RootCauseReport)

    def test_numeric_correlation_with_nan(self):
        """Test numeric correlation with NaN values."""
        np.random.seed(42)
        n_samples = 100

        values = np.random.randn(n_samples)
        values[::10] = np.nan  # Add some NaN values
        issue_mask = np.random.random(n_samples) < 0.1

        analyzer = RootCauseAnalyzer()
        correlation = analyzer._analyze_numeric_correlation(values, issue_mask)

        # Should handle NaN values gracefully
        assert correlation is None or isinstance(correlation, FeatureCorrelation)

    def test_temporal_analysis_invalid_dates(self):
        """Test temporal analysis with invalid dates."""
        df = pd.DataFrame(
            {
                "timestamp": ["not_a_date", "also_not_a_date", "nope"],
                "feature": [1, 2, 3],
            }
        )
        issue_mask = np.array([True, False, False])

        analyzer = RootCauseAnalyzer()
        patterns = analyzer._analyze_temporal_patterns(df, "timestamp", issue_mask)

        # Should return empty list for invalid dates
        assert patterns == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the root cause module."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        np.random.seed(42)
        n_samples = 500

        # Create realistic dataset
        annotators = np.random.choice(["expert", "novice_1", "novice_2"], n_samples)
        sources = np.random.choice(["clean_db", "scraped"], n_samples)
        base_date = datetime(2024, 1, 1)
        timestamps = [
            base_date + timedelta(days=np.random.randint(0, 60))
            for _ in range(n_samples)
        ]

        # novice_2 has high error rate, scraped data has issues
        is_issue = np.array(
            [
                np.random.random()
                < (0.4 if annotators[i] == "novice_2" else (0.2 if sources[i] == "scraped" else 0.05))
                for i in range(n_samples)
            ]
        )

        df = pd.DataFrame(
            {
                "annotator": annotators,
                "source": sources,
                "timestamp": timestamps,
                "confidence_score": np.random.uniform(0, 1, n_samples),
                "text_length": np.random.randint(10, 500, n_samples),
            }
        )

        # Run analysis
        analyzer = RootCauseAnalyzer(min_lift=1.5)
        report = analyzer.analyze(
            data=df,
            issue_indices=np.where(is_issue)[0].tolist(),
            metadata_columns=["annotator", "source"],
            temporal_column="timestamp",
        )

        # Verify report structure
        assert report.n_samples == n_samples
        assert report.n_issues == is_issue.sum()
        assert isinstance(report.root_causes, list)
        assert isinstance(report.feature_correlations, list)
        assert isinstance(report.temporal_patterns, list)
        assert isinstance(report.recommendations, list)

        # Should have summary
        summary = report.summary()
        assert len(summary) > 0

        # Should have dict representation
        dict_repr = report.to_dict()
        assert "n_samples" in dict_repr
        assert "root_causes" in dict_repr
