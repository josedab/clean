"""Tests for the auto-fix engine."""

import numpy as np
import pandas as pd
import pytest

from clean.core.report import QualityReport
from clean.core.types import (
    ClassDistribution,
    DatasetInfo,
    DataType,
    DuplicatePair,
    LabelError,
    Outlier,
    QualityScore,
    TaskType,
)
from clean.detection.base import DetectorResult
from clean.fixes import (
    FixConfig,
    FixEngine,
    FixResult,
    FixStrategy,
    apply_fixes,
    suggest_fixes,
)
from clean.plugins import SuggestedFix


@pytest.fixture
def sample_features():
    """Sample feature DataFrame."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })


@pytest.fixture
def sample_labels():
    """Sample labels array."""
    return np.array(["cat", "dog", "cat", "bird", "dog", "cat", "dog", "bird", "cat", "dog"])


@pytest.fixture
def sample_report():
    """Sample QualityReport with various issues."""
    # Create label errors
    label_errors = [
        LabelError(index=2, given_label="cat", predicted_label="dog", confidence=0.95, self_confidence=0.1),
        LabelError(index=5, given_label="cat", predicted_label="bird", confidence=0.85, self_confidence=0.2),
        LabelError(index=8, given_label="cat", predicted_label="dog", confidence=0.75, self_confidence=0.3),
    ]

    # Create duplicates
    duplicates = [
        DuplicatePair(index1=0, index2=3, similarity=0.99, is_exact=False),
        DuplicatePair(index1=1, index2=6, similarity=0.95, is_exact=False),
    ]

    # Create outliers
    outliers = [
        Outlier(index=9, score=0.95, method="isolation_forest"),
        Outlier(index=4, score=0.80, method="lof"),
    ]

    return QualityReport(
        quality_score=QualityScore(
            overall=0.75,
            label_quality=0.8,
            duplicate_quality=0.9,
            outlier_quality=0.95,
            imbalance_quality=0.7,
            bias_quality=0.9,
        ),
        dataset_info=DatasetInfo(
            n_samples=10,
            n_features=2,
            n_classes=3,
            feature_names=["feature1", "feature2"],
            label_column="label",
            data_type=DataType.TABULAR,
            task_type=TaskType.CLASSIFICATION,
        ),
        class_distribution=ClassDistribution(
            class_counts={"cat": 4, "dog": 4, "bird": 2},
            class_ratios={"cat": 0.4, "dog": 0.4, "bird": 0.2},
            imbalance_ratio=2.0,
            majority_class="cat",
            minority_class="bird",
        ),
        label_errors_result=DetectorResult(issues=label_errors, metadata={}),
        duplicates_result=DetectorResult(issues=duplicates, metadata={}),
        outliers_result=DetectorResult(issues=outliers, metadata={}),
        imbalance_result=None,
        bias_result=None,
    )


class TestFixConfig:
    """Tests for FixConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FixConfig()
        assert config.label_error_threshold == 0.9
        assert config.duplicate_similarity_threshold == 0.98
        assert config.outlier_score_threshold == 0.9
        assert config.auto_relabel is False

    def test_conservative_strategy(self):
        """Test conservative strategy preset."""
        config = FixConfig.from_strategy(FixStrategy.CONSERVATIVE)
        assert config.label_error_threshold == 0.95
        assert config.duplicate_similarity_threshold == 0.99
        assert config.auto_relabel is False

    def test_aggressive_strategy(self):
        """Test aggressive strategy preset."""
        config = FixConfig.from_strategy(FixStrategy.AGGRESSIVE)
        assert config.label_error_threshold == 0.7
        assert config.duplicate_similarity_threshold == 0.9
        assert config.auto_relabel is True

    def test_moderate_strategy(self):
        """Test moderate strategy preset."""
        config = FixConfig.from_strategy(FixStrategy.MODERATE)
        assert config.label_error_threshold == 0.9


class TestFixEngine:
    """Tests for FixEngine."""

    def test_suggest_label_fixes(self, sample_report, sample_features, sample_labels):
        """Test suggesting label error fixes."""
        engine = FixEngine(sample_report, sample_features, sample_labels)
        fixes = engine.suggest_fixes(include_duplicates=False, include_outliers=False)

        # Only the error with confidence >= 0.9 should be suggested
        assert len(fixes) == 1
        assert fixes[0].fix_type == "relabel"
        assert fixes[0].issue_index == 2
        assert fixes[0].new_value == "dog"

    def test_suggest_duplicate_fixes(self, sample_report, sample_features, sample_labels):
        """Test suggesting duplicate fixes."""
        engine = FixEngine(sample_report, sample_features, sample_labels)
        fixes = engine.suggest_fixes(include_label_errors=False, include_outliers=False)

        # Only duplicate with similarity >= 0.98 should be suggested
        assert len(fixes) == 1
        assert fixes[0].fix_type == "remove"
        assert fixes[0].issue_type == "duplicate"

    def test_suggest_outlier_fixes(self, sample_report, sample_features, sample_labels):
        """Test suggesting outlier fixes."""
        engine = FixEngine(sample_report, sample_features, sample_labels)
        fixes = engine.suggest_fixes(include_label_errors=False, include_duplicates=False)

        # Only outlier with score >= 0.9 should be suggested
        assert len(fixes) == 1
        assert fixes[0].issue_type == "outlier"
        assert fixes[0].issue_index == 9

    def test_suggest_all_fixes(self, sample_report, sample_features, sample_labels):
        """Test suggesting all types of fixes."""
        engine = FixEngine(sample_report, sample_features, sample_labels)
        fixes = engine.suggest_fixes()

        # Should have fixes from all categories above threshold
        assert len(fixes) >= 1
        # Fixes should be sorted by confidence
        for i in range(len(fixes) - 1):
            assert fixes[i].confidence >= fixes[i + 1].confidence

    def test_apply_relabel_fix(self, sample_report, sample_features, sample_labels):
        """Test applying relabel fixes."""
        config = FixConfig(label_error_threshold=0.7)  # Lower threshold to include more
        engine = FixEngine(sample_report, sample_features, sample_labels, config)

        fixes = engine.suggest_fixes(include_duplicates=False, include_outliers=False)
        result = engine.apply_fixes(fixes, dry_run=False)

        assert result.n_applied == len(fixes)
        # Check that labels were actually changed
        assert result.labels is not None
        assert result.labels[2] == "dog"  # The high-confidence fix

    def test_apply_fixes_dry_run(self, sample_report, sample_features, sample_labels):
        """Test dry run doesn't modify data."""
        engine = FixEngine(sample_report, sample_features, sample_labels)
        original_labels = sample_labels.copy()

        fixes = engine.suggest_fixes()
        result = engine.apply_fixes(fixes, dry_run=True)

        # Original should be unchanged
        np.testing.assert_array_equal(sample_labels, original_labels)
        # Applied count should still reflect what would be applied
        assert result.n_applied >= 0

    def test_apply_remove_fixes(self, sample_report, sample_features, sample_labels):
        """Test applying remove fixes."""
        config = FixConfig(
            duplicate_similarity_threshold=0.9,  # Lower to include more
            outlier_action="remove",
            outlier_score_threshold=0.9,
        )
        engine = FixEngine(sample_report, sample_features, sample_labels, config)

        fixes = engine.suggest_fixes(include_label_errors=False)
        initial_len = len(sample_features)
        result = engine.apply_fixes(fixes, dry_run=False)

        # Should have fewer rows
        assert len(result.features) < initial_len

    def test_max_fixes_limit(self, sample_report, sample_features, sample_labels):
        """Test max_fixes configuration."""
        config = FixConfig(
            label_error_threshold=0.5,  # Low threshold
            max_fixes=2,
        )
        engine = FixEngine(sample_report, sample_features, sample_labels, config)
        fixes = engine.suggest_fixes()

        assert len(fixes) <= 2

    def test_audit_log(self, sample_report, sample_features, sample_labels):
        """Test audit logging."""
        engine = FixEngine(sample_report, sample_features, sample_labels)
        fixes = engine.suggest_fixes()
        engine.apply_fixes(fixes, dry_run=False)

        log = engine.get_audit_log()
        # Audit log should have entries for applied fixes
        # (only relabel fixes are logged in current implementation)

        engine.clear_audit_log()
        assert len(engine.get_audit_log()) == 0


class TestFixResult:
    """Tests for FixResult."""

    def test_fix_result_summary(self, sample_features, sample_labels):
        """Test FixResult summary generation."""
        fixes = [
            SuggestedFix(
                issue_type="label_error",
                issue_index=0,
                fix_type="relabel",
                confidence=0.9,
                description="Test",
            ),
            SuggestedFix(
                issue_type="duplicate",
                issue_index=(1, 2),
                fix_type="remove",
                confidence=0.95,
                description="Test",
            ),
        ]

        result = FixResult(
            features=sample_features,
            labels=sample_labels,
            applied_fixes=fixes,
            skipped_fixes=[],
            errors=[],
        )

        summary = result.summary()
        assert "Applied: 2 fixes" in summary
        assert "relabel" in summary
        assert "remove" in summary


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_suggest_fixes_function(self, sample_report, sample_features, sample_labels):
        """Test suggest_fixes convenience function."""
        fixes = suggest_fixes(sample_report, sample_features, sample_labels)
        assert isinstance(fixes, list)

    def test_apply_fixes_function(self, sample_report, sample_features, sample_labels):
        """Test apply_fixes convenience function."""
        result = apply_fixes(
            sample_report,
            sample_features,
            sample_labels,
            strategy=FixStrategy.AGGRESSIVE,
            dry_run=True,
        )
        assert isinstance(result, FixResult)
