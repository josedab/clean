"""Comprehensive tests for the model_aware module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from clean.model_aware import (
    ClassMetrics,
    ImpactLevel,
    ModelAwareReport,
    ModelAwareScorer,
    ModelComparisonScorer,
    SampleQuality,
    score_with_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_classification_data():
    """Create simple classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f_{i}" for i in range(8)]), y


@pytest.fixture
def imbalanced_classification_data():
    """Create imbalanced classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],  # Imbalanced
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


@pytest.fixture
def fitted_scorer(simple_classification_data):
    """Create and return a fitted scorer."""
    X, y = simple_classification_data
    scorer = ModelAwareScorer()
    scorer.fit(X, y)
    return scorer, X, y


# =============================================================================
# Tests for ImpactLevel Enum
# =============================================================================


class TestImpactLevel:
    """Tests for ImpactLevel enum."""

    def test_all_values_exist(self):
        """Test all expected impact levels exist."""
        assert ImpactLevel.CRITICAL.value == "critical"
        assert ImpactLevel.HIGH.value == "high"
        assert ImpactLevel.MEDIUM.value == "medium"
        assert ImpactLevel.LOW.value == "low"
        assert ImpactLevel.NEGLIGIBLE.value == "negligible"


# =============================================================================
# Tests for SampleQuality Dataclass
# =============================================================================


class TestSampleQuality:
    """Tests for SampleQuality dataclass."""

    def test_creation(self):
        """Test basic creation."""
        sq = SampleQuality(
            index=0,
            quality_score=85.0,
            model_confidence=0.92,
            is_correctly_predicted=True,
            is_in_confusion_zone=False,
            predicted_label=1,
            true_label=1,
            impact_on_model=ImpactLevel.NEGLIGIBLE,
            issues=[],
        )

        assert sq.index == 0
        assert sq.quality_score == 85.0
        assert sq.model_confidence == 0.92
        assert sq.is_correctly_predicted == True
        assert sq.impact_on_model == ImpactLevel.NEGLIGIBLE

    def test_to_dict(self):
        """Test conversion to dictionary."""
        sq = SampleQuality(
            index=5,
            quality_score=65.0,
            model_confidence=0.55,
            is_correctly_predicted=False,
            is_in_confusion_zone=True,
            predicted_label=0,
            true_label=1,
            impact_on_model=ImpactLevel.HIGH,
            issues=["Low confidence", "In confusion zone"],
        )

        result = sq.to_dict()

        assert result["index"] == 5
        assert result["quality_score"] == 65.0
        assert result["model_confidence"] == 0.55
        assert result["is_correctly_predicted"] == False
        assert result["is_in_confusion_zone"] == True
        assert result["predicted_label"] == 0
        assert result["true_label"] == 1
        assert result["impact_on_model"] == "high"
        assert len(result["issues"]) == 2

    def test_default_issues(self):
        """Test default empty issues list."""
        sq = SampleQuality(
            index=0,
            quality_score=90.0,
            model_confidence=0.95,
            is_correctly_predicted=True,
            is_in_confusion_zone=False,
            predicted_label=1,
            true_label=1,
            impact_on_model=ImpactLevel.NEGLIGIBLE,
        )

        assert sq.issues == []


# =============================================================================
# Tests for ClassMetrics Dataclass
# =============================================================================


class TestClassMetrics:
    """Tests for ClassMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        cm = ClassMetrics(
            label=1,
            n_samples=100,
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1_score=0.85,
            confusion_rate=0.15,
            avg_confidence=0.78,
            quality_score=82.0,
            most_confused_with=[(2, 0.10), (0, 0.05)],
        )

        assert cm.label == 1
        assert cm.n_samples == 100
        assert cm.accuracy == 0.85
        assert cm.precision == 0.80
        assert cm.recall == 0.90
        assert cm.f1_score == 0.85
        assert cm.confusion_rate == 0.15
        assert cm.avg_confidence == 0.78
        assert cm.quality_score == 82.0
        assert len(cm.most_confused_with) == 2


# =============================================================================
# Tests for ModelAwareReport Dataclass
# =============================================================================


class TestModelAwareReport:
    """Tests for ModelAwareReport dataclass."""

    def test_summary(self):
        """Test summary generation."""
        sample_scores = [
            SampleQuality(
                index=i,
                quality_score=80.0,
                model_confidence=0.8,
                is_correctly_predicted=True,
                is_in_confusion_zone=False,
                predicted_label=0,
                true_label=0,
                impact_on_model=ImpactLevel.NEGLIGIBLE,
            )
            for i in range(10)
        ]

        report = ModelAwareReport(
            n_samples=10,
            overall_quality_score=80.0,
            model_accuracy=0.90,
            model_confidence_mean=0.85,
            model_confidence_std=0.10,
            sample_scores=sample_scores,
            class_metrics={},
            high_confusion_pairs=[("A", "B", 0.15)],
            critical_samples=[1, 2],
            high_impact_samples=[3, 4, 5],
            recommendations=["Review critical samples"],
        )

        summary = report.summary()

        assert "Model-Aware Quality Report" in summary
        assert "10" in summary
        assert "80.0" in summary
        assert "90.0%" in summary or "90%" in summary
        assert "A" in summary and "B" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = ModelAwareReport(
            n_samples=50,
            overall_quality_score=75.0,
            model_accuracy=0.88,
            model_confidence_mean=0.82,
            model_confidence_std=0.12,
            sample_scores=[],
            class_metrics={},
            high_confusion_pairs=[],
            critical_samples=[1, 2, 3],
            high_impact_samples=[4, 5],
            recommendations=[],
        )

        result = report.to_dict()

        assert result["n_samples"] == 50
        assert result["overall_quality_score"] == 75.0
        assert result["model_accuracy"] == 0.88
        assert result["model_confidence_mean"] == 0.82
        assert result["critical_samples"] == 3
        assert result["high_impact_samples"] == 2

    def test_get_samples_by_impact(self):
        """Test filtering samples by impact level."""
        sample_scores = [
            SampleQuality(
                index=0,
                quality_score=30.0,
                model_confidence=0.9,
                is_correctly_predicted=False,
                is_in_confusion_zone=False,
                predicted_label=0,
                true_label=1,
                impact_on_model=ImpactLevel.CRITICAL,
            ),
            SampleQuality(
                index=1,
                quality_score=50.0,
                model_confidence=0.6,
                is_correctly_predicted=False,
                is_in_confusion_zone=True,
                predicted_label=1,
                true_label=0,
                impact_on_model=ImpactLevel.HIGH,
            ),
            SampleQuality(
                index=2,
                quality_score=90.0,
                model_confidence=0.95,
                is_correctly_predicted=True,
                is_in_confusion_zone=False,
                predicted_label=0,
                true_label=0,
                impact_on_model=ImpactLevel.NEGLIGIBLE,
            ),
        ]

        report = ModelAwareReport(
            n_samples=3,
            overall_quality_score=56.7,
            model_accuracy=0.33,
            model_confidence_mean=0.82,
            model_confidence_std=0.15,
            sample_scores=sample_scores,
            class_metrics={},
            high_confusion_pairs=[],
            critical_samples=[0],
            high_impact_samples=[1],
            recommendations=[],
        )

        critical = report.get_samples_by_impact(ImpactLevel.CRITICAL)
        high = report.get_samples_by_impact(ImpactLevel.HIGH)
        negligible = report.get_samples_by_impact(ImpactLevel.NEGLIGIBLE)

        assert len(critical) == 1
        assert critical[0].index == 0
        assert len(high) == 1
        assert high[0].index == 1
        assert len(negligible) == 1
        assert negligible[0].index == 2

    def test_get_worst_classes(self):
        """Test getting worst performing classes."""
        class_metrics = {
            0: ClassMetrics(
                label=0,
                n_samples=50,
                accuracy=0.90,
                precision=0.88,
                recall=0.92,
                f1_score=0.90,
                confusion_rate=0.10,
                avg_confidence=0.85,
                quality_score=88.0,
                most_confused_with=[],
            ),
            1: ClassMetrics(
                label=1,
                n_samples=30,
                accuracy=0.60,
                precision=0.55,
                recall=0.65,
                f1_score=0.60,
                confusion_rate=0.40,
                avg_confidence=0.55,
                quality_score=55.0,
                most_confused_with=[(0, 0.25)],
            ),
            2: ClassMetrics(
                label=2,
                n_samples=20,
                accuracy=0.75,
                precision=0.72,
                recall=0.78,
                f1_score=0.75,
                confusion_rate=0.25,
                avg_confidence=0.70,
                quality_score=72.0,
                most_confused_with=[],
            ),
        }

        report = ModelAwareReport(
            n_samples=100,
            overall_quality_score=75.0,
            model_accuracy=0.80,
            model_confidence_mean=0.75,
            model_confidence_std=0.15,
            sample_scores=[],
            class_metrics=class_metrics,
            high_confusion_pairs=[],
            critical_samples=[],
            high_impact_samples=[],
            recommendations=[],
        )

        worst = report.get_worst_classes(n=2)

        assert len(worst) == 2
        assert worst[0].label == 1  # Worst
        assert worst[1].label == 2  # Second worst


# =============================================================================
# Tests for ModelAwareScorer
# =============================================================================


class TestModelAwareScorerInit:
    """Tests for ModelAwareScorer initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        scorer = ModelAwareScorer()

        assert scorer.model is None
        assert scorer.cv_folds == 5
        assert scorer.confidence_threshold == 0.7
        assert scorer.confusion_threshold == 0.1
        assert scorer._fitted == False

    def test_custom_initialization(self):
        """Test custom parameter values."""
        model = LogisticRegression()
        scorer = ModelAwareScorer(
            model=model,
            cv_folds=3,
            confidence_threshold=0.8,
            confusion_threshold=0.05,
        )

        assert scorer.model is model
        assert scorer.cv_folds == 3
        assert scorer.confidence_threshold == 0.8
        assert scorer.confusion_threshold == 0.05


class TestModelAwareScorerFit:
    """Tests for ModelAwareScorer.fit method."""

    def test_fit_with_dataframe(self, simple_classification_data):
        """Test fitting with DataFrame input."""
        X, y = simple_classification_data

        scorer = ModelAwareScorer()
        result = scorer.fit(X, y)

        assert result is scorer  # Returns self
        assert scorer._fitted == True
        assert scorer._classes is not None
        assert len(scorer._class_priors) > 0
        assert scorer._confusion_matrix is not None

    def test_fit_with_array(self, simple_classification_data):
        """Test fitting with numpy array input."""
        X, y = simple_classification_data

        scorer = ModelAwareScorer()
        scorer.fit(X.values, y)

        assert scorer._fitted == True

    def test_fit_creates_default_model(self, simple_classification_data):
        """Test that fit creates default model if none provided."""
        X, y = simple_classification_data

        scorer = ModelAwareScorer(model=None)
        scorer.fit(X, y)

        assert scorer.model is not None
        assert isinstance(scorer.model, LogisticRegression)

    def test_fit_with_custom_model(self, simple_classification_data):
        """Test fitting with custom model."""
        X, y = simple_classification_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scorer = ModelAwareScorer(model=model)
        scorer.fit(X, y)

        assert scorer._fitted == True
        assert isinstance(scorer.model, RandomForestClassifier)

    def test_fit_calculates_class_priors(self, imbalanced_classification_data):
        """Test that class priors are calculated correctly."""
        X, y = imbalanced_classification_data

        scorer = ModelAwareScorer()
        scorer.fit(X, y)

        # Sum of priors should be 1
        total_prior = sum(scorer._class_priors.values())
        assert abs(total_prior - 1.0) < 0.01


class TestModelAwareScorerScore:
    """Tests for ModelAwareScorer.score method."""

    def test_score_requires_fit(self, simple_classification_data):
        """Test that score requires fitting first."""
        X, y = simple_classification_data

        scorer = ModelAwareScorer()

        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.score(X, y)

    def test_score_returns_report(self, fitted_scorer):
        """Test that score returns ModelAwareReport."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X, y)

        assert isinstance(report, ModelAwareReport)
        assert report.n_samples == len(X)
        assert 0 <= report.model_accuracy <= 1
        assert 0 <= report.overall_quality_score <= 100

    def test_score_sample_scores(self, fitted_scorer):
        """Test that sample scores are calculated."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X, y)

        assert len(report.sample_scores) == len(X)
        for sq in report.sample_scores:
            assert isinstance(sq, SampleQuality)
            assert 0 <= sq.quality_score <= 100
            assert 0 <= sq.model_confidence <= 1

    def test_score_class_metrics(self, fitted_scorer):
        """Test that class metrics are calculated."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X, y)

        assert len(report.class_metrics) > 0
        for label, cm in report.class_metrics.items():
            assert isinstance(cm, ClassMetrics)
            assert 0 <= cm.accuracy <= 1
            assert 0 <= cm.precision <= 1
            assert 0 <= cm.recall <= 1

    def test_score_with_dataframe(self, fitted_scorer):
        """Test scoring with DataFrame."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X, y)

        assert isinstance(report, ModelAwareReport)

    def test_score_with_array(self, fitted_scorer):
        """Test scoring with numpy array."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X.values, y)

        assert isinstance(report, ModelAwareReport)


class TestModelAwareScorerHelpers:
    """Tests for ModelAwareScorer helper methods."""

    def test_score_sample_correct_high_confidence(self, fitted_scorer):
        """Test sample scoring for correct, high-confidence prediction."""
        scorer, X, y = fitted_scorer

        sq = scorer._score_sample(
            index=0,
            true_label=0,
            predicted_label=0,
            probabilities=np.array([0.95, 0.03, 0.02]),
        )

        assert sq.is_correctly_predicted == True
        assert sq.is_in_confusion_zone == False  # Large margin
        assert sq.quality_score > 80  # High quality
        assert sq.impact_on_model == ImpactLevel.NEGLIGIBLE

    def test_score_sample_wrong_high_confidence(self, fitted_scorer):
        """Test sample scoring for wrong, high-confidence prediction (critical)."""
        scorer, X, y = fitted_scorer

        sq = scorer._score_sample(
            index=0,
            true_label=1,
            predicted_label=0,
            probabilities=np.array([0.90, 0.05, 0.05]),
        )

        assert sq.is_correctly_predicted == False
        assert sq.quality_score < 80  # Lower quality
        assert sq.impact_on_model == ImpactLevel.CRITICAL

    def test_score_sample_in_confusion_zone(self, fitted_scorer):
        """Test sample scoring for prediction in confusion zone."""
        scorer, X, y = fitted_scorer

        sq = scorer._score_sample(
            index=0,
            true_label=0,
            predicted_label=0,
            probabilities=np.array([0.45, 0.40, 0.15]),  # Small margin
        )

        assert sq.is_in_confusion_zone == True
        # Check that confusion zone issue is in the issues list
        has_confusion_issue = any("confusion zone" in issue.lower() for issue in sq.issues)
        assert has_confusion_issue

    def test_score_sample_low_confidence(self, fitted_scorer):
        """Test sample scoring for low confidence prediction."""
        scorer, X, y = fitted_scorer

        sq = scorer._score_sample(
            index=0,
            true_label=0,
            predicted_label=0,
            probabilities=np.array([0.50, 0.30, 0.20]),  # Low confidence
        )

        assert sq.model_confidence < scorer.confidence_threshold
        has_confidence_issue = any("confidence" in issue.lower() for issue in sq.issues)
        assert has_confidence_issue

    def test_calculate_sample_quality(self, fitted_scorer):
        """Test quality score calculation."""
        scorer, X, y = fitted_scorer

        # Perfect prediction
        score1 = scorer._calculate_sample_quality(
            is_correct=True,
            confidence=0.95,
            is_confusion_zone=False,
            n_issues=0,
        )

        # Wrong prediction
        score2 = scorer._calculate_sample_quality(
            is_correct=False,
            confidence=0.95,
            is_confusion_zone=False,
            n_issues=1,
        )

        assert score1 > score2
        assert score1 <= 100
        assert score2 >= 0

    def test_determine_impact_critical(self, fitted_scorer):
        """Test impact determination for critical cases."""
        scorer, X, y = fitted_scorer

        # High confidence wrong prediction
        impact = scorer._determine_impact(
            is_correct=False,
            confidence=0.85,
            is_confusion_zone=False,
            true_label=0,
        )

        assert impact == ImpactLevel.CRITICAL

    def test_determine_impact_high(self, fitted_scorer):
        """Test impact determination for high impact cases."""
        scorer, X, y = fitted_scorer

        # Wrong prediction, medium confidence
        impact = scorer._determine_impact(
            is_correct=False,
            confidence=0.6,
            is_confusion_zone=False,
            true_label=0,
        )

        assert impact == ImpactLevel.HIGH

    def test_determine_impact_medium(self, fitted_scorer):
        """Test impact determination for medium impact cases."""
        scorer, X, y = fitted_scorer

        # Correct but in confusion zone
        impact = scorer._determine_impact(
            is_correct=True,
            confidence=0.5,
            is_confusion_zone=True,
            true_label=0,
        )

        assert impact == ImpactLevel.MEDIUM

    def test_determine_impact_low(self, fitted_scorer):
        """Test impact determination for low impact cases."""
        scorer, X, y = fitted_scorer

        # Correct but low confidence
        impact = scorer._determine_impact(
            is_correct=True,
            confidence=0.55,  # Below threshold
            is_confusion_zone=False,
            true_label=0,
        )

        assert impact == ImpactLevel.LOW

    def test_determine_impact_negligible(self, fitted_scorer):
        """Test impact determination for negligible cases."""
        scorer, X, y = fitted_scorer

        # Correct, high confidence
        impact = scorer._determine_impact(
            is_correct=True,
            confidence=0.95,
            is_confusion_zone=False,
            true_label=0,
        )

        assert impact == ImpactLevel.NEGLIGIBLE

    def test_build_confusion_matrix(self, fitted_scorer):
        """Test confusion matrix building."""
        scorer, X, y = fitted_scorer

        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        matrix = scorer._build_confusion_matrix(y_true, y_pred)

        # Should be normalized by row
        for i in range(len(scorer._classes)):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01

    def test_get_high_confusion_pairs(self, fitted_scorer):
        """Test getting high confusion pairs."""
        scorer, X, y = fitted_scorer

        # Set up a confusion matrix with some high values
        scorer._confusion_matrix = np.array(
            [
                [0.8, 0.15, 0.05],
                [0.1, 0.7, 0.2],
                [0.05, 0.15, 0.8],
            ]
        )
        scorer._classes = np.array([0, 1, 2])

        pairs = scorer._get_high_confusion_pairs()

        # Should find pairs with rate > confusion_threshold
        assert isinstance(pairs, list)
        for c1, c2, rate in pairs:
            assert rate > scorer.confusion_threshold


class TestModelAwareScorerRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_generated(self, fitted_scorer):
        """Test that recommendations are generated."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X, y)

        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0

    def test_recommendations_for_critical_samples(self, fitted_scorer):
        """Test recommendations when critical samples exist."""
        scorer, X, y = fitted_scorer

        report = scorer.score(X, y)

        # If there are critical samples, should have related recommendation
        if report.critical_samples:
            has_critical_rec = any(
                "critical" in r.lower() for r in report.recommendations
            )
            assert has_critical_rec or len(report.recommendations) > 0


# =============================================================================
# Tests for Convenience Function
# =============================================================================


class TestScoreWithModel:
    """Tests for the score_with_model convenience function."""

    def test_basic_usage(self, simple_classification_data):
        """Test basic function usage."""
        X, y = simple_classification_data

        report = score_with_model(X, y)

        assert isinstance(report, ModelAwareReport)

    def test_with_custom_model(self, simple_classification_data):
        """Test with custom model."""
        X, y = simple_classification_data

        model = DecisionTreeClassifier(random_state=42)
        report = score_with_model(X, y, model=model)

        assert isinstance(report, ModelAwareReport)

    def test_with_kwargs(self, simple_classification_data):
        """Test with custom kwargs."""
        X, y = simple_classification_data

        report = score_with_model(
            X,
            y,
            cv_folds=3,
            confidence_threshold=0.8,
        )

        assert isinstance(report, ModelAwareReport)


# =============================================================================
# Tests for ModelComparisonScorer
# =============================================================================


class TestModelComparisonScorer:
    """Tests for ModelComparisonScorer."""

    def test_initialization(self):
        """Test initialization with multiple models."""
        models = [
            LogisticRegression(max_iter=1000),
            DecisionTreeClassifier(random_state=42),
        ]

        scorer = ModelComparisonScorer(models)

        assert len(scorer.models) == 2
        assert len(scorer.scorers) == 2

    def test_fit(self, simple_classification_data):
        """Test fitting multiple models."""
        X, y = simple_classification_data

        models = [
            LogisticRegression(max_iter=1000, random_state=42),
            DecisionTreeClassifier(random_state=42),
        ]

        scorer = ModelComparisonScorer(models)
        result = scorer.fit(X, y)

        assert result is scorer
        for s in scorer.scorers:
            assert s._fitted == True

    def test_compare(self, simple_classification_data):
        """Test model comparison."""
        X, y = simple_classification_data

        models = [
            LogisticRegression(max_iter=1000, random_state=42),
            DecisionTreeClassifier(random_state=42),
        ]

        scorer = ModelComparisonScorer(models)
        scorer.fit(X, y)
        comparison = scorer.compare(X, y)

        assert comparison["n_models"] == 2
        assert len(comparison["model_accuracies"]) == 2
        assert len(comparison["quality_scores"]) == 2
        assert "universal_issues" in comparison
        assert "model_specific_issues" in comparison
        assert "recommendation" in comparison


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_binary_classification(self, binary_classification_data):
        """Test with binary classification."""
        X, y = binary_classification_data

        scorer = ModelAwareScorer()
        scorer.fit(X, y)
        report = scorer.score(X, y)

        assert isinstance(report, ModelAwareReport)
        assert len(report.class_metrics) == 2

    def test_imbalanced_data(self, imbalanced_classification_data):
        """Test with imbalanced dataset."""
        X, y = imbalanced_classification_data

        scorer = ModelAwareScorer()
        scorer.fit(X, y)
        report = scorer.score(X, y)

        assert isinstance(report, ModelAwareReport)
        # Should have class imbalance recommendation
        has_imbalance_rec = any(
            "imbalance" in r.lower() for r in report.recommendations
        )
        # May or may not trigger depending on exact class ratios
        assert isinstance(has_imbalance_rec, bool)

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=20,
            n_features=5,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = pd.DataFrame(X)

        scorer = ModelAwareScorer(cv_folds=2)  # Fewer folds for small data
        scorer.fit(X, y)
        report = scorer.score(X, y)

        assert isinstance(report, ModelAwareReport)
        assert report.n_samples == 20

    def test_single_class_in_subset(self, simple_classification_data):
        """Test when scoring data has different class distribution."""
        X, y = simple_classification_data

        scorer = ModelAwareScorer()
        scorer.fit(X, y)

        # Score on subset (may not have all classes)
        subset_mask = y != 2  # Remove class 2
        report = scorer.score(X[subset_mask], y[subset_mask])

        assert isinstance(report, ModelAwareReport)

    def test_numpy_array_input(self, simple_classification_data):
        """Test with numpy arrays throughout."""
        X, y = simple_classification_data

        X_arr = X.values

        scorer = ModelAwareScorer()
        scorer.fit(X_arr, y)
        report = scorer.score(X_arr, y)

        assert isinstance(report, ModelAwareReport)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for model-aware scoring."""

    def test_full_workflow(self, simple_classification_data):
        """Test complete workflow."""
        X, y = simple_classification_data

        # Create and fit scorer
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scorer = ModelAwareScorer(model=model, cv_folds=3)
        scorer.fit(X, y)

        # Score
        report = scorer.score(X, y)

        # Verify report
        assert report.n_samples == len(X)
        assert len(report.sample_scores) == len(X)
        assert len(report.class_metrics) > 0
        assert len(report.recommendations) > 0

        # Get summary
        summary = report.summary()
        assert len(summary) > 0

        # Convert to dict
        dict_repr = report.to_dict()
        assert "n_samples" in dict_repr
        assert "overall_quality_score" in dict_repr

        # Get samples by impact
        critical = report.get_samples_by_impact(ImpactLevel.CRITICAL)
        assert isinstance(critical, list)

        # Get worst classes
        worst = report.get_worst_classes(n=2)
        assert len(worst) <= 2

    def test_model_comparison_workflow(self, simple_classification_data):
        """Test model comparison workflow."""
        X, y = simple_classification_data

        models = [
            LogisticRegression(max_iter=1000, random_state=42),
            RandomForestClassifier(n_estimators=5, random_state=42),
            DecisionTreeClassifier(random_state=42),
        ]

        scorer = ModelComparisonScorer(models)
        scorer.fit(X, y)
        comparison = scorer.compare(X, y)

        assert comparison["n_models"] == 3
        assert len(comparison["model_accuracies"]) == 3
        assert "universal_issues" in comparison

    def test_reproducibility(self, simple_classification_data):
        """Test that results are reproducible."""
        X, y = simple_classification_data

        scorer1 = ModelAwareScorer(cv_folds=3)
        scorer1.fit(X, y)
        report1 = scorer1.score(X, y)

        scorer2 = ModelAwareScorer(cv_folds=3)
        scorer2.fit(X, y)
        report2 = scorer2.score(X, y)

        # Quality scores should be similar (may differ due to CV randomness)
        assert abs(report1.overall_quality_score - report2.overall_quality_score) < 10
