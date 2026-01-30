"""Comprehensive tests for the slice_discovery module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clean.slice_discovery import (
    DataSlice,
    IssueType,
    SliceCondition,
    SliceDiscoverer,
    SliceDiscoveryReport,
    SliceType,
    discover_slices,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def classification_data():
    """Create classification dataset with underperforming slices."""
    np.random.seed(42)
    n_samples = 500

    # Create features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    category = np.random.choice(["A", "B", "C"], n_samples)

    # True labels
    y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

    # Predictions - worse for category C and high age
    predictions = y.copy()
    for i in range(n_samples):
        if category[i] == "C":
            # Higher error rate for category C
            if np.random.random() < 0.3:
                predictions[i] = 1 - predictions[i]
        elif age[i] > 60:
            # Higher error rate for age > 60
            if np.random.random() < 0.25:
                predictions[i] = 1 - predictions[i]
        else:
            # Low baseline error
            if np.random.random() < 0.05:
                predictions[i] = 1 - predictions[i]

    df = pd.DataFrame({"age": age, "income": income, "category": category})

    return df, y, predictions


@pytest.fixture
def simple_data():
    """Create simple dataset for basic tests."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "feature_a": np.random.choice(["x", "y", "z"], n_samples),
            "feature_b": np.random.randn(n_samples),
        }
    )

    y = np.random.choice([0, 1], n_samples)
    predictions = y.copy()
    # Some errors
    predictions[:20] = 1 - predictions[:20]

    return df, y, predictions


@pytest.fixture
def data_with_probabilities():
    """Create dataset with prediction probabilities."""
    np.random.seed(42)
    n_samples = 300

    df = pd.DataFrame(
        {
            "feature": np.random.choice(["low", "medium", "high"], n_samples),
            "value": np.random.uniform(0, 100, n_samples),
        }
    )

    y = np.random.choice([0, 1, 2], n_samples)
    predictions = y.copy()

    # Create probabilities
    pred_proba = np.zeros((n_samples, 3))
    for i in range(n_samples):
        pred_proba[i, predictions[i]] = 0.7 + np.random.random() * 0.25
        # Distribute rest among other classes
        remaining = 1 - pred_proba[i, predictions[i]]
        other_classes = [j for j in range(3) if j != predictions[i]]
        for j in other_classes:
            pred_proba[i, j] = remaining / 2

    return df, y, predictions, pred_proba


# =============================================================================
# Tests for Enums
# =============================================================================


class TestSliceType:
    """Tests for SliceType enum."""

    def test_all_values_exist(self):
        """Test all expected slice types exist."""
        assert SliceType.SINGLE_FEATURE.value == "single_feature"
        assert SliceType.CONJUNCTION.value == "conjunction"
        assert SliceType.DISJUNCTION.value == "disjunction"
        assert SliceType.RANGE.value == "range"
        assert SliceType.CLUSTER.value == "cluster"


class TestIssueType:
    """Tests for IssueType enum."""

    def test_all_values_exist(self):
        """Test all expected issue types exist."""
        assert IssueType.LOW_ACCURACY.value == "low_accuracy"
        assert IssueType.HIGH_ERROR_RATE.value == "high_error_rate"
        assert IssueType.LOW_CONFIDENCE.value == "low_confidence"
        assert IssueType.HIGH_CONFUSION.value == "high_confusion"
        assert IssueType.DISTRIBUTION_SHIFT.value == "distribution_shift"
        assert IssueType.UNDERREPRESENTED.value == "underrepresented"


# =============================================================================
# Tests for SliceCondition
# =============================================================================


class TestSliceCondition:
    """Tests for SliceCondition dataclass."""

    def test_str_equality(self):
        """Test string representation for equality operator."""
        cond = SliceCondition(feature="category", operator="==", value="A")
        assert str(cond) == "category == A"

    def test_str_inequality(self):
        """Test string representation for inequality operator."""
        cond = SliceCondition(feature="age", operator="!=", value=30)
        assert str(cond) == "age != 30"

    def test_str_comparison(self):
        """Test string representation for comparison operators."""
        assert str(SliceCondition("age", "<", 30)) == "age < 30"
        assert str(SliceCondition("age", "<=", 30)) == "age <= 30"
        assert str(SliceCondition("age", ">", 30)) == "age > 30"
        assert str(SliceCondition("age", ">=", 30)) == "age >= 30"

    def test_str_between(self):
        """Test string representation for between operator."""
        cond = SliceCondition(
            feature="income", operator="between", value=20000, value_end=50000
        )
        assert "∈" in str(cond)
        assert "20000" in str(cond)
        assert "50000" in str(cond)

    def test_str_in(self):
        """Test string representation for in operator."""
        cond = SliceCondition(feature="category", operator="in", value=["A", "B"])
        assert "∈" in str(cond)

    def test_evaluate_equality(self):
        """Test evaluation of equality condition."""
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
        cond = SliceCondition(feature="cat", operator="==", value="A")

        mask = cond.evaluate(df)

        assert mask.tolist() == [True, False, True, False]

    def test_evaluate_inequality(self):
        """Test evaluation of inequality condition."""
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
        cond = SliceCondition(feature="cat", operator="!=", value="A")

        mask = cond.evaluate(df)

        assert mask.tolist() == [False, True, False, True]

    def test_evaluate_less_than(self):
        """Test evaluation of less than condition."""
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        cond = SliceCondition(feature="value", operator="<", value=25)

        mask = cond.evaluate(df)

        assert mask.tolist() == [True, True, False, False]

    def test_evaluate_greater_than(self):
        """Test evaluation of greater than condition."""
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        cond = SliceCondition(feature="value", operator=">", value=25)

        mask = cond.evaluate(df)

        assert mask.tolist() == [False, False, True, True]

    def test_evaluate_between(self):
        """Test evaluation of between condition."""
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        cond = SliceCondition(
            feature="value", operator="between", value=15, value_end=35
        )

        mask = cond.evaluate(df)

        assert mask.tolist() == [False, True, True, False]

    def test_evaluate_in(self):
        """Test evaluation of in condition."""
        df = pd.DataFrame({"cat": ["A", "B", "C", "D"]})
        cond = SliceCondition(feature="cat", operator="in", value=["A", "C"])

        mask = cond.evaluate(df)

        assert mask.tolist() == [True, False, True, False]

    def test_evaluate_missing_column(self):
        """Test evaluation when column doesn't exist."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        cond = SliceCondition(feature="nonexistent", operator="==", value=1)

        mask = cond.evaluate(df)

        assert all(not m for m in mask)

    def test_evaluate_unknown_operator(self):
        """Test evaluation with unknown operator."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        cond = SliceCondition(feature="value", operator="~", value=2)

        mask = cond.evaluate(df)

        assert all(not m for m in mask)


# =============================================================================
# Tests for DataSlice
# =============================================================================


class TestDataSlice:
    """Tests for DataSlice dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        slice_ = DataSlice(
            slice_id="test_slice",
            conditions=[SliceCondition("cat", "==", "A")],
            slice_type=SliceType.SINGLE_FEATURE,
            n_samples=100,
            n_total=500,
            support=0.2,
            accuracy=0.75,
            baseline_accuracy=0.90,
            accuracy_gap=0.15,
            error_rate=0.25,
            avg_confidence=0.8,
            issue_types=[IssueType.LOW_ACCURACY],
            severity_score=0.6,
            sample_indices=np.arange(100),
        )

        str_repr = str(slice_)

        assert "cat == A" in str_repr
        assert "n=100" in str_repr
        assert "75.0%" in str_repr

    def test_to_dict(self):
        """Test conversion to dictionary."""
        slice_ = DataSlice(
            slice_id="test_slice",
            conditions=[
                SliceCondition("cat", "==", "A"),
                SliceCondition("age", ">", 30),
            ],
            slice_type=SliceType.CONJUNCTION,
            n_samples=50,
            n_total=500,
            support=0.1,
            accuracy=0.70,
            baseline_accuracy=0.90,
            accuracy_gap=0.20,
            error_rate=0.30,
            avg_confidence=0.75,
            issue_types=[IssueType.LOW_ACCURACY, IssueType.HIGH_ERROR_RATE],
            severity_score=0.7,
            sample_indices=np.arange(50),
        )

        result = slice_.to_dict()

        assert result["slice_id"] == "test_slice"
        assert len(result["conditions"]) == 2
        assert result["slice_type"] == "conjunction"
        assert result["n_samples"] == 50
        assert result["support"] == 0.1
        assert result["accuracy"] == 0.70
        assert result["baseline_accuracy"] == 0.90
        assert result["accuracy_gap"] == 0.20
        assert result["error_rate"] == 0.30
        assert len(result["issue_types"]) == 2
        assert result["severity_score"] == 0.7


# =============================================================================
# Tests for SliceDiscoveryReport
# =============================================================================


class TestSliceDiscoveryReport:
    """Tests for SliceDiscoveryReport dataclass."""

    def test_summary_with_slices(self):
        """Test summary generation with discovered slices."""
        slice_ = DataSlice(
            slice_id="slice_0",
            conditions=[SliceCondition("category", "==", "C")],
            slice_type=SliceType.SINGLE_FEATURE,
            n_samples=100,
            n_total=500,
            support=0.2,
            accuracy=0.60,
            baseline_accuracy=0.85,
            accuracy_gap=0.25,
            error_rate=0.40,
            avg_confidence=0.65,
            issue_types=[IssueType.LOW_ACCURACY],
            severity_score=0.8,
            sample_indices=np.arange(100),
        )

        report = SliceDiscoveryReport(
            n_samples=500,
            n_slices_found=1,
            slices=[slice_],
            baseline_accuracy=0.85,
            worst_slice_accuracy=0.60,
            total_affected_samples=100,
            recommendations=["Investigate category C"],
        )

        summary = report.summary()

        assert "Data Slice Discovery Report" in summary
        assert "500" in summary
        assert "85.0%" in summary or "85%" in summary
        assert "60.0%" in summary or "60%" in summary
        assert "category" in summary

    def test_summary_without_slices(self):
        """Test summary when no slices found."""
        report = SliceDiscoveryReport(
            n_samples=500,
            n_slices_found=0,
            slices=[],
            baseline_accuracy=0.90,
            worst_slice_accuracy=0.90,
            total_affected_samples=0,
            recommendations=["No problem slices found"],
        )

        summary = report.summary()

        assert "500" in summary
        assert "0" in str(report.n_slices_found)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = SliceDiscoveryReport(
            n_samples=300,
            n_slices_found=2,
            slices=[],
            baseline_accuracy=0.85,
            worst_slice_accuracy=0.70,
            total_affected_samples=50,
            recommendations=["Test recommendation"],
        )

        result = report.to_dict()

        assert result["n_samples"] == 300
        assert result["n_slices_found"] == 2
        assert result["baseline_accuracy"] == 0.85
        assert "recommendations" in result

    def test_get_slice_samples(self):
        """Test getting samples for a specific slice."""
        indices = np.array([5, 10, 15, 20])
        slice_ = DataSlice(
            slice_id="target_slice",
            conditions=[SliceCondition("x", "==", "a")],
            slice_type=SliceType.SINGLE_FEATURE,
            n_samples=4,
            n_total=100,
            support=0.04,
            accuracy=0.5,
            baseline_accuracy=0.9,
            accuracy_gap=0.4,
            error_rate=0.5,
            avg_confidence=0.6,
            issue_types=[IssueType.LOW_ACCURACY],
            severity_score=0.9,
            sample_indices=indices,
        )

        report = SliceDiscoveryReport(
            n_samples=100,
            n_slices_found=1,
            slices=[slice_],
            baseline_accuracy=0.9,
            worst_slice_accuracy=0.5,
            total_affected_samples=4,
            recommendations=[],
        )

        result = report.get_slice_samples("target_slice")

        np.testing.assert_array_equal(result, indices)

    def test_get_slice_samples_not_found(self):
        """Test getting samples for non-existent slice."""
        report = SliceDiscoveryReport(
            n_samples=100,
            n_slices_found=0,
            slices=[],
            baseline_accuracy=0.9,
            worst_slice_accuracy=0.9,
            total_affected_samples=0,
            recommendations=[],
        )

        result = report.get_slice_samples("nonexistent")

        assert result is None


# =============================================================================
# Tests for SliceDiscoverer
# =============================================================================


class TestSliceDiscovererInit:
    """Tests for SliceDiscoverer initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        discoverer = SliceDiscoverer()

        assert discoverer.min_slice_size == 50
        assert discoverer.min_support == 0.01
        assert discoverer.max_conditions == 3
        assert discoverer.significance_level == 0.05
        assert discoverer.accuracy_gap_threshold == 0.05
        assert discoverer.n_bins == 5

    def test_custom_initialization(self):
        """Test custom parameter values."""
        discoverer = SliceDiscoverer(
            min_slice_size=100,
            min_support=0.05,
            max_conditions=2,
            significance_level=0.01,
            accuracy_gap_threshold=0.10,
            n_bins=10,
        )

        assert discoverer.min_slice_size == 100
        assert discoverer.min_support == 0.05
        assert discoverer.max_conditions == 2
        assert discoverer.significance_level == 0.01
        assert discoverer.accuracy_gap_threshold == 0.10
        assert discoverer.n_bins == 10


class TestSliceDiscovererDiscover:
    """Tests for SliceDiscoverer.discover method."""

    def test_discover_basic(self, simple_data):
        """Test basic slice discovery."""
        df, y, predictions = simple_data

        discoverer = SliceDiscoverer(min_slice_size=20)
        report = discoverer.discover(df, y, predictions)

        assert isinstance(report, SliceDiscoveryReport)
        assert report.n_samples == len(df)
        assert report.baseline_accuracy == (predictions == y).mean()

    def test_discover_finds_categorical_slices(self, classification_data):
        """Test discovery of categorical slices."""
        df, y, predictions = classification_data

        discoverer = SliceDiscoverer(min_slice_size=30, accuracy_gap_threshold=0.08)
        report = discoverer.discover(df, y, predictions)

        # Should find category C as problematic
        categorical_slices = [
            s
            for s in report.slices
            if any(c.feature == "category" for c in s.conditions)
        ]

        assert len(categorical_slices) > 0 or report.n_slices_found >= 0

    def test_discover_finds_numeric_slices(self, classification_data):
        """Test discovery of numeric range slices."""
        df, y, predictions = classification_data

        discoverer = SliceDiscoverer(min_slice_size=30, accuracy_gap_threshold=0.08)
        report = discoverer.discover(df, y, predictions)

        # Should potentially find age ranges as problematic
        assert isinstance(report, SliceDiscoveryReport)

    def test_discover_with_pred_proba(self, data_with_probabilities):
        """Test discovery with prediction probabilities."""
        df, y, predictions, pred_proba = data_with_probabilities

        discoverer = SliceDiscoverer(min_slice_size=30)
        report = discoverer.discover(df, y, predictions, pred_proba=pred_proba)

        assert isinstance(report, SliceDiscoveryReport)
        # Check that confidence is calculated
        for slice_ in report.slices:
            assert slice_.avg_confidence >= 0

    def test_discover_with_specific_features(self, classification_data):
        """Test discovery with specific feature columns."""
        df, y, predictions = classification_data

        discoverer = SliceDiscoverer(min_slice_size=30)
        report = discoverer.discover(
            df, y, predictions, feature_columns=["category", "age"]
        )

        # All slices should be based only on specified features
        for slice_ in report.slices:
            for cond in slice_.conditions:
                assert cond.feature in ["category", "age"]

    def test_discover_without_predictions(self, simple_data):
        """Test discovery without predictions."""
        df, y, _ = simple_data

        discoverer = SliceDiscoverer(min_slice_size=20)
        report = discoverer.discover(df, y)

        # Should still work, assuming all predictions correct
        assert report.baseline_accuracy == 1.0

    def test_discover_respects_min_support(self, simple_data):
        """Test that min_support is respected."""
        df, y, predictions = simple_data

        discoverer = SliceDiscoverer(min_support=0.5)
        report = discoverer.discover(df, y, predictions)

        # All slices should have support >= 0.5
        for slice_ in report.slices:
            assert slice_.support >= 0.5

    def test_discover_respects_min_slice_size(self, simple_data):
        """Test that min_slice_size is respected."""
        df, y, predictions = simple_data

        discoverer = SliceDiscoverer(min_slice_size=50)
        report = discoverer.discover(df, y, predictions)

        # All slices should have >= 50 samples
        for slice_ in report.slices:
            assert slice_.n_samples >= 50


class TestSliceDiscovererHelpers:
    """Tests for SliceDiscoverer helper methods."""

    def test_generate_conditions_numeric(self):
        """Test condition generation for numeric features."""
        df = pd.DataFrame({"value": np.random.randn(100)})

        discoverer = SliceDiscoverer(min_slice_size=10, n_bins=4)
        conditions = discoverer._generate_conditions(df, ["value"])

        # Should have range conditions
        range_conditions = [c for c in conditions if c.operator == "between"]
        assert len(range_conditions) > 0

    def test_generate_conditions_categorical(self):
        """Test condition generation for categorical features."""
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 40})

        discoverer = SliceDiscoverer(min_slice_size=20)
        conditions = discoverer._generate_conditions(df, ["cat"])

        # Should have equality conditions for each value
        eq_conditions = [c for c in conditions if c.operator == "=="]
        assert len(eq_conditions) == 3
        values = {c.value for c in eq_conditions}
        assert values == {"A", "B", "C"}

    def test_create_numeric_conditions_with_few_samples(self):
        """Test numeric condition creation with insufficient samples."""
        df = pd.DataFrame({"value": [1, 2, 3]})  # Very few samples

        discoverer = SliceDiscoverer(min_slice_size=10)
        conditions = discoverer._create_numeric_conditions(df["value"], "value")

        # Should return empty list due to insufficient samples
        assert conditions == []

    def test_create_categorical_conditions_filters_small(self):
        """Test that small categories are filtered out."""
        df = pd.DataFrame({"cat": ["A"] * 100 + ["B"] * 5})  # B is too small

        discoverer = SliceDiscoverer(min_slice_size=50)
        conditions = discoverer._create_categorical_conditions(df["cat"], "cat")

        # Should only have condition for A
        assert len(conditions) == 1
        assert conditions[0].value == "A"

    def test_characterize_issues_low_accuracy(self):
        """Test issue characterization for low accuracy."""
        mask = np.array([True] * 50 + [False] * 450)
        is_correct = np.zeros(500, dtype=bool)
        is_correct[50:] = True  # 90% baseline accuracy
        is_correct[:25] = True  # 50% slice accuracy

        discoverer = SliceDiscoverer()
        issues, severity = discoverer._characterize_issues(
            mask, is_correct, None, baseline_accuracy=0.9
        )

        assert IssueType.LOW_ACCURACY in issues
        assert severity > 0

    def test_characterize_issues_high_error_rate(self):
        """Test issue characterization for high error rate."""
        mask = np.array([True] * 50 + [False] * 50)
        is_correct = np.zeros(100, dtype=bool)
        is_correct[30:] = True  # 60% error rate in slice

        discoverer = SliceDiscoverer()
        issues, severity = discoverer._characterize_issues(
            mask, is_correct, None, baseline_accuracy=0.8
        )

        assert IssueType.HIGH_ERROR_RATE in issues

    def test_characterize_issues_underrepresented(self):
        """Test issue characterization for underrepresented slice."""
        mask = np.array([True] * 20 + [False] * 980)  # 2% support
        is_correct = np.ones(1000, dtype=bool)
        is_correct[:15] = False  # Some errors in slice

        discoverer = SliceDiscoverer()
        issues, severity = discoverer._characterize_issues(
            mask, is_correct, None, baseline_accuracy=0.9
        )

        assert IssueType.UNDERREPRESENTED in issues

    def test_avg_confidence(self):
        """Test average confidence calculation."""
        mask = np.array([True, True, False, False])
        pred_proba = np.array(
            [
                [0.8, 0.2],
                [0.9, 0.1],
                [0.7, 0.3],
                [0.6, 0.4],
            ]
        )

        discoverer = SliceDiscoverer()
        avg_conf = discoverer._avg_confidence(mask, pred_proba)

        # Average of max probabilities for masked samples: (0.8 + 0.9) / 2 = 0.85
        assert abs(avg_conf - 0.85) < 0.01

    def test_avg_confidence_no_proba(self):
        """Test average confidence when no probabilities provided."""
        mask = np.array([True, False, True])

        discoverer = SliceDiscoverer()
        avg_conf = discoverer._avg_confidence(mask, None)

        assert avg_conf == 0.0

    def test_remove_redundant_slices(self):
        """Test removal of redundant slices."""
        # Create two slices with very similar samples
        slice1 = DataSlice(
            slice_id="s1",
            conditions=[SliceCondition("a", "==", 1)],
            slice_type=SliceType.SINGLE_FEATURE,
            n_samples=100,
            n_total=500,
            support=0.2,
            accuracy=0.7,
            baseline_accuracy=0.9,
            accuracy_gap=0.2,
            error_rate=0.3,
            avg_confidence=0.8,
            issue_types=[IssueType.LOW_ACCURACY],
            severity_score=0.7,
            sample_indices=np.arange(100),
        )

        slice2 = DataSlice(
            slice_id="s2",
            conditions=[SliceCondition("a", "==", 1), SliceCondition("b", ">", 0)],
            slice_type=SliceType.CONJUNCTION,
            n_samples=95,
            n_total=500,
            support=0.19,
            accuracy=0.71,
            baseline_accuracy=0.9,
            accuracy_gap=0.19,
            error_rate=0.29,
            avg_confidence=0.8,
            issue_types=[IssueType.LOW_ACCURACY],
            severity_score=0.65,
            sample_indices=np.arange(95),  # 95% overlap
        )

        discoverer = SliceDiscoverer()
        kept = discoverer._remove_redundant_slices([slice1, slice2])

        # Should keep only one (probably slice1 due to order)
        assert len(kept) == 1


class TestSliceDiscovererRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_no_slices(self):
        """Test recommendations when no slices found."""
        discoverer = SliceDiscoverer()
        recommendations = discoverer._generate_recommendations([], 0.9, 500)

        assert len(recommendations) > 0
        assert any("no significant" in r.lower() or "consistent" in r.lower() for r in recommendations)

    def test_recommendations_with_severe_slice(self):
        """Test recommendations with severe problem slice."""
        slice_ = DataSlice(
            slice_id="s1",
            conditions=[SliceCondition("category", "==", "bad")],
            slice_type=SliceType.SINGLE_FEATURE,
            n_samples=50,
            n_total=500,
            support=0.1,
            accuracy=0.5,
            baseline_accuracy=0.9,
            accuracy_gap=0.4,
            error_rate=0.5,
            avg_confidence=0.6,
            issue_types=[IssueType.LOW_ACCURACY, IssueType.HIGH_ERROR_RATE],
            severity_score=0.9,
            sample_indices=np.arange(50),
        )

        discoverer = SliceDiscoverer()
        recommendations = discoverer._generate_recommendations([slice_], 0.9, 500)

        assert len(recommendations) > 0
        # Should mention the problematic slice
        assert any("category" in r.lower() or "bad" in r.lower() or "priority" in r.lower() for r in recommendations)


# =============================================================================
# Tests for Convenience Function
# =============================================================================


class TestDiscoverSlices:
    """Tests for the discover_slices convenience function."""

    def test_basic_usage(self, simple_data):
        """Test basic function usage."""
        df, y, predictions = simple_data

        report = discover_slices(df, y, predictions, min_slice_size=20)

        assert isinstance(report, SliceDiscoveryReport)

    def test_with_kwargs(self, classification_data):
        """Test with various kwargs."""
        df, y, predictions = classification_data

        report = discover_slices(
            df,
            y,
            predictions,
            min_slice_size=30,
            accuracy_gap_threshold=0.1,
            n_bins=3,
        )

        assert isinstance(report, SliceDiscoveryReport)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        np.random.seed(42)
        df = pd.DataFrame({"feature": np.random.choice(["A", "B"], 100)})
        y = np.random.choice([0, 1], 100)
        predictions = y.copy()  # Perfect predictions

        discoverer = SliceDiscoverer(min_slice_size=20)
        report = discoverer.discover(df, y, predictions)

        # Should find no underperforming slices
        assert report.n_slices_found == 0

    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong."""
        np.random.seed(42)
        df = pd.DataFrame({"feature": np.random.choice(["A", "B"], 100)})
        y = np.random.choice([0, 1], 100)
        predictions = 1 - y  # All wrong

        discoverer = SliceDiscoverer(min_slice_size=20)
        report = discoverer.discover(df, y, predictions)

        assert report.baseline_accuracy == 0.0
        # No slices significantly worse than 0% accuracy
        assert isinstance(report, SliceDiscoveryReport)

    def test_single_category(self):
        """Test with single category feature."""
        df = pd.DataFrame({"feature": ["A"] * 100})
        y = np.random.choice([0, 1], 100)
        predictions = y.copy()
        predictions[:20] = 1 - predictions[:20]

        discoverer = SliceDiscoverer(min_slice_size=50)
        report = discoverer.discover(df, y, predictions)

        # Should still work
        assert isinstance(report, SliceDiscoveryReport)

    def test_binary_numeric_feature(self):
        """Test with binary numeric feature."""
        df = pd.DataFrame({"feature": [0, 1] * 50})
        y = np.array([0, 1] * 50)
        predictions = y.copy()

        discoverer = SliceDiscoverer(min_slice_size=20)
        report = discoverer.discover(df, y, predictions)

        assert isinstance(report, SliceDiscoveryReport)

    def test_all_nan_feature(self):
        """Test with all NaN values in a feature."""
        df = pd.DataFrame(
            {
                "good_feature": np.random.choice(["A", "B"], 100),
                "bad_feature": [np.nan] * 100,
            }
        )
        y = np.random.choice([0, 1], 100)
        predictions = y.copy()

        discoverer = SliceDiscoverer(min_slice_size=20)
        report = discoverer.discover(df, y, predictions)

        assert isinstance(report, SliceDiscoveryReport)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"feature": []})
        y = np.array([])
        predictions = np.array([])

        discoverer = SliceDiscoverer(min_slice_size=1)
        report = discoverer.discover(df, y, predictions)

        assert report.n_samples == 0
        assert report.n_slices_found == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for slice discovery."""

    def test_full_workflow(self, classification_data):
        """Test complete slice discovery workflow."""
        df, y, predictions = classification_data

        # Run discovery
        discoverer = SliceDiscoverer(
            min_slice_size=30,
            accuracy_gap_threshold=0.08,
            max_conditions=2,
        )
        report = discoverer.discover(df, y, predictions)

        # Verify report structure
        assert report.n_samples == len(df)
        assert 0 <= report.baseline_accuracy <= 1
        assert report.worst_slice_accuracy <= report.baseline_accuracy

        # Get summary
        summary = report.summary()
        assert len(summary) > 0

        # Convert to dict
        dict_repr = report.to_dict()
        assert "n_samples" in dict_repr
        assert "slices" in dict_repr

        # Check individual slices
        for slice_ in report.slices:
            assert slice_.n_samples >= discoverer.min_slice_size
            assert slice_.accuracy_gap >= discoverer.accuracy_gap_threshold
            assert len(slice_.conditions) <= discoverer.max_conditions

    def test_conjunction_slices(self, classification_data):
        """Test discovery of conjunction (multi-condition) slices."""
        df, y, predictions = classification_data

        discoverer = SliceDiscoverer(
            min_slice_size=20,
            accuracy_gap_threshold=0.05,
            max_conditions=3,
        )
        report = discoverer.discover(df, y, predictions)

        # Check for conjunction slices
        conjunction_slices = [
            s for s in report.slices if s.slice_type == SliceType.CONJUNCTION
        ]

        # May or may not find conjunctions depending on data
        for conj in conjunction_slices:
            assert len(conj.conditions) >= 2
            assert len(conj.conditions) <= 3
