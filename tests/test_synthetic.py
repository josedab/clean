"""Tests for synthetic data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clean.synthetic import (
    SyntheticDataValidator,
    SyntheticIssueType,
    SyntheticValidationReport,
    validate_synthetic_data,
)


class TestSyntheticDataValidator:
    """Tests for SyntheticDataValidator class."""

    @pytest.fixture
    def real_data(self) -> pd.DataFrame:
        """Create realistic reference data."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "age": np.random.normal(35, 10, n),
            "income": np.random.lognormal(10, 1, n),
            "category": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        })

    @pytest.fixture
    def good_synthetic(self, real_data: pd.DataFrame) -> pd.DataFrame:
        """Create good synthetic data similar to real."""
        np.random.seed(43)
        n = 500
        return pd.DataFrame({
            "age": np.random.normal(35, 10, n),
            "income": np.random.lognormal(10, 1, n),
            "category": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        })

    @pytest.fixture
    def collapsed_synthetic(self) -> pd.DataFrame:
        """Create synthetic data with mode collapse."""
        n = 500
        return pd.DataFrame({
            "age": np.full(n, 35.0),  # No variance - collapsed
            "income": np.random.lognormal(10, 0.1, n),  # Very low variance
            "category": np.array(["A"] * n),  # Single category
        })

    @pytest.fixture
    def memorized_synthetic(self, real_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic data that memorizes real data."""
        # Copy first 200 samples exactly
        memorized = real_data.head(200).copy()
        # Add some noise to make it slightly different
        memorized["age"] = memorized["age"] + np.random.normal(0, 0.01, len(memorized))
        return memorized

    def test_validator_init(self) -> None:
        validator = SyntheticDataValidator()
        assert validator is not None

    def test_set_reference(self, real_data: pd.DataFrame) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        assert validator._reference is not None

    def test_validate_requires_reference(self, good_synthetic: pd.DataFrame) -> None:
        validator = SyntheticDataValidator()
        with pytest.raises(RuntimeError, match="Reference data not set"):
            validator.validate(good_synthetic)

    def test_validate_good_synthetic(
        self, real_data: pd.DataFrame, good_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(good_synthetic)

        assert isinstance(report, SyntheticValidationReport)
        # Good synthetic data should have high fidelity at minimum
        assert report.fidelity_score > 80

    def test_detect_mode_collapse(
        self, real_data: pd.DataFrame, collapsed_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(collapsed_synthetic)

        # Should detect mode collapse
        collapse_issues = [
            i for i in report.issues
            if i.issue_type == SyntheticIssueType.MODE_COLLAPSE
        ]
        assert len(collapse_issues) > 0

    def test_detect_memorization(
        self, real_data: pd.DataFrame, memorized_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(memorized_synthetic)

        # Privacy score should be low due to memorization
        assert report.privacy_score < 80

    def test_fidelity_score(
        self, real_data: pd.DataFrame, good_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(good_synthetic)

        assert 0 <= report.fidelity_score <= 100

    def test_diversity_score(
        self, real_data: pd.DataFrame, good_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(good_synthetic)

        assert 0 <= report.diversity_score <= 100

    def test_feature_scores(
        self, real_data: pd.DataFrame, good_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(good_synthetic)

        assert len(report.feature_scores) > 0
        for score in report.feature_scores.values():
            assert 0 <= score <= 100

    def test_report_summary(
        self, real_data: pd.DataFrame, good_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(good_synthetic)

        summary = report.summary()
        assert "Synthetic Data Validation Report" in summary
        assert "Quality Scores" in summary

    def test_report_to_dict(
        self, real_data: pd.DataFrame, good_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(good_synthetic)

        d = report.to_dict()
        assert "quality_score" in d
        assert "fidelity_score" in d
        assert "issues" in d

    def test_recommendations_generated(
        self, real_data: pd.DataFrame, collapsed_synthetic: pd.DataFrame
    ) -> None:
        validator = SyntheticDataValidator()
        validator.set_reference(real_data)
        report = validator.validate(collapsed_synthetic)

        assert len(report.recommendations) > 0

    def test_custom_thresholds(self, real_data: pd.DataFrame) -> None:
        validator = SyntheticDataValidator(
            memorization_threshold=0.99,
            mode_collapse_threshold=0.05,
            distribution_threshold=0.05,
        )
        validator.set_reference(real_data)

        # Just verify it works with custom thresholds
        assert validator.memorization_threshold == 0.99


class TestValidateSyntheticData:
    """Tests for validate_synthetic_data convenience function."""

    def test_function_works(self) -> None:
        np.random.seed(42)
        real = pd.DataFrame({"x": np.random.randn(500)})
        synth = pd.DataFrame({"x": np.random.randn(500)})

        report = validate_synthetic_data(real, synth)
        assert isinstance(report, SyntheticValidationReport)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_synthetic(self) -> None:
        np.random.seed(42)
        real = pd.DataFrame({"x": np.random.randn(100)})
        synth = pd.DataFrame({"x": []})

        validator = SyntheticDataValidator()
        validator.set_reference(real)
        report = validator.validate(synth)

        assert report is not None

    def test_no_common_columns(self) -> None:
        real = pd.DataFrame({"a": [1, 2, 3]})
        synth = pd.DataFrame({"b": [1, 2, 3]})

        validator = SyntheticDataValidator()
        validator.set_reference(real)

        with pytest.raises(ValueError, match="No common columns"):
            validator.validate(synth)

    def test_categorical_only(self) -> None:
        real = pd.DataFrame({
            "cat1": np.random.choice(["A", "B"], 100),
            "cat2": np.random.choice(["X", "Y", "Z"], 100),
        })
        synth = pd.DataFrame({
            "cat1": np.random.choice(["A", "B"], 100),
            "cat2": np.random.choice(["X", "Y", "Z"], 100),
        })

        validator = SyntheticDataValidator()
        validator.set_reference(real)
        report = validator.validate(synth)

        assert report is not None

    def test_numerical_only(self) -> None:
        np.random.seed(42)
        real = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
        })
        synth = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
        })

        validator = SyntheticDataValidator()
        validator.set_reference(real)
        report = validator.validate(synth)

        assert report is not None
        assert len(report.feature_scores) == 2

    def test_single_row(self) -> None:
        np.random.seed(42)
        real = pd.DataFrame({"x": np.random.randn(100)})
        synth = pd.DataFrame({"x": [0.5]})

        validator = SyntheticDataValidator()
        validator.set_reference(real)
        report = validator.validate(synth)

        assert report is not None

    def test_missing_values(self) -> None:
        np.random.seed(42)
        real = pd.DataFrame({"x": [1, 2, np.nan, 4, 5] * 20})
        synth = pd.DataFrame({"x": [1, np.nan, 3, 4, 5] * 20})

        validator = SyntheticDataValidator()
        validator.set_reference(real)
        report = validator.validate(synth)

        assert report is not None
