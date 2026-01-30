"""Tests for imbalance and bias detection."""

import numpy as np
import pandas as pd

from clean.detection import BiasDetector, ImbalanceDetector, analyze_bias, analyze_imbalance


class TestImbalanceDetector:
    """Tests for ImbalanceDetector."""

    def test_detect_imbalance(self, sample_imbalanced_data):
        """Test imbalance detection."""
        labels = sample_imbalanced_data["label"].values

        detector = ImbalanceDetector(imbalance_threshold=5.0)
        result = detector.fit_detect(sample_imbalanced_data, labels)

        # 9:1 ratio should trigger imbalance
        assert result.metadata["is_imbalanced"]
        assert result.metadata["imbalance_ratio"] > 5.0

    def test_balanced_data(self):
        """Test with balanced data."""
        labels = np.array([0] * 50 + [1] * 50)
        X = np.random.randn(100, 3)

        detector = ImbalanceDetector()
        result = detector.fit_detect(X, labels)

        assert not result.metadata["is_imbalanced"]
        assert result.metadata["imbalance_ratio"] == 1.0

    def test_get_distribution(self, sample_imbalanced_data):
        """Test getting class distribution."""
        labels = sample_imbalanced_data["label"].values

        detector = ImbalanceDetector()
        detector.fit(sample_imbalanced_data, labels)

        dist = detector.get_distribution()

        assert dist is not None
        assert 0 in dist.class_counts
        assert 1 in dist.class_counts
        assert dist.imbalance_ratio > 1.0

    def test_resampling_suggestions(self, sample_imbalanced_data):
        """Test resampling suggestions."""
        labels = sample_imbalanced_data["label"].values

        detector = ImbalanceDetector()
        detector.fit(sample_imbalanced_data, labels)

        suggestions = detector.get_resampling_suggestion()

        assert "strategies" in suggestions
        assert len(suggestions["strategies"]) > 0

    def test_multiclass_imbalance(self):
        """Test with multiclass imbalanced data."""
        labels = np.array([0] * 80 + [1] * 15 + [2] * 5)  # 80-15-5 split
        X = np.random.randn(100, 3)

        detector = ImbalanceDetector()
        result = detector.fit_detect(X, labels)

        assert result.metadata["n_classes"] == 3
        assert result.metadata["imbalance_ratio"] > 10

    def test_convenience_function(self, sample_imbalanced_data):
        """Test analyze_imbalance function."""
        labels = sample_imbalanced_data["label"].values

        result = analyze_imbalance(labels)

        assert "distribution" in result
        assert "metadata" in result
        assert "suggestions" in result


class TestBiasDetector:
    """Tests for BiasDetector."""

    def test_detect_demographic_parity(self, sample_biased_data):
        """Test demographic parity detection."""
        labels = sample_biased_data["label"].values

        detector = BiasDetector(
            sensitive_features=["gender"],
            demographic_parity_threshold=0.1,
        )
        result = detector.fit_detect(sample_biased_data, labels)

        # Should detect bias in the gender-label relationship
        dp_issues = [i for i in result.issues if i.metric == "demographic_parity"]
        assert len(dp_issues) > 0

    def test_correlation_detection(self, sample_biased_data):
        """Test feature-label correlation detection."""
        labels = sample_biased_data["label"].values

        # Income is correlated with label
        detector = BiasDetector(
            sensitive_features=["income"],
            correlation_threshold=0.2,
        )
        result = detector.fit_detect(sample_biased_data, labels)

        # Should find correlation issues
        assert result.metadata["n_features_analyzed"] > 0

    def test_auto_detect_sensitive_features(self, sample_biased_data):
        """Test automatic detection of sensitive features."""
        labels = sample_biased_data["label"].values

        detector = BiasDetector()  # No sensitive_features specified
        detector.fit(sample_biased_data, labels)

        # Should auto-detect gender
        assert "gender" in detector.sensitive_features

    def test_no_bias(self):
        """Test with unbiased data."""
        np.random.seed(42)

        df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "gender": np.random.choice(["M", "F"], 100),
        })
        labels = np.random.choice([0, 1], 100)  # Random, no bias

        detector = BiasDetector(sensitive_features=["gender"])
        result = detector.fit_detect(df, labels)

        # May or may not find issues depending on random sampling
        # But should not fail
        assert result is not None

    def test_compute_demographic_parity(self, sample_biased_data):
        """Test computing demographic parity."""
        labels = sample_biased_data["label"].values

        detector = BiasDetector(sensitive_features=["gender"])
        detector.fit(sample_biased_data, labels)

        dp = detector.compute_demographic_parity(
            sample_biased_data["gender"],
            labels,
        )

        assert "M" in dp
        assert "F" in dp
        # Rates should be different due to bias
        assert abs(dp["M"] - dp["F"]) > 0.1

    def test_compute_equalized_odds(self, sample_biased_data):
        """Test computing equalized odds."""
        labels = sample_biased_data["label"].values
        # Create predictions (same as labels for simplicity)
        predictions = labels.copy()

        detector = BiasDetector(sensitive_features=["gender"])
        detector.fit(sample_biased_data, labels)

        eo = detector.compute_equalized_odds(
            sample_biased_data["gender"],
            labels,
            predictions,
        )

        assert "M" in eo
        assert "F" in eo
        assert "tpr" in eo["M"]
        assert "fpr" in eo["M"]

    def test_convenience_function(self, sample_biased_data):
        """Test analyze_bias function."""
        labels = sample_biased_data["label"].values

        result = analyze_bias(
            sample_biased_data,
            labels,
            sensitive_features=["gender"],
        )

        assert "issues" in result
        assert "metadata" in result


class TestBiasDetectorEdgeCases:
    """Edge case tests for BiasDetector."""

    def test_no_sensitive_features(self):
        """Test with no sensitive features detected."""
        df = pd.DataFrame({
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
        })
        labels = np.random.choice([0, 1], 100)

        detector = BiasDetector()
        result = detector.fit_detect(df, labels)

        # Should not fail, may find no issues
        assert result is not None

    def test_categorical_sensitive_feature(self):
        """Test with categorical sensitive feature."""
        df = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], 100),
            "value": np.random.randn(100),
        })
        labels = np.random.choice([0, 1], 100)

        detector = BiasDetector(sensitive_features=["category"])
        result = detector.fit_detect(df, labels)

        assert result is not None
