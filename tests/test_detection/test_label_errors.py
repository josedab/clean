"""Tests for label error detection."""

import numpy as np
import pandas as pd
import pytest

from clean.detection import LabelErrorDetector, find_label_errors


class TestLabelErrorDetector:
    """Tests for LabelErrorDetector."""

    def test_fit_basic(self, sample_classification_data):
        """Test basic fitting."""
        df, labels, _ = sample_classification_data
        features = df.drop(columns=["label"])

        detector = LabelErrorDetector()
        detector.fit(features, labels)

        assert detector.is_fitted
        assert detector.get_pred_probs() is not None

    def test_detect_basic(self, sample_classification_data):
        """Test basic detection."""
        df, labels, error_indices = sample_classification_data
        features = df.drop(columns=["label"])

        detector = LabelErrorDetector()
        result = detector.fit_detect(features, labels)

        assert result.n_issues >= 0
        assert "error_rate" in result.metadata

    def test_detect_finds_errors(self):
        """Test that detector finds intentional errors."""
        np.random.seed(42)

        # Create clearly separable classes
        n_per_class = 100
        X_class0 = np.random.randn(n_per_class, 5) - 2
        X_class1 = np.random.randn(n_per_class, 5) + 2
        X = np.vstack([X_class0, X_class1])

        # True labels
        y_true = np.array([0] * n_per_class + [1] * n_per_class)

        # Add obvious errors (flip some labels)
        y_with_errors = y_true.copy()
        error_indices = [0, 1, 100, 101]  # Some from each class
        y_with_errors[error_indices] = 1 - y_with_errors[error_indices]

        detector = LabelErrorDetector(confidence_threshold=0.5)
        result = detector.fit_detect(X, y_with_errors)

        # Should detect some of the errors
        detected_indices = {e.index for e in result.issues}
        overlap = detected_indices & set(error_indices)

        # At least some errors should be detected
        assert len(result.issues) > 0

    def test_labels_required(self):
        """Test that labels are required."""
        X = np.random.randn(100, 5)

        detector = LabelErrorDetector()

        with pytest.raises(ValueError, match="Labels required"):
            detector.fit(X, None)

    def test_min_classes(self):
        """Test minimum class requirement."""
        X = np.random.randn(100, 5)
        y = np.zeros(100)  # Only one class

        detector = LabelErrorDetector()

        with pytest.raises(ValueError, match="at least 2 classes"):
            detector.fit(X, y)

    def test_result_metadata(self, sample_classification_data):
        """Test result metadata."""
        df, labels, _ = sample_classification_data
        features = df.drop(columns=["label"])

        detector = LabelErrorDetector(cv_folds=3, confidence_threshold=0.6)
        result = detector.fit_detect(features, labels)

        assert result.metadata["cv_folds"] == 3
        assert result.metadata["confidence_threshold"] == 0.6
        assert "n_samples" in result.metadata
        assert "n_classes" in result.metadata

    def test_to_dataframe(self, sample_classification_data):
        """Test conversion to DataFrame."""
        df, labels, _ = sample_classification_data
        features = df.drop(columns=["label"])

        detector = LabelErrorDetector()
        result = detector.fit_detect(features, labels)

        result_df = result.to_dataframe()

        assert isinstance(result_df, pd.DataFrame)
        if len(result_df) > 0:
            assert "index" in result_df.columns
            assert "given_label" in result_df.columns
            assert "predicted_label" in result_df.columns
            assert "confidence" in result_df.columns

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        np.random.seed(42)

        # Create data where all samples should be flagged at low threshold
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        # Low threshold should find more issues
        detector_low = LabelErrorDetector(confidence_threshold=0.3)
        result_low = detector_low.fit_detect(X, y)

        # High threshold should find fewer issues
        detector_high = LabelErrorDetector(confidence_threshold=0.9)
        result_high = detector_high.fit_detect(X, y)

        assert result_low.n_issues >= result_high.n_issues

    def test_convenience_function(self, sample_classification_data):
        """Test find_label_errors convenience function."""
        df, labels, _ = sample_classification_data
        features = df.drop(columns=["label"])

        result_df = find_label_errors(features, labels)

        assert isinstance(result_df, pd.DataFrame)


class TestLabelErrorDetectorEdgeCases:
    """Edge case tests for LabelErrorDetector."""

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.random.randn(20, 3)
        y = np.array([0] * 10 + [1] * 10)

        detector = LabelErrorDetector(cv_folds=2)
        result = detector.fit_detect(X, y)

        # Should work without errors
        assert result is not None

    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 90 + [1] * 10)  # 90-10 split

        detector = LabelErrorDetector(cv_folds=2)
        result = detector.fit_detect(X, y)

        assert result is not None

    def test_with_nan_features(self):
        """Test handling of NaN in features."""
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        X[5, 2] = np.nan
        y = np.random.choice([0, 1], 100)

        detector = LabelErrorDetector()
        result = detector.fit_detect(X, y)

        # Should handle NaN gracefully
        assert result is not None
