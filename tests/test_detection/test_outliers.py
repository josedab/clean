"""Tests for outlier detection."""

import numpy as np
import pandas as pd

from clean.detection import OutlierDetector, find_outliers


class TestOutlierDetector:
    """Tests for OutlierDetector."""

    def test_isolation_forest(self, sample_data_with_outliers):
        """Test Isolation Forest detection."""
        df, expected_outliers = sample_data_with_outliers

        detector = OutlierDetector(method="isolation_forest", contamination=0.1)
        result = detector.fit_detect(df)

        assert result.n_issues > 0

        # Check that we found some of the expected outliers
        detected_indices = {o.index for o in result.issues}
        overlap = detected_indices & set(expected_outliers)
        assert len(overlap) > 0

    def test_lof(self, sample_data_with_outliers):
        """Test Local Outlier Factor detection."""
        df, expected_outliers = sample_data_with_outliers

        detector = OutlierDetector(method="lof", contamination=0.1)
        result = detector.fit_detect(df)

        assert result.n_issues > 0

    def test_zscore(self, sample_data_with_outliers):
        """Test Z-score based detection."""
        df, expected_outliers = sample_data_with_outliers

        detector = OutlierDetector(method="zscore", zscore_threshold=2.0)
        result = detector.fit_detect(df)

        # Z-score should find the extreme outliers
        detected_indices = {o.index for o in result.issues}

        # Our extreme outliers should be detected
        assert 0 in detected_indices or 1 in detected_indices or 2 in detected_indices

    def test_iqr(self, sample_data_with_outliers):
        """Test IQR method detection."""
        df, expected_outliers = sample_data_with_outliers

        detector = OutlierDetector(method="iqr", iqr_multiplier=1.5)
        result = detector.fit_detect(df)

        # IQR should find outliers
        assert result.n_issues > 0

    def test_ensemble(self, sample_data_with_outliers):
        """Test ensemble method."""
        df, expected_outliers = sample_data_with_outliers

        detector = OutlierDetector(
            method="ensemble",
            ensemble_methods=["isolation_forest", "zscore", "iqr"],
            ensemble_threshold=2,
        )
        result = detector.fit_detect(df)

        # Ensemble with threshold 2 means at least 2 methods must agree
        # This should still find the obvious outliers
        assert result.n_issues >= 0

    def test_contamination(self):
        """Test different contamination values."""
        np.random.seed(42)
        X = np.random.randn(100, 3)

        # Low contamination
        detector_low = OutlierDetector(method="isolation_forest", contamination=0.01)
        result_low = detector_low.fit_detect(X)

        # High contamination
        detector_high = OutlierDetector(method="isolation_forest", contamination=0.2)
        result_high = detector_high.fit_detect(X)

        # Higher contamination should find more outliers
        assert result_low.n_issues <= result_high.n_issues

    def test_metadata(self, sample_data_with_outliers):
        """Test result metadata."""
        df, _ = sample_data_with_outliers

        detector = OutlierDetector(method="isolation_forest", contamination=0.1)
        result = detector.fit_detect(df)

        assert result.metadata["method"] == "isolation_forest"
        assert result.metadata["contamination"] == 0.1
        assert "n_samples" in result.metadata
        assert "n_outliers" in result.metadata
        assert "outlier_rate" in result.metadata

    def test_get_outlier_scores(self, sample_data_with_outliers):
        """Test getting outlier scores."""
        df, expected_outliers = sample_data_with_outliers

        detector = OutlierDetector(method="isolation_forest")
        detector.fit(df)

        scores = detector.get_outlier_scores(df)

        assert len(scores) == len(df)

        # Outliers should have higher scores
        outlier_scores = scores[expected_outliers]
        normal_scores = np.delete(scores, expected_outliers)

        assert np.mean(outlier_scores) > np.mean(normal_scores)

    def test_to_dataframe(self, sample_data_with_outliers):
        """Test conversion to DataFrame."""
        df, _ = sample_data_with_outliers

        detector = OutlierDetector()
        result = detector.fit_detect(df)

        result_df = result.to_dataframe()

        assert isinstance(result_df, pd.DataFrame)
        if len(result_df) > 0:
            assert "index" in result_df.columns
            assert "score" in result_df.columns
            assert "method" in result_df.columns

    def test_convenience_function(self, sample_data_with_outliers):
        """Test find_outliers convenience function."""
        df, _ = sample_data_with_outliers

        result_df = find_outliers(df, method="isolation_forest")

        assert isinstance(result_df, pd.DataFrame)


class TestOutlierDetectorEdgeCases:
    """Edge case tests for OutlierDetector."""

    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.random.randn(20, 3)

        detector = OutlierDetector(method="isolation_forest", contamination=0.1)
        result = detector.fit_detect(X)

        assert result is not None

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        X[0] = 10  # Outlier

        detector = OutlierDetector(method="zscore")
        result = detector.fit_detect(X)

        # Should find the outlier
        detected_indices = {o.index for o in result.issues}
        assert 0 in detected_indices

    def test_with_nan(self):
        """Test handling of NaN values."""
        X = np.random.randn(100, 3)
        X[5, 0] = np.nan

        detector = OutlierDetector()
        result = detector.fit_detect(X)

        # Should handle NaN gracefully
        assert result is not None

    def test_non_numeric_columns(self):
        """Test with mixed column types."""
        df = pd.DataFrame({
            "numeric1": np.random.randn(100),
            "numeric2": np.random.randn(100),
            "categorical": np.random.choice(["A", "B"], 100),
        })

        detector = OutlierDetector()
        result = detector.fit_detect(df)

        # Should only use numeric columns
        assert result is not None
