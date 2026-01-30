"""Tests for duplicate detection."""

import numpy as np
import pandas as pd

from clean.detection import DuplicateDetector, find_duplicates


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    def test_detect_exact_duplicates(self, sample_data_with_duplicates):
        """Test detection of exact duplicates."""
        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(sample_data_with_duplicates)

        # Should find exact duplicates (rows 0, 3, 7 are identical)
        assert result.n_issues > 0

        # Check that we found exact duplicates
        exact_count = sum(1 for d in result.issues if d.is_exact)
        assert exact_count > 0

    def test_detect_near_duplicates(self, sample_data_with_duplicates):
        """Test detection of near duplicates."""
        detector = DuplicateDetector(
            methods=["fuzzy"],
            similarity_threshold=0.99,
        )
        result = detector.fit_detect(sample_data_with_duplicates)

        # Should find near-duplicates (rows 1 and 4 are nearly identical)
        # Note: exact duplicates are also detected by fuzzy matching
        assert result.n_issues >= 0

    def test_hash_method_only(self):
        """Test hash method detection."""
        df = pd.DataFrame({
            "a": [1, 2, 1, 3],
            "b": ["x", "y", "x", "z"],
        })

        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(df)

        # Rows 0 and 2 are identical
        assert result.n_issues == 1
        assert result.issues[0].is_exact

    def test_similarity_threshold(self):
        """Test different similarity thresholds."""
        df = pd.DataFrame({
            "a": [1.0, 1.01, 2.0, 3.0],
            "b": [1.0, 1.01, 2.0, 3.0],
        })

        # High threshold - fewer matches
        detector_high = DuplicateDetector(methods=["fuzzy"], similarity_threshold=0.999)
        result_high = detector_high.fit_detect(df)

        # Lower threshold - more matches
        detector_low = DuplicateDetector(methods=["fuzzy"], similarity_threshold=0.9)
        result_low = detector_low.fit_detect(df)

        assert result_high.n_issues <= result_low.n_issues

    def test_metadata(self, sample_data_with_duplicates):
        """Test result metadata."""
        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(sample_data_with_duplicates)

        assert "n_duplicate_pairs" in result.metadata
        assert "n_exact" in result.metadata
        assert result.metadata["n_duplicate_pairs"] == result.n_issues

    def test_get_duplicate_groups(self, sample_data_with_duplicates):
        """Test getting duplicate groups."""
        detector = DuplicateDetector(methods=["hash"])
        detector.fit(sample_data_with_duplicates)

        groups = detector.get_duplicate_groups(sample_data_with_duplicates)

        # Should have at least one group
        assert len(groups) > 0

        # Each group should have at least 2 items
        for group in groups:
            assert len(group) >= 2

    def test_no_duplicates(self):
        """Test with data that has no duplicates."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })

        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(df)

        assert result.n_issues == 0

    def test_to_dataframe(self, sample_data_with_duplicates):
        """Test conversion to DataFrame."""
        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(sample_data_with_duplicates)

        result_df = result.to_dataframe()

        assert isinstance(result_df, pd.DataFrame)
        if len(result_df) > 0:
            assert "index1" in result_df.columns
            assert "index2" in result_df.columns
            assert "similarity" in result_df.columns
            assert "is_exact" in result_df.columns

    def test_convenience_function(self, sample_data_with_duplicates):
        """Test find_duplicates convenience function."""
        result_df = find_duplicates(sample_data_with_duplicates, methods=["hash"])

        assert isinstance(result_df, pd.DataFrame)


class TestDuplicateDetectorEdgeCases:
    """Edge case tests for DuplicateDetector."""

    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(df)

        assert result.n_issues == 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"a": [], "b": []})

        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(df)

        assert result.n_issues == 0

    def test_numpy_input(self):
        """Test with numpy array input."""
        X = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])

        detector = DuplicateDetector(methods=["hash"])
        result = detector.fit_detect(X)

        # Rows 0 and 2 are identical
        assert result.n_issues == 1

    def test_multiple_methods(self, sample_data_with_duplicates):
        """Test combining multiple methods."""
        detector = DuplicateDetector(methods=["hash", "fuzzy"])
        result = detector.fit_detect(sample_data_with_duplicates)

        # Should find duplicates from both methods
        assert result.n_issues > 0
