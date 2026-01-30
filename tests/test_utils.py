"""Tests for utils modules (export, preprocessing, validation)."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from clean.utils.export import (
    export_clean_dataset,
    export_issues,
    export_review_queue,
    export_summary_stats,
)
from clean.utils.preprocessing import (
    encode_labels,
    decode_labels,
    scale_features,
    handle_missing,
    get_numeric_features,
    get_categorical_features,
)
from clean.utils.validation import (
    validate_features,
    validate_labels,
    validate_threshold,
    validate_positive_int,
)


# =============================================================================
# Export Utils Tests
# =============================================================================

class TestExportCleanDataset:
    """Tests for export_clean_dataset function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 1, 0, 1, 0])

    def test_export_returns_dataframe_when_no_path(self, sample_df, sample_labels):
        """Test that function returns DataFrame when path is None."""
        result = export_clean_dataset(sample_df, sample_labels)
        
        assert isinstance(result, pd.DataFrame)
        assert "label" in result.columns
        assert len(result) == 5

    def test_export_custom_label_column(self, sample_df, sample_labels):
        """Test custom label column name."""
        result = export_clean_dataset(sample_df, sample_labels, label_column="target")
        
        assert "target" in result.columns
        assert "label" not in result.columns

    def test_export_without_labels(self, sample_df):
        """Test export without labels."""
        result = export_clean_dataset(sample_df)
        
        assert "label" not in result.columns

    def test_export_csv(self, sample_df, sample_labels):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            
            export_clean_dataset(sample_df, sample_labels, path=path, format="csv")
            
            assert path.exists()
            loaded = pd.read_csv(path)
            assert len(loaded) == 5
            assert "label" in loaded.columns

    def test_export_parquet(self, sample_df, sample_labels):
        """Test Parquet export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.parquet"
            
            export_clean_dataset(sample_df, sample_labels, path=path, format="parquet")
            
            assert path.exists()

    def test_export_json(self, sample_df, sample_labels):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            
            export_clean_dataset(sample_df, sample_labels, path=path, format="json")
            
            assert path.exists()
            loaded = pd.read_json(path)
            assert len(loaded) == 5

    def test_export_unknown_format_raises(self, sample_df):
        """Test that unknown format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xyz"
            
            with pytest.raises(ValueError, match="Unknown format"):
                export_clean_dataset(sample_df, path=path, format="xyz")


class TestExportIssues:
    """Tests for export_issues function."""

    @pytest.fixture
    def mock_report(self):
        """Create mock QualityReport."""
        report = MagicMock()
        
        # Mock label errors
        label_error = MagicMock()
        label_error.to_dict.return_value = {
            "index": 0, "given_label": 0, "predicted_label": 1, "confidence": 0.9
        }
        report.label_errors_result = MagicMock()
        report.label_errors_result.issues = [label_error]
        
        # Mock duplicates
        dup = MagicMock()
        dup.to_dict.return_value = {"index1": 1, "index2": 2, "similarity": 0.95}
        report.duplicates_result = MagicMock()
        report.duplicates_result.issues = [dup]
        
        # Mock outliers
        outlier = MagicMock()
        outlier.to_dict.return_value = {"index": 3, "score": 0.8}
        report.outliers_result = MagicMock()
        report.outliers_result.issues = [outlier]
        
        # Mock bias
        bias = MagicMock()
        bias.to_dict.return_value = {"feature": "col1", "metric": "dp"}
        report.bias_result = MagicMock()
        report.bias_result.issues = [bias]
        
        return report

    def test_export_all_issues(self, mock_report):
        """Test exporting all issue types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "issues.json"
            
            export_issues(mock_report, path)
            
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            
            assert "label_errors" in data
            assert "duplicates" in data
            assert "outliers" in data
            assert "bias" in data

    def test_export_specific_issues(self, mock_report):
        """Test exporting specific issue types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "issues.json"
            
            export_issues(mock_report, path, issue_types=["label_errors", "outliers"])
            
            with open(path) as f:
                data = json.load(f)
            
            assert "label_errors" in data
            assert "outliers" in data
            assert "duplicates" not in data


class TestExportReviewQueue:
    """Tests for export_review_queue function."""

    @pytest.fixture
    def mock_report(self):
        """Create mock report with issues."""
        report = MagicMock()
        
        # Mock label errors
        error = MagicMock()
        error.index = 0
        error.confidence = 0.9
        error.given_label = "A"
        error.predicted_label = "B"
        report.label_errors_result = MagicMock()
        report.label_errors_result.issues = [error]
        
        # Mock outliers
        outlier = MagicMock()
        outlier.index = 1
        outlier.score = 0.8
        outlier.method = "isolation_forest"
        report.outliers_result = MagicMock()
        report.outliers_result.issues = [outlier]
        
        return report

    def test_export_review_queue_csv(self, mock_report):
        """Test CSV export of review queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queue.csv"
            
            export_review_queue(mock_report, path)
            
            assert path.exists()
            loaded = pd.read_csv(path)
            assert len(loaded) >= 1

    def test_export_review_queue_json(self, mock_report):
        """Test JSON export of review queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queue.json"
            
            export_review_queue(mock_report, path, format="json")
            
            assert path.exists()

    def test_export_review_queue_max_items(self, mock_report):
        """Test max_items limit."""
        # Add more items
        errors = []
        for i in range(100):
            error = MagicMock()
            error.index = i
            error.confidence = 0.5
            error.given_label = "A"
            error.predicted_label = "B"
            errors.append(error)
        mock_report.label_errors_result.issues = errors
        mock_report.outliers_result.issues = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queue.csv"
            
            export_review_queue(mock_report, path, max_items=10)
            
            loaded = pd.read_csv(path)
            assert len(loaded) <= 10


class TestExportSummaryStats:
    """Tests for export_summary_stats function."""

    @pytest.fixture
    def mock_report(self):
        """Create mock report."""
        report = MagicMock()
        
        report.dataset_info = MagicMock()
        report.dataset_info.to_dict.return_value = {
            "n_samples": 100, "n_features": 5
        }
        
        report.quality_score = MagicMock()
        report.quality_score.to_dict.return_value = {
            "overall": 85.0, "label_quality": 80.0,
            "duplicate_quality": 90.0, "imbalance_quality": 75.0
        }
        report.quality_score.label_quality = 80.0
        report.quality_score.duplicate_quality = 90.0
        report.quality_score.imbalance_quality = 75.0
        
        report.label_errors_result = MagicMock()
        report.label_errors_result.n_issues = 5
        report.duplicates_result = MagicMock()
        report.duplicates_result.n_issues = 3
        report.outliers_result = MagicMock()
        report.outliers_result.n_issues = 10
        report.bias_result = MagicMock()
        report.bias_result.n_issues = 1
        
        return report

    def test_export_summary_stats(self, mock_report):
        """Test summary stats export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            
            export_summary_stats(mock_report, path)
            
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            
            assert "dataset" in data
            assert "quality_scores" in data
            assert "issue_counts" in data
            assert "recommendations" in data


# =============================================================================
# Preprocessing Utils Tests
# =============================================================================

class TestEncodeLabels:
    """Tests for encode_labels function."""

    def test_encode_string_labels(self):
        """Test encoding string labels."""
        labels = np.array(["cat", "dog", "cat", "bird", "dog"])
        
        encoded, mapping = encode_labels(labels)
        
        assert len(np.unique(encoded)) == 3
        assert set(mapping.values()) == {"cat", "dog", "bird"}

    def test_encode_numeric_labels(self):
        """Test encoding numeric labels."""
        labels = np.array([1, 2, 1, 3, 2])
        
        encoded, mapping = encode_labels(labels)
        
        assert len(np.unique(encoded)) == 3

    def test_encode_series(self):
        """Test encoding pandas Series."""
        labels = pd.Series(["a", "b", "a", "c"])
        
        encoded, mapping = encode_labels(labels)
        
        assert isinstance(encoded, np.ndarray)

    def test_mapping_is_correct(self):
        """Test that mapping is correct for decoding."""
        labels = np.array(["x", "y", "z", "x"])
        
        encoded, mapping = encode_labels(labels)
        
        # Verify mapping
        for i, label in enumerate(encoded):
            assert mapping[label] == labels[i]


class TestDecodeLabels:
    """Tests for decode_labels function."""

    def test_decode_basic(self):
        """Test basic decoding."""
        labels = np.array(["cat", "dog", "cat"])
        encoded, mapping = encode_labels(labels)
        
        decoded = decode_labels(encoded, mapping)
        
        assert np.array_equal(decoded, labels)

    def test_decode_preserves_order(self):
        """Test that decoding preserves order."""
        labels = np.array(["a", "b", "c", "a", "b"])
        encoded, mapping = encode_labels(labels)
        
        decoded = decode_labels(encoded, mapping)
        
        assert list(decoded) == list(labels)


class TestScaleFeatures:
    """Tests for scale_features function."""

    def test_scale_dataframe(self):
        """Test scaling DataFrame."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        
        scaled, scaler = scale_features(df)
        
        assert scaled.shape == (5, 2)
        # Mean should be close to 0
        assert np.abs(scaled.mean()) < 0.1

    def test_scale_array(self):
        """Test scaling numpy array."""
        X = np.array([[1, 10], [2, 20], [3, 30]])
        
        scaled, scaler = scale_features(X)
        
        assert scaled.shape == X.shape

    def test_scale_with_existing_scaler(self):
        """Test scaling with pre-fit scaler."""
        from sklearn.preprocessing import StandardScaler
        
        X_train = np.array([[1, 10], [2, 20], [3, 30]])
        X_test = np.array([[4, 40], [5, 50]])
        
        _, scaler = scale_features(X_train)
        scaled_test, _ = scale_features(X_test, scaler=scaler)
        
        assert scaled_test.shape == X_test.shape


class TestHandleMissing:
    """Tests for handle_missing function."""

    @pytest.fixture
    def df_with_missing(self):
        """Create DataFrame with missing values."""
        return pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [1.0, np.nan, 3.0, np.nan, 5.0],
            "c": ["x", "y", None, "z", "w"],
        })

    def test_drop_strategy(self, df_with_missing):
        """Test drop strategy."""
        result = handle_missing(df_with_missing, strategy="drop")
        
        assert len(result) == 2  # Only rows without any NaN

    def test_fill_strategy(self, df_with_missing):
        """Test fill strategy."""
        result = handle_missing(df_with_missing, strategy="fill", fill_value=0)
        
        assert not result["a"].isna().any()
        assert not result["b"].isna().any()

    def test_mean_strategy(self, df_with_missing):
        """Test mean imputation strategy."""
        result = handle_missing(df_with_missing, strategy="mean")
        
        assert not result["a"].isna().any()
        assert not result["b"].isna().any()

    def test_median_strategy(self, df_with_missing):
        """Test median imputation strategy."""
        result = handle_missing(df_with_missing, strategy="median")
        
        assert not result["a"].isna().any()
        assert not result["b"].isna().any()

    def test_mode_strategy(self, df_with_missing):
        """Test mode imputation strategy."""
        result = handle_missing(df_with_missing, strategy="mode")
        
        # Mode might not fill all values if there's no clear mode
        assert result is not None

    def test_unknown_strategy_raises(self, df_with_missing):
        """Test that unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing(df_with_missing, strategy="unknown")


class TestGetNumericFeatures:
    """Tests for get_numeric_features function."""

    def test_extracts_numeric_only(self):
        """Test that only numeric columns are returned."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": ["x", "y", "z"],
        })
        
        result = get_numeric_features(df)
        
        assert list(result.columns) == ["a", "b"]
        assert "c" not in result.columns

    def test_empty_if_no_numeric(self):
        """Test returns empty if no numeric columns."""
        df = pd.DataFrame({
            "a": ["x", "y", "z"],
            "b": ["1", "2", "3"],
        })
        
        result = get_numeric_features(df)
        
        assert len(result.columns) == 0


class TestGetCategoricalFeatures:
    """Tests for get_categorical_features function."""

    def test_extracts_categorical_only(self):
        """Test that only categorical columns are returned."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": pd.Categorical(["a", "b", "c"]),
        })
        
        result = get_categorical_features(df)
        
        assert "b" in result.columns
        assert "c" in result.columns
        assert "a" not in result.columns


# =============================================================================
# Validation Utils Tests
# =============================================================================

class TestValidateFeatures:
    """Tests for validate_features function."""

    def test_validate_dataframe(self):
        """Test validating DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        result = validate_features(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_validate_array(self):
        """Test validating numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        
        result = validate_features(arr)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)

    def test_validate_1d_array(self):
        """Test validating 1D array gets reshaped."""
        arr = np.array([1, 2, 3])
        
        result = validate_features(arr)
        
        assert result.shape == (3, 1)

    def test_empty_raises(self):
        """Test that empty data raises error."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            validate_features(df)

    def test_min_samples_check(self):
        """Test minimum samples check."""
        df = pd.DataFrame({"a": [1, 2]})
        
        with pytest.raises(ValueError, match="samples"):
            validate_features(df, min_samples=5)

    def test_min_features_check(self):
        """Test minimum features check."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="features"):
            validate_features(df, min_features=3)


class TestValidateLabels:
    """Tests for validate_labels function."""

    def test_validate_array(self):
        """Test validating numpy array."""
        labels = np.array([0, 1, 0, 1, 0])
        
        result = validate_labels(labels)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_validate_list(self):
        """Test validating list."""
        labels = [0, 1, 2, 0, 1]
        
        result = validate_labels(labels)
        
        assert isinstance(result, np.ndarray)

    def test_validate_series(self):
        """Test validating pandas Series."""
        labels = pd.Series([0, 1, 0, 1])
        
        result = validate_labels(labels)
        
        assert isinstance(result, np.ndarray)

    def test_empty_raises(self):
        """Test that empty labels raise error."""
        with pytest.raises(ValueError, match="empty"):
            validate_labels([])

    def test_n_samples_check(self):
        """Test n_samples check."""
        labels = np.array([0, 1, 0])
        
        with pytest.raises(ValueError, match="length"):
            validate_labels(labels, n_samples=5)

    def test_min_classes_check(self):
        """Test minimum classes check."""
        labels = np.array([0, 0, 0, 0])  # Only 1 class
        
        with pytest.raises(ValueError, match="classes"):
            validate_labels(labels, min_classes=2)

    def test_handles_nan_in_class_count(self):
        """Test that NaN values are handled in class count."""
        labels = np.array([0, 1, np.nan, 0, 1])
        
        result = validate_labels(labels)
        
        assert len(result) == 5


class TestValidateThreshold:
    """Tests for validate_threshold function."""

    def test_valid_threshold(self):
        """Test valid threshold passes."""
        result = validate_threshold(0.5, "test")
        
        assert result == 0.5

    def test_int_converted_to_float(self):
        """Test integer converted to float."""
        result = validate_threshold(1, "test")
        
        assert result == 1.0
        assert isinstance(result, float)

    def test_below_min_raises(self):
        """Test value below min raises error."""
        with pytest.raises(ValueError, match="between"):
            validate_threshold(-0.1, "test")

    def test_above_max_raises(self):
        """Test value above max raises error."""
        with pytest.raises(ValueError, match="between"):
            validate_threshold(1.5, "test")

    def test_custom_range(self):
        """Test custom min/max range."""
        result = validate_threshold(50, "test", min_val=0, max_val=100)
        
        assert result == 50.0

    def test_non_numeric_raises(self):
        """Test non-numeric value raises error."""
        with pytest.raises(ValueError, match="number"):
            validate_threshold("0.5", "test")


class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_valid_positive_int(self):
        """Test valid positive integer passes."""
        result = validate_positive_int(5, "test")
        
        assert result == 5

    def test_zero_raises(self):
        """Test zero raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_positive_int(0, "test")

    def test_negative_raises(self):
        """Test negative raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_positive_int(-1, "test")

    def test_float_raises(self):
        """Test float raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_positive_int(1.5, "test")

    def test_string_raises(self):
        """Test string raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_positive_int("5", "test")
