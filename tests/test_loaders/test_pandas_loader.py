"""Tests for data loaders."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from clean.core.types import DataType, TaskType
from clean.loaders import (
    CSVLoader,
    NumpyLoader,
    PandasLoader,
    load_arrays,
    load_csv,
    load_dataframe,
)


class TestPandasLoader:
    """Tests for PandasLoader."""

    def test_load_basic(self, sample_dataframe):
        """Test basic loading."""
        loader = PandasLoader(sample_dataframe, label_column="label")
        features, labels = loader.load()

        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, np.ndarray)
        assert len(features) == len(labels)
        assert "label" not in features.columns

    def test_load_without_labels(self, sample_dataframe):
        """Test loading without label column."""
        loader = PandasLoader(sample_dataframe)
        features, labels = loader.load()

        assert isinstance(features, pd.DataFrame)
        assert labels is None

    def test_get_info(self, sample_dataframe):
        """Test getting dataset info."""
        loader = PandasLoader(sample_dataframe, label_column="label")
        loader.load()
        info = loader.get_info()

        assert info.n_samples == len(sample_dataframe)
        assert info.n_features == len(sample_dataframe.columns) - 1
        assert info.label_column == "label"

    def test_feature_columns_selection(self, sample_dataframe):
        """Test selecting specific feature columns."""
        loader = PandasLoader(
            sample_dataframe,
            label_column="label",
            feature_columns=["feature_0", "feature_1"],
        )
        features, _ = loader.load()

        assert len(features.columns) == 2
        assert "feature_0" in features.columns
        assert "feature_1" in features.columns

    def test_exclude_columns(self, sample_dataframe):
        """Test excluding columns."""
        loader = PandasLoader(
            sample_dataframe,
            label_column="label",
            exclude_columns=["category"],
        )
        features, _ = loader.load()

        assert "category" not in features.columns

    def test_invalid_label_column(self, sample_dataframe):
        """Test with invalid label column."""
        loader = PandasLoader(sample_dataframe, label_column="nonexistent")

        with pytest.raises(ValueError):
            loader.load()

    def test_convenience_function(self, sample_dataframe):
        """Test load_dataframe convenience function."""
        features, labels, info = load_dataframe(sample_dataframe, label_column="label")

        assert features is not None
        assert labels is not None
        assert info is not None


class TestNumpyLoader:
    """Tests for NumpyLoader."""

    def test_load_basic(self):
        """Test basic loading."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        loader = NumpyLoader(X, labels=y)
        features, labels = loader.load()

        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, np.ndarray)
        assert len(features) == 100
        assert len(features.columns) == 5

    def test_load_1d_array(self):
        """Test loading 1D feature array."""
        X = np.random.randn(100)
        y = np.random.choice([0, 1], 100)

        loader = NumpyLoader(X, labels=y)
        features, _ = loader.load()

        assert len(features.columns) == 1

    def test_custom_feature_names(self):
        """Test with custom feature names."""
        X = np.random.randn(100, 3)
        names = ["a", "b", "c"]

        loader = NumpyLoader(X, feature_names=names)
        features, _ = loader.load()

        assert list(features.columns) == names

    def test_get_info(self):
        """Test getting dataset info."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        loader = NumpyLoader(X, labels=y)
        loader.load()
        info = loader.get_info()

        assert info.n_samples == 100
        assert info.n_features == 5
        assert info.data_type == DataType.TABULAR

    def test_convenience_function(self):
        """Test load_arrays convenience function."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        features, labels, info = load_arrays(X, labels=y)

        assert features is not None
        assert labels is not None
        assert info is not None


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_load_basic(self, sample_dataframe):
        """Test basic CSV loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loader = CSVLoader(temp_path, label_column="label")
            features, labels = loader.load()

            assert isinstance(features, pd.DataFrame)
            assert isinstance(labels, np.ndarray)
            assert len(features) == len(sample_dataframe)
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test with non-existent file."""
        loader = CSVLoader("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_convenience_function(self, sample_dataframe):
        """Test load_csv convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            features, labels, info = load_csv(temp_path, label_column="label")

            assert features is not None
            assert labels is not None
            assert info is not None
        finally:
            os.unlink(temp_path)


class TestDataTypeDetection:
    """Tests for automatic data type detection."""

    def test_detect_tabular(self):
        """Test detecting tabular data."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })

        loader = PandasLoader(df)
        loader.load()
        info = loader.get_info()

        assert info.data_type == DataType.TABULAR

    def test_detect_text(self):
        """Test detecting text data."""
        df = pd.DataFrame({
            "text": [
                "This is a long text that should be detected as text data because it has many characters.",
                "Another long text string that contains enough characters to be identified as text content.",
                "A third text sample with sufficient length to trigger text detection in the loader.",
            ],
        })

        loader = PandasLoader(df)
        loader.load()
        info = loader.get_info()

        assert info.data_type == DataType.TEXT

    def test_detect_task_classification(self):
        """Test detecting classification task."""
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5],
            "label": [0, 1, 0, 1, 0],
        })

        loader = PandasLoader(df, label_column="label")
        loader.load()
        info = loader.get_info()

        assert info.task_type == TaskType.CLASSIFICATION
