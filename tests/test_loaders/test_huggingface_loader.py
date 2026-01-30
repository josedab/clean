"""Tests for HuggingFace loader module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from clean.core.types import DataType, TaskType


class TestHuggingFaceLoaderImport:
    """Test import handling."""

    def test_import_error_when_datasets_not_available(self):
        """Test ImportError when datasets library not available."""
        import importlib
        import sys

        # Remove datasets from modules if present
        modules_to_remove = [k for k in sys.modules if k.startswith("datasets")]
        original_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            # Patch the loader module
            import clean.loaders.huggingface_loader as hf_module

            original_has = hf_module.HAS_DATASETS
            hf_module.HAS_DATASETS = False

            try:
                with pytest.raises(ImportError) as exc_info:
                    from clean.loaders.huggingface_loader import HuggingFaceLoader

                    HuggingFaceLoader(MagicMock())
                assert "datasets" in str(exc_info.value).lower()
            finally:
                hf_module.HAS_DATASETS = original_has
        finally:
            # Restore original modules
            sys.modules.update(original_modules)


class MockDataset:
    """Mock HuggingFace Dataset for testing."""

    def __init__(self, data: dict):
        self._data = data
        self._df = pd.DataFrame(data)

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)


class MockDatasetDict(dict):
    """Mock HuggingFace DatasetDict for testing."""

    pass


class TestHuggingFaceLoaderBasic:
    """Test basic HuggingFaceLoader functionality."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        return MockDataset({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "label": ["a", "b", "a", "b", "a"],
        })

    @pytest.fixture
    def mock_dataset_dict(self, mock_dataset):
        """Create a mock DatasetDict."""
        dd = MockDatasetDict()
        dd["train"] = mock_dataset
        dd["test"] = MockDataset({
            "feature1": [6.0, 7.0],
            "feature2": [0.0, -1.0],
            "label": ["a", "b"],
        })
        return dd

    def test_load_basic(self, mock_dataset):
        """Test basic loading."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset,
                label_column="label",
            )

            features, labels = loader.load()

            assert isinstance(features, pd.DataFrame)
            assert len(features) == 5
            assert "feature1" in features.columns
            assert "feature2" in features.columns
            # Label column should be excluded from features
            assert "label" not in features.columns
            assert labels is not None
            assert len(labels) == 5

    def test_load_with_feature_columns(self, mock_dataset):
        """Test loading with specific feature columns."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset,
                label_column="label",
                feature_columns=["feature1"],  # Only use feature1
            )

            features, labels = loader.load()

            assert list(features.columns) == ["feature1"]
            assert len(labels) == 5

    def test_load_no_labels(self, mock_dataset):
        """Test loading without label column."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(mock_dataset)

            features, labels = loader.load()

            assert len(features) == 5
            assert labels is None
            # All columns should be features
            assert "label" in features.columns

    def test_load_invalid_label_column(self, mock_dataset):
        """Test loading with non-existent label column."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset,
                label_column="nonexistent",
            )

            with pytest.raises(ValueError) as exc_info:
                loader.load()

            assert "nonexistent" in str(exc_info.value)


class TestDatasetDictHandling:
    """Test DatasetDict handling."""

    @pytest.fixture
    def mock_dataset_dict(self):
        """Create a mock DatasetDict."""
        dd = MockDatasetDict()
        dd["train"] = MockDataset({
            "x": [1.0, 2.0, 3.0],
            "y": ["a", "b", "c"],
        })
        dd["test"] = MockDataset({
            "x": [4.0, 5.0],
            "y": ["a", "b"],
        })
        return dd

    def test_dataset_dict_with_split(self, mock_dataset_dict):
        """Test loading specific split from DatasetDict."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", MockDatasetDict):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset_dict,
                label_column="y",
                split="test",
            )

            features, labels = loader.load()

            assert len(features) == 2  # test split has 2 samples
            assert len(labels) == 2

    def test_dataset_dict_default_split(self, mock_dataset_dict):
        """Test loading default split from DatasetDict."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", MockDatasetDict):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset_dict,
                label_column="y",
            )

            features, labels = loader.load()

            # Should use first split (train)
            assert len(features) == 3

    def test_dataset_dict_invalid_split(self, mock_dataset_dict):
        """Test loading with invalid split name."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", MockDatasetDict):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset_dict,
                label_column="y",
                split="validation",  # Doesn't exist
            )

            with pytest.raises(ValueError) as exc_info:
                loader.load()

            assert "validation" in str(exc_info.value)
            assert "train" in str(exc_info.value)  # Should show available splits


class TestGetInfo:
    """Test get_info method."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        return MockDataset({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "label": ["x", "y", "x", "z"],
        })

    def test_get_info_basic(self, mock_dataset):
        """Test getting dataset info."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(mock_dataset, label_column="label")
            info = loader.get_info()

            assert info.n_samples == 4
            assert info.n_features == 2
            assert info.n_classes == 3  # x, y, z
            assert "a" in info.feature_names
            assert "b" in info.feature_names
            assert info.label_column == "label"
            assert info.task_type == TaskType.CLASSIFICATION

    def test_get_info_with_explicit_task_type(self, mock_dataset):
        """Test getting info with explicit task type."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset,
                label_column="label",
                task_type=TaskType.CLASSIFICATION,
            )
            info = loader.get_info()

            assert info.task_type == TaskType.CLASSIFICATION

    def test_get_info_with_data_type(self, mock_dataset):
        """Test getting info with explicit data type."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(
                mock_dataset,
                data_type=DataType.TABULAR,
            )
            info = loader.get_info()

            assert info.data_type == DataType.TABULAR

    def test_get_info_calls_load_if_needed(self, mock_dataset):
        """Test that get_info calls load if not already loaded."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            loader = HuggingFaceLoader(mock_dataset)

            # Call get_info without calling load first
            info = loader.get_info()

            assert info.n_samples == 4


class TestConvenienceFunction:
    """Test load_huggingface convenience function."""

    def test_load_huggingface(self):
        """Test convenience function."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import load_huggingface

            mock_ds = MockDataset({
                "f1": [1.0, 2.0],
                "f2": [3.0, 4.0],
                "target": ["a", "b"],
            })

            features, labels, info = load_huggingface(
                mock_ds,
                label_column="target",
            )

            assert len(features) == 2
            assert len(labels) == 2
            assert info.n_samples == 2


class TestDataTypeDetection:
    """Test data type detection."""

    def test_detect_tabular_data(self):
        """Test tabular data detection."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            mock_ds = MockDataset({
                "num1": [1.0, 2.0, 3.0],
                "num2": [4.0, 5.0, 6.0],
                "cat": ["a", "b", "c"],
            })

            loader = HuggingFaceLoader(mock_ds)
            info = loader.get_info()

            # Should detect as tabular
            assert info.data_type == DataType.TABULAR

    def test_detect_text_data(self):
        """Test text data detection."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            # Text needs to be at least 50 chars on average to be detected as TEXT
            mock_ds = MockDataset({
                "text": [
                    "This is a long sentence with many words that should exceed the fifty character minimum for text detection",
                    "Another sentence that is quite long and definitely exceeds fifty characters to trigger text detection correctly",
                    "Yet another text document here that must also be long enough for the text detection algorithm to work properly",
                ],
            })

            loader = HuggingFaceLoader(mock_ds)
            info = loader.get_info()

            # Should detect as text
            assert info.data_type == DataType.TEXT


class TestNullLabelHandling:
    """Test handling of null labels."""

    def test_labels_with_nans(self):
        """Test counting classes with NaN labels."""
        with patch("clean.loaders.huggingface_loader.HAS_DATASETS", True), \
             patch("clean.loaders.huggingface_loader.DatasetDict", None):

            from clean.loaders.huggingface_loader import HuggingFaceLoader

            mock_ds = MockDataset({
                "feature": [1.0, 2.0, 3.0, 4.0],
                "label": ["a", "b", np.nan, "a"],
            })

            loader = HuggingFaceLoader(
                mock_ds,
                label_column="label",
                task_type=TaskType.CLASSIFICATION,
            )
            info = loader.get_info()

            # Should only count non-nan classes
            assert info.n_classes == 2  # a, b
