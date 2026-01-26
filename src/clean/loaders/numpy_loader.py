"""NumPy array loader."""

from typing import Any

import numpy as np
import pandas as pd

from clean.core.types import DatasetInfo, DataType, TaskType
from clean.loaders.base import BaseLoader, LoaderConfig


class NumpyLoader(BaseLoader):
    """Load data from NumPy arrays."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        task_type: TaskType | None = None,
    ):
        """Initialize the NumPy loader.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,) or None
            feature_names: Names for features (auto-generated if None)
            task_type: Type of ML task (auto-detected if None)
        """
        self.raw_features = np.asarray(features)
        self.raw_labels = np.asarray(labels) if labels is not None else None
        self.feature_names = feature_names
        self.config = LoaderConfig(task_type=task_type)
        self._features: pd.DataFrame | None = None
        self._labels: np.ndarray | None = None
        self._info: DatasetInfo | None = None

    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Load and convert NumPy arrays to DataFrame format.

        Returns:
            Tuple of (features DataFrame, labels array or None)
        """
        # Ensure 2D
        features = self.raw_features
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Generate feature names if not provided
        if self.feature_names is None:
            n_features = features.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        # Convert to DataFrame
        self._features = pd.DataFrame(features, columns=self.feature_names)
        self._labels = self.raw_labels

        # Validate
        self._features, self._labels = self.validate_data(self._features, self._labels)

        return self._features, self._labels

    def get_info(self) -> DatasetInfo:
        """Get information about the dataset.

        Returns:
            DatasetInfo with metadata
        """
        if self._features is None:
            self.load()

        assert self._features is not None

        # Auto-detect task type
        task_type = self.config.task_type
        if task_type is None:
            task_type = self.detect_task_type(self._labels)

        # Count classes if classification
        n_classes = None
        if self._labels is not None and task_type == TaskType.CLASSIFICATION:
            n_classes = len(np.unique(self._labels[~pd.isna(self._labels)]))

        self._info = DatasetInfo(
            n_samples=len(self._features),
            n_features=len(self._features.columns),
            n_classes=n_classes,
            feature_names=list(self._features.columns),
            label_column=None,
            data_type=DataType.TABULAR,
            task_type=task_type,
        )

        return self._info


def load_arrays(
    features: np.ndarray,
    labels: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray | None, DatasetInfo]:
    """Convenience function to load NumPy arrays.

    Args:
        features: Feature array
        labels: Label array or None
        **kwargs: Additional arguments for NumpyLoader

    Returns:
        Tuple of (features DataFrame, labels, info)
    """
    loader = NumpyLoader(features, labels=labels, **kwargs)
    features_df, labels = loader.load()
    info = loader.get_info()
    return features_df, labels, info
