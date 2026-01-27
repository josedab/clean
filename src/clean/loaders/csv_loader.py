"""CSV file loader."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clean.core.types import DatasetInfo, DataType, TaskType
from clean.loaders.base import BaseLoader, LoaderConfig


class CSVLoader(BaseLoader):
    """Load data from CSV files."""

    def __init__(
        self,
        path: str | Path,
        label_column: str | None = None,
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        task_type: TaskType | None = None,
        data_type: DataType | None = None,
        **read_csv_kwargs: Any,
    ):
        """Initialize the CSV loader.

        Args:
            path: Path to CSV file
            label_column: Name of the label column
            feature_columns: Columns to use as features
            exclude_columns: Columns to exclude from features
            task_type: Type of ML task
            data_type: Type of data
            **read_csv_kwargs: Additional arguments for pd.read_csv
        """
        self.path = Path(path)
        self.config = LoaderConfig(
            label_column=label_column,
            feature_columns=feature_columns,
            exclude_columns=exclude_columns,
            task_type=task_type,
            data_type=data_type,
        )
        self.read_csv_kwargs = read_csv_kwargs
        self._data: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._labels: np.ndarray | None = None
        self._info: DatasetInfo | None = None

    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Load CSV file and separate features and labels.

        Returns:
            Tuple of (features DataFrame, labels array or None)
        """
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        # Read CSV
        self._data = pd.read_csv(self.path, **self.read_csv_kwargs)

        # Extract labels
        labels = None
        if self.config.label_column:
            if self.config.label_column not in self._data.columns:
                raise ValueError(
                    f"Label column '{self.config.label_column}' not found in CSV"
                )
            labels = self._data[self.config.label_column].values

        # Determine feature columns
        if self.config.feature_columns:
            feature_cols = [c for c in self.config.feature_columns if c in self._data.columns]
        else:
            exclude = set(self.config.exclude_columns or [])
            if self.config.label_column:
                exclude.add(self.config.label_column)
            feature_cols = [c for c in self._data.columns if c not in exclude]

        features = self._data[feature_cols].copy()

        # Validate
        features, labels = self.validate_data(features, labels)

        self._features = features
        self._labels = labels

        return features, labels

    def get_info(self) -> DatasetInfo:
        """Get information about the dataset.

        Returns:
            DatasetInfo with metadata
        """
        if self._features is None:
            self.load()

        assert self._features is not None

        # Auto-detect data type
        data_type = self.config.data_type
        if data_type is None:
            data_type = self.detect_data_type(self._features)

        # Auto-detect task type
        task_type = self.config.task_type
        if task_type is None:
            task_type = self.detect_task_type(self._labels)

        # Count classes
        n_classes = None
        if self._labels is not None and task_type == TaskType.CLASSIFICATION:
            n_classes = len(np.unique(self._labels[~pd.isna(self._labels)]))

        self._info = DatasetInfo(
            n_samples=len(self._features),
            n_features=len(self._features.columns),
            n_classes=n_classes,
            feature_names=list(self._features.columns),
            label_column=self.config.label_column,
            data_type=data_type,
            task_type=task_type,
        )

        return self._info


def load_csv(
    path: str | Path,
    label_column: str | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray | None, DatasetInfo]:
    """Convenience function to load a CSV file.

    Args:
        path: Path to CSV file
        label_column: Name of the label column
        **kwargs: Additional arguments for CSVLoader

    Returns:
        Tuple of (features, labels, info)
    """
    loader = CSVLoader(path, label_column=label_column, **kwargs)
    features, labels = loader.load()
    info = loader.get_info()
    return features, labels, info
