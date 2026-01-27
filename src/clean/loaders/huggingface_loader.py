"""HuggingFace datasets loader."""

from typing import Any

import numpy as np
import pandas as pd

from clean.core.types import DatasetInfo, DataType, TaskType
from clean.loaders.base import BaseLoader, LoaderConfig

# Optional import
try:
    from datasets import Dataset, DatasetDict

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None
    DatasetDict = None


class HuggingFaceLoader(BaseLoader):
    """Load data from HuggingFace datasets."""

    def __init__(
        self,
        dataset: Any,  # datasets.Dataset or DatasetDict
        label_column: str | None = None,
        feature_columns: list[str] | None = None,
        split: str | None = None,
        task_type: TaskType | None = None,
        data_type: DataType | None = None,
    ):
        """Initialize the HuggingFace loader.

        Args:
            dataset: HuggingFace Dataset or DatasetDict
            label_column: Name of the label column
            feature_columns: Columns to use as features
            split: Split to use if DatasetDict (e.g., 'train')
            task_type: Type of ML task
            data_type: Type of data

        Raises:
            ImportError: If datasets library not installed
        """
        if not HAS_DATASETS:
            raise ImportError(
                "HuggingFace datasets library required. "
                "Install with: pip install clean-data-quality[huggingface]"
            )

        self.raw_dataset = dataset
        self.split = split
        self.config = LoaderConfig(
            label_column=label_column,
            feature_columns=feature_columns,
            task_type=task_type,
            data_type=data_type,
        )
        self._features: pd.DataFrame | None = None
        self._labels: np.ndarray | None = None
        self._info: DatasetInfo | None = None

    def _get_dataset(self) -> Any:
        """Get the actual dataset, handling DatasetDict."""
        dataset = self.raw_dataset

        # Handle DatasetDict
        if DatasetDict is not None and isinstance(dataset, DatasetDict):
            if self.split:
                if self.split not in dataset:
                    raise ValueError(
                        f"Split '{self.split}' not found. "
                        f"Available: {list(dataset.keys())}"
                    )
                dataset = dataset[self.split]
            else:
                # Default to first split
                first_split = list(dataset.keys())[0]
                dataset = dataset[first_split]

        return dataset

    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Load HuggingFace dataset and convert to DataFrame.

        Returns:
            Tuple of (features DataFrame, labels array or None)
        """
        dataset = self._get_dataset()

        # Convert to pandas
        df = dataset.to_pandas()

        # Extract labels
        labels = None
        if self.config.label_column:
            if self.config.label_column not in df.columns:
                raise ValueError(
                    f"Label column '{self.config.label_column}' not found in dataset"
                )
            labels = df[self.config.label_column].values

        # Determine feature columns
        if self.config.feature_columns:
            feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        else:
            exclude = set()
            if self.config.label_column:
                exclude.add(self.config.label_column)
            feature_cols = [c for c in df.columns if c not in exclude]

        features = df[feature_cols].copy()

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


def load_huggingface(
    dataset: Any,
    label_column: str | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray | None, DatasetInfo]:
    """Convenience function to load a HuggingFace dataset.

    Args:
        dataset: HuggingFace Dataset or DatasetDict
        label_column: Name of the label column
        **kwargs: Additional arguments for HuggingFaceLoader

    Returns:
        Tuple of (features, labels, info)
    """
    loader = HuggingFaceLoader(dataset, label_column=label_column, **kwargs)
    features, labels = loader.load()
    info = loader.get_info()
    return features, labels, info
