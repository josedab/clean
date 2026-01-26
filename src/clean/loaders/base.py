"""Base loader interface for Clean."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from clean.constants import (
    DATA_TYPE_SAMPLE_SIZE,
    REGRESSION_UNIQUE_THRESHOLD,
    TEXT_COLUMN_MIN_AVG_LENGTH,
)
from clean.core.types import DatasetInfo, DataType, TaskType


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Load data and return features DataFrame and optional labels array.

        Returns:
            Tuple of (features DataFrame, labels array or None)
        """
        pass

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Get information about the loaded dataset.

        Returns:
            DatasetInfo with dataset metadata
        """
        pass

    @staticmethod
    def detect_data_type(df: pd.DataFrame) -> DataType:
        """Detect the type of data in the DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Detected DataType
        """
        has_text = False
        has_numeric = False

        for col in df.columns:
            if df[col].dtype == object:
                # Check if it's long text
                sample = df[col].dropna().head(DATA_TYPE_SAMPLE_SIZE)
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    if avg_len > TEXT_COLUMN_MIN_AVG_LENGTH:
                        has_text = True
            elif np.issubdtype(df[col].dtype, np.number):
                has_numeric = True

        if has_text and has_numeric:
            return DataType.MIXED
        if has_text:
            return DataType.TEXT
        return DataType.TABULAR

    @staticmethod
    def detect_task_type(labels: np.ndarray | pd.Series | None) -> TaskType | None:
        """Detect the ML task type from labels.

        Args:
            labels: Label array or Series

        Returns:
            Detected TaskType or None if no labels
        """
        if labels is None:
            return None

        labels_array = np.asarray(labels)

        # Check for multi-label (2D array with binary values)
        if labels_array.ndim == 2:
            unique_vals = np.unique(labels_array)
            if set(unique_vals).issubset({0, 1}):
                return TaskType.MULTI_LABEL
            return TaskType.CLASSIFICATION

        # Check unique values
        unique_vals = np.unique(labels_array[~pd.isna(labels_array)])

        # If continuous values, likely regression
        if np.issubdtype(labels_array.dtype, np.floating):
            if len(unique_vals) > REGRESSION_UNIQUE_THRESHOLD:
                return TaskType.REGRESSION

        return TaskType.CLASSIFICATION

    def validate_data(
        self, features: pd.DataFrame, labels: np.ndarray | None
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Validate loaded data.

        Args:
            features: Feature DataFrame
            labels: Label array or None

        Returns:
            Validated features and labels

        Raises:
            ValueError: If data is invalid
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty")

        if labels is not None:
            if len(labels) != len(features):
                raise ValueError(
                    f"Features ({len(features)}) and labels ({len(labels)}) "
                    "have different lengths"
                )

        return features, labels


class LoaderConfig:
    """Configuration for data loaders."""

    def __init__(
        self,
        label_column: str | None = None,
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        task_type: TaskType | None = None,
        data_type: DataType | None = None,
        **kwargs: Any,
    ):
        """Initialize loader configuration.

        Args:
            label_column: Name of the label column
            feature_columns: List of feature columns to use (None = all except label)
            exclude_columns: Columns to exclude from features
            task_type: Type of ML task
            data_type: Type of data
            **kwargs: Additional loader-specific options
        """
        self.label_column = label_column
        self.feature_columns = feature_columns
        self.exclude_columns = exclude_columns or []
        self.task_type = task_type
        self.data_type = data_type
        self.extra = kwargs
