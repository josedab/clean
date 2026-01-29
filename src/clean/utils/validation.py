"""Input validation utilities."""

from typing import Any

import numpy as np
import pandas as pd


def validate_features(
    features: pd.DataFrame | np.ndarray,
    min_samples: int = 1,
    min_features: int = 1,
) -> pd.DataFrame:
    """Validate feature data.

    Args:
        features: Feature data
        min_samples: Minimum number of samples
        min_features: Minimum number of features

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If validation fails
    """
    if isinstance(features, np.ndarray):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        features = pd.DataFrame(features)

    if features.empty:
        raise ValueError("Features cannot be empty")

    if len(features) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {len(features)}")

    if len(features.columns) < min_features:
        raise ValueError(
            f"Need at least {min_features} features, got {len(features.columns)}"
        )

    return features


def validate_labels(
    labels: np.ndarray | pd.Series | list,
    n_samples: int | None = None,
    min_classes: int = 2,
) -> np.ndarray:
    """Validate label data.

    Args:
        labels: Label data
        n_samples: Expected number of samples
        min_classes: Minimum number of classes

    Returns:
        Validated numpy array

    Raises:
        ValueError: If validation fails
    """
    labels = np.asarray(labels)

    if labels.size == 0:
        raise ValueError("Labels cannot be empty")

    if n_samples is not None and len(labels) != n_samples:
        raise ValueError(
            f"Labels length ({len(labels)}) doesn't match "
            f"expected samples ({n_samples})"
        )

    n_classes = len(np.unique(labels[~pd.isna(labels)]))
    if n_classes < min_classes:
        raise ValueError(
            f"Need at least {min_classes} classes, got {n_classes}"
        )

    return labels


def validate_threshold(
    value: float,
    name: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """Validate a threshold value.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated value

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")

    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

    return float(value)


def validate_positive_int(value: Any, name: str) -> int:
    """Validate a positive integer.

    Args:
        value: Value to validate
        name: Parameter name

    Returns:
        Validated integer

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")

    return value
