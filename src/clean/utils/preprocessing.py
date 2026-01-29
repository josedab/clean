"""Data preprocessing utilities."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_labels(labels: np.ndarray | pd.Series) -> tuple[np.ndarray, dict[int, Any]]:
    """Encode labels to integers.

    Args:
        labels: Original labels

    Returns:
        Tuple of (encoded labels, mapping from int to original)
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    mapping = {i: label for i, label in enumerate(encoder.classes_)}
    return encoded, mapping


def decode_labels(encoded: np.ndarray, mapping: dict[int, Any]) -> np.ndarray:
    """Decode labels back to original values.

    Args:
        encoded: Encoded labels
        mapping: Integer to original mapping

    Returns:
        Decoded labels
    """
    return np.array([mapping[i] for i in encoded])


def scale_features(
    features: pd.DataFrame | np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Scale features using StandardScaler.

    Args:
        features: Feature data
        scaler: Existing scaler (fit_transform if None)

    Returns:
        Tuple of (scaled features, scaler)
    """
    if isinstance(features, pd.DataFrame):
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_cols].values
    else:
        X = features

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
    else:
        scaled = scaler.transform(X)

    return scaled, scaler


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value: Any = None,
) -> pd.DataFrame:
    """Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        strategy: 'drop', 'fill', 'mean', 'median', 'mode'
        fill_value: Value to use for 'fill' strategy

    Returns:
        DataFrame with missing values handled
    """
    result = df.copy()

    if strategy == "drop":
        return result.dropna()

    if strategy == "fill":
        return result.fillna(fill_value)

    if strategy == "mean":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col].fillna(result[col].mean(), inplace=True)
        return result

    if strategy == "median":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col].fillna(result[col].median(), inplace=True)
        return result

    if strategy == "mode":
        for col in result.columns:
            mode_val = result[col].mode()
            if len(mode_val) > 0:
                result[col].fillna(mode_val[0], inplace=True)
        return result

    raise ValueError(f"Unknown strategy: {strategy}")


def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Get only numeric features from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with only numeric columns
    """
    return df.select_dtypes(include=[np.number])


def get_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Get only categorical features from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with only categorical columns
    """
    return df.select_dtypes(include=["object", "category"])
