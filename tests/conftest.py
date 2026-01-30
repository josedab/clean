"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data with some issues."""
    np.random.seed(42)
    n_samples = 200

    # Features
    X = np.random.randn(n_samples, 5)

    # Labels with some errors (5% wrong)
    y = np.array([0] * 100 + [1] * 100)
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    y_with_errors = y.copy()
    y_with_errors[error_indices] = 1 - y_with_errors[error_indices]

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["label"] = y_with_errors

    return df, y_with_errors, error_indices


@pytest.fixture
def sample_dataframe():
    """Generate a sample DataFrame."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        "feature_0": np.random.randn(n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "label": np.random.choice([0, 1], n_samples),
    })

    return df


@pytest.fixture
def sample_data_with_duplicates():
    """Generate data with exact and near duplicates."""
    np.random.seed(42)

    df = pd.DataFrame({
        "feature_0": [1.0, 2.0, 3.0, 1.0, 2.001, 5.0, 6.0, 1.0],
        "feature_1": [1.0, 2.0, 3.0, 1.0, 2.001, 5.0, 6.0, 1.0],
        "label": [0, 0, 1, 0, 0, 1, 1, 0],
    })

    return df


@pytest.fixture
def sample_data_with_outliers():
    """Generate data with outliers."""
    np.random.seed(42)
    n_samples = 100

    # Normal data
    X = np.random.randn(n_samples, 3)

    # Add outliers
    outlier_indices = [0, 1, 2]
    X[0] = [10, 10, 10]  # Far from normal distribution
    X[1] = [-10, -10, -10]
    X[2] = [0, 0, 20]

    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["label"] = np.random.choice([0, 1], n_samples)

    return df, outlier_indices


@pytest.fixture
def sample_imbalanced_data():
    """Generate imbalanced classification data."""
    np.random.seed(42)

    # 90% class 0, 10% class 1
    n_majority = 180
    n_minority = 20

    X_majority = np.random.randn(n_majority, 3)
    X_minority = np.random.randn(n_minority, 3) + 2

    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)

    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["label"] = y

    return df


@pytest.fixture
def sample_text_data():
    """Generate sample text data."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",  # Near duplicate
        "Machine learning is transforming technology.",
        "Deep learning neural networks are powerful.",
        "The quick brown fox jumps over the lazy dog.",  # Exact duplicate
        "Python is a great programming language.",
    ]

    df = pd.DataFrame({
        "text": texts,
        "label": [0, 0, 1, 1, 0, 2],
    })

    return df


@pytest.fixture
def sample_biased_data():
    """Generate data with potential bias."""
    np.random.seed(42)
    n_samples = 200

    # Sensitive feature: gender
    gender = np.random.choice(["M", "F"], n_samples)

    # Feature correlated with gender
    income = np.where(gender == "M", np.random.randn(n_samples) + 1, np.random.randn(n_samples))

    # Label correlated with gender
    label_prob = np.where(gender == "M", 0.7, 0.3)
    label = np.random.binomial(1, label_prob)

    df = pd.DataFrame({
        "income": income,
        "age": np.random.randint(20, 60, n_samples),
        "gender": gender,
        "label": label,
    })

    return df
