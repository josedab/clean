"""Base detector interface for Clean."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DetectorResult:
    """Result from a detector."""

    issues: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_issues(self) -> int:
        """Number of issues found."""
        return len(self.issues)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to DataFrame."""
        if not self.issues:
            return pd.DataFrame()
        return pd.DataFrame([issue.to_dict() for issue in self.issues])


class BaseDetector(ABC):
    """Abstract base class for issue detectors."""

    def __init__(self, **kwargs: Any):
        """Initialize detector with configuration.

        Args:
            **kwargs: Detector-specific configuration
        """
        self.config = kwargs
        self._is_fitted = False

    @abstractmethod
    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> "BaseDetector":
        """Fit the detector to data.

        Args:
            features: Feature data
            labels: Optional label data

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Detect issues in the data.

        Args:
            features: Feature data
            labels: Optional label data

        Returns:
            DetectorResult with found issues
        """
        pass

    def fit_detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Fit and detect in one step.

        Args:
            features: Feature data
            labels: Optional label data

        Returns:
            DetectorResult with found issues
        """
        self.fit(features, labels)
        return self.detect(features, labels)

    @property
    def is_fitted(self) -> bool:
        """Check if detector has been fitted."""
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Check if detector is fitted, raise if not."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before detecting. "
                "Call fit() or fit_detect() first."
            )

    @staticmethod
    def _to_numpy(data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)

    @staticmethod
    def _get_feature_names(data: pd.DataFrame | np.ndarray) -> list[str]:
        """Get feature names from data."""
        if isinstance(data, pd.DataFrame):
            return list(data.columns)
        return [f"feature_{i}" for i in range(data.shape[1] if data.ndim > 1 else 1)]
