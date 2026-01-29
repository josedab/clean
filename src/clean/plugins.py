"""Plugin architecture for Clean.

This module provides a registry system for extending Clean with custom:
- Detectors (for new issue types)
- Loaders (for new data sources)
- Exporters (for new output formats)
- Fixers (for auto-fix strategies)

Example:
    >>> from clean.plugins import registry, DetectorPlugin
    >>>
    >>> @registry.detector("my_detector")
    >>> class MyDetector(DetectorPlugin):
    ...     def detect(self, features, labels):
    ...         # Custom detection logic
    ...         return DetectorResult(issues=[], metadata={})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from clean.detection.base import DetectorResult


T = TypeVar("T")


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""

    name: str
    plugin_type: str
    cls: type
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)


class DetectorPlugin(ABC):
    """Base class for custom detector plugins.

    Subclass this to create custom issue detectors that integrate
    with Clean's analysis pipeline.

    Example:
        >>> class DriftDetector(DetectorPlugin):
        ...     def __init__(self, threshold: float = 0.1):
        ...         self.threshold = threshold
        ...
        ...     def fit(self, features, labels=None):
        ...         self._baseline = features.mean()
        ...         return self
        ...
        ...     def detect(self, features, labels=None):
        ...         drift = abs(features.mean() - self._baseline)
        ...         issues = [...]  # Create issues for drifted features
        ...         return DetectorResult(issues=issues, metadata={"drift": drift})
    """

    @abstractmethod
    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorPlugin:
        """Fit the detector on training data.

        Args:
            features: Feature data
            labels: Optional label data

        Returns:
            Self for method chaining
        """
        ...

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
        ...

    def fit_detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Fit and detect in one call."""
        return self.fit(features, labels).detect(features, labels)


class LoaderPlugin(ABC):
    """Base class for custom data loader plugins.

    Subclass this to create loaders for custom data sources.
    """

    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
        """Load data from source.

        Returns:
            Tuple of (features DataFrame, labels array, info dict)
        """
        ...

    def validate(self) -> bool:
        """Validate the data source is accessible.

        Returns:
            True if source is valid
        """
        return True


class ExporterPlugin(ABC):
    """Base class for custom report exporter plugins.

    Subclass this to create exporters for custom output formats.
    """

    @abstractmethod
    def export(self, report: Any, path: str) -> None:
        """Export a report to the specified path.

        Args:
            report: QualityReport to export
            path: Output file path
        """
        ...

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this exporter (e.g., '.xlsx', '.parquet')."""
        ...


class FixerPlugin(ABC):
    """Base class for custom auto-fix plugins.

    Subclass this to create custom fix strategies for detected issues.
    """

    @abstractmethod
    def suggest_fixes(
        self,
        issues: list[Any],
        features: pd.DataFrame,
        labels: np.ndarray | None = None,
    ) -> list[SuggestedFix]:
        """Generate fix suggestions for detected issues.

        Args:
            issues: List of detected issues
            features: Original feature data
            labels: Original labels

        Returns:
            List of suggested fixes
        """
        ...

    @abstractmethod
    def apply_fix(
        self,
        fix: SuggestedFix,
        features: pd.DataFrame,
        labels: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Apply a single fix to the data.

        Args:
            fix: Fix to apply
            features: Feature data
            labels: Label data

        Returns:
            Tuple of (modified features, modified labels)
        """
        ...


@dataclass
class SuggestedFix:
    """A suggested fix for a detected issue."""

    issue_type: str
    issue_index: int | tuple[int, int]
    fix_type: str  # 'relabel', 'remove', 'merge', 'impute', etc.
    confidence: float
    description: str
    old_value: Any = None
    new_value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SuggestedFix({self.fix_type} for {self.issue_type}@{self.issue_index}, conf={self.confidence:.2f})"


class PluginRegistry:
    """Central registry for all Clean plugins.

    This class manages registration and discovery of plugins.
    Use the global `registry` instance.

    Example:
        >>> from clean.plugins import registry
        >>>
        >>> # Register a detector
        >>> @registry.detector("drift")
        >>> class DriftDetector(DetectorPlugin):
        ...     ...
        >>>
        >>> # Get registered detector
        >>> detector_cls = registry.get_detector("drift")
        >>> detector = detector_cls()
    """

    def __init__(self) -> None:
        self._detectors: dict[str, PluginInfo] = {}
        self._loaders: dict[str, PluginInfo] = {}
        self._exporters: dict[str, PluginInfo] = {}
        self._fixers: dict[str, PluginInfo] = {}

    def detector(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a detector plugin.

        Args:
            name: Unique name for the detector
            description: Human-readable description
            version: Plugin version
            author: Plugin author
            tags: Searchable tags

        Returns:
            Decorator function
        """

        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, DetectorPlugin):
                raise TypeError(f"{cls.__name__} must subclass DetectorPlugin")
            self._detectors[name] = PluginInfo(
                name=name,
                plugin_type="detector",
                cls=cls,
                description=description or cls.__doc__ or "",
                version=version,
                author=author,
                tags=tags or [],
            )
            return cls

        return decorator

    def loader(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a loader plugin."""

        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, LoaderPlugin):
                raise TypeError(f"{cls.__name__} must subclass LoaderPlugin")
            self._loaders[name] = PluginInfo(
                name=name,
                plugin_type="loader",
                cls=cls,
                description=description or cls.__doc__ or "",
                version=version,
                author=author,
                tags=tags or [],
            )
            return cls

        return decorator

    def exporter(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register an exporter plugin."""

        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, ExporterPlugin):
                raise TypeError(f"{cls.__name__} must subclass ExporterPlugin")
            self._exporters[name] = PluginInfo(
                name=name,
                plugin_type="exporter",
                cls=cls,
                description=description or cls.__doc__ or "",
                version=version,
                author=author,
                tags=tags or [],
            )
            return cls

        return decorator

    def fixer(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a fixer plugin."""

        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, FixerPlugin):
                raise TypeError(f"{cls.__name__} must subclass FixerPlugin")
            self._fixers[name] = PluginInfo(
                name=name,
                plugin_type="fixer",
                cls=cls,
                description=description or cls.__doc__ or "",
                version=version,
                author=author,
                tags=tags or [],
            )
            return cls

        return decorator

    def get_detector(self, name: str) -> type[DetectorPlugin]:
        """Get a registered detector by name."""
        if name not in self._detectors:
            raise KeyError(f"Detector '{name}' not found. Available: {list(self._detectors.keys())}")
        return self._detectors[name].cls

    def get_loader(self, name: str) -> type[LoaderPlugin]:
        """Get a registered loader by name."""
        if name not in self._loaders:
            raise KeyError(f"Loader '{name}' not found. Available: {list(self._loaders.keys())}")
        return self._loaders[name].cls

    def get_exporter(self, name: str) -> type[ExporterPlugin]:
        """Get a registered exporter by name."""
        if name not in self._exporters:
            raise KeyError(f"Exporter '{name}' not found. Available: {list(self._exporters.keys())}")
        return self._exporters[name].cls

    def get_fixer(self, name: str) -> type[FixerPlugin]:
        """Get a registered fixer by name."""
        if name not in self._fixers:
            raise KeyError(f"Fixer '{name}' not found. Available: {list(self._fixers.keys())}")
        return self._fixers[name].cls

    def list_detectors(self) -> list[PluginInfo]:
        """List all registered detectors."""
        return list(self._detectors.values())

    def list_loaders(self) -> list[PluginInfo]:
        """List all registered loaders."""
        return list(self._loaders.values())

    def list_exporters(self) -> list[PluginInfo]:
        """List all registered exporters."""
        return list(self._exporters.values())

    def list_fixers(self) -> list[PluginInfo]:
        """List all registered fixers."""
        return list(self._fixers.values())

    def list_all(self) -> dict[str, list[PluginInfo]]:
        """List all registered plugins by type."""
        return {
            "detectors": self.list_detectors(),
            "loaders": self.list_loaders(),
            "exporters": self.list_exporters(),
            "fixers": self.list_fixers(),
        }

    def clear(self) -> None:
        """Clear all registered plugins (useful for testing)."""
        self._detectors.clear()
        self._loaders.clear()
        self._exporters.clear()
        self._fixers.clear()


# Global registry instance
registry = PluginRegistry()

__all__ = [
    "registry",
    "PluginRegistry",
    "PluginInfo",
    "DetectorPlugin",
    "LoaderPlugin",
    "ExporterPlugin",
    "FixerPlugin",
    "SuggestedFix",
]
