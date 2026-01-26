"""Domain-specific exceptions for Clean library.

This module provides a hierarchy of exceptions for better error handling
and more informative error messages throughout the Clean library.

Example:
    >>> from clean.exceptions import DetectionError, ValidationError
    >>>
    >>> try:
    ...     detector.fit(features, labels)
    ... except DetectionError as e:
    ...     print(f"Detection failed: {e}")
"""

from __future__ import annotations

from typing import Any


class CleanError(Exception):
    """Base exception for all Clean library errors.

    All Clean-specific exceptions inherit from this class, allowing
    users to catch all library errors with a single except clause.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ValidationError(CleanError):
    """Raised when input data or configuration is invalid.

    This exception is raised during data validation, configuration
    parsing, or when required inputs are missing or malformed.

    Example:
        >>> raise ValidationError(
        ...     "Label column not found",
        ...     details={"column": "target", "available": ["a", "b", "c"]}
        ... )
    """

    pass


class DetectionError(CleanError):
    """Raised when issue detection fails.

    This exception is raised when a detector encounters an error
    during fitting or detection, such as insufficient data or
    incompatible data types.

    Example:
        >>> raise DetectionError(
        ...     "Insufficient samples for cross-validation",
        ...     details={"n_samples": 5, "cv_folds": 10}
        ... )
    """

    pass


class ConfigurationError(CleanError):
    """Raised when configuration is invalid or inconsistent.

    This exception is raised when plugin configuration, strategy
    settings, or other configuration options are invalid.
    """

    pass


class PluginError(CleanError):
    """Raised when plugin registration or loading fails.

    This exception is raised when a plugin cannot be registered,
    loaded, or when a requested plugin is not found.
    """

    pass


class FixError(CleanError):
    """Raised when applying fixes fails.

    This exception is raised when the FixEngine encounters an error
    while suggesting or applying fixes to the dataset.
    """

    pass


class LoaderError(CleanError):
    """Raised when data loading fails.

    This exception is raised when a loader cannot read, parse,
    or validate the input data source.
    """

    pass


class ExportError(CleanError):
    """Raised when report export fails.

    This exception is raised when exporting reports to JSON, HTML,
    or other formats fails.
    """

    pass


class DependencyError(CleanError):
    """Raised when an optional dependency is missing.

    This exception provides helpful installation instructions
    when optional packages are not available.

    Example:
        >>> raise DependencyError(
        ...     "sentence-transformers",
        ...     extra="text",
        ...     feature="embedding-based duplicate detection"
        ... )
    """

    def __init__(
        self,
        package: str,
        extra: str,
        feature: str | None = None,
    ):
        """Initialize the dependency error.

        Args:
            package: Name of the missing package
            extra: pip extra that includes this dependency
            feature: Optional description of the feature requiring this package
        """
        feature_msg = f" for {feature}" if feature else ""
        message = (
            f"{package} is required{feature_msg}. "
            f"Install with: pip install clean-data-quality[{extra}]"
        )
        super().__init__(message, details={"package": package, "extra": extra})
        self.package = package
        self.extra = extra


class StreamingError(CleanError):
    """Raised when streaming analysis fails.

    This exception is raised during chunked or streaming analysis
    when processing a chunk fails or the stream is interrupted.
    """

    pass


def require_package(package: str, extra: str, feature: str | None = None) -> None:
    """Raise DependencyError for a missing optional package.

    This is a convenience function for checking optional dependencies
    at the point of use.

    Args:
        package: Name of the missing package
        extra: pip extra that includes this dependency
        feature: Optional description of the feature

    Raises:
        DependencyError: Always raised with installation instructions
    """
    raise DependencyError(package, extra, feature)


__all__ = [
    "CleanError",
    "ConfigurationError",
    "DependencyError",
    "DetectionError",
    "ExportError",
    "FixError",
    "LoaderError",
    "PluginError",
    "StreamingError",
    "ValidationError",
    "require_package",
]
