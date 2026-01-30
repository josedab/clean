"""Tests for Clean exception hierarchy."""

import pytest

from clean.exceptions import (
    CleanError,
    ConfigurationError,
    DependencyError,
    DetectionError,
    ExportError,
    FixError,
    LoaderError,
    PluginError,
    StreamingError,
    ValidationError,
    require_package,
)


class TestCleanError:
    """Tests for base CleanError."""

    def test_basic_message(self) -> None:
        """Test exception with just a message."""
        error = CleanError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_message_with_details(self) -> None:
        """Test exception with message and details."""
        error = CleanError(
            "Validation failed",
            details={"field": "label", "reason": "not found"},
        )
        assert "Validation failed" in str(error)
        assert "field=label" in str(error)
        assert "reason=not found" in str(error)
        assert error.details == {"field": "label", "reason": "not found"}

    def test_inheritance(self) -> None:
        """Test that CleanError inherits from Exception."""
        error = CleanError("test")
        assert isinstance(error, Exception)


class TestValidationError:
    """Tests for ValidationError."""

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        error = ValidationError("Invalid input")
        assert isinstance(error, CleanError)
        assert isinstance(error, Exception)

    def test_with_details(self) -> None:
        """Test with validation-specific details."""
        error = ValidationError(
            "Label column not found",
            details={"column": "target", "available": ["a", "b"]},
        )
        assert error.details["column"] == "target"


class TestDetectionError:
    """Tests for DetectionError."""

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        error = DetectionError("Detection failed")
        assert isinstance(error, CleanError)

    def test_typical_usage(self) -> None:
        """Test typical detection error scenario."""
        error = DetectionError(
            "Insufficient samples for cross-validation",
            details={"n_samples": 5, "cv_folds": 10},
        )
        assert "Insufficient samples" in str(error)
        assert error.details["n_samples"] == 5


class TestDependencyError:
    """Tests for DependencyError."""

    def test_basic_initialization(self) -> None:
        """Test basic dependency error."""
        error = DependencyError("pandas", "core")
        assert "pandas" in str(error)
        assert "pip install clean-data-quality[core]" in str(error)
        assert error.package == "pandas"
        assert error.extra == "core"

    def test_with_feature(self) -> None:
        """Test with feature description."""
        error = DependencyError(
            "sentence-transformers",
            "text",
            feature="embedding-based duplicate detection",
        )
        assert "sentence-transformers" in str(error)
        assert "embedding-based duplicate detection" in str(error)
        assert "pip install clean-data-quality[text]" in str(error)

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        error = DependencyError("pkg", "extra")
        assert isinstance(error, CleanError)


class TestRequirePackage:
    """Tests for require_package helper."""

    def test_raises_dependency_error(self) -> None:
        """Test that require_package raises DependencyError."""
        with pytest.raises(DependencyError) as exc_info:
            require_package("torch", "image", "image analysis")

        assert exc_info.value.package == "torch"
        assert exc_info.value.extra == "image"
        assert "image analysis" in str(exc_info.value)


class TestOtherExceptions:
    """Tests for other exception types."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, CleanError)

    def test_plugin_error(self) -> None:
        """Test PluginError."""
        error = PluginError("Plugin not found", details={"name": "my_plugin"})
        assert isinstance(error, CleanError)

    def test_fix_error(self) -> None:
        """Test FixError."""
        error = FixError("Cannot apply fix")
        assert isinstance(error, CleanError)

    def test_loader_error(self) -> None:
        """Test LoaderError."""
        error = LoaderError("Failed to load CSV")
        assert isinstance(error, CleanError)

    def test_export_error(self) -> None:
        """Test ExportError."""
        error = ExportError("Cannot export to HTML")
        assert isinstance(error, CleanError)

    def test_streaming_error(self) -> None:
        """Test StreamingError."""
        error = StreamingError("Chunk processing failed")
        assert isinstance(error, CleanError)


class TestExceptionCatching:
    """Test catching exceptions at different levels."""

    def test_catch_specific_exception(self) -> None:
        """Test catching a specific exception type."""
        with pytest.raises(ValidationError):
            raise ValidationError("Invalid input")

    def test_catch_base_exception(self) -> None:
        """Test catching all Clean exceptions via base class."""
        exceptions_to_test = [
            ValidationError("test"),
            DetectionError("test"),
            ConfigurationError("test"),
            PluginError("test"),
            FixError("test"),
            LoaderError("test"),
            ExportError("test"),
            StreamingError("test"),
            DependencyError("pkg", "extra"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except CleanError as caught:
                assert caught is exc
