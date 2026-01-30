"""Tests for the plugin architecture."""

import pandas as pd
import pytest

from clean.detection.base import DetectorResult
from clean.plugins import (
    DetectorPlugin,
    ExporterPlugin,
    FixerPlugin,
    LoaderPlugin,
    PluginRegistry,
    SuggestedFix,
    registry,
)


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        self.registry = PluginRegistry()

    def test_register_detector(self):
        """Test registering a detector plugin."""

        @self.registry.detector("test_detector", description="A test detector")
        class TestDetector(DetectorPlugin):
            def fit(self, features, labels=None):
                return self

            def detect(self, features, labels=None):
                return DetectorResult(issues=[], metadata={})

        assert "test_detector" in [p.name for p in self.registry.list_detectors()]
        detector_cls = self.registry.get_detector("test_detector")
        assert detector_cls == TestDetector

    def test_register_loader(self):
        """Test registering a loader plugin."""

        @self.registry.loader("test_loader")
        class TestLoader(LoaderPlugin):
            def load(self):
                return pd.DataFrame(), None, {}

        assert "test_loader" in [p.name for p in self.registry.list_loaders()]

    def test_register_exporter(self):
        """Test registering an exporter plugin."""

        @self.registry.exporter("test_exporter")
        class TestExporter(ExporterPlugin):
            def export(self, report, path):
                pass

            @property
            def extension(self):
                return ".test"

        assert "test_exporter" in [p.name for p in self.registry.list_exporters()]

    def test_register_fixer(self):
        """Test registering a fixer plugin."""

        @self.registry.fixer("test_fixer")
        class TestFixer(FixerPlugin):
            def suggest_fixes(self, issues, features, labels=None):
                return []

            def apply_fix(self, fix, features, labels=None):
                return features, labels

        assert "test_fixer" in [p.name for p in self.registry.list_fixers()]

    def test_get_nonexistent_detector(self):
        """Test getting a non-existent detector raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            self.registry.get_detector("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_detector_must_subclass(self):
        """Test that registered detector must subclass DetectorPlugin."""
        with pytest.raises(TypeError):

            @self.registry.detector("bad_detector")
            class BadDetector:
                pass

    def test_plugin_info_metadata(self):
        """Test plugin info contains correct metadata."""

        @self.registry.detector(
            "meta_detector",
            description="Detector with metadata",
            version="2.0.0",
            author="Test Author",
            tags=["test", "example"],
        )
        class MetaDetector(DetectorPlugin):
            def fit(self, features, labels=None):
                return self

            def detect(self, features, labels=None):
                return DetectorResult(issues=[], metadata={})

        info = self.registry.list_detectors()[0]
        assert info.name == "meta_detector"
        assert info.description == "Detector with metadata"
        assert info.version == "2.0.0"
        assert info.author == "Test Author"
        assert info.tags == ["test", "example"]

    def test_list_all(self):
        """Test listing all plugins."""

        @self.registry.detector("d1")
        class D1(DetectorPlugin):
            def fit(self, features, labels=None):
                return self

            def detect(self, features, labels=None):
                return DetectorResult(issues=[], metadata={})

        @self.registry.loader("l1")
        class L1(LoaderPlugin):
            def load(self):
                return pd.DataFrame(), None, {}

        all_plugins = self.registry.list_all()
        assert len(all_plugins["detectors"]) == 1
        assert len(all_plugins["loaders"]) == 1
        assert len(all_plugins["exporters"]) == 0
        assert len(all_plugins["fixers"]) == 0

    def test_clear_registry(self):
        """Test clearing the registry."""

        @self.registry.detector("to_clear")
        class ToClear(DetectorPlugin):
            def fit(self, features, labels=None):
                return self

            def detect(self, features, labels=None):
                return DetectorResult(issues=[], metadata={})

        assert len(self.registry.list_detectors()) == 1
        self.registry.clear()
        assert len(self.registry.list_detectors()) == 0


class TestDetectorPlugin:
    """Tests for DetectorPlugin base class."""

    def test_fit_detect(self):
        """Test fit_detect convenience method."""

        class SimpleDetector(DetectorPlugin):
            def fit(self, features, labels=None):
                self.fitted = True
                return self

            def detect(self, features, labels=None):
                return DetectorResult(
                    issues=[], metadata={"fitted": getattr(self, "fitted", False)}
                )

        detector = SimpleDetector()
        features = pd.DataFrame({"a": [1, 2, 3]})
        result = detector.fit_detect(features)

        assert result.metadata["fitted"] is True


class TestSuggestedFix:
    """Tests for SuggestedFix dataclass."""

    def test_suggested_fix_creation(self):
        """Test creating a suggested fix."""
        fix = SuggestedFix(
            issue_type="label_error",
            issue_index=42,
            fix_type="relabel",
            confidence=0.95,
            description="Change label from 'cat' to 'dog'",
            old_value="cat",
            new_value="dog",
        )

        assert fix.issue_type == "label_error"
        assert fix.issue_index == 42
        assert fix.confidence == 0.95
        assert fix.old_value == "cat"
        assert fix.new_value == "dog"

    def test_suggested_fix_repr(self):
        """Test SuggestedFix string representation."""
        fix = SuggestedFix(
            issue_type="duplicate",
            issue_index=(10, 20),
            fix_type="merge",
            confidence=0.85,
            description="Merge duplicate samples",
        )

        repr_str = repr(fix)
        assert "merge" in repr_str
        assert "duplicate" in repr_str
        assert "(10, 20)" in repr_str
        assert "0.85" in repr_str


class TestGlobalRegistry:
    """Tests for the global registry instance."""

    def test_global_registry_exists(self):
        """Test that global registry is available."""
        assert registry is not None
        assert isinstance(registry, PluginRegistry)
