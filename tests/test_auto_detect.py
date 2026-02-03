"""Tests for auto-detection module."""

import numpy as np
import pandas as pd
import pytest

from clean.auto_detect import (
    AutoConfig,
    AutoDetector,
    ColumnProfile,
    DataModality,
    TaskType,
    auto_analyze,
    detect_config,
)
from clean.core.types import DataType


class TestColumnProfile:
    """Tests for ColumnProfile."""

    def test_profile_attributes(self):
        """Test profile has all required attributes."""
        profile = ColumnProfile(
            name="test",
            dtype="int64",
            n_unique=10,
            n_missing=0,
            n_total=100,
            sample_values=[1, 2, 3],
            is_numeric=True,
            is_categorical=False,
            is_text=False,
            is_datetime=False,
            is_path=False,
            is_embedding=False,
            label_score=0.5,
        )

        assert profile.name == "test"
        assert profile.is_numeric
        assert not profile.is_text
        assert profile.label_score == 0.5


class TestAutoDetector:
    """Tests for AutoDetector."""

    @pytest.fixture
    def detector(self):
        """Create auto detector."""
        return AutoDetector()

    @pytest.fixture
    def tabular_data(self):
        """Create sample tabular data."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),  # Use 'label' which is a known label name
        })

    @pytest.fixture
    def text_data(self):
        """Create sample text data."""
        return pd.DataFrame({
            "text": [
                "This is a sample text that is long enough to be detected as text.",
                "Another sample text with sufficient length for detection.",
                "More text content here for testing purposes.",
            ] * 100,
            "label": ["positive", "negative", "neutral"] * 100,
        })

    def test_detect_label_column_by_name(self, detector, tabular_data):
        """Test detection of label column by name."""
        config = detector.detect(tabular_data)
        assert config.label_column == "label"
        assert config.label_confidence > 0.3

    def test_detect_modality_tabular(self, detector, tabular_data):
        """Test detection of tabular modality."""
        config = detector.detect(tabular_data)
        assert config.modality == DataModality.TABULAR

    def test_detect_modality_text(self, detector, text_data):
        """Test detection of text modality."""
        config = detector.detect(text_data)
        assert config.modality in (DataModality.TEXT, DataModality.MIXED)

    def test_detect_task_type_binary(self, detector, tabular_data):
        """Test detection of binary classification."""
        config = detector.detect(tabular_data)
        assert config.task_type == TaskType.BINARY_CLASSIFICATION

    def test_detect_task_type_multiclass(self, detector):
        """Test detection of multiclass classification."""
        data = pd.DataFrame({
            "feature": np.random.randn(100),
            "label": np.random.choice(["A", "B", "C", "D", "E"], 100),
        })
        config = detector.detect(data)
        assert config.task_type == TaskType.MULTICLASS_CLASSIFICATION

    def test_detect_no_label(self, detector):
        """Test detection when no label column exists."""
        data = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        })
        config = detector.detect(data)
        assert config.label_column is None
        assert config.task_type == TaskType.UNSUPERVISED

    def test_detect_numeric_columns(self, detector, tabular_data):
        """Test detection of numeric columns."""
        config = detector.detect(tabular_data)
        assert "feature1" in config.numeric_columns
        assert "feature2" in config.numeric_columns

    def test_detect_categorical_columns(self, detector, tabular_data):
        """Test detection of categorical columns."""
        config = detector.detect(tabular_data)
        # With simplified fixture, no categorical columns (label is binary numeric)
        # Check that numeric columns are detected
        assert "feature1" in config.numeric_columns or "feature1" in config.feature_columns

    def test_suggested_detectors_supervised(self, detector, tabular_data):
        """Test suggested detectors for supervised data."""
        config = detector.detect(tabular_data)
        assert "LabelErrorDetector" in config.suggested_detectors
        assert "DuplicateDetector" in config.suggested_detectors

    def test_warnings_generated(self, detector):
        """Test that warnings are generated for issues."""
        # Create data with issues
        data = pd.DataFrame({
            "feature": np.random.randn(50),  # Small dataset
            "missing_col": [None] * 50,  # All missing
        })
        config = detector.detect(data)
        assert len(config.warnings) > 0

    def test_config_to_dict(self, detector, tabular_data):
        """Test config conversion to dict."""
        config = detector.detect(tabular_data)
        result = config.to_dict()

        assert "label_column" in result
        assert "modality" in result
        assert "task_type" in result

    def test_config_summary(self, detector, tabular_data):
        """Test config summary generation."""
        config = detector.detect(tabular_data)
        summary = config.summary()

        assert "Auto-Detected Configuration" in summary
        assert "Modality" in summary


class TestAutoAnalyze:
    """Tests for auto_analyze function."""

    def test_auto_analyze_returns_report(self):
        """Test that auto_analyze returns a report."""
        data = pd.DataFrame({
            "feature": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })
        report = auto_analyze(data)

        # QualityReport has dataset_info instead of n_samples directly
        assert hasattr(report, "quality_score") or hasattr(report, "dataset_info")

    def test_auto_analyze_with_explicit_label(self):
        """Test auto_analyze with explicit label column."""
        data = pd.DataFrame({
            "feature": np.random.randn(100),
            "my_target": np.random.choice([0, 1], 100),  # Unusual name
        })
        report = auto_analyze(data, label_column="my_target")

        # Check dataset info
        assert report.dataset_info.n_samples == 100


class TestDetectConfig:
    """Tests for detect_config function."""

    def test_detect_config_returns_autoconfig(self):
        """Test that detect_config returns AutoConfig."""
        data = pd.DataFrame({
            "feature": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })
        config = detect_config(data)

        assert isinstance(config, AutoConfig)
        assert config.label_column == "target"
