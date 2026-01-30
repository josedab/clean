"""Tests for the DatasetCleaner class."""

import numpy as np
import pandas as pd

from clean import DatasetCleaner, QualityReport


class TestDatasetCleaner:
    """Tests for DatasetCleaner."""

    def test_init_with_dataframe(self, sample_dataframe):
        """Test initialization with DataFrame."""
        cleaner = DatasetCleaner(
            data=sample_dataframe,
            label_column="label",
            task="classification",
        )

        assert cleaner.features is not None
        assert cleaner.labels is not None
        assert len(cleaner.features) == len(sample_dataframe)
        assert "label" not in cleaner.features.columns

    def test_init_with_numpy(self):
        """Test initialization with numpy arrays."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        cleaner = DatasetCleaner(data=X, labels=y, task="classification")

        assert cleaner.features is not None
        assert cleaner.labels is not None
        assert len(cleaner.features) == 100

    def test_analyze_basic(self, sample_classification_data):
        """Test basic analysis."""
        df, labels, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
            task="classification",
        )

        report = cleaner.analyze(show_progress=False)

        assert isinstance(report, QualityReport)
        assert report.quality_score.overall >= 0
        assert report.quality_score.overall <= 100

    def test_analyze_detectors(self, sample_classification_data):
        """Test that all detectors run."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
            task="classification",
        )

        report = cleaner.analyze(
            detect_label_errors=True,
            detect_duplicates=True,
            detect_outliers=True,
            detect_imbalance=True,
            detect_bias=True,
            show_progress=False,
        )

        # Check all results exist
        assert report.label_errors_result is not None
        assert report.duplicates_result is not None
        assert report.outliers_result is not None
        assert report.imbalance_result is not None
        assert report.bias_result is not None

    def test_get_clean_data(self, sample_data_with_duplicates):
        """Test getting cleaned data."""
        cleaner = DatasetCleaner(
            data=sample_data_with_duplicates,
            label_column="label",
            task="classification",
        )

        cleaner.analyze(
            detect_label_errors=False,
            detect_outliers=False,
            detect_bias=False,
            detect_imbalance=False,
            show_progress=False,
        )

        clean_df = cleaner.get_clean_data(remove_duplicates=True)

        # Should have fewer rows due to duplicate removal
        assert len(clean_df) < len(sample_data_with_duplicates)

    def test_get_review_queue(self, sample_classification_data):
        """Test review queue generation."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
            task="classification",
        )

        cleaner.analyze(show_progress=False)

        queue = cleaner.get_review_queue(max_items=10)

        assert isinstance(queue, pd.DataFrame)
        assert len(queue) <= 10
        if len(queue) > 0:
            assert "index" in queue.columns
            assert "issue_type" in queue.columns

    def test_repr(self, sample_dataframe):
        """Test string representation."""
        cleaner = DatasetCleaner(
            data=sample_dataframe,
            label_column="label",
        )

        repr_str = repr(cleaner)

        assert "DatasetCleaner" in repr_str
        assert "samples=" in repr_str


class TestQualityReport:
    """Tests for QualityReport."""

    def test_summary(self, sample_classification_data):
        """Test report summary generation."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        summary = report.summary()

        assert isinstance(summary, str)
        assert "Data Quality Report" in summary
        assert "Samples analyzed" in summary

    def test_to_dict(self, sample_classification_data):
        """Test conversion to dictionary."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "dataset_info" in report_dict
        assert "quality_score" in report_dict

    def test_to_json(self, sample_classification_data):
        """Test JSON serialization."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        json_str = report.to_json()

        assert isinstance(json_str, str)
        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_label_errors_dataframe(self, sample_classification_data):
        """Test label errors as DataFrame."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        errors_df = report.label_errors()

        assert isinstance(errors_df, pd.DataFrame)
        if len(errors_df) > 0:
            assert "index" in errors_df.columns
            assert "given_label" in errors_df.columns
            assert "predicted_label" in errors_df.columns

    def test_get_all_issue_indices(self, sample_classification_data):
        """Test getting all issue indices."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        indices = report.get_all_issue_indices()

        assert isinstance(indices, set)
