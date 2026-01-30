"""Tests for QualityReport."""

import os
import tempfile

import pandas as pd
import pytest

from clean import DatasetCleaner


class TestQualityReportMethods:
    """Tests for QualityReport methods."""

    @pytest.fixture
    def report(self, sample_classification_data):
        """Create a report for testing."""
        df, _, _ = sample_classification_data
        cleaner = DatasetCleaner(data=df, label_column="label")
        return cleaner.analyze(show_progress=False)

    def test_summary_format(self, report):
        """Test summary has correct format."""
        summary = report.summary()

        assert "Data Quality Report" in summary
        assert "Samples analyzed" in summary
        assert "Overall Quality Score" in summary
        assert "Component Scores" in summary
        assert "Issues Found" in summary

    def test_to_html(self, report):
        """Test HTML export."""
        html = report.to_html()

        assert "<html>" in html
        assert "Quality Scores" in html
        assert "</html>" in html

    def test_save_html(self, report):
        """Test saving HTML to file."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            report.save_html(f.name)
            assert os.path.exists(f.name)
            with open(f.name) as rf:
                content = rf.read()
            assert "<html>" in content
            os.unlink(f.name)

    def test_save_json(self, report):
        """Test saving JSON to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.save_json(f.name)
            assert os.path.exists(f.name)
            import json
            with open(f.name) as rf:
                data = json.load(rf)
            assert "quality_score" in data
            os.unlink(f.name)

    def test_duplicates_dataframe(self, sample_data_with_duplicates):
        """Test duplicates as DataFrame."""
        cleaner = DatasetCleaner(
            data=sample_data_with_duplicates,
            label_column="label",
        )
        report = cleaner.analyze(
            detect_label_errors=False,
            detect_outliers=False,
            detect_bias=False,
            show_progress=False,
        )

        dups = report.duplicates()
        assert isinstance(dups, pd.DataFrame)
        assert len(dups) > 0
        assert "index1" in dups.columns
        assert "index2" in dups.columns

    def test_outliers_dataframe(self, sample_data_with_outliers):
        """Test outliers as DataFrame."""
        df, _ = sample_data_with_outliers

        cleaner = DatasetCleaner(data=df, label_column="label")
        report = cleaner.analyze(
            detect_label_errors=False,
            detect_duplicates=False,
            detect_bias=False,
            show_progress=False,
        )

        outliers = report.outliers()
        assert isinstance(outliers, pd.DataFrame)
        assert len(outliers) > 0
        assert "index" in outliers.columns
        assert "score" in outliers.columns

    def test_bias_issues_dataframe(self, sample_biased_data):
        """Test bias issues as DataFrame."""
        cleaner = DatasetCleaner(
            data=sample_biased_data,
            label_column="label",
        )
        report = cleaner.analyze(
            detect_label_errors=False,
            detect_duplicates=False,
            detect_outliers=False,
            show_progress=False,
        )

        bias = report.bias_issues()
        assert isinstance(bias, pd.DataFrame)
