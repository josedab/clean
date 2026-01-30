"""Integration tests for the full pipeline."""

import os
import tempfile

import numpy as np
import pandas as pd

from clean import DatasetCleaner, QualityReport


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_classification(self, sample_classification_data):
        """Test complete classification workflow."""
        df, labels, error_indices = sample_classification_data

        # 1. Initialize cleaner
        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
            task="classification",
        )

        # 2. Run analysis
        report = cleaner.analyze(show_progress=False)

        # 3. Check report
        assert isinstance(report, QualityReport)
        assert report.quality_score.overall >= 0
        assert report.quality_score.overall <= 100

        # 4. Get review queue
        queue = cleaner.get_review_queue(max_items=50)
        assert isinstance(queue, pd.DataFrame)

        # 5. Get clean data
        clean_df = cleaner.get_clean_data(
            remove_duplicates=True,
            remove_outliers="conservative",
        )
        assert len(clean_df) <= len(df)

        # 6. Export report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.save_json(f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_end_to_end_with_all_detectors(self):
        """Test with all detectors enabled."""
        np.random.seed(42)

        # Create dataset with multiple issues
        n_samples = 200

        # Features
        X = np.random.randn(n_samples, 5)
        X[0] = [10, 10, 10, 10, 10]  # Outlier

        # Labels with some errors
        y = np.array([0] * 100 + [1] * 100)
        y[50] = 1  # Label error
        y[150] = 0  # Label error

        # Create duplicates
        X[10] = X[0]
        X[20] = X[0]

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["gender"] = np.random.choice(["M", "F"], n_samples)
        df["label"] = y

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

        # All detectors should have run
        assert report.label_errors_result is not None
        assert report.duplicates_result is not None
        assert report.outliers_result is not None
        assert report.imbalance_result is not None
        assert report.bias_result is not None

        # Should find some issues
        total_issues = (
            report.label_errors_result.n_issues +
            report.duplicates_result.n_issues +
            report.outliers_result.n_issues
        )
        assert total_issues > 0

    def test_pipeline_with_numpy_input(self):
        """Test pipeline with numpy array input."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        cleaner = DatasetCleaner(
            data=X,
            labels=y,
            task="classification",
        )

        report = cleaner.analyze(show_progress=False)

        assert report is not None
        assert report.dataset_info.n_samples == 100

    def test_pipeline_with_imbalanced_data(self, sample_imbalanced_data):
        """Test pipeline with imbalanced data."""
        cleaner = DatasetCleaner(
            data=sample_imbalanced_data,
            label_column="label",
            task="classification",
        )

        report = cleaner.analyze(show_progress=False)

        # Should detect imbalance
        assert report.imbalance_result is not None
        assert report.imbalance_result.metadata["is_imbalanced"]
        assert report.quality_score.imbalance_quality < 100

    def test_summary_and_exports(self, sample_classification_data):
        """Test summary and export functionality."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        # Test summary
        summary = report.summary()
        assert "Data Quality Report" in summary
        assert "Overall Quality Score" in summary

        # Test JSON export
        json_str = report.to_json()
        import json
        parsed = json.loads(json_str)
        assert "quality_score" in parsed

        # Test HTML export
        html = report.to_html()
        assert "<html>" in html
        assert "Quality Scores" in html

    def test_relabeling_workflow(self, sample_classification_data):
        """Test relabeling workflow."""
        df, labels, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        cleaner.analyze(show_progress=False)

        # Get relabeling suggestions
        suggestions = cleaner.relabel(apply_suggestions=False)

        assert isinstance(suggestions, pd.DataFrame)
        if len(suggestions) > 0:
            assert "current_label" in suggestions.columns
            assert "suggested_label" in suggestions.columns
            assert "confidence" in suggestions.columns

    def test_cleaning_workflow(self, sample_data_with_duplicates):
        """Test data cleaning workflow."""
        cleaner = DatasetCleaner(
            data=sample_data_with_duplicates,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        original_len = len(sample_data_with_duplicates)

        # Clean with duplicate removal
        clean_df = cleaner.get_clean_data(remove_duplicates=True)

        # Should have fewer rows
        assert len(clean_df) < original_len

    def test_multiple_analyses(self, sample_classification_data):
        """Test running multiple analyses."""
        df, _, _ = sample_classification_data

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )

        # First analysis
        report1 = cleaner.analyze(show_progress=False)

        # Second analysis
        report2 = cleaner.analyze(show_progress=False)

        # Should produce consistent results
        assert report1.quality_score.overall == report2.quality_score.overall


class TestEdgeCases:
    """Edge case integration tests."""

    def test_empty_issues(self):
        """Test with data that has no issues."""
        np.random.seed(42)

        # Create clean, well-separated data
        X0 = np.random.randn(50, 5) - 3
        X1 = np.random.randn(50, 5) + 3
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["label"] = y

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )
        report = cleaner.analyze(
            detect_duplicates=False,  # No duplicates by construction
            show_progress=False,
        )

        # Should have high quality score
        assert report.quality_score.overall > 70

    def test_small_dataset(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5],
            "f2": [5, 4, 3, 2, 1],
            "label": [0, 0, 1, 1, 0],
        })

        cleaner = DatasetCleaner(
            data=df,
            label_column="label",
        )

        # Should not fail on small data
        report = cleaner.analyze(show_progress=False)
        assert report is not None

    def test_many_features(self):
        """Test with many features."""
        np.random.seed(42)

        X = np.random.randn(100, 50)  # 50 features
        y = np.random.choice([0, 1], 100)

        cleaner = DatasetCleaner(
            data=X,
            labels=y,
        )

        report = cleaner.analyze(show_progress=False)
        assert report.dataset_info.n_features == 50

    def test_multiclass_classification(self):
        """Test with multiclass classification."""
        np.random.seed(42)

        X = np.random.randn(150, 5)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)

        cleaner = DatasetCleaner(
            data=X,
            labels=y,
        )

        report = cleaner.analyze(show_progress=False)
        assert report.dataset_info.n_classes == 3
