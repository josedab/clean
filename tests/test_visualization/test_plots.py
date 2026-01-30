"""Tests for visualization module."""

import pytest

from clean import DatasetCleaner


class TestStaticPlots:
    """Tests for static matplotlib plots."""

    @pytest.fixture
    def report(self, sample_classification_data):
        """Create a report for testing."""
        df, _, _ = sample_classification_data
        cleaner = DatasetCleaner(data=df, label_column="label")
        return cleaner.analyze(show_progress=False)

    def test_plot_quality_scores(self, report):
        """Test quality scores plot."""
        from clean.visualization import plot_quality_scores

        fig = plot_quality_scores(report)
        assert fig is not None

        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_class_distribution(self, sample_classification_data):
        """Test class distribution plot."""
        from clean.visualization import plot_class_distribution

        _, labels, _ = sample_classification_data
        fig = plot_class_distribution(labels)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_report_summary(self, report):
        """Test report summary plot."""
        from clean.visualization import plot_report_summary

        fig = plot_report_summary(report)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_outlier_distribution(self, report, sample_classification_data):
        """Test outlier distribution plot."""
        from clean.visualization import plot_outlier_distribution

        df, _, _ = sample_classification_data
        features = df.drop(columns=["label"])

        fig = plot_outlier_distribution(report, features)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_label_error_confusion(self, report):
        """Test label error confusion matrix."""
        from clean.visualization import plot_label_error_confusion

        # May return None if no errors
        fig = plot_label_error_confusion(report)
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_plot_duplicate_similarity(self, sample_data_with_duplicates):
        """Test duplicate similarity histogram."""
        from clean.visualization import plot_duplicate_similarity

        cleaner = DatasetCleaner(
            data=sample_data_with_duplicates,
            label_column="label",
        )
        report = cleaner.analyze(show_progress=False)

        fig = plot_duplicate_similarity(report)
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


class TestInteractivePlots:
    """Tests for interactive plotly plots (if available)."""

    @pytest.fixture
    def report(self, sample_classification_data):
        """Create a report for testing."""
        df, _, _ = sample_classification_data
        cleaner = DatasetCleaner(data=df, label_column="label")
        return cleaner.analyze(show_progress=False)

    def test_interactive_import(self):
        """Test that interactive module can be imported."""
        try:
            from clean.visualization.interactive import HAS_PLOTLY
            # Just check import works, HAS_PLOTLY may be True or False
            assert isinstance(HAS_PLOTLY, bool)
        except ImportError:
            pytest.skip("Plotly not installed")
