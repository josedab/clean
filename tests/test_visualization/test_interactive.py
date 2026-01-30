"""Tests for visualization interactive module (plotly-based)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestPlotlyAvailability:
    """Test plotly availability checking."""

    def test_check_plotly_raises_when_not_available(self):
        """Test _check_plotly raises ImportError when plotly not available."""
        import clean.visualization.interactive as interactive_module

        original_has_plotly = interactive_module.HAS_PLOTLY
        interactive_module.HAS_PLOTLY = False

        try:
            with pytest.raises(ImportError) as exc_info:
                interactive_module._check_plotly()
            assert "plotly" in str(exc_info.value).lower()
        finally:
            interactive_module.HAS_PLOTLY = original_has_plotly


class MockFigure:
    """Mock plotly figure for testing."""

    def __init__(self):
        self.traces = []
        self.layout = {}
        self.annotations = []

    def add_trace(self, trace, row=None, col=None):
        self.traces.append({"trace": trace, "row": row, "col": col})

    def add_annotation(self, **kwargs):
        self.annotations.append(kwargs)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)

    def update_traces(self, **kwargs):
        pass

    def write_html(self, path):
        pass


class MockQualityScore:
    """Mock QualityScore for testing."""

    def __init__(self):
        self.overall = 85.0
        self.label_quality = 90.0
        self.duplicate_quality = 80.0
        self.outlier_quality = 75.0
        self.imbalance_quality = 95.0
        self.bias_quality = 88.0


class MockDetectorResult:
    """Mock detector result."""

    def __init__(self, n_issues: int):
        self.n_issues = n_issues


class MockQualityReport:
    """Mock QualityReport for testing."""

    def __init__(
        self,
        label_errors_df=None,
        outliers_df=None,
        duplicates_df=None,
    ):
        self._label_errors = label_errors_df if label_errors_df is not None else pd.DataFrame()
        self._outliers = outliers_df if outliers_df is not None else pd.DataFrame()
        self._duplicates = duplicates_df if duplicates_df is not None else pd.DataFrame()

        self.quality_score = MockQualityScore()
        self.label_errors_result = MockDetectorResult(len(self._label_errors))
        self.outliers_result = MockDetectorResult(len(self._outliers))
        self.duplicates_result = MockDetectorResult(len(self._duplicates))
        self.bias_result = MockDetectorResult(0)

    def label_errors(self):
        return self._label_errors

    def outliers(self):
        return self._outliers

    def duplicates(self):
        return self._duplicates


class TestPlotQualityScores:
    """Test plot_quality_scores_interactive function."""

    def test_creates_figure_with_six_indicators(self):
        """Test that quality scores creates 6 gauge indicators."""
        mock_go = MagicMock()
        mock_make_subplots = MagicMock(return_value=MockFigure())
        mock_indicator = MagicMock()
        mock_go.Indicator = mock_indicator

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go), \
             patch("clean.visualization.interactive.make_subplots", mock_make_subplots):

            from clean.visualization.interactive import plot_quality_scores_interactive

            report = MockQualityReport()
            fig = plot_quality_scores_interactive(report)

            # Should create subplots with 2x3 layout
            mock_make_subplots.assert_called_once()
            assert mock_indicator.call_count == 6  # 6 score indicators


class TestPlotLabelErrors:
    """Test plot_label_errors_interactive function."""

    def test_empty_label_errors(self):
        """Test handling empty label errors."""
        mock_go = MagicMock()
        mock_fig = MockFigure()
        mock_go.Figure.return_value = mock_fig

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go):

            from clean.visualization.interactive import plot_label_errors_interactive

            report = MockQualityReport(label_errors_df=pd.DataFrame())
            fig = plot_label_errors_interactive(report)

            # Should add annotation about no errors
            assert len(mock_fig.annotations) > 0

    def test_with_label_errors(self):
        """Test with actual label errors."""
        mock_go = MagicMock()
        mock_fig = MockFigure()
        mock_go.Figure.return_value = mock_fig
        mock_go.Heatmap.return_value = MagicMock()

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go):

            from clean.visualization.interactive import plot_label_errors_interactive

            errors_df = pd.DataFrame({
                "index": [0, 1, 2],
                "given_label": ["a", "b", "a"],
                "predicted_label": ["b", "a", "c"],
                "confidence": [0.9, 0.8, 0.7],
            })

            report = MockQualityReport(label_errors_df=errors_df)
            fig = plot_label_errors_interactive(report)

            # Should create heatmap
            mock_go.Heatmap.assert_called_once()


class TestPlotOutliers:
    """Test plot_outliers_interactive function."""

    def test_with_outliers(self):
        """Test plotting outliers."""
        mock_px = MagicMock()
        mock_go = MagicMock()
        mock_fig = MockFigure()
        mock_px.scatter.return_value = mock_fig

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.px", mock_px), \
             patch("clean.visualization.interactive.go", mock_go):

            from clean.visualization.interactive import plot_outliers_interactive

            outliers_df = pd.DataFrame({
                "index": [5, 10],
                "score": [0.9, 0.85],
                "method": ["isolation_forest", "isolation_forest"],
            })

            features = pd.DataFrame({
                "feature1": np.random.randn(20),
                "feature2": np.random.randn(20),
            })

            report = MockQualityReport(outliers_df=outliers_df)
            fig = plot_outliers_interactive(report, features)

            mock_px.scatter.assert_called_once()

    def test_insufficient_features(self):
        """Test with less than 2 numeric features."""
        mock_go = MagicMock()
        mock_fig = MockFigure()
        mock_go.Figure.return_value = mock_fig

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go):

            from clean.visualization.interactive import plot_outliers_interactive

            features = pd.DataFrame({
                "text_only": ["a", "b", "c"],
            })

            report = MockQualityReport()
            fig = plot_outliers_interactive(report, features)

            # Should add annotation about needing features
            assert len(mock_fig.annotations) > 0


class TestPlotDuplicates:
    """Test plot_duplicates_interactive function."""

    def test_empty_duplicates(self):
        """Test handling empty duplicates."""
        mock_go = MagicMock()
        mock_fig = MockFigure()
        mock_go.Figure.return_value = mock_fig

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go):

            from clean.visualization.interactive import plot_duplicates_interactive

            report = MockQualityReport(duplicates_df=pd.DataFrame())
            fig = plot_duplicates_interactive(report)

            assert len(mock_fig.annotations) > 0

    def test_with_duplicates(self):
        """Test with actual duplicates."""
        mock_px = MagicMock()
        mock_fig = MockFigure()
        mock_px.histogram.return_value = mock_fig

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.px", mock_px):

            from clean.visualization.interactive import plot_duplicates_interactive

            dups_df = pd.DataFrame({
                "index1": [0, 2],
                "index2": [1, 3],
                "similarity": [0.95, 0.88],
                "is_exact": [True, False],
            })

            report = MockQualityReport(duplicates_df=dups_df)
            fig = plot_duplicates_interactive(report)

            mock_px.histogram.assert_called_once()


class TestPlotClassDistribution:
    """Test plot_class_distribution_interactive function."""

    def test_class_distribution(self):
        """Test class distribution plotting."""
        mock_px = MagicMock()
        mock_fig = MockFigure()
        mock_px.bar.return_value = mock_fig

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.px", mock_px):

            from clean.visualization.interactive import plot_class_distribution_interactive

            labels = np.array(["a", "b", "a", "c", "a", "b"])
            fig = plot_class_distribution_interactive(labels)

            mock_px.bar.assert_called_once()
            call_args = mock_px.bar.call_args
            # Should have class and count columns
            df = call_args[1].get("data_frame", call_args[0][0] if call_args[0] else None)
            assert df is not None or "x" in call_args[1]


class TestPlotReportDashboard:
    """Test plot_report_dashboard function."""

    def test_dashboard_creation(self):
        """Test dashboard creates multiple subplots."""
        mock_go = MagicMock()
        mock_make_subplots = MagicMock(return_value=MockFigure())

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go), \
             patch("clean.visualization.interactive.make_subplots", mock_make_subplots):

            from clean.visualization.interactive import plot_report_dashboard

            report = MockQualityReport()
            labels = np.array(["a", "b", "a"])

            fig = plot_report_dashboard(report, labels=labels)

            # Should create 2x2 subplots
            mock_make_subplots.assert_called_once()

    def test_dashboard_without_labels(self):
        """Test dashboard without labels."""
        mock_go = MagicMock()
        mock_make_subplots = MagicMock(return_value=MockFigure())

        with patch("clean.visualization.interactive.HAS_PLOTLY", True), \
             patch("clean.visualization.interactive.go", mock_go), \
             patch("clean.visualization.interactive.make_subplots", mock_make_subplots):

            from clean.visualization.interactive import plot_report_dashboard

            report = MockQualityReport()
            fig = plot_report_dashboard(report)

            mock_make_subplots.assert_called_once()


class TestSaveInteractiveHtml:
    """Test save_interactive_html function."""

    def test_save_html(self):
        """Test saving figure to HTML."""
        with patch("clean.visualization.interactive.HAS_PLOTLY", True):
            from clean.visualization.interactive import save_interactive_html

            mock_fig = MagicMock()
            save_interactive_html(mock_fig, "/tmp/test.html")

            mock_fig.write_html.assert_called_once_with("/tmp/test.html")
