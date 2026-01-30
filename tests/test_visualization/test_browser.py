"""Tests for visualization browser module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# We need to mock ipywidgets before importing the module
mock_widgets = MagicMock()
mock_widgets.Dropdown = MagicMock
mock_widgets.Button = MagicMock
mock_widgets.Label = MagicMock
mock_widgets.Output = MagicMock
mock_widgets.HBox = MagicMock
mock_widgets.VBox = MagicMock
mock_widgets.IntProgress = MagicMock


class TestCheckWidgetsAvailability:
    """Test widget availability checking."""

    def test_check_widgets_raises_when_not_available(self):
        """Test that _check_widgets raises ImportError when widgets not available."""
        with patch.dict("sys.modules", {"ipywidgets": None}):
            # Import fresh with widgets unavailable
            import importlib
            import clean.visualization.browser as browser_module

            # Patch HAS_WIDGETS to False
            original_has_widgets = browser_module.HAS_WIDGETS
            browser_module.HAS_WIDGETS = False

            try:
                with pytest.raises(ImportError) as exc_info:
                    browser_module._check_widgets()
                assert "ipywidgets" in str(exc_info.value)
            finally:
                browser_module.HAS_WIDGETS = original_has_widgets


class TestIssueBrowserLogic:
    """Test IssueBrowser logic without actual widgets."""

    @pytest.fixture
    def mock_report(self):
        """Create a mock QualityReport."""
        report = MagicMock()

        # Mock label errors
        report.label_errors.return_value = pd.DataFrame({
            "index": [0, 5, 10],
            "given_label": ["a", "b", "c"],
            "predicted_label": ["b", "c", "a"],
            "confidence": [0.9, 0.8, 0.7],
        })

        # Mock duplicates
        report.duplicates.return_value = pd.DataFrame({
            "index1": [1, 3],
            "index2": [2, 4],
            "similarity": [0.95, 0.92],
        })

        # Mock outliers
        report.outliers.return_value = pd.DataFrame({
            "index": [7, 8],
            "score": [0.85, 0.75],
            "method": ["isolation_forest", "isolation_forest"],
        })

        # Mock bias
        report.bias_issues.return_value = pd.DataFrame()

        return report

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
            "label": ["a"] * 10 + ["b"] * 10,
        })

    def test_get_current_issues_label_errors(self, mock_report):
        """Test getting label errors."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.report = mock_report
            browser.current_issue_type = "label_errors"

            issues = browser._get_current_issues()

            assert len(issues) == 3
            assert "confidence" in issues.columns

    def test_get_current_issues_duplicates(self, mock_report):
        """Test getting duplicates."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.report = mock_report
            browser.current_issue_type = "duplicates"

            issues = browser._get_current_issues()

            assert len(issues) == 2
            assert "similarity" in issues.columns

    def test_get_current_issues_outliers(self, mock_report):
        """Test getting outliers."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.report = mock_report
            browser.current_issue_type = "outliers"

            issues = browser._get_current_issues()

            assert len(issues) == 2
            assert "score" in issues.columns

    def test_sort_issues_by_confidence_desc(self, mock_report):
        """Test sorting by confidence descending."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            # Create mock dropdown
            sort_dropdown = MagicMock()
            sort_dropdown.value = "confidence_desc"

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.report = mock_report
            browser.sort_dropdown = sort_dropdown

            issues = mock_report.label_errors()
            sorted_issues = browser._sort_issues(issues)

            # First should have highest confidence
            assert sorted_issues.iloc[0]["confidence"] == 0.9

    def test_sort_issues_by_confidence_asc(self, mock_report):
        """Test sorting by confidence ascending."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            sort_dropdown = MagicMock()
            sort_dropdown.value = "confidence_asc"

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.report = mock_report
            browser.sort_dropdown = sort_dropdown

            issues = mock_report.label_errors()
            sorted_issues = browser._sort_issues(issues)

            # First should have lowest confidence
            assert sorted_issues.iloc[0]["confidence"] == 0.7

    def test_sort_issues_empty_dataframe(self, mock_report):
        """Test sorting empty DataFrame."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            sort_dropdown = MagicMock()
            sort_dropdown.value = "confidence_desc"

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.sort_dropdown = sort_dropdown

            empty_df = pd.DataFrame()
            result = browser._sort_issues(empty_df)

            assert result.empty


class TestReviewQueueLogic:
    """Test ReviewQueue logic without actual widgets."""

    @pytest.fixture
    def mock_report(self):
        """Create a mock QualityReport."""
        report = MagicMock()

        report.label_errors.return_value = pd.DataFrame({
            "index": [0, 1],
            "given_label": ["a", "b"],
            "predicted_label": ["b", "a"],
            "confidence": [0.9, 0.8],
        })

        report.outliers.return_value = pd.DataFrame({
            "index": [5],
            "score": [0.7],
            "method": ["lof"],
        })

        return report

    def test_collect_review_items(self, mock_report):
        """Test collecting review items."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import ReviewQueue

            queue = ReviewQueue.__new__(ReviewQueue)
            queue.report = mock_report

            items = queue._collect_review_items()

            assert len(items) == 3  # 2 label errors + 1 outlier
            assert all("index" in item for item in items)
            assert all("type" in item for item in items)
            assert all("priority" in item for item in items)

    def test_review_items_sorted_by_priority(self, mock_report):
        """Test that items are sorted by priority."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import ReviewQueue

            queue = ReviewQueue.__new__(ReviewQueue)
            queue.report = mock_report

            items = queue._collect_review_items()

            # Check sorted descending by priority
            priorities = [item["priority"] for item in items]
            assert priorities == sorted(priorities, reverse=True)

    def test_get_decisions(self, mock_report):
        """Test getting decisions."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import ReviewQueue

            queue = ReviewQueue.__new__(ReviewQueue)
            queue.decisions = {
                (0, "label_error"): "keep",
                (1, "label_error"): "remove",
                (5, "outlier"): "skip",
            }

            decisions = queue.get_decisions()

            assert len(decisions) == 3
            assert decisions[(0, "label_error")] == "keep"
            assert decisions[(1, "label_error")] == "remove"

    def test_get_indices_to_remove(self, mock_report):
        """Test getting indices marked for removal."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import ReviewQueue

            queue = ReviewQueue.__new__(ReviewQueue)
            queue.decisions = {
                (0, "label_error"): "keep",
                (1, "label_error"): "remove",
                (5, "outlier"): "remove",
                (7, "outlier"): "skip",
            }

            to_remove = queue.get_indices_to_remove()

            assert sorted(to_remove) == [1, 5]


class TestDecisionCallback:
    """Test decision callback functionality."""

    def test_decision_callback_called(self):
        """Test that on_decision callback is called."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import ReviewQueue

            callback = MagicMock()

            queue = ReviewQueue.__new__(ReviewQueue)
            queue.decisions = {}
            queue.current_idx = 0
            queue.review_items = [
                {"index": 0, "type": "label_error", "priority": 0.9, "details": "test"}
            ]
            queue.on_decision = callback
            queue.progress = MagicMock()
            queue.item_output = MagicMock()

            # Mock _display_current to avoid widget operations
            queue._display_current = MagicMock()

            queue._decide("keep")

            callback.assert_called_once_with(0, "label_error", "keep")

    def test_decision_recorded(self):
        """Test that decision is recorded."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import ReviewQueue

            queue = ReviewQueue.__new__(ReviewQueue)
            queue.decisions = {}
            queue.current_idx = 0
            queue.review_items = [
                {"index": 5, "type": "outlier", "priority": 0.5, "details": "test"}
            ]
            queue.on_decision = None
            queue.progress = MagicMock()
            queue._display_current = MagicMock()

            queue._decide("remove")

            assert queue.decisions[(5, "outlier")] == "remove"
            assert queue.current_idx == 1


class TestPagination:
    """Test pagination logic."""

    def test_page_navigation(self):
        """Test page navigation."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.current_page = 0
            browser._display_page = MagicMock()

            # Test next
            browser._on_next(None)
            assert browser.current_page == 1

            # Test prev
            browser._on_prev(None)
            assert browser.current_page == 0

            # Test prev at 0 (should stay at 0)
            browser._on_prev(None)
            assert browser.current_page == 0

    def test_issue_type_change_resets_page(self):
        """Test that changing issue type resets to page 0."""
        with patch("clean.visualization.browser.HAS_WIDGETS", True), \
             patch("clean.visualization.browser.widgets", mock_widgets):

            from clean.visualization.browser import IssueBrowser

            browser = IssueBrowser.__new__(IssueBrowser)
            browser.current_page = 5
            browser.current_issue_type = "label_errors"
            browser._display_page = MagicMock()

            browser._on_issue_type_change({"new": "duplicates"})

            assert browser.current_issue_type == "duplicates"
            assert browser.current_page == 0
