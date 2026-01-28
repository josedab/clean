"""Issue browser widget for Jupyter notebooks."""

from typing import Any, Callable

import pandas as pd

from clean.core.report import QualityReport

# Optional ipywidgets import
try:
    import ipywidgets as widgets
    from IPython.display import HTML, display

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    widgets = None


def _check_widgets() -> None:
    """Check if ipywidgets is available."""
    if not HAS_WIDGETS:
        raise ImportError(
            "ipywidgets required for issue browser. "
            "Install with: pip install clean-data-quality[interactive]"
        )


class IssueBrowser:
    """Interactive widget for browsing data quality issues."""

    def __init__(
        self,
        report: QualityReport,
        data: pd.DataFrame | None = None,
        page_size: int = 10,
    ):
        """Initialize the issue browser.

        Args:
            report: QualityReport with analysis results
            data: Original data for viewing samples
            page_size: Number of issues per page
        """
        _check_widgets()

        self.report = report
        self.data = data
        self.page_size = page_size

        self.current_issue_type = "label_errors"
        self.current_page = 0

        self._build_widgets()

    def _build_widgets(self) -> None:
        """Build the widget interface."""
        # Issue type selector
        self.issue_type_dropdown = widgets.Dropdown(
            options=[
                ("Label Errors", "label_errors"),
                ("Duplicates", "duplicates"),
                ("Outliers", "outliers"),
                ("Bias Issues", "bias"),
            ],
            value="label_errors",
            description="Issue Type:",
        )
        self.issue_type_dropdown.observe(self._on_issue_type_change, names="value")

        # Page navigation
        self.prev_button = widgets.Button(description="← Previous")
        self.next_button = widgets.Button(description="Next →")
        self.page_label = widgets.Label(value="Page 1")

        self.prev_button.on_click(self._on_prev)
        self.next_button.on_click(self._on_next)

        # Sort dropdown
        self.sort_dropdown = widgets.Dropdown(
            options=[
                ("Confidence (High to Low)", "confidence_desc"),
                ("Confidence (Low to High)", "confidence_asc"),
                ("Index", "index"),
            ],
            value="confidence_desc",
            description="Sort by:",
        )
        self.sort_dropdown.observe(self._on_sort_change, names="value")

        # Output area for issues
        self.output = widgets.Output()

        # Layout
        nav_box = widgets.HBox([self.prev_button, self.page_label, self.next_button])
        controls = widgets.HBox([self.issue_type_dropdown, self.sort_dropdown])

        self.container = widgets.VBox([
            controls,
            nav_box,
            self.output,
        ])

    def _get_current_issues(self) -> pd.DataFrame:
        """Get issues for current type."""
        if self.current_issue_type == "label_errors":
            return self.report.label_errors()
        elif self.current_issue_type == "duplicates":
            return self.report.duplicates()
        elif self.current_issue_type == "outliers":
            return self.report.outliers()
        elif self.current_issue_type == "bias":
            return self.report.bias_issues()
        return pd.DataFrame()

    def _sort_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort issues based on current sort setting."""
        if df.empty:
            return df

        sort_val = self.sort_dropdown.value

        if sort_val == "confidence_desc" and "confidence" in df.columns:
            return df.sort_values("confidence", ascending=False)
        elif sort_val == "confidence_asc" and "confidence" in df.columns:
            return df.sort_values("confidence", ascending=True)
        elif sort_val == "index" and "index" in df.columns:
            return df.sort_values("index")
        elif "similarity" in df.columns:
            return df.sort_values("similarity", ascending=sort_val.endswith("asc"))

        return df

    def _display_page(self) -> None:
        """Display current page of issues."""
        self.output.clear_output()

        issues = self._get_current_issues()
        issues = self._sort_issues(issues)

        total = len(issues)
        total_pages = max(1, (total + self.page_size - 1) // self.page_size)

        self.current_page = min(self.current_page, total_pages - 1)
        self.current_page = max(0, self.current_page)

        start = self.current_page * self.page_size
        end = start + self.page_size

        page_issues = issues.iloc[start:end]

        self.page_label.value = f"Page {self.current_page + 1} of {total_pages} ({total} issues)"

        with self.output:
            if page_issues.empty:
                display(HTML("<p><i>No issues of this type</i></p>"))
            else:
                display(page_issues)

                # Show sample data if available
                if self.data is not None and "index" in page_issues.columns:
                    indices = page_issues["index"].values[:3]
                    valid_indices = [i for i in indices if i < len(self.data)]
                    if valid_indices:
                        display(HTML("<h4>Sample Data:</h4>"))
                        display(self.data.iloc[valid_indices])

    def _on_issue_type_change(self, change: dict) -> None:
        """Handle issue type change."""
        self.current_issue_type = change["new"]
        self.current_page = 0
        self._display_page()

    def _on_sort_change(self, change: dict) -> None:
        """Handle sort change."""
        self._display_page()

    def _on_prev(self, button: Any) -> None:
        """Handle previous button."""
        self.current_page = max(0, self.current_page - 1)
        self._display_page()

    def _on_next(self, button: Any) -> None:
        """Handle next button."""
        self.current_page += 1
        self._display_page()

    def display(self) -> None:
        """Display the browser widget."""
        self._display_page()
        display(self.container)


class ReviewQueue:
    """Interactive widget for reviewing and marking issues."""

    def __init__(
        self,
        report: QualityReport,
        data: pd.DataFrame | None = None,
        on_decision: Callable[[int, str, str], None] | None = None,
    ):
        """Initialize the review queue.

        Args:
            report: QualityReport
            data: Original data
            on_decision: Callback(index, issue_type, decision) when decision made
        """
        _check_widgets()

        self.report = report
        self.data = data
        self.on_decision = on_decision

        self.decisions: dict[tuple[int, str], str] = {}
        self.current_idx = 0

        # Get all issues as review items
        self.review_items = self._collect_review_items()

        self._build_widgets()

    def _collect_review_items(self) -> list[dict]:
        """Collect all issues for review."""
        items = []

        # Label errors
        for _, row in self.report.label_errors().iterrows():
            items.append({
                "index": row["index"],
                "type": "label_error",
                "priority": row.get("confidence", 0.5),
                "details": f"Given: {row['given_label']} → Suggested: {row['predicted_label']}",
            })

        # Outliers
        for _, row in self.report.outliers().iterrows():
            items.append({
                "index": row["index"],
                "type": "outlier",
                "priority": row.get("score", 0.5) * 0.5,
                "details": f"Method: {row.get('method', 'unknown')}, Score: {row.get('score', 0):.2f}",
            })

        # Sort by priority
        items.sort(key=lambda x: x["priority"], reverse=True)

        return items

    def _build_widgets(self) -> None:
        """Build review widgets."""
        # Progress
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=max(1, len(self.review_items)),
            description="Progress:",
        )

        # Current item display
        self.item_output = widgets.Output()

        # Decision buttons
        self.keep_button = widgets.Button(
            description="Keep",
            button_style="success",
            icon="check",
        )
        self.remove_button = widgets.Button(
            description="Remove",
            button_style="danger",
            icon="trash",
        )
        self.skip_button = widgets.Button(
            description="Skip",
            button_style="warning",
            icon="forward",
        )

        self.keep_button.on_click(lambda b: self._decide("keep"))
        self.remove_button.on_click(lambda b: self._decide("remove"))
        self.skip_button.on_click(lambda b: self._decide("skip"))

        # Layout
        buttons = widgets.HBox([self.keep_button, self.remove_button, self.skip_button])

        self.container = widgets.VBox([
            self.progress,
            self.item_output,
            buttons,
        ])

    def _display_current(self) -> None:
        """Display current review item."""
        self.item_output.clear_output()

        with self.item_output:
            if self.current_idx >= len(self.review_items):
                display(HTML("<h3>Review Complete!</h3>"))
                display(HTML(f"<p>Reviewed {len(self.decisions)} items</p>"))

                # Summary
                keeps = sum(1 for d in self.decisions.values() if d == "keep")
                removes = sum(1 for d in self.decisions.values() if d == "remove")
                skips = sum(1 for d in self.decisions.values() if d == "skip")
                display(HTML(f"<p>Keep: {keeps}, Remove: {removes}, Skip: {skips}</p>"))
                return

            item = self.review_items[self.current_idx]

            display(HTML(f"<h4>Item {self.current_idx + 1} of {len(self.review_items)}</h4>"))
            display(HTML(f"<p><b>Type:</b> {item['type']}</p>"))
            display(HTML(f"<p><b>Index:</b> {item['index']}</p>"))
            display(HTML(f"<p><b>Priority:</b> {item['priority']:.2f}</p>"))
            display(HTML(f"<p><b>Details:</b> {item['details']}</p>"))

            # Show data if available
            if self.data is not None and item["index"] < len(self.data):
                display(HTML("<h5>Sample Data:</h5>"))
                display(self.data.iloc[[item["index"]]])

    def _decide(self, decision: str) -> None:
        """Record decision and move to next."""
        if self.current_idx < len(self.review_items):
            item = self.review_items[self.current_idx]
            self.decisions[(item["index"], item["type"])] = decision

            if self.on_decision:
                self.on_decision(item["index"], item["type"], decision)

            self.current_idx += 1
            self.progress.value = self.current_idx

        self._display_current()

    def display(self) -> None:
        """Display the review queue."""
        self._display_current()
        display(self.container)

    def get_decisions(self) -> dict[tuple[int, str], str]:
        """Get all decisions made."""
        return self.decisions.copy()

    def get_indices_to_remove(self) -> list[int]:
        """Get indices marked for removal."""
        return [idx for (idx, _), dec in self.decisions.items() if dec == "remove"]


def browse_issues(
    report: QualityReport,
    data: pd.DataFrame | None = None,
    page_size: int = 10,
) -> IssueBrowser:
    """Create and display an issue browser.

    Args:
        report: QualityReport
        data: Original data
        page_size: Items per page

    Returns:
        IssueBrowser instance
    """
    browser = IssueBrowser(report, data, page_size)
    browser.display()
    return browser


def review_issues(
    report: QualityReport,
    data: pd.DataFrame | None = None,
) -> ReviewQueue:
    """Create and display a review queue.

    Args:
        report: QualityReport
        data: Original data

    Returns:
        ReviewQueue instance
    """
    queue = ReviewQueue(report, data)
    queue.display()
    return queue
