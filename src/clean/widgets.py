"""Notebook-Native Widget Experience for Clean.

Rich interactive widgets for Jupyter with inline fix suggestions,
one-click corrections, and visual explorers.

Example:
    >>> from clean.widgets import QualityExplorer, FixWidget, show_report
    >>>
    >>> # Interactive quality explorer
    >>> explorer = QualityExplorer(report)
    >>> explorer.show()
    >>>
    >>> # One-click fix widget
    >>> fix_widget = FixWidget(report, data)
    >>> fix_widget.show()
    >>>
    >>> # Quick display
    >>> show_report(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from clean.core.report import QualityReport
from clean.exceptions import DependencyError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _check_ipywidgets() -> None:
    """Check if ipywidgets is available."""
    try:
        import ipywidgets  # noqa: F401
    except ImportError as e:
        raise DependencyError(
            "ipywidgets",
            "pip install clean-data-quality[interactive]",
            "Jupyter widgets",
        ) from e


class QualityExplorer:
    """Interactive widget for exploring quality reports.

    Provides:
    - Score overview with gauges
    - Issue breakdown by type
    - Sample browser for each issue
    - Filter and search capabilities
    """

    def __init__(
        self,
        report: QualityReport,
        data: pd.DataFrame | None = None,
        height: str = "600px",
    ):
        """Initialize quality explorer.

        Args:
            report: Quality report to explore
            data: Optional original data for sample viewing
            height: Widget height
        """
        _check_ipywidgets()

        import ipywidgets as widgets
        from IPython.display import HTML, display

        self.report = report
        self.data = data
        self.height = height
        self._widgets = widgets
        self._display = display
        self._HTML = HTML
        self._widget: Any = None

    def _create_score_gauge(self, score: float, label: str, max_val: float = 100) -> Any:
        """Create a score gauge widget."""
        widgets = self._widgets

        # Color based on score
        if score >= 80:
            color = "#28a745"  # Green
        elif score >= 60:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red

        bar_width = int(score / max_val * 100)

        html = f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold;">{label}</span>
                <span style="font-weight: bold; color: {color};">{score:.1f}/{max_val:.0f}</span>
            </div>
            <div style="background: #e9ecef; border-radius: 5px; height: 20px;">
                <div style="background: {color}; width: {bar_width}%; height: 100%;
                            border-radius: 5px; transition: width 0.3s;"></div>
            </div>
        </div>
        """
        return widgets.HTML(value=html)

    def _create_issue_cards(self) -> Any:
        """Create issue summary cards."""
        widgets = self._widgets

        cards = []
        issue_data = [
            ("🏷️ Label Errors", self.report.label_error_count, "#dc3545"),
            ("🔍 Duplicates", self.report.duplicate_count, "#17a2b8"),
            ("📊 Outliers", self.report.outlier_count, "#ffc107"),
        ]

        for label, count, color in issue_data:
            if count > 0:
                pct = count / self.report.n_samples * 100
                html = f"""
                <div style="background: white; border-radius: 8px; padding: 15px;
                            margin: 5px; min-width: 150px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #666;">{label}</div>
                    <div style="font-size: 28px; font-weight: bold; color: {color};">{count:,}</div>
                    <div style="font-size: 12px; color: #999;">{pct:.1f}% of samples</div>
                </div>
                """
                cards.append(widgets.HTML(value=html))

        return widgets.HBox(cards, layout=widgets.Layout(flex_wrap="wrap"))

    def _create_sample_browser(self) -> Any:
        """Create sample browser for viewing issues."""
        widgets = self._widgets

        # Issue type selector
        issue_types = ["Label Errors", "Duplicates", "Outliers"]
        type_dropdown = widgets.Dropdown(
            options=issue_types,
            value=issue_types[0],
            description="Issue Type:",
            style={"description_width": "initial"},
        )

        # Sample index slider
        sample_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max(self.report.label_error_count - 1, 0),
            description="Sample #:",
            style={"description_width": "initial"},
        )

        # Sample display area
        sample_output = widgets.Output()

        def update_display(change: Any = None) -> None:
            sample_output.clear_output()
            with sample_output:
                issue_type = type_dropdown.value
                idx = sample_slider.value

                if issue_type == "Label Errors" and self.report.label_error_count > 0:
                    errors = self.report.label_errors()
                    if idx < len(errors):
                        sample = errors.iloc[idx]
                        self._display(self._format_label_error(sample))
                elif issue_type == "Duplicates" and self.report.duplicate_count > 0:
                    dups = self.report.duplicates()
                    if idx < len(dups):
                        sample = dups.iloc[idx]
                        self._display(self._format_duplicate(sample))
                elif issue_type == "Outliers" and self.report.outlier_count > 0:
                    outliers = self.report.outliers()
                    if idx < len(outliers):
                        sample = outliers.iloc[idx]
                        self._display(self._format_outlier(sample))

        def update_slider_max(change: Any) -> None:
            issue_type = type_dropdown.value
            if issue_type == "Label Errors":
                sample_slider.max = max(self.report.label_error_count - 1, 0)
            elif issue_type == "Duplicates":
                sample_slider.max = max(self.report.duplicate_count - 1, 0)
            elif issue_type == "Outliers":
                sample_slider.max = max(self.report.outlier_count - 1, 0)
            sample_slider.value = 0
            update_display()

        type_dropdown.observe(update_slider_max, names="value")
        sample_slider.observe(update_display, names="value")

        # Initial display
        update_display()

        return widgets.VBox([
            widgets.HBox([type_dropdown, sample_slider]),
            sample_output,
        ])

    def _format_label_error(self, sample: pd.Series) -> Any:
        """Format a label error for display."""
        html = f"""
        <div style="background: #fff3cd; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #856404;">⚠️ Potential Label Error</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Index:</td>
                    <td style="padding: 8px;">{sample.get('index', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Given Label:</td>
                    <td style="padding: 8px; color: #dc3545;">{sample.get('given_label', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Suggested Label:</td>
                    <td style="padding: 8px; color: #28a745;">{sample.get('suggested_label', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Confidence:</td>
                    <td style="padding: 8px;">{sample.get('confidence', 0):.1%}</td>
                </tr>
            </table>
        </div>
        """
        return self._HTML(html)

    def _format_duplicate(self, sample: pd.Series) -> Any:
        """Format a duplicate pair for display."""
        html = f"""
        <div style="background: #cce5ff; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #004085;">🔍 Duplicate Pair</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Index 1:</td>
                    <td style="padding: 8px;">{sample.get('index1', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Index 2:</td>
                    <td style="padding: 8px;">{sample.get('index2', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Similarity:</td>
                    <td style="padding: 8px;">{sample.get('similarity', 0):.1%}</td>
                </tr>
            </table>
        </div>
        """
        return self._HTML(html)

    def _format_outlier(self, sample: pd.Series) -> Any:
        """Format an outlier for display."""
        html = f"""
        <div style="background: #fff3cd; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #856404;">📊 Outlier</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Index:</td>
                    <td style="padding: 8px;">{sample.get('index', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Score:</td>
                    <td style="padding: 8px;">{sample.get('score', 0):.3f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Method:</td>
                    <td style="padding: 8px;">{sample.get('method', 'N/A')}</td>
                </tr>
            </table>
        </div>
        """
        return self._HTML(html)

    def show(self) -> None:
        """Display the quality explorer widget."""
        widgets = self._widgets

        # Header
        header = widgets.HTML(value="""
        <h2 style="margin: 0; padding: 15px 0; border-bottom: 2px solid #eee;">
            🧹 Data Quality Explorer
        </h2>
        """)

        # Score section
        score_section = widgets.VBox([
            widgets.HTML(value="<h3>Quality Score</h3>"),
            self._create_score_gauge(self.report.quality_score, "Overall Score"),
        ])

        # Issues section
        issues_section = widgets.VBox([
            widgets.HTML(value="<h3>Issues Found</h3>"),
            self._create_issue_cards(),
        ])

        # Sample browser
        browser_section = widgets.VBox([
            widgets.HTML(value="<h3>Sample Browser</h3>"),
            self._create_sample_browser(),
        ])

        # Create tabs
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox([score_section, issues_section]),
            browser_section,
        ]
        tab.set_title(0, "Overview")
        tab.set_title(1, "Browse Issues")

        # Main container
        self._widget = widgets.VBox(
            [header, tab],
            layout=widgets.Layout(
                border="1px solid #ddd",
                border_radius="10px",
                padding="15px",
                background_color="#f8f9fa",
            ),
        )

        self._display(self._widget)


class FixWidget:
    """Interactive widget for applying fixes with one-click.

    Provides:
    - Fix suggestions with confidence scores
    - One-click apply/reject buttons
    - Batch operations
    - Undo functionality
    """

    def __init__(
        self,
        report: QualityReport,
        data: pd.DataFrame,
        label_column: str | None = None,
    ):
        """Initialize fix widget.

        Args:
            report: Quality report
            data: Original data
            label_column: Label column name
        """
        _check_ipywidgets()

        import ipywidgets as widgets
        from IPython.display import HTML, display

        self.report = report
        self.data = data.copy()
        self.original_data = data.copy()
        self.label_column = label_column
        self._widgets = widgets
        self._display = display
        self._HTML = HTML
        self._applied_fixes: list[dict[str, Any]] = []

    def _create_fix_card(
        self,
        idx: int,
        issue_type: str,
        description: str,
        confidence: float,
        on_apply: Callable,
        on_reject: Callable,
    ) -> Any:
        """Create a fix suggestion card."""
        widgets = self._widgets

        # Confidence color
        if confidence >= 0.9:
            conf_color = "#28a745"
        elif confidence >= 0.7:
            conf_color = "#ffc107"
        else:
            conf_color = "#dc3545"

        # Card HTML
        card_html = f"""
        <div style="background: white; border-radius: 8px; padding: 15px;
                    margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 12px; color: #666; text-transform: uppercase;">
                        {issue_type}
                    </span>
                    <div style="font-size: 14px; margin-top: 5px;">{description}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: bold; color: {conf_color};">
                        {confidence:.0%}
                    </div>
                    <div style="font-size: 10px; color: #999;">confidence</div>
                </div>
            </div>
        </div>
        """

        card = widgets.HTML(value=card_html)

        apply_btn = widgets.Button(
            description="✓ Apply",
            button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        apply_btn.on_click(lambda b: on_apply(idx))

        reject_btn = widgets.Button(
            description="✗ Reject",
            button_style="danger",
            layout=widgets.Layout(width="80px"),
        )
        reject_btn.on_click(lambda b: on_reject(idx))

        return widgets.HBox([
            card,
            widgets.VBox([apply_btn, reject_btn]),
        ])

    def _apply_label_fix(self, idx: int) -> None:
        """Apply a label fix."""
        errors = self.report.label_errors()
        if idx < len(errors):
            error = errors.iloc[idx]
            sample_idx = error.get("index")
            new_label = error.get("suggested_label")

            if sample_idx is not None and new_label is not None and self.label_column:
                self.data.loc[sample_idx, self.label_column] = new_label
                self._applied_fixes.append({
                    "type": "label",
                    "index": sample_idx,
                    "old_value": error.get("given_label"),
                    "new_value": new_label,
                })
                logger.info(f"Applied label fix at index {sample_idx}")

    def _remove_duplicate(self, idx: int) -> None:
        """Remove a duplicate."""
        dups = self.report.duplicates()
        if idx < len(dups):
            dup = dups.iloc[idx]
            idx_to_remove = dup.get("index2")  # Keep first, remove second

            if idx_to_remove is not None and idx_to_remove in self.data.index:
                self.data = self.data.drop(idx_to_remove)
                self._applied_fixes.append({
                    "type": "duplicate",
                    "index": idx_to_remove,
                    "action": "removed",
                })
                logger.info(f"Removed duplicate at index {idx_to_remove}")

    def _undo_last_fix(self) -> None:
        """Undo the last applied fix."""
        if self._applied_fixes:
            last_fix = self._applied_fixes.pop()
            # Restore from original data
            if last_fix["type"] == "label":
                idx = last_fix["index"]
                self.data.loc[idx, self.label_column] = last_fix["old_value"]
            elif last_fix["type"] == "duplicate":
                idx = last_fix["index"]
                self.data = pd.concat([self.data, self.original_data.loc[[idx]]])
            logger.info(f"Undid fix: {last_fix}")

    def get_fixed_data(self) -> pd.DataFrame:
        """Get the data with all fixes applied.

        Returns:
            Fixed DataFrame
        """
        return self.data.copy()

    def show(self) -> None:
        """Display the fix widget."""
        widgets = self._widgets

        # Header
        header = widgets.HTML(value="""
        <h2 style="margin: 0; padding: 15px 0; border-bottom: 2px solid #eee;">
            🔧 One-Click Fix Suggestions
        </h2>
        """)

        # Status bar
        self._status = widgets.HTML(value="")

        # Undo button
        undo_btn = widgets.Button(
            description="↩ Undo Last",
            button_style="warning",
            layout=widgets.Layout(width="120px"),
        )
        undo_btn.on_click(lambda b: self._undo_last_fix())

        # Export button
        export_btn = widgets.Button(
            description="📥 Export Fixed Data",
            button_style="info",
            layout=widgets.Layout(width="150px"),
        )

        export_output = widgets.Output()

        def export_data(b: Any) -> None:
            export_output.clear_output()
            with export_output:
                self._display(self._HTML(
                    f"<p style='color: #28a745;'>✓ Fixed data available via "
                    f"<code>widget.get_fixed_data()</code> ({len(self.data)} rows)</p>"
                ))

        export_btn.on_click(export_data)

        # Fix suggestions
        fix_cards = []

        # Label error fixes
        if self.report.label_error_count > 0:
            errors = self.report.label_errors()
            for i, (_, error) in enumerate(errors.head(10).iterrows()):
                card = self._create_fix_card(
                    idx=i,
                    issue_type="Label Error",
                    description=f"Change '{error.get('given_label')}' → '{error.get('suggested_label')}' at index {error.get('index')}",
                    confidence=error.get("confidence", 0.5),
                    on_apply=self._apply_label_fix,
                    on_reject=lambda x: None,  # Just dismiss
                )
                fix_cards.append(card)

        # Duplicate fixes
        if self.report.duplicate_count > 0:
            dups = self.report.duplicates()
            for i, (_, dup) in enumerate(dups.head(10).iterrows()):
                card = self._create_fix_card(
                    idx=i,
                    issue_type="Duplicate",
                    description=f"Remove duplicate at index {dup.get('index2')} (similar to {dup.get('index1')})",
                    confidence=dup.get("similarity", 0.9),
                    on_apply=self._remove_duplicate,
                    on_reject=lambda x: None,
                )
                fix_cards.append(card)

        if not fix_cards:
            fix_cards.append(widgets.HTML(
                value="<p style='color: #28a745; text-align: center; padding: 20px;'>"
                "✓ No high-confidence fixes suggested!</p>"
            ))

        # Main container
        main = widgets.VBox([
            header,
            widgets.HBox([undo_btn, export_btn]),
            self._status,
            widgets.VBox(fix_cards),
            export_output,
        ], layout=widgets.Layout(
            border="1px solid #ddd",
            border_radius="10px",
            padding="15px",
            background_color="#f8f9fa",
        ))

        self._display(main)


def show_report(
    report: QualityReport,
    data: pd.DataFrame | None = None,
    interactive: bool = True,
) -> None:
    """Quick display of a quality report.

    Args:
        report: Quality report to display
        data: Optional original data
        interactive: Use interactive widgets if available

    Example:
        >>> report = cleaner.analyze()
        >>> show_report(report)
    """
    try:
        if interactive:
            explorer = QualityExplorer(report, data)
            explorer.show()
    except DependencyError:
        # Fall back to text display
        from IPython.display import display, Markdown

        display(Markdown(f"```\n{report.summary()}\n```"))


__all__ = [
    "QualityExplorer",
    "FixWidget",
    "show_report",
]
