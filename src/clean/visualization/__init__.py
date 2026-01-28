"""Visualization module for Clean."""

from clean.visualization.plots import (
    plot_class_distribution,
    plot_duplicate_similarity,
    plot_label_error_confusion,
    plot_outlier_distribution,
    plot_quality_scores,
    plot_report_summary,
    save_plots,
)

__all__ = [
    "plot_class_distribution",
    "plot_duplicate_similarity",
    "plot_label_error_confusion",
    "plot_outlier_distribution",
    "plot_quality_scores",
    "plot_report_summary",
    "save_plots",
]

# Optional interactive imports
try:
    from clean.visualization.interactive import (
        plot_class_distribution_interactive,
        plot_duplicates_interactive,
        plot_label_errors_interactive,
        plot_outliers_interactive,
        plot_quality_scores_interactive,
        plot_report_dashboard,
        save_interactive_html,
    )

    __all__.extend([
        "plot_class_distribution_interactive",
        "plot_duplicates_interactive",
        "plot_label_errors_interactive",
        "plot_outliers_interactive",
        "plot_quality_scores_interactive",
        "plot_report_dashboard",
        "save_interactive_html",
    ])
except ImportError:
    pass

# Optional widget imports
try:
    from clean.visualization.browser import (
        IssueBrowser,
        ReviewQueue,
        browse_issues,
        review_issues,
    )

    __all__.extend([
        "IssueBrowser",
        "ReviewQueue",
        "browse_issues",
        "review_issues",
    ])
except ImportError:
    pass
