"""Interactive visualization plots using plotly."""

from typing import Any

import numpy as np
import pandas as pd

from clean.core.report import QualityReport

# Optional plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    px = None
    go = None


def _check_plotly() -> None:
    """Check if plotly is available."""
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly required for interactive visualizations. "
            "Install with: pip install clean-data-quality[interactive]"
        )


def plot_quality_scores_interactive(report: QualityReport) -> Any:
    """Create interactive quality score gauge chart.

    Args:
        report: QualityReport

    Returns:
        Plotly figure
    """
    _check_plotly()

    scores = report.quality_score

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        ],
        subplot_titles=["Overall", "Labels", "Duplicates", "Outliers", "Imbalance", "Bias"],
    )

    indicators = [
        (scores.overall, 1, 1),
        (scores.label_quality, 1, 2),
        (scores.duplicate_quality, 1, 3),
        (scores.outlier_quality, 2, 1),
        (scores.imbalance_quality, 2, 2),
        (scores.bias_quality, 2, 3),
    ]

    for value, row, col in indicators:
        color = (
            "#27ae60" if value >= 80 else
            "#f39c12" if value >= 60 else
            "#e74c3c"
        )
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#ffebee"},
                        {"range": [40, 60], "color": "#fff3e0"},
                        {"range": [60, 80], "color": "#fff8e1"},
                        {"range": [80, 100], "color": "#e8f5e9"},
                    ],
                },
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title="Data Quality Scores",
        height=500,
    )

    return fig


def plot_label_errors_interactive(
    report: QualityReport,
    features: pd.DataFrame | None = None,
) -> Any:
    """Create interactive label error visualization.

    Args:
        report: QualityReport
        features: Optional feature DataFrame for scatter

    Returns:
        Plotly figure
    """
    _check_plotly()

    errors_df = report.label_errors()
    if errors_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No label errors detected", showarrow=False)
        return fig

    # Confusion heatmap
    given = errors_df["given_label"].values
    predicted = errors_df["predicted_label"].values
    labels = sorted(set(given) | set(predicted))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for g, p in zip(given, predicted):
        matrix[label_to_idx[g], label_to_idx[p]] += 1

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        colorscale="Blues",
        text=matrix,
        texttemplate="%{text}",
        hovertemplate="Given: %{y}<br>Suggested: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Label Error Confusion Matrix",
        xaxis_title="Suggested Label",
        yaxis_title="Given Label",
        height=500,
    )

    return fig


def plot_outliers_interactive(
    report: QualityReport,
    features: pd.DataFrame,
    feature_x: str | None = None,
    feature_y: str | None = None,
    feature_color: str | None = None,
) -> Any:
    """Create interactive outlier scatter plot.

    Args:
        report: QualityReport
        features: Feature DataFrame
        feature_x: X-axis feature
        feature_y: Y-axis feature
        feature_color: Feature for color encoding

    Returns:
        Plotly figure
    """
    _check_plotly()

    outliers_df = report.outliers()
    outlier_indices = set(outliers_df["index"].values) if not outliers_df.empty else set()

    # Auto-select features
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric features", showarrow=False)
        return fig

    if feature_x is None:
        feature_x = numeric_cols[0]
    if feature_y is None:
        feature_y = numeric_cols[1]

    # Create DataFrame for plotting
    plot_df = features[[feature_x, feature_y]].copy()
    plot_df["is_outlier"] = plot_df.index.isin(outlier_indices)
    plot_df["point_type"] = plot_df["is_outlier"].map({True: "Outlier", False: "Normal"})

    fig = px.scatter(
        plot_df,
        x=feature_x,
        y=feature_y,
        color="point_type",
        color_discrete_map={"Normal": "#3498db", "Outlier": "#e74c3c"},
        title="Outlier Detection Results",
        hover_data={"is_outlier": False, "point_type": True},
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=500)

    return fig


def plot_duplicates_interactive(report: QualityReport) -> Any:
    """Create interactive duplicate similarity histogram.

    Args:
        report: QualityReport

    Returns:
        Plotly figure
    """
    _check_plotly()

    duplicates_df = report.duplicates()
    if duplicates_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No duplicates detected", showarrow=False)
        return fig

    fig = px.histogram(
        duplicates_df,
        x="similarity",
        nbins=30,
        color="is_exact",
        color_discrete_map={True: "#9b59b6", False: "#3498db"},
        labels={"is_exact": "Exact Match"},
        title="Duplicate Similarity Distribution",
    )

    fig.update_layout(
        xaxis_title="Similarity Score",
        yaxis_title="Count",
        height=400,
    )

    return fig


def plot_class_distribution_interactive(
    labels: np.ndarray,
    title: str = "Class Distribution",
) -> Any:
    """Create interactive class distribution bar chart.

    Args:
        labels: Label array
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    unique, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"class": unique, "count": counts})
    df = df.sort_values("count", ascending=False)

    fig = px.bar(
        df,
        x="class",
        y="count",
        title=title,
        color="count",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        height=400,
    )

    return fig


def plot_report_dashboard(
    report: QualityReport,
    features: pd.DataFrame | None = None,
    labels: np.ndarray | None = None,
) -> Any:
    """Create comprehensive interactive dashboard.

    Args:
        report: QualityReport
        features: Optional feature DataFrame
        labels: Optional label array

    Returns:
        Plotly figure with multiple subplots
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Quality Scores",
            "Issues by Type",
            "Class Distribution" if labels is not None else "N/A",
            "Duplicate Similarity",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "histogram"}],
        ],
    )

    # Quality scores bar chart
    scores = report.quality_score
    score_names = ["Overall", "Labels", "Duplicates", "Outliers", "Imbalance", "Bias"]
    score_values = [
        scores.overall, scores.label_quality, scores.duplicate_quality,
        scores.outlier_quality, scores.imbalance_quality, scores.bias_quality,
    ]
    colors = [
        "#27ae60" if v >= 80 else "#f39c12" if v >= 60 else "#e74c3c"
        for v in score_values
    ]

    fig.add_trace(
        go.Bar(x=score_names, y=score_values, marker_color=colors, name="Scores"),
        row=1, col=1,
    )

    # Issues by type
    issue_types = ["Label Errors", "Duplicates", "Outliers", "Bias"]
    issue_counts = [
        report.label_errors_result.n_issues if report.label_errors_result else 0,
        report.duplicates_result.n_issues if report.duplicates_result else 0,
        report.outliers_result.n_issues if report.outliers_result else 0,
        report.bias_result.n_issues if report.bias_result else 0,
    ]

    fig.add_trace(
        go.Bar(
            x=issue_types, y=issue_counts,
            marker_color=["#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"],
            name="Issues",
        ),
        row=1, col=2,
    )

    # Class distribution
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        fig.add_trace(
            go.Bar(x=[str(u) for u in unique], y=counts, marker_color="#3498db", name="Classes"),
            row=2, col=1,
        )

    # Duplicate similarity
    dups_df = report.duplicates()
    if not dups_df.empty:
        fig.add_trace(
            go.Histogram(x=dups_df["similarity"], nbinsx=20, marker_color="#9b59b6", name="Similarity"),
            row=2, col=2,
        )

    fig.update_layout(
        title="Data Quality Dashboard",
        height=700,
        showlegend=False,
    )

    return fig


def save_interactive_html(fig: Any, path: str) -> None:
    """Save interactive plot to HTML file.

    Args:
        fig: Plotly figure
        path: Output file path
    """
    _check_plotly()
    fig.write_html(path)
