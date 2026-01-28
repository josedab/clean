"""Static visualization plots using matplotlib."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean.core.report import QualityReport


def plot_class_distribution(
    labels: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Class Distribution",
    figsize: tuple[int, int] = (10, 6),
    color: str = "#3498db",
) -> plt.Figure | None:
    """Plot class distribution bar chart.

    Args:
        labels: Label array
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title
        figsize: Figure size
        color: Bar color

    Returns:
        Figure if ax was None, else None
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]

    ax.bar(range(len(unique)), counts[sorted_idx], color=color, edgecolor="white")
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([str(unique[i]) for i in sorted_idx], rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Add count labels on bars
    for i, (idx, count) in enumerate(zip(sorted_idx, counts[sorted_idx])):
        ax.annotate(
            f"{count:,}",
            xy=(i, count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if created_fig:
        plt.tight_layout()
        return fig
    return None


def plot_label_error_confusion(
    report: QualityReport,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure | None:
    """Plot confusion matrix of label errors.

    Shows given labels vs suggested labels for detected errors.

    Args:
        report: QualityReport with label error results
        ax: Matplotlib axes
        figsize: Figure size
        cmap: Colormap

    Returns:
        Figure if ax was None
    """
    errors_df = report.label_errors()
    if errors_df.empty:
        return None

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Build confusion matrix
    given = errors_df["given_label"].values
    predicted = errors_df["predicted_label"].values
    labels = sorted(set(given) | set(predicted))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for g, p in zip(given, predicted):
        matrix[label_to_idx[g], label_to_idx[p]] += 1

    # Plot
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_xlabel("Suggested Label")
    ax.set_ylabel("Given Label")
    ax.set_title("Label Error Confusion Matrix")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )

    plt.colorbar(im, ax=ax, label="Count")

    if created_fig:
        plt.tight_layout()
        return fig
    return None


def plot_outlier_distribution(
    report: QualityReport,
    features: pd.DataFrame,
    feature_x: str | None = None,
    feature_y: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure | None:
    """Plot scatter plot highlighting outliers.

    Args:
        report: QualityReport with outlier results
        features: Feature DataFrame
        feature_x: X-axis feature (auto-selected if None)
        feature_y: Y-axis feature (auto-selected if None)
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Figure if ax was None
    """
    outliers_df = report.outliers()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Auto-select features
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        if created_fig:
            ax.text(0.5, 0.5, "Need at least 2 numeric features", ha="center", va="center")
            return fig
        return None

    if feature_x is None:
        feature_x = numeric_cols[0]
    if feature_y is None:
        feature_y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

    x = features[feature_x].values
    y = features[feature_y].values

    # Plot normal points
    outlier_indices = set(outliers_df["index"].values) if not outliers_df.empty else set()
    normal_mask = ~np.isin(np.arange(len(features)), list(outlier_indices))

    ax.scatter(
        x[normal_mask], y[normal_mask],
        c="#3498db", alpha=0.6, label="Normal", s=30,
    )

    # Plot outliers
    if len(outlier_indices) > 0:
        outlier_mask = np.isin(np.arange(len(features)), list(outlier_indices))
        ax.scatter(
            x[outlier_mask], y[outlier_mask],
            c="#e74c3c", alpha=0.8, label="Outlier", s=50, marker="x",
        )

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title("Outlier Distribution")
    ax.legend()

    if created_fig:
        plt.tight_layout()
        return fig
    return None


def plot_quality_scores(
    report: QualityReport,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure | None:
    """Plot quality score breakdown.

    Args:
        report: QualityReport
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Figure if ax was None
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    scores = report.quality_score
    categories = ["Label", "Duplicate", "Outlier", "Imbalance", "Bias", "Overall"]
    values = [
        scores.label_quality,
        scores.duplicate_quality,
        scores.outlier_quality,
        scores.imbalance_quality,
        scores.bias_quality,
        scores.overall,
    ]

    colors = []
    for v in values:
        if v >= 90:
            colors.append("#27ae60")
        elif v >= 75:
            colors.append("#2ecc71")
        elif v >= 60:
            colors.append("#f39c12")
        elif v >= 40:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")

    bars = ax.barh(categories, values, color=colors, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Quality Score")
    ax.set_title("Data Quality Score Breakdown")

    # Add score labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 2, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}",
            va="center", fontsize=10, fontweight="bold",
        )

    # Add threshold line
    ax.axvline(x=75, color="gray", linestyle="--", alpha=0.5, label="Good threshold")

    if created_fig:
        plt.tight_layout()
        return fig
    return None


def plot_duplicate_similarity(
    report: QualityReport,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    bins: int = 20,
) -> plt.Figure | None:
    """Plot histogram of duplicate similarity scores.

    Args:
        report: QualityReport
        ax: Matplotlib axes
        figsize: Figure size
        bins: Number of histogram bins

    Returns:
        Figure if ax was None
    """
    duplicates_df = report.duplicates()
    if duplicates_df.empty:
        return None

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    similarities = duplicates_df["similarity"].values

    ax.hist(
        similarities, bins=bins,
        color="#9b59b6", edgecolor="white", alpha=0.7,
    )
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title("Duplicate Similarity Distribution")

    # Add statistics
    ax.axvline(x=np.mean(similarities), color="red", linestyle="--", label=f"Mean: {np.mean(similarities):.2f}")
    ax.legend()

    if created_fig:
        plt.tight_layout()
        return fig
    return None


def plot_report_summary(
    report: QualityReport,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create comprehensive report visualization with multiple plots.

    Args:
        report: QualityReport
        figsize: Figure size

    Returns:
        Figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)

    # Quality scores (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_quality_scores(report, ax=ax1)

    # Class distribution (top right) - if we have class info
    ax2 = fig.add_subplot(2, 2, 2)
    if report.class_distribution:
        counts = list(report.class_distribution.class_counts.values())
        classes = list(report.class_distribution.class_counts.keys())
        ax2.bar(range(len(classes)), counts, color="#3498db")
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
        ax2.set_title("Class Distribution")
        ax2.set_ylabel("Count")
    else:
        ax2.text(0.5, 0.5, "No class distribution data", ha="center", va="center")
        ax2.set_title("Class Distribution")

    # Issues breakdown (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    issue_types = ["Label\nErrors", "Duplicates", "Outliers", "Bias\nIssues"]
    issue_counts = [
        report.label_errors_result.n_issues if report.label_errors_result else 0,
        report.duplicates_result.n_issues if report.duplicates_result else 0,
        report.outliers_result.n_issues if report.outliers_result else 0,
        report.bias_result.n_issues if report.bias_result else 0,
    ]
    colors = ["#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
    ax3.bar(issue_types, issue_counts, color=colors)
    ax3.set_ylabel("Count")
    ax3.set_title("Issues by Type")
    for i, count in enumerate(issue_counts):
        ax3.annotate(str(count), xy=(i, count), ha="center", va="bottom")

    # Summary text (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    summary_text = [
        f"Dataset: {report.dataset_info.n_samples:,} samples, {report.dataset_info.n_features} features",
        f"Overall Quality: {report.quality_score.overall:.0f}/100",
        "",
        "Issue Summary:",
        f"  • Label errors: {report.label_errors_result.n_issues if report.label_errors_result else 0}",
        f"  • Duplicate pairs: {report.duplicates_result.n_issues if report.duplicates_result else 0}",
        f"  • Outliers: {report.outliers_result.n_issues if report.outliers_result else 0}",
    ]

    if report.imbalance_result:
        ratio = report.imbalance_result.metadata.get("imbalance_ratio", 1)
        summary_text.append(f"  • Imbalance ratio: {ratio:.1f}:1")

    ax4.text(
        0.1, 0.9, "\n".join(summary_text),
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    return fig


def save_plots(
    report: QualityReport,
    output_dir: str,
    features: pd.DataFrame | None = None,
    labels: np.ndarray | None = None,
    format: str = "png",
    dpi: int = 150,
) -> list[str]:
    """Save all relevant plots to files.

    Args:
        report: QualityReport
        output_dir: Output directory
        features: Feature DataFrame for scatter plots
        labels: Label array for distribution
        format: Image format (png, pdf, svg)
        dpi: Image DPI

    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved = []

    # Quality scores
    fig = plot_quality_scores(report)
    if fig:
        path = os.path.join(output_dir, f"quality_scores.{format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # Summary
    fig = plot_report_summary(report)
    path = os.path.join(output_dir, f"summary.{format}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Label errors confusion
    fig = plot_label_error_confusion(report)
    if fig:
        path = os.path.join(output_dir, f"label_errors.{format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # Class distribution
    if labels is not None:
        fig = plot_class_distribution(labels)
        if fig:
            path = os.path.join(output_dir, f"class_distribution.{format}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    # Outliers
    if features is not None:
        fig = plot_outlier_distribution(report, features)
        if fig:
            path = os.path.join(output_dir, f"outliers.{format}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    # Duplicates
    fig = plot_duplicate_similarity(report)
    if fig:
        path = os.path.join(output_dir, f"duplicates.{format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved
