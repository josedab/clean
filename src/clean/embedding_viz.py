"""Embedding Space Visualizer.

This module provides interactive visualization of data in embedding space
with quality overlays for debugging data quality issues.

Example:
    >>> from clean.embedding_viz import EmbeddingVisualizer
    >>>
    >>> viz = EmbeddingVisualizer()
    >>> fig = viz.visualize(embeddings, labels, quality_scores)
    >>> fig.show()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clean.core.report import QualityReport

logger = logging.getLogger(__name__)


class ReductionMethod(Enum):
    """Dimensionality reduction methods."""

    UMAP = "umap"
    TSNE = "tsne"
    PCA = "pca"
    TRIMAP = "trimap"


class ColorScheme(Enum):
    """Color schemes for visualization."""

    BY_LABEL = "by_label"
    BY_QUALITY = "by_quality"
    BY_ISSUE_TYPE = "by_issue_type"
    BY_CLUSTER = "by_cluster"
    BY_CONFIDENCE = "by_confidence"


@dataclass
class VisualizationConfig:
    """Configuration for embedding visualization."""

    reduction_method: ReductionMethod = ReductionMethod.UMAP
    n_components: int = 3  # 2 or 3
    color_scheme: ColorScheme = ColorScheme.BY_LABEL

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # t-SNE parameters
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0

    # Display options
    point_size: int = 5
    opacity: float = 0.7
    show_legend: bool = True
    interactive: bool = True

    # Sampling for large datasets
    max_points: int = 10000
    sample_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reduction_method": self.reduction_method.value,
            "n_components": self.n_components,
            "color_scheme": self.color_scheme.value,
            "max_points": self.max_points,
        }


@dataclass
class EmbeddingPoint:
    """A single point in the embedding visualization."""

    index: int
    coordinates: np.ndarray  # 2D or 3D
    original_embedding: np.ndarray
    label: Any
    quality_score: float | None
    issue_types: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationResult:
    """Result of embedding visualization."""

    n_points: int
    n_dimensions: int
    reduction_method: str

    coordinates: np.ndarray
    labels: np.ndarray | None
    quality_scores: np.ndarray | None
    issue_flags: np.ndarray | None

    # Statistics
    cluster_info: dict[str, Any] | None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy export."""
        data = {
            f"dim_{i}": self.coordinates[:, i]
            for i in range(self.coordinates.shape[1])
        }

        if self.labels is not None:
            data["label"] = self.labels

        if self.quality_scores is not None:
            data["quality_score"] = self.quality_scores

        if self.issue_flags is not None:
            data["has_issue"] = self.issue_flags

        return pd.DataFrame(data)


class DimensionalityReducer:
    """Reduce high-dimensional embeddings for visualization."""

    def __init__(self, config: VisualizationConfig):
        """Initialize reducer.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self._reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform embeddings.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            Low-dimensional coordinates
        """
        method = self.config.reduction_method

        if method == ReductionMethod.UMAP:
            return self._reduce_umap(embeddings)
        elif method == ReductionMethod.TSNE:
            return self._reduce_tsne(embeddings)
        elif method == ReductionMethod.PCA:
            return self._reduce_pca(embeddings)
        else:
            return self._reduce_pca(embeddings)

    def _reduce_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce using UMAP."""
        try:
            import umap

            self._reducer = umap.UMAP(
                n_components=self.config.n_components,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric=self.config.umap_metric,
                random_state=self.config.sample_seed,
            )
            return self._reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("UMAP not available, falling back to t-SNE")
            return self._reduce_tsne(embeddings)

    def _reduce_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce using t-SNE."""
        from sklearn.manifold import TSNE

        # t-SNE only supports 2D, use PCA first if 3D needed
        n_components = min(self.config.n_components, 2)

        self._reducer = TSNE(
            n_components=n_components,
            perplexity=min(self.config.tsne_perplexity, len(embeddings) - 1),
            learning_rate=self.config.tsne_learning_rate,
            random_state=self.config.sample_seed,
        )
        return self._reducer.fit_transform(embeddings)

    def _reduce_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce using PCA."""
        from sklearn.decomposition import PCA

        self._reducer = PCA(
            n_components=self.config.n_components,
            random_state=self.config.sample_seed,
        )
        return self._reducer.fit_transform(embeddings)


class EmbeddingVisualizer:
    """Visualize embeddings with quality overlays.

    Creates interactive 2D/3D scatter plots of embedding space
    with color coding for quality issues.
    """

    def __init__(
        self,
        config: VisualizationConfig | None = None,
    ):
        """Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.reducer = DimensionalityReducer(self.config)

    def visualize(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray | None = None,
        quality_scores: np.ndarray | None = None,
        report: QualityReport | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> Any:
        """Create embedding visualization.

        Args:
            embeddings: High-dimensional embeddings
            labels: Optional class labels
            quality_scores: Optional quality scores
            report: Optional quality report for issue highlighting
            metadata: Optional metadata DataFrame

        Returns:
            Plotly figure (or matplotlib if plotly unavailable)
        """
        # Sample if too many points
        n_samples = len(embeddings)
        if n_samples > self.config.max_points:
            logger.info(
                f"Sampling {self.config.max_points} of {n_samples} points"
            )
            np.random.seed(self.config.sample_seed)
            indices = np.random.choice(
                n_samples,
                self.config.max_points,
                replace=False,
            )
            embeddings = embeddings[indices]
            if labels is not None:
                labels = np.array(labels)[indices]
            if quality_scores is not None:
                quality_scores = np.array(quality_scores)[indices]
        else:
            indices = np.arange(n_samples)

        # Reduce dimensionality
        coordinates = self.reducer.fit_transform(embeddings)

        # Extract issue information from report
        issue_flags = None
        issue_types = None
        if report is not None:
            issue_flags, issue_types = self._extract_issues(report, indices)

        # Create result
        result = VisualizationResult(
            n_points=len(coordinates),
            n_dimensions=self.config.n_components,
            reduction_method=self.config.reduction_method.value,
            coordinates=coordinates,
            labels=labels,
            quality_scores=quality_scores,
            issue_flags=issue_flags,
            cluster_info=None,
        )

        # Create plot
        try:
            return self._create_plotly_figure(
                result, issue_types, metadata, indices
            )
        except ImportError:
            return self._create_matplotlib_figure(result)

    def _extract_issues(
        self,
        report: QualityReport,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        """Extract issue information from report."""
        n_points = len(indices)
        issue_flags = np.zeros(n_points, dtype=bool)
        issue_types: dict[int, list[str]] = {i: [] for i in range(n_points)}

        index_to_pos = {idx: pos for pos, idx in enumerate(indices)}

        # Label errors
        if hasattr(report, "label_errors"):
            errors = report.label_errors()
            if errors is not None and "index" in errors.columns:
                for idx in errors["index"]:
                    if idx in index_to_pos:
                        pos = index_to_pos[idx]
                        issue_flags[pos] = True
                        issue_types[pos].append("label_error")

        # Outliers
        if hasattr(report, "outliers"):
            outliers = report.outliers()
            if outliers is not None:
                for idx in outliers:
                    if idx in index_to_pos:
                        pos = index_to_pos[idx]
                        issue_flags[pos] = True
                        issue_types[pos].append("outlier")

        # Duplicates
        if hasattr(report, "duplicates"):
            duplicates = report.duplicates()
            if duplicates is not None:
                for col in ["index_1", "index_2", "index"]:
                    if col in duplicates.columns:
                        for idx in duplicates[col]:
                            if idx in index_to_pos:
                                pos = index_to_pos[idx]
                                issue_flags[pos] = True
                                issue_types[pos].append("duplicate")

        return issue_flags, issue_types

    def _create_plotly_figure(
        self,
        result: VisualizationResult,
        issue_types: dict[int, list[str]] | None,
        metadata: pd.DataFrame | None,
        indices: np.ndarray,
    ) -> Any:
        """Create interactive Plotly figure."""
        import plotly.express as px
        import plotly.graph_objects as go

        coords = result.coordinates
        n_dims = result.n_dimensions

        # Prepare data for plotting
        plot_data = {
            "x": coords[:, 0],
            "y": coords[:, 1],
        }

        if n_dims >= 3:
            plot_data["z"] = coords[:, 2]

        # Add index for hover
        plot_data["index"] = indices

        # Color by scheme
        if self.config.color_scheme == ColorScheme.BY_LABEL and result.labels is not None:
            plot_data["color"] = [str(l) for l in result.labels]
            color_col = "color"
        elif self.config.color_scheme == ColorScheme.BY_QUALITY and result.quality_scores is not None:
            plot_data["color"] = result.quality_scores
            color_col = "color"
        elif self.config.color_scheme == ColorScheme.BY_ISSUE_TYPE and result.issue_flags is not None:
            plot_data["color"] = ["issue" if f else "clean" for f in result.issue_flags]
            color_col = "color"
        else:
            plot_data["color"] = ["point"] * len(coords)
            color_col = "color"

        # Add issue type info for hover
        if issue_types:
            plot_data["issues"] = [
                ", ".join(issue_types.get(i, [])) or "none"
                for i in range(len(coords))
            ]

        # Add quality scores for hover
        if result.quality_scores is not None:
            plot_data["quality"] = result.quality_scores

        df = pd.DataFrame(plot_data)

        # Create figure
        if n_dims >= 3:
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color=color_col,
                opacity=self.config.opacity,
                hover_data=["index", "issues", "quality"] if "issues" in df.columns else ["index"],
                title="Embedding Space Visualization",
            )
        else:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color=color_col,
                opacity=self.config.opacity,
                hover_data=["index", "issues", "quality"] if "issues" in df.columns else ["index"],
                title="Embedding Space Visualization",
            )

        # Update layout
        fig.update_traces(marker=dict(size=self.config.point_size))
        fig.update_layout(
            showlegend=self.config.show_legend,
            legend_title="Legend",
            hovermode="closest",
        )

        return fig

    def _create_matplotlib_figure(self, result: VisualizationResult) -> Any:
        """Create matplotlib figure (fallback)."""
        import matplotlib.pyplot as plt

        coords = result.coordinates
        n_dims = result.n_dimensions

        if n_dims >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            colors = result.quality_scores if result.quality_scores is not None else None

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors,
                cmap="RdYlGn",
                alpha=self.config.opacity,
                s=self.config.point_size,
            )

            if colors is not None:
                plt.colorbar(scatter, label="Quality Score")

            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_zlabel("Dimension 3")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

            colors = result.quality_scores if result.quality_scores is not None else None

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                cmap="RdYlGn",
                alpha=self.config.opacity,
                s=self.config.point_size,
            )

            if colors is not None:
                plt.colorbar(scatter, label="Quality Score")

            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

        plt.title("Embedding Space Visualization")
        return fig

    def visualize_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray | None = None,
        n_clusters: int = 10,
    ) -> Any:
        """Visualize with automatic clustering.

        Args:
            embeddings: Embeddings to visualize
            labels: Optional true labels for comparison
            n_clusters: Number of clusters

        Returns:
            Plotly figure
        """
        from sklearn.cluster import KMeans

        # Reduce dimensions
        coordinates = self.reducer.fit_transform(embeddings)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.sample_seed)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Create result
        result = VisualizationResult(
            n_points=len(coordinates),
            n_dimensions=self.config.n_components,
            reduction_method=self.config.reduction_method.value,
            coordinates=coordinates,
            labels=cluster_labels,
            quality_scores=None,
            issue_flags=None,
            cluster_info={"n_clusters": n_clusters},
        )

        # Plot
        self.config.color_scheme = ColorScheme.BY_CLUSTER
        return self._create_plotly_figure(result, None, None, np.arange(len(embeddings)))

    def create_quality_heatmap(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray,
        grid_size: int = 50,
    ) -> Any:
        """Create heatmap of quality scores in embedding space.

        Args:
            embeddings: Embeddings
            quality_scores: Quality scores
            grid_size: Grid resolution

        Returns:
            Plotly figure
        """
        # Reduce to 2D
        self.config.n_components = 2
        coords = self.reducer.fit_transform(embeddings)

        # Create grid
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)

        # Aggregate quality scores into grid cells
        heatmap = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))

        for (x, y), q in zip(coords, quality_scores):
            xi = min(int((x - x_min) / (x_max - x_min) * (grid_size - 1)), grid_size - 1)
            yi = min(int((y - y_min) / (y_max - y_min) * (grid_size - 1)), grid_size - 1)
            heatmap[yi, xi] += q
            counts[yi, xi] += 1

        # Average
        with np.errstate(divide="ignore", invalid="ignore"):
            heatmap = np.where(counts > 0, heatmap / counts, np.nan)

        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Heatmap(
                z=heatmap,
                x=x_grid,
                y=y_grid,
                colorscale="RdYlGn",
                colorbar=dict(title="Quality Score"),
            ))

            fig.update_layout(
                title="Quality Score Heatmap in Embedding Space",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
            )

            return fig
        except ImportError:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                heatmap,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                cmap="RdYlGn",
                aspect="auto",
            )
            plt.colorbar(im, label="Quality Score")
            plt.title("Quality Score Heatmap in Embedding Space")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            return fig


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    quality_scores: np.ndarray | None = None,
    method: str = "umap",
    n_components: int = 3,
    **kwargs: Any,
) -> Any:
    """Convenience function to visualize embeddings.

    Args:
        embeddings: High-dimensional embeddings
        labels: Optional labels
        quality_scores: Optional quality scores
        method: Reduction method (umap, tsne, pca)
        n_components: Number of dimensions (2 or 3)
        **kwargs: Additional config parameters

    Returns:
        Interactive figure
    """
    config = VisualizationConfig(
        reduction_method=ReductionMethod(method),
        n_components=n_components,
        **kwargs,
    )

    viz = EmbeddingVisualizer(config=config)
    return viz.visualize(embeddings, labels, quality_scores)


def create_embedding_visualizer(
    method: str = "umap",
    n_components: int = 3,
    **kwargs: Any,
) -> EmbeddingVisualizer:
    """Create an embedding visualizer.

    Args:
        method: Reduction method
        n_components: Number of dimensions
        **kwargs: Additional config parameters

    Returns:
        EmbeddingVisualizer
    """
    config = VisualizationConfig(
        reduction_method=ReductionMethod(method),
        n_components=n_components,
        **kwargs,
    )
    return EmbeddingVisualizer(config=config)
