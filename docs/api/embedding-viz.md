# Embedding Visualization

Interactive visualization of data embeddings with quality overlays.

## Quick Example

```python
from clean.embedding_viz import EmbeddingVisualizer

# Create visualizer
viz = EmbeddingVisualizer()

# Visualize embeddings with labels
result = viz.visualize(embeddings, labels=labels)

# Visualize with quality scores
result = viz.visualize(
    embeddings, 
    labels=labels, 
    quality_scores=quality_scores
)
```

## API Reference

### VisualizationConfig

Configuration for visualization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reduction_method` | ReductionMethod | `UMAP` | Dimensionality reduction method |
| `n_components` | int | `2` | Target dimensions (2 or 3) |
| `color_scheme` | ColorScheme | `QUALITY` | Default coloring scheme |
| `umap_n_neighbors` | int | `15` | UMAP neighbor parameter |
| `umap_min_dist` | float | `0.1` | UMAP min distance |
| `umap_metric` | str | `"euclidean"` | UMAP distance metric |
| `tsne_perplexity` | float | `30.0` | t-SNE perplexity |
| `tsne_learning_rate` | float | `200.0` | t-SNE learning rate |
| `point_size` | int | `5` | Point size in plot |
| `opacity` | float | `0.7` | Point opacity |
| `show_legend` | bool | `True` | Show legend |
| `interactive` | bool | `True` | Interactive plot (Plotly) |
| `max_points` | int | `10000` | Max points to display |
| `sample_seed` | int | `42` | Seed for sampling |

### ReductionMethod (Enum)

Dimensionality reduction methods.

| Value | Description |
|-------|-------------|
| `UMAP` | Uniform Manifold Approximation |
| `TSNE` | t-Distributed Stochastic Neighbor Embedding |
| `PCA` | Principal Component Analysis |

### ColorScheme (Enum)

Color scheme options.

| Value | Description |
|-------|-------------|
| `QUALITY` | Color by quality scores |
| `LABELS` | Color by class labels |
| `ISSUES` | Highlight quality issues |
| `CLUSTERS` | Color by clusters |

### EmbeddingVisualizer

Main visualizer class.

#### `__init__(config: VisualizationConfig | None = None)`

Initialize with optional configuration.

#### `visualize(embeddings, labels=None, quality_scores=None, report=None, metadata=None) -> VisualizationResult`

Create visualization of embeddings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | `np.ndarray` | High-dimensional embeddings (N x D) |
| `labels` | `np.ndarray \| None` | Class labels for coloring |
| `quality_scores` | `np.ndarray \| None` | Quality scores per sample |
| `report` | `QualityReport \| None` | Quality report for issue flags |
| `metadata` | `pd.DataFrame \| None` | Additional metadata for tooltips |

#### `create_quality_heatmap(embeddings, quality_scores) -> Any`

Create a quality heatmap overlay.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | `np.ndarray` | Embeddings (N x D) |
| `quality_scores` | `np.ndarray` | Quality scores |

#### `visualize_clusters(embeddings, cluster_labels) -> VisualizationResult`

Visualize with cluster coloring.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | `np.ndarray` | Embeddings (N x D) |
| `cluster_labels` | `np.ndarray` | Cluster assignments |

### VisualizationResult

Visualization result dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `n_points` | int | Number of points |
| `n_dimensions` | int | Reduced dimensions |
| `reduction_method` | ReductionMethod | Method used |
| `coordinates` | np.ndarray | Reduced coordinates |
| `labels` | np.ndarray \| None | Point labels |
| `quality_scores` | np.ndarray \| None | Quality scores |
| `issue_flags` | np.ndarray | Boolean issue flags |
| `cluster_info` | dict \| None | Cluster information |

## Example Workflows

### Basic Visualization

```python
from clean.embedding_viz import EmbeddingVisualizer, VisualizationConfig, ReductionMethod

config = VisualizationConfig(
    reduction_method=ReductionMethod.UMAP,
    n_components=2,
    interactive=True
)

viz = EmbeddingVisualizer(config=config)
result = viz.visualize(embeddings, labels=y)
```

### Quality-Colored Visualization

```python
from clean import DatasetCleaner
from clean.embedding_viz import EmbeddingVisualizer

# Get quality analysis
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Create embeddings (from your model)
embeddings = model.encode(df["text"])

# Visualize with quality context
viz = EmbeddingVisualizer()
result = viz.visualize(
    embeddings, 
    report=report,
    quality_scores=quality_scores
)
```

### Cluster Analysis

```python
from clean.embedding_viz import EmbeddingVisualizer
from sklearn.cluster import KMeans

# Cluster the embeddings
kmeans = KMeans(n_clusters=5)
cluster_labels = kmeans.fit_predict(embeddings)

# Visualize clusters
viz = EmbeddingVisualizer()
result = viz.visualize_clusters(embeddings, cluster_labels)
```

### 3D Visualization

```python
from clean.embedding_viz import EmbeddingVisualizer, VisualizationConfig

config = VisualizationConfig(
    n_components=3,
    interactive=True
)

viz = EmbeddingVisualizer(config=config)
result = viz.visualize(embeddings, labels=labels)
```

### Export for Custom Plotting

```python
viz = EmbeddingVisualizer()
result = viz.visualize(embeddings, labels=labels)

# Use coordinates for custom matplotlib plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    result.coordinates[:, 0],
    result.coordinates[:, 1],
    c=result.labels,
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(scatter)
plt.title("Embedding Space")
plt.savefig("embeddings.png")
```
