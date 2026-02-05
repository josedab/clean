"""Tests for embedding_viz module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestEmbeddingVizModule:
    """Tests for embedding visualization module."""

    def test_imports(self) -> None:
        from clean.embedding_viz import (
            EmbeddingVisualizer,
            VisualizationConfig,
            VisualizationResult,
            ReductionMethod,
            ColorScheme,
        )
        assert EmbeddingVisualizer is not None
        assert VisualizationConfig is not None
        assert VisualizationResult is not None

    def test_reduction_methods(self) -> None:
        from clean.embedding_viz import ReductionMethod
        
        assert ReductionMethod.UMAP is not None
        assert ReductionMethod.TSNE is not None
        assert ReductionMethod.PCA is not None

    def test_color_schemes(self) -> None:
        from clean.embedding_viz import ColorScheme
        
        # Check actual enum values
        assert hasattr(ColorScheme, 'BY_QUALITY') or hasattr(ColorScheme, 'QUALITY')
        assert hasattr(ColorScheme, 'BY_LABEL') or hasattr(ColorScheme, 'LABELS')

    def test_config_defaults(self) -> None:
        from clean.embedding_viz import VisualizationConfig
        
        config = VisualizationConfig()
        assert config.n_components > 0
        assert config.point_size > 0
        assert config.opacity > 0

    def test_visualizer_init(self) -> None:
        from clean.embedding_viz import EmbeddingVisualizer, VisualizationConfig
        
        viz = EmbeddingVisualizer()
        assert viz is not None
        
        config = VisualizationConfig(n_components=3)
        viz_with_config = EmbeddingVisualizer(config=config)
        assert viz_with_config is not None

    def test_visualize_basic(self) -> None:
        from clean.embedding_viz import EmbeddingVisualizer, VisualizationConfig
        
        np.random.seed(42)
        # Use 2D embeddings with 2D config
        embeddings = np.random.randn(100, 2)
        labels = np.random.choice([0, 1, 2], 100)
        
        config = VisualizationConfig(n_components=2)
        viz = EmbeddingVisualizer(config=config)
        result = viz.visualize(embeddings, labels=labels)
        
        assert result is not None

    def test_visualize_with_quality_scores(self) -> None:
        from clean.embedding_viz import EmbeddingVisualizer, VisualizationConfig
        
        np.random.seed(42)
        embeddings = np.random.randn(100, 2)
        labels = np.random.choice([0, 1, 2], 100)
        quality_scores = np.random.rand(100)
        
        config = VisualizationConfig(n_components=2)
        viz = EmbeddingVisualizer(config=config)
        result = viz.visualize(embeddings, labels=labels, quality_scores=quality_scores)
        
        assert result is not None

    def test_visualization_result_fields(self) -> None:
        from clean.embedding_viz import VisualizationResult, ReductionMethod
        
        result = VisualizationResult(
            n_points=100,
            n_dimensions=2,
            reduction_method=ReductionMethod.PCA,
            coordinates=np.random.randn(100, 2),
            labels=np.random.choice([0, 1], 100),
            quality_scores=np.random.rand(100),
            issue_flags=np.zeros(100, dtype=bool),
            cluster_info=None,
        )
        
        assert result.n_points == 100
        assert result.n_dimensions == 2
        assert result.reduction_method == ReductionMethod.PCA
