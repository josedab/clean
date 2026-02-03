"""Tests for foundation model embeddings."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

from clean.embeddings.foundation import (
    EmbeddingProvider,
    EmbeddingStats,
    OpenAIEmbedder,
    CohereEmbedder,
    VoyageEmbedder,
    CachedEmbedder,
    create_embedder,
)
from clean.exceptions import ConfigurationError, DependencyError


class TestEmbeddingStats:
    """Tests for EmbeddingStats."""

    def test_record_request(self):
        """Test recording request statistics."""
        stats = EmbeddingStats()
        stats.record_request(tokens=100, cost=0.001, latency_ms=50, cached=False)

        assert stats.total_tokens == 100
        assert stats.total_requests == 1
        assert stats.total_cost_usd == 0.001
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0

    def test_record_cached_request(self):
        """Test recording cached request."""
        stats = EmbeddingStats()
        stats.record_request(tokens=0, cost=0, latency_ms=1, cached=True)

        assert stats.cache_hits == 1
        assert stats.total_tokens == 0
        assert stats.total_requests == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = EmbeddingStats()
        stats.record_request(tokens=100, cost=0.001, latency_ms=50, cached=False)

        result = stats.to_dict()
        assert "total_tokens" in result
        assert "total_cost_usd" in result
        assert result["total_tokens"] == 100


class TestOpenAIEmbedder:
    """Tests for OpenAI embedder."""

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="API key required"):
                OpenAIEmbedder()

    def test_embedding_dim(self):
        """Test embedding dimension property."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            embedder = OpenAIEmbedder()
            assert embedder.embedding_dim == 1536

    def test_embedding_dim_custom(self):
        """Test custom embedding dimensions."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            embedder = OpenAIEmbedder(dimensions=512)
            assert embedder.embedding_dim == 512

    def test_provider(self):
        """Test provider property."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            embedder = OpenAIEmbedder()
            assert embedder.provider == EmbeddingProvider.OPENAI

    def test_estimate_cost(self):
        """Test cost estimation."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            embedder = OpenAIEmbedder()
            cost = embedder.estimate_cost(n_texts=1000, avg_tokens_per_text=100)
            assert cost > 0
            assert cost < 1  # Should be relatively cheap


class TestCohereEmbedder:
    """Tests for Cohere embedder."""

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="API key required"):
                CohereEmbedder()

    def test_embedding_dim(self):
        """Test embedding dimension."""
        with patch.dict("os.environ", {"CO_API_KEY": "test-key"}):
            embedder = CohereEmbedder()
            assert embedder.embedding_dim == 1024

    def test_provider(self):
        """Test provider property."""
        with patch.dict("os.environ", {"CO_API_KEY": "test-key"}):
            embedder = CohereEmbedder()
            assert embedder.provider == EmbeddingProvider.COHERE


class TestVoyageEmbedder:
    """Tests for Voyage embedder."""

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="API key required"):
                VoyageEmbedder()

    def test_embedding_dim(self):
        """Test embedding dimension."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            embedder = VoyageEmbedder()
            assert embedder.embedding_dim == 1024

    def test_provider(self):
        """Test provider property."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            embedder = VoyageEmbedder()
            assert embedder.provider == EmbeddingProvider.VOYAGE


class TestCachedEmbedder:
    """Tests for cached embedder."""

    def test_caching_works(self):
        """Test that caching actually caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock embedder
            mock_embedder = MagicMock()
            mock_embedder.model = "test-model"
            mock_embedder.embedding_dim = 384
            mock_embedder.encode.return_value = np.random.rand(2, 384).astype(np.float32)
            mock_embedder.stats = EmbeddingStats()

            cached = CachedEmbedder(mock_embedder, cache_dir=tmpdir)

            # First call should use embedder
            texts = ["hello", "world"]
            result1 = cached.encode(texts)
            assert mock_embedder.encode.call_count == 1

            # Second call should use cache
            result2 = cached.encode(texts)
            assert mock_embedder.encode.call_count == 1  # Still 1

            # Results should be same
            np.testing.assert_array_almost_equal(result1, result2)

    def test_clear_cache(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_embedder = MagicMock()
            mock_embedder.model = "test-model"
            mock_embedder.embedding_dim = 384
            mock_embedder.encode.return_value = np.random.rand(2, 384).astype(np.float32)
            mock_embedder.stats = EmbeddingStats()

            cached = CachedEmbedder(mock_embedder, cache_dir=tmpdir)
            cached.encode(["hello", "world"])

            count = cached.clear_cache()
            assert count == 2


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_create_openai(self):
        """Test creating OpenAI embedder."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            embedder = create_embedder("openai", cached=False)
            assert isinstance(embedder, OpenAIEmbedder)

    def test_create_cohere(self):
        """Test creating Cohere embedder."""
        with patch.dict("os.environ", {"CO_API_KEY": "test-key"}):
            embedder = create_embedder("cohere", cached=False)
            assert isinstance(embedder, CohereEmbedder)

    def test_create_voyage(self):
        """Test creating Voyage embedder."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            embedder = create_embedder("voyage", cached=False)
            assert isinstance(embedder, VoyageEmbedder)

    def test_create_with_caching(self):
        """Test creating embedder with caching enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                embedder = create_embedder("openai", cached=True, cache_dir=tmpdir)
                assert isinstance(embedder, CachedEmbedder)

    def test_create_unknown_raises(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ConfigurationError, match="Unknown"):
            create_embedder("unknown_provider")
