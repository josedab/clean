"""Tests for the embeddings module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from clean.embeddings.base import BaseEmbedder


class ConcreteEmbedder(BaseEmbedder):
    """Concrete implementation for testing."""

    def __init__(self, dim: int = 128):
        self._dim = dim

    def encode(
        self,
        data: list,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate random embeddings for testing."""
        return np.random.randn(len(data), self._dim).astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._dim


class TestBaseEmbedder:
    """Tests for BaseEmbedder abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseEmbedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""
        embedder = ConcreteEmbedder(dim=64)
        assert embedder.embedding_dim == 64

    def test_encode(self):
        """Test encoding produces correct shape."""
        embedder = ConcreteEmbedder(dim=128)
        data = ["text1", "text2", "text3"]
        
        embeddings = embedder.encode(data)
        
        assert embeddings.shape == (3, 128)
        assert embeddings.dtype == np.float32

    def test_encode_batch_size(self):
        """Test encoding with different batch sizes."""
        embedder = ConcreteEmbedder(dim=64)
        data = list(range(100))
        
        embeddings = embedder.encode(data, batch_size=16)
        
        assert embeddings.shape == (100, 64)

    def test_similarity(self):
        """Test similarity computation."""
        embedder = ConcreteEmbedder(dim=128)
        
        # Create known embeddings
        emb1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        emb2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        
        sim = embedder.similarity(emb1, emb2)
        
        # First row of emb1 should be identical to first row of emb2
        assert np.isclose(sim[0, 0], 1.0)
        # First row of emb1 should be orthogonal to second row of emb2
        assert np.isclose(sim[0, 1], 0.0)

    def test_similarity_normalized(self):
        """Test that similarity normalizes vectors."""
        embedder = ConcreteEmbedder()
        
        # Non-unit vectors
        emb1 = np.array([[2.0, 0.0], [0.0, 3.0]])
        emb2 = np.array([[4.0, 0.0], [0.0, 5.0]])
        
        sim = embedder.similarity(emb1, emb2)
        
        # Should still get 1.0 for parallel vectors after normalization
        assert np.isclose(sim[0, 0], 1.0)
        assert np.isclose(sim[1, 1], 1.0)

    def test_similarity_matrix_shape(self):
        """Test similarity matrix has correct shape."""
        embedder = ConcreteEmbedder(dim=64)
        
        emb1 = np.random.randn(5, 64)
        emb2 = np.random.randn(10, 64)
        
        sim = embedder.similarity(emb1, emb2)
        
        assert sim.shape == (5, 10)

    def test_similarity_symmetric(self):
        """Test self-similarity is symmetric."""
        embedder = ConcreteEmbedder(dim=32)
        
        emb = np.random.randn(5, 32)
        sim = embedder.similarity(emb, emb)
        
        # Self-similarity matrix should be symmetric
        assert np.allclose(sim, sim.T)

    def test_empty_input(self):
        """Test encoding empty list."""
        embedder = ConcreteEmbedder(dim=64)
        
        embeddings = embedder.encode([])
        
        assert embeddings.shape == (0, 64)


class TestTextEmbedder:
    """Tests for TextEmbedder (mocked to avoid model downloads)."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock sentence transformer model."""
        with patch("clean.embeddings.text.HAS_SENTENCE_TRANSFORMERS", True):
            with patch("clean.embeddings.text.SentenceTransformer") as mock_st:
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_model.encode.return_value = np.random.randn(3, 384)
                mock_st.return_value = mock_model
                yield mock_st, mock_model

    def test_text_embedder_import_error(self):
        """Test DependencyError when sentence-transformers not available."""
        with patch("clean.embeddings.text.HAS_SENTENCE_TRANSFORMERS", False):
            from clean.embeddings.text import TextEmbedder
            from clean.exceptions import DependencyError
            
            with pytest.raises(DependencyError) as exc_info:
                TextEmbedder()
            
            assert "sentence-transformers" in str(exc_info.value)
            assert exc_info.value.extra == "text"

    def test_text_embedder_init(self, mock_sentence_transformer):
        """Test TextEmbedder initialization."""
        mock_st, mock_model = mock_sentence_transformer
        
        from clean.embeddings.text import TextEmbedder
        
        embedder = TextEmbedder()
        
        assert embedder.embedding_dim == 384
        mock_st.assert_called_once()

    def test_text_embedder_custom_model(self, mock_sentence_transformer):
        """Test TextEmbedder with custom model name."""
        mock_st, mock_model = mock_sentence_transformer
        
        from clean.embeddings.text import TextEmbedder
        
        embedder = TextEmbedder(model_name="custom/model")
        
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert call_args[0][0] == "custom/model"

    def test_text_embedder_encode(self, mock_sentence_transformer):
        """Test TextEmbedder encoding."""
        mock_st, mock_model = mock_sentence_transformer
        
        from clean.embeddings.text import TextEmbedder
        
        embedder = TextEmbedder()
        texts = ["Hello world", "Test text", "Another example"]
        
        embeddings = embedder.encode(texts)
        
        mock_model.encode.assert_called_once()
        assert embeddings.shape[0] == 3

    def test_text_embedder_encode_query(self, mock_sentence_transformer):
        """Test TextEmbedder single query encoding."""
        mock_st, mock_model = mock_sentence_transformer
        mock_model.encode.return_value = np.random.randn(1, 384)
        
        from clean.embeddings.text import TextEmbedder
        
        embedder = TextEmbedder()
        query = "search query"
        
        embedding = embedder.encode_query(query)
        
        assert embedding.shape == (384,)

    def test_get_text_embeddings_function(self, mock_sentence_transformer):
        """Test convenience function."""
        mock_st, mock_model = mock_sentence_transformer
        
        from clean.embeddings.text import get_text_embeddings
        
        texts = ["text1", "text2"]
        embeddings = get_text_embeddings(texts)
        
        assert embeddings is not None


class TestImageEmbedder:
    """Tests for ImageEmbedder (mocked to avoid model downloads)."""

    @pytest.fixture
    def mock_torch_and_clip(self):
        """Mock torch and CLIP model."""
        with patch("clean.embeddings.image.HAS_CLIP", True):
            with patch("clean.embeddings.image.torch") as mock_torch:
                with patch("clean.embeddings.image.CLIPProcessor") as mock_proc:
                    with patch("clean.embeddings.image.CLIPModel") as mock_model:
                        mock_torch.no_grad.return_value.__enter__ = MagicMock()
                        mock_torch.no_grad.return_value.__exit__ = MagicMock()
                        
                        yield mock_torch, mock_proc, mock_model

    def test_image_embedder_requires_dependencies(self):
        """Test that ImageEmbedder requires torch and transformers."""
        with patch("clean.embeddings.image.HAS_CLIP", False):
            # Need to reimport to pick up the patched value
            import importlib
            import clean.embeddings.image as img_module
            importlib.reload(img_module)
            
            with pytest.raises(ImportError):
                img_module.ImageEmbedder()


class TestEmbeddingIntegration:
    """Integration tests for embeddings (with mocks)."""

    def test_embedding_pipeline(self):
        """Test basic embedding pipeline."""
        embedder = ConcreteEmbedder(dim=256)
        
        texts = ["This is document one", "This is document two", "A completely different text"]
        embeddings = embedder.encode(texts)
        
        # Compute similarities
        sim = embedder.similarity(embeddings, embeddings)
        
        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(texts)):
            assert np.isclose(sim[i, i], 1.0, atol=1e-5)

    def test_duplicate_detection_with_embeddings(self):
        """Test using embeddings for duplicate detection."""
        embedder = ConcreteEmbedder(dim=64)
        
        # Create documents where first two are "similar" (same random seed)
        np.random.seed(42)
        emb1 = np.random.randn(64)
        emb2 = emb1 + np.random.randn(64) * 0.1  # Similar to emb1
        emb3 = np.random.randn(64)  # Different
        
        embeddings = np.vstack([emb1, emb2, emb3])
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        sim = embedder.similarity(embeddings, embeddings)
        
        # emb1 and emb2 should be more similar than emb1 and emb3
        assert sim[0, 1] > sim[0, 2]

    def test_batch_processing(self):
        """Test batch processing of embeddings."""
        embedder = ConcreteEmbedder(dim=128)
        
        # Large batch
        data = [f"text_{i}" for i in range(1000)]
        
        # Process in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            emb = embedder.encode(batch, batch_size=batch_size)
            all_embeddings.append(emb)
        
        full_embeddings = np.vstack(all_embeddings)
        
        assert full_embeddings.shape == (1000, 128)


class TestEdgeCases:
    """Edge case tests for embeddings."""

    def test_single_item(self):
        """Test encoding a single item."""
        embedder = ConcreteEmbedder(dim=64)
        
        embeddings = embedder.encode(["single item"])
        
        assert embeddings.shape == (1, 64)

    def test_very_large_dimension(self):
        """Test large embedding dimension."""
        embedder = ConcreteEmbedder(dim=4096)
        
        embeddings = embedder.encode(["text"])
        
        assert embeddings.shape == (1, 4096)

    def test_similarity_with_self(self):
        """Test similarity of embedding with itself."""
        embedder = ConcreteEmbedder(dim=32)
        
        emb = np.random.randn(1, 32)
        sim = embedder.similarity(emb, emb)
        
        assert np.isclose(sim[0, 0], 1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        embedder = ConcreteEmbedder()
        
        # Orthogonal unit vectors
        emb1 = np.array([[1.0, 0.0, 0.0, 0.0]])
        emb2 = np.array([[0.0, 1.0, 0.0, 0.0]])
        
        sim = embedder.similarity(emb1, emb2)
        
        assert np.isclose(sim[0, 0], 0.0)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        embedder = ConcreteEmbedder()
        
        emb1 = np.array([[1.0, 0.0, 0.0]])
        emb2 = np.array([[-1.0, 0.0, 0.0]])
        
        sim = embedder.similarity(emb1, emb2)
        
        assert np.isclose(sim[0, 0], -1.0)
