"""Text embedding using sentence-transformers."""

from __future__ import annotations

import numpy as np

from clean.embeddings.base import BaseEmbedder
from clean.exceptions import DependencyError

# Optional import
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore[misc, assignment]


class TextEmbedder(BaseEmbedder):
    """Generate embeddings for text using sentence-transformers."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
    ):
        """Initialize the text embedder.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ('cpu', 'cuda', etc.)
            cache_folder: Folder to cache models

        Raises:
            DependencyError: If sentence-transformers not installed
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise DependencyError(
                "sentence-transformers",
                "text",
                feature="text embeddings",
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device

        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=cache_folder,
        )
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        data: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text to embeddings.

        Args:
            data: List of text strings
            batch_size: Batch size
            show_progress: Show progress bar

        Returns:
            Embedding matrix
        """
        embeddings = self._model.encode(
            data,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedding_dim

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query text.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self._model.encode([query], convert_to_numpy=True)[0]


def get_text_embeddings(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Get embeddings for a list of texts.

    Args:
        texts: List of text strings
        model_name: Model name
        batch_size: Batch size
        show_progress: Show progress

    Returns:
        Embedding matrix
    """
    embedder = TextEmbedder(model_name=model_name)
    return embedder.encode(texts, batch_size=batch_size, show_progress=show_progress)
