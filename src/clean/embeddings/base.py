"""Base embedding interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def encode(
        self,
        data: list[Any],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode data to embeddings.

        Args:
            data: List of items to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embedding matrix of shape (n_samples, embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        pass

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between embeddings.

        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix

        Returns:
            Similarity matrix
        """
        # Normalize
        norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        return np.dot(norm1, norm2.T)
