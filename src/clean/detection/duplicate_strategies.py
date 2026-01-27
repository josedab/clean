"""Duplicate detection strategies.

This module provides strategy classes for different duplicate detection algorithms,
implementing the Strategy pattern to reduce complexity in DuplicateDetector.

Example:
    >>> strategy = HashStrategy()
    >>> strategy.fit(df)
    >>> duplicates = strategy.detect(df)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from clean.exceptions import DependencyError

if TYPE_CHECKING:
    pass

# Optional imports
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore[misc, assignment]


@dataclass
class DuplicateCandidate:
    """A candidate duplicate pair identified by a detection strategy."""

    index1: int
    index2: int
    similarity: float
    is_exact: bool
    method: str


class DuplicateStrategy(ABC):
    """Abstract base class for duplicate detection strategies.

    Each strategy implements a specific duplicate detection algorithm.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> DuplicateStrategy:
        """Fit the strategy on data.

        Args:
            df: DataFrame to analyze

        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def detect(
        self,
        df: pd.DataFrame,
        similarity_threshold: float,
        seen_pairs: set[tuple[int, int]],
    ) -> list[DuplicateCandidate]:
        """Detect duplicates using this strategy.

        Args:
            df: DataFrame to analyze
            similarity_threshold: Minimum similarity for near-duplicates
            seen_pairs: Set of already-seen pairs to avoid duplicating

        Returns:
            List of duplicate candidates
        """
        ...


class HashStrategy(DuplicateStrategy):
    """Exact duplicate detection using content hashing.

    Computes MD5 hash of row contents to find exact duplicates.
    Fast and memory-efficient for large datasets.
    """

    def __init__(self, hash_columns: list[str] | None = None):
        """Initialize hash strategy.

        Args:
            hash_columns: Columns to use for hashing (None = all)
        """
        self.hash_columns = hash_columns
        self._hashes: list[str] = []

    @property
    def name(self) -> str:
        return "hash"

    def fit(self, df: pd.DataFrame) -> HashStrategy:
        """Compute content hashes for each row."""
        cols = self.hash_columns or list(df.columns)
        cols = [c for c in cols if c in df.columns]

        self._hashes = []
        for _, row in df[cols].iterrows():
            content = "|".join(str(v) for v in row.values)
            hash_val = hashlib.md5(content.encode()).hexdigest()
            self._hashes.append(hash_val)

        return self

    def detect(
        self,
        df: pd.DataFrame,
        similarity_threshold: float,
        seen_pairs: set[tuple[int, int]],
    ) -> list[DuplicateCandidate]:
        """Find exact duplicates via hash comparison."""
        if not self._hashes:
            return []

        # Group indices by hash
        hash_to_indices: dict[str, list[int]] = {}
        for idx, h in enumerate(self._hashes):
            if h not in hash_to_indices:
                hash_to_indices[h] = []
            hash_to_indices[h].append(idx)

        # Find duplicates
        candidates = []
        for indices in hash_to_indices.values():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pair = (indices[i], indices[j])
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            candidates.append(
                                DuplicateCandidate(
                                    index1=indices[i],
                                    index2=indices[j],
                                    similarity=1.0,
                                    is_exact=True,
                                    method=self.name,
                                )
                            )

        return candidates


class FuzzyStrategy(DuplicateStrategy):
    """Fuzzy duplicate detection using cosine similarity on numeric features.

    Normalizes numeric columns and computes pairwise similarity.
    Good for finding near-duplicates in tabular data.
    """

    def __init__(self, max_samples: int = 10000):
        """Initialize fuzzy strategy.

        Args:
            max_samples: Maximum samples for full pairwise comparison
        """
        self.max_samples = max_samples
        self._normalized_data: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "fuzzy"

    def fit(self, df: pd.DataFrame) -> FuzzyStrategy:
        """Prepare normalized numeric data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self._normalized_data = None
            return self

        numeric_data = df[numeric_cols].values

        # Normalize
        data_min = np.nanmin(numeric_data, axis=0)
        data_max = np.nanmax(numeric_data, axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        normalized = (numeric_data - data_min) / data_range

        # Handle NaN
        self._normalized_data = np.nan_to_num(normalized, nan=0.0)

        return self

    def detect(
        self,
        df: pd.DataFrame,
        similarity_threshold: float,
        seen_pairs: set[tuple[int, int]],
    ) -> list[DuplicateCandidate]:
        """Find near-duplicates via cosine similarity."""
        if self._normalized_data is None:
            return []

        n_samples = len(df)
        if n_samples > self.max_samples:
            # Skip for very large datasets
            return []

        candidates = []
        sim_matrix = cosine_similarity(self._normalized_data)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if sim_matrix[i, j] >= similarity_threshold:
                    pair = (i, j)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidates.append(
                            DuplicateCandidate(
                                index1=i,
                                index2=j,
                                similarity=float(sim_matrix[i, j]),
                                is_exact=False,
                                method=self.name,
                            )
                        )

        return candidates


class EmbeddingStrategy(DuplicateStrategy):
    """Semantic duplicate detection using text embeddings.

    Uses sentence transformers to compute embeddings and find
    semantically similar text entries.
    """

    def __init__(
        self,
        text_column: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_samples: int = 10000,
    ):
        """Initialize embedding strategy.

        Args:
            text_column: Column containing text for embedding
            embedding_model: Sentence transformer model name
            batch_size: Batch size for embedding generation
            max_samples: Maximum samples for full pairwise comparison
        """
        self.text_column = text_column
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_samples = max_samples
        self._embeddings: np.ndarray | None = None
        self._encoder: Any = None

    @property
    def name(self) -> str:
        return "embedding"

    def fit(self, df: pd.DataFrame) -> EmbeddingStrategy:
        """Compute text embeddings."""
        text_col = self._find_text_column(df)
        if text_col is None:
            self._embeddings = None
            return self

        if not HAS_SENTENCE_TRANSFORMERS:
            raise DependencyError(
                "sentence-transformers",
                "text",
                feature="embedding-based duplicate detection",
            )

        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)

        texts = df[text_col].fillna("").astype(str).tolist()
        self._embeddings = self._encoder.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )

        return self

    def _find_text_column(self, df: pd.DataFrame) -> str | None:
        """Find appropriate text column."""
        if self.text_column is not None:
            return self.text_column if self.text_column in df.columns else None

        # Auto-detect text column
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if len(sample) > 0 and sample.astype(str).str.len().mean() > 20:
                    return col
        return None

    def detect(
        self,
        df: pd.DataFrame,
        similarity_threshold: float,
        seen_pairs: set[tuple[int, int]],
    ) -> list[DuplicateCandidate]:
        """Find semantic duplicates via embedding similarity."""
        if self._embeddings is None:
            return []

        n_samples = len(self._embeddings)
        if n_samples > self.max_samples:
            return []

        candidates = []
        sim_matrix = cosine_similarity(self._embeddings)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if sim_matrix[i, j] >= similarity_threshold:
                    pair = (i, j)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidates.append(
                            DuplicateCandidate(
                                index1=i,
                                index2=j,
                                similarity=float(sim_matrix[i, j]),
                                is_exact=False,
                                method=self.name,
                            )
                        )

        return candidates


def create_duplicate_strategy(
    method: str,
    hash_columns: list[str] | None = None,
    text_column: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> DuplicateStrategy:
    """Factory function to create duplicate detection strategy.

    Args:
        method: One of 'hash', 'fuzzy', 'embedding'
        hash_columns: Columns for hash strategy
        text_column: Column for embedding strategy
        embedding_model: Model for embedding strategy
        batch_size: Batch size for embedding strategy

    Returns:
        Appropriate duplicate strategy instance
    """
    if method == "hash":
        return HashStrategy(hash_columns=hash_columns)
    elif method == "fuzzy":
        return FuzzyStrategy()
    elif method == "embedding":
        return EmbeddingStrategy(
            text_column=text_column,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unknown duplicate detection method: {method}. "
            f"Available: hash, fuzzy, embedding"
        )
