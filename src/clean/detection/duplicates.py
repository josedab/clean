"""Duplicate detection using hashing and embeddings."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from clean.core.types import DuplicatePair
from clean.detection.base import BaseDetector, DetectorResult
from clean.exceptions import DependencyError

# Optional imports
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore[misc, assignment]


class DuplicateDetector(BaseDetector):
    """Detect duplicate and near-duplicate samples.

    Supports multiple detection methods:
    - hash: Exact duplicates via content hashing
    - fuzzy: Fuzzy matching for tabular data
    - embedding: Semantic similarity using embeddings
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        similarity_threshold: float = 0.9,
        hash_columns: list[str] | None = None,
        text_column: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        """Initialize the duplicate detector.

        Args:
            methods: Detection methods to use ('hash', 'fuzzy', 'embedding')
            similarity_threshold: Threshold for near-duplicate detection (0-1)
            hash_columns: Columns to use for hashing (None = all)
            text_column: Column containing text for embedding similarity
            embedding_model: Sentence transformer model name
            batch_size: Batch size for embedding generation
        """
        super().__init__(
            methods=methods,
            similarity_threshold=similarity_threshold,
            hash_columns=hash_columns,
            text_column=text_column,
        )
        self.methods = methods or ["hash"]
        self.similarity_threshold = similarity_threshold
        self.hash_columns = hash_columns
        self.text_column = text_column
        self.embedding_model = embedding_model
        self.batch_size = batch_size

        self._embeddings: np.ndarray | None = None
        self._hashes: list[str] | None = None
        self._encoder: Any = None

    def fit(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DuplicateDetector:
        """Fit the detector by computing hashes and/or embeddings.

        Args:
            features: Feature data
            labels: Not used for duplicate detection

        Returns:
            Self for chaining
        """
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        # Compute hashes
        if "hash" in self.methods:
            self._compute_hashes(features)

        # Compute embeddings for text
        if "embedding" in self.methods:
            self._compute_embeddings(features)

        self._is_fitted = True
        return self

    def _compute_hashes(self, df: pd.DataFrame) -> None:
        """Compute content hashes for each row."""
        cols = self.hash_columns or list(df.columns)
        cols = [c for c in cols if c in df.columns]

        self._hashes = []
        for _, row in df[cols].iterrows():
            content = "|".join(str(v) for v in row.values)
            hash_val = hashlib.md5(content.encode()).hexdigest()
            self._hashes.append(hash_val)

    def _compute_embeddings(self, df: pd.DataFrame) -> None:
        """Compute text embeddings."""
        if self.text_column is None:
            # Find text column
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0 and sample.astype(str).str.len().mean() > 20:
                        self.text_column = col
                        break

        if self.text_column is None:
            return

        if not HAS_SENTENCE_TRANSFORMERS:
            raise DependencyError(
                "sentence-transformers",
                "text",
                feature="embedding-based duplicate detection",
            )

        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)

        texts = df[self.text_column].fillna("").astype(str).tolist()
        self._embeddings = self._encoder.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )

    def detect(
        self, features: pd.DataFrame | np.ndarray, labels: np.ndarray | None = None
    ) -> DetectorResult:
        """Detect duplicates in the data.

        Args:
            features: Feature data
            labels: Not used

        Returns:
            DetectorResult with DuplicatePair objects
        """
        self._check_fitted()

        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        duplicates: list[DuplicatePair] = []
        seen_pairs: set[tuple[int, int]] = set()

        # Find exact duplicates via hash
        if "hash" in self.methods and self._hashes:
            hash_to_indices: dict[str, list[int]] = {}
            for idx, h in enumerate(self._hashes):
                if h not in hash_to_indices:
                    hash_to_indices[h] = []
                hash_to_indices[h].append(idx)

            for indices in hash_to_indices.values():
                if len(indices) > 1:
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            pair = (indices[i], indices[j])
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                duplicates.append(
                                    DuplicatePair(
                                        index1=indices[i],
                                        index2=indices[j],
                                        similarity=1.0,
                                        is_exact=True,
                                    )
                                )

        # Find near-duplicates via fuzzy matching on numeric columns
        if "fuzzy" in self.methods:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_data = features[numeric_cols].values

                # Normalize
                data_min = np.nanmin(numeric_data, axis=0)
                data_max = np.nanmax(numeric_data, axis=0)
                data_range = data_max - data_min
                data_range[data_range == 0] = 1
                normalized = (numeric_data - data_min) / data_range

                # Handle NaN
                normalized = np.nan_to_num(normalized, nan=0.0)

                # Compute pairwise similarities (for small datasets)
                n_samples = len(features)
                if n_samples <= 10000:
                    sim_matrix = cosine_similarity(normalized)
                    for i in range(n_samples):
                        for j in range(i + 1, n_samples):
                            if sim_matrix[i, j] >= self.similarity_threshold:
                                pair = (i, j)
                                if pair not in seen_pairs:
                                    seen_pairs.add(pair)
                                    duplicates.append(
                                        DuplicatePair(
                                            index1=i,
                                            index2=j,
                                            similarity=float(sim_matrix[i, j]),
                                            is_exact=False,
                                        )
                                    )

        # Find semantic duplicates via embeddings
        if "embedding" in self.methods and self._embeddings is not None:
            embeddings = self._embeddings
            n_samples = len(embeddings)

            if n_samples <= 10000:
                sim_matrix = cosine_similarity(embeddings)
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        if sim_matrix[i, j] >= self.similarity_threshold:
                            pair = (i, j)
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                duplicates.append(
                                    DuplicatePair(
                                        index1=i,
                                        index2=j,
                                        similarity=float(sim_matrix[i, j]),
                                        is_exact=False,
                                    )
                                )

        # Sort by similarity
        duplicates.sort(key=lambda d: d.similarity, reverse=True)

        # Count statistics
        n_exact = sum(1 for d in duplicates if d.is_exact)
        n_near_high = sum(
            1 for d in duplicates if not d.is_exact and d.similarity >= 0.95
        )
        n_near_med = sum(
            1
            for d in duplicates
            if not d.is_exact and 0.9 <= d.similarity < 0.95
        )

        metadata = {
            "methods": self.methods,
            "similarity_threshold": self.similarity_threshold,
            "n_samples": len(features),
            "n_duplicate_pairs": len(duplicates),
            "n_exact": n_exact,
            "n_near_high": n_near_high,
            "n_near_medium": n_near_med,
        }

        return DetectorResult(issues=duplicates, metadata=metadata)

    def get_duplicate_groups(
        self, features: pd.DataFrame | np.ndarray
    ) -> list[list[int]]:
        """Get groups of duplicate samples.

        Args:
            features: Feature data

        Returns:
            List of lists, each containing indices of duplicates
        """
        result = self.detect(features)
        duplicates = result.issues

        # Build adjacency
        from collections import defaultdict

        adj: dict[int, set[int]] = defaultdict(set)
        for dup in duplicates:
            adj[dup.index1].add(dup.index2)
            adj[dup.index2].add(dup.index1)

        # Find connected components
        visited: set[int] = set()
        groups: list[list[int]] = []

        for node in adj:
            if node not in visited:
                group: list[int] = []
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        group.append(curr)
                        stack.extend(adj[curr] - visited)
                groups.append(sorted(group))

        return groups


def find_duplicates(
    features: pd.DataFrame | np.ndarray,
    methods: list[str] | None = None,
    similarity_threshold: float = 0.9,
    **kwargs: Any,
) -> pd.DataFrame:
    """Find duplicates in a dataset.

    Args:
        features: Feature data
        methods: Detection methods
        similarity_threshold: Similarity threshold
        **kwargs: Additional arguments for DuplicateDetector

    Returns:
        DataFrame with duplicate pairs
    """
    detector = DuplicateDetector(
        methods=methods, similarity_threshold=similarity_threshold, **kwargs
    )
    result = detector.fit_detect(features)
    return result.to_dataframe()
