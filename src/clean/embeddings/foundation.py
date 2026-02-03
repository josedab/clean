"""Foundation Model Embeddings for production-grade semantic similarity.

This module provides pluggable embedding backends using foundation model APIs
(OpenAI, Cohere, Voyage) for 10x better semantic similarity vs local models.

Example:
    >>> from clean.embeddings.foundation import (
    ...     OpenAIEmbedder, CohereEmbedder, VoyageEmbedder, create_embedder
    ... )
    >>>
    >>> # Use OpenAI embeddings
    >>> embedder = OpenAIEmbedder(api_key="sk-...")
    >>> embeddings = embedder.encode(["hello world", "goodbye world"])
    >>>
    >>> # Use factory function
    >>> embedder = create_embedder("openai", api_key="sk-...")
    >>>
    >>> # Cached embedder for efficiency
    >>> from clean.embeddings.foundation import CachedEmbedder
    >>> cached = CachedEmbedder(embedder, cache_dir=".embeddings_cache")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from clean.embeddings.base import BaseEmbedder
from clean.exceptions import CleanError, ConfigurationError, DependencyError

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Supported foundation model embedding providers."""

    OPENAI = "openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    LOCAL = "local"  # sentence-transformers fallback


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations."""

    total_tokens: int = 0
    total_requests: int = 0
    total_cost_usd: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list)

    def record_request(
        self, tokens: int, cost: float, latency_ms: float, cached: bool = False
    ) -> None:
        """Record a request's statistics."""
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.total_tokens += tokens
            self.total_requests += 1
            self.total_cost_usd += cost
            self._latencies.append(latency_ms)
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "total_cost_usd": self.total_cost_usd,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "avg_latency_ms": self.avg_latency_ms,
        }


class FoundationEmbedder(BaseEmbedder, ABC):
    """Abstract base class for foundation model embedders."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize foundation model embedder.

        Args:
            api_key: API key (or set via environment variable)
            model: Model name to use
            batch_size: Batch size for API calls
            max_retries: Maximum retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stats = EmbeddingStats()

    @abstractmethod
    def _get_embeddings_batch(self, texts: list[str]) -> tuple[np.ndarray, int, float]:
        """Get embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Tuple of (embeddings, tokens_used, cost_usd)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        pass

    @property
    @abstractmethod
    def provider(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        pass

    def encode(
        self,
        data: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            data: List of text strings
            batch_size: Override default batch size
            show_progress: Show progress bar

        Returns:
            Embedding matrix of shape (n_samples, embedding_dim)
        """
        if not data:
            return np.array([])

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            start_time = time.time()

            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    embeddings, tokens, cost = self._get_embeddings_batch(batch)
                    latency_ms = (time.time() - start_time) * 1000
                    self.stats.record_request(tokens, cost, latency_ms)
                    all_embeddings.append(embeddings)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise CleanError(f"Failed to get embeddings after {self.max_retries} attempts: {e}") from e
                    logger.warning(f"Embedding request failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))

            if show_progress:
                logger.info(f"Embedded {min(i + batch_size, len(data))}/{len(data)} texts")

        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query text.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.encode([query])[0]

    def get_stats(self) -> EmbeddingStats:
        """Get usage statistics."""
        return self.stats

    def estimate_cost(self, n_texts: int, avg_tokens_per_text: int = 100) -> float:
        """Estimate cost for embedding texts.

        Args:
            n_texts: Number of texts to embed
            avg_tokens_per_text: Average tokens per text

        Returns:
            Estimated cost in USD
        """
        # Override in subclasses with provider-specific pricing
        return 0.0


class OpenAIEmbedder(FoundationEmbedder):
    """OpenAI embeddings (text-embedding-3-small/large, ada-002)."""

    DEFAULT_MODEL = "text-embedding-3-small"

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            model: Model name (default: text-embedding-3-small)
            dimensions: Output dimensions (for text-embedding-3-* models)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model=model or self.DEFAULT_MODEL,
            **kwargs,
        )
        self.dimensions = dimensions
        self._client: Any = None

        if not self.api_key:
            raise ConfigurationError("OpenAI API key required (set OPENAI_API_KEY or pass api_key)")

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise DependencyError("openai", "pip install openai", "OpenAI embeddings") from e

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _get_embeddings_batch(self, texts: list[str]) -> tuple[np.ndarray, int, float]:
        """Get embeddings from OpenAI API."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }
        if self.dimensions and self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(**kwargs)

        embeddings = np.array([d.embedding for d in response.data])
        tokens = response.usage.total_tokens

        # Calculate cost
        price_per_million = self.PRICING.get(self.model, 0.02)
        cost = (tokens / 1_000_000) * price_per_million

        return embeddings, tokens, cost

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if self.dimensions:
            return self.dimensions
        return self.DIMENSIONS.get(self.model, 1536)

    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.OPENAI

    def estimate_cost(self, n_texts: int, avg_tokens_per_text: int = 100) -> float:
        """Estimate cost for embedding texts."""
        total_tokens = n_texts * avg_tokens_per_text
        price_per_million = self.PRICING.get(self.model, 0.02)
        return (total_tokens / 1_000_000) * price_per_million


class CohereEmbedder(FoundationEmbedder):
    """Cohere embeddings (embed-english-v3.0, embed-multilingual-v3.0)."""

    DEFAULT_MODEL = "embed-english-v3.0"

    # Pricing per 1M tokens
    PRICING = {
        "embed-english-v3.0": 0.10,
        "embed-multilingual-v3.0": 0.10,
        "embed-english-light-v3.0": 0.10,
        "embed-multilingual-light-v3.0": 0.10,
    }

    DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        input_type: str = "search_document",
        **kwargs: Any,
    ):
        """Initialize Cohere embedder.

        Args:
            api_key: Cohere API key (or CO_API_KEY env var)
            model: Model name
            input_type: Input type (search_document, search_query, classification, clustering)
            **kwargs: Additional arguments
        """
        super().__init__(
            api_key=api_key or os.environ.get("CO_API_KEY"),
            model=model or self.DEFAULT_MODEL,
            **kwargs,
        )
        self.input_type = input_type
        self._client: Any = None

        if not self.api_key:
            raise ConfigurationError("Cohere API key required (set CO_API_KEY or pass api_key)")

    def _get_client(self) -> Any:
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError as e:
                raise DependencyError("cohere", "pip install cohere", "Cohere embeddings") from e

            self._client = cohere.Client(self.api_key)
        return self._client

    def _get_embeddings_batch(self, texts: list[str]) -> tuple[np.ndarray, int, float]:
        """Get embeddings from Cohere API."""
        client = self._get_client()

        response = client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
        )

        embeddings = np.array(response.embeddings)

        # Estimate tokens (Cohere doesn't always return token count)
        tokens = sum(len(t.split()) * 1.3 for t in texts)  # rough estimate
        price_per_million = self.PRICING.get(self.model, 0.10)
        cost = (tokens / 1_000_000) * price_per_million

        return embeddings, int(tokens), cost

    @property
    def embedding_dim(self) -> int:
        return self.DIMENSIONS.get(self.model, 1024)

    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.COHERE


class VoyageEmbedder(FoundationEmbedder):
    """Voyage AI embeddings (voyage-2, voyage-large-2, voyage-code-2)."""

    DEFAULT_MODEL = "voyage-2"

    # Pricing per 1M tokens
    PRICING = {
        "voyage-2": 0.10,
        "voyage-large-2": 0.12,
        "voyage-code-2": 0.12,
        "voyage-lite-02-instruct": 0.02,
    }

    DIMENSIONS = {
        "voyage-2": 1024,
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
        "voyage-lite-02-instruct": 1024,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        input_type: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Voyage embedder.

        Args:
            api_key: Voyage API key (or VOYAGE_API_KEY env var)
            model: Model name
            input_type: Input type (query, document)
            **kwargs: Additional arguments
        """
        super().__init__(
            api_key=api_key or os.environ.get("VOYAGE_API_KEY"),
            model=model or self.DEFAULT_MODEL,
            **kwargs,
        )
        self.input_type = input_type
        self._client: Any = None

        if not self.api_key:
            raise ConfigurationError("Voyage API key required (set VOYAGE_API_KEY or pass api_key)")

    def _get_client(self) -> Any:
        """Get or create Voyage client."""
        if self._client is None:
            try:
                import voyageai
            except ImportError as e:
                raise DependencyError("voyageai", "pip install voyageai", "Voyage embeddings") from e

            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def _get_embeddings_batch(self, texts: list[str]) -> tuple[np.ndarray, int, float]:
        """Get embeddings from Voyage API."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "texts": texts,
            "model": self.model,
        }
        if self.input_type:
            kwargs["input_type"] = self.input_type

        response = client.embed(**kwargs)

        embeddings = np.array(response.embeddings)
        tokens = response.total_tokens

        price_per_million = self.PRICING.get(self.model, 0.10)
        cost = (tokens / 1_000_000) * price_per_million

        return embeddings, tokens, cost

    @property
    def embedding_dim(self) -> int:
        return self.DIMENSIONS.get(self.model, 1024)

    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.VOYAGE


class CachedEmbedder(BaseEmbedder):
    """Wrapper that caches embeddings to avoid re-computation.

    Uses SQLite for persistent caching across sessions.
    """

    def __init__(
        self,
        embedder: FoundationEmbedder,
        cache_dir: str | Path | None = None,
        cache_name: str = "embeddings_cache",
    ):
        """Initialize cached embedder.

        Args:
            embedder: Underlying embedder to cache
            cache_dir: Directory for cache (default: .clean_cache)
            cache_name: Name for cache database
        """
        self.embedder = embedder
        self.cache_dir = Path(cache_dir or ".clean_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / f"{cache_name}.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                model TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)")
        conn.commit()
        conn.close()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached(self, text_hashes: list[str], model: str) -> dict[str, np.ndarray]:
        """Get cached embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cached = {}
        for h in text_hashes:
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE text_hash = ? AND model = ?",
                (h, model),
            )
            row = cursor.fetchone()
            if row:
                cached[h] = np.frombuffer(row[0], dtype=np.float32)

        conn.close()
        return cached

    def _cache_embeddings(
        self, text_hashes: list[str], embeddings: np.ndarray, model: str
    ) -> None:
        """Cache embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for h, emb in zip(text_hashes, embeddings):
            cursor.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding) VALUES (?, ?, ?)",
                (h, model, emb.astype(np.float32).tobytes()),
            )

        conn.commit()
        conn.close()

    def encode(
        self,
        data: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode with caching."""
        if not data:
            return np.array([])

        model = self.embedder.model or "default"
        text_hashes = [self._hash_text(t) for t in data]

        # Get cached embeddings
        cached = self._get_cached(text_hashes, model)

        # Find texts that need embedding
        uncached_indices = [i for i, h in enumerate(text_hashes) if h not in cached]
        uncached_texts = [data[i] for i in uncached_indices]

        # Get new embeddings
        if uncached_texts:
            new_embeddings = self.embedder.encode(
                uncached_texts, batch_size=batch_size, show_progress=show_progress
            )
            uncached_hashes = [text_hashes[i] for i in uncached_indices]
            self._cache_embeddings(uncached_hashes, new_embeddings, model)

            # Add to cached dict
            for h, emb in zip(uncached_hashes, new_embeddings):
                cached[h] = emb

        # Reconstruct in original order
        embeddings = np.array([cached[h] for h in text_hashes])

        # Update stats
        if hasattr(self.embedder, "stats"):
            self.embedder.stats.cache_hits += len(data) - len(uncached_texts)

        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self.embedder.embedding_dim

    def get_stats(self) -> EmbeddingStats:
        """Get underlying embedder stats."""
        return self.embedder.get_stats()

    def clear_cache(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of entries cleared
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        return count


def create_embedder(
    provider: str | EmbeddingProvider,
    api_key: str | None = None,
    model: str | None = None,
    cached: bool = True,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> BaseEmbedder:
    """Factory function to create an embedder.

    Args:
        provider: Embedding provider (openai, cohere, voyage, local)
        api_key: API key for the provider
        model: Model name
        cached: Whether to use caching
        cache_dir: Cache directory
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedder instance

    Example:
        >>> embedder = create_embedder("openai", api_key="sk-...")
        >>> embeddings = embedder.encode(["hello", "world"])
    """
    if isinstance(provider, str):
        try:
            provider = EmbeddingProvider(provider.lower())
        except ValueError:
            valid = [p.value for p in EmbeddingProvider]
            raise ConfigurationError(
                f"Unknown embedding provider: {provider}. "
                f"Valid options: {valid}"
            )

    embedder: FoundationEmbedder

    if provider == EmbeddingProvider.OPENAI:
        embedder = OpenAIEmbedder(api_key=api_key, model=model, **kwargs)
    elif provider == EmbeddingProvider.COHERE:
        embedder = CohereEmbedder(api_key=api_key, model=model, **kwargs)
    elif provider == EmbeddingProvider.VOYAGE:
        embedder = VoyageEmbedder(api_key=api_key, model=model, **kwargs)
    elif provider == EmbeddingProvider.LOCAL:
        # Fall back to sentence-transformers
        from clean.embeddings.text import TextEmbedder

        return TextEmbedder(model_name=model)
    else:
        raise ConfigurationError(f"Unknown embedding provider: {provider}")

    if cached:
        return CachedEmbedder(embedder, cache_dir=cache_dir)

    return embedder


__all__ = [
    "EmbeddingProvider",
    "EmbeddingStats",
    "FoundationEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    "CachedEmbedder",
    "create_embedder",
]
