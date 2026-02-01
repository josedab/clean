"""Vector database integration for embedding-based analysis.

This module provides native connectors for vector databases like Pinecone,
Weaviate, Milvus, and Qdrant for efficient embedding-based duplicate and
outlier detection.

Example:
    >>> from clean.vectordb import VectorStore, PineconeStore
    >>>
    >>> store = PineconeStore(api_key="...", index_name="embeddings")
    >>> await store.connect()
    >>> duplicates = await store.find_duplicates(embeddings, threshold=0.95)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


class VectorDBBackend(Enum):
    """Supported vector database backends."""

    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    CHROMADB = "chromadb"
    MEMORY = "memory"  # For testing


class DistanceMetric(Enum):
    """Distance metrics for similarity search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: np.ndarray | None = None


@dataclass
class DuplicatePair:
    """A pair of duplicate vectors."""

    id1: str
    id2: str
    similarity: float
    metadata1: dict[str, Any] = field(default_factory=dict)
    metadata2: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierResult:
    """Outlier detection result."""

    id: str
    outlier_score: float
    avg_distance_to_neighbors: float
    nearest_neighbor_distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        pass

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors.

        Args:
            ids: Vector IDs
            vectors: Embedding vectors (n_samples, n_dims)
            metadata: Optional metadata per vector

        Returns:
            Number of vectors upserted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        pass

    async def find_duplicates(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        threshold: float = 0.95,
        batch_size: int = 100,
    ) -> list[DuplicatePair]:
        """Find duplicate vectors based on similarity threshold.

        Args:
            vectors: Embedding vectors to check
            ids: Optional vector IDs (default: indices)
            threshold: Similarity threshold for duplicates
            batch_size: Batch size for queries

        Returns:
            List of duplicate pairs
        """
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]

        # First upsert all vectors
        await self.upsert(ids, vectors)

        duplicates = []
        seen_pairs = set()

        for i in range(0, len(vectors), batch_size):
            batch_end = min(i + batch_size, len(vectors))
            batch_vectors = vectors[i:batch_end]
            batch_ids = ids[i:batch_end]

            for j, (vec, vec_id) in enumerate(zip(batch_vectors, batch_ids)):
                results = await self.search(vec, top_k=10)

                for result in results:
                    if result.id == vec_id:
                        continue  # Skip self

                    if result.score >= threshold:
                        pair_key = tuple(sorted([vec_id, result.id]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            duplicates.append(DuplicatePair(
                                id1=vec_id,
                                id2=result.id,
                                similarity=result.score,
                            ))

        return duplicates

    async def find_outliers(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        k_neighbors: int = 10,
        contamination: float = 0.1,
    ) -> list[OutlierResult]:
        """Find outlier vectors based on neighbor distances.

        Args:
            vectors: Embedding vectors to check
            ids: Optional vector IDs
            k_neighbors: Number of neighbors for LOF-style detection
            contamination: Expected fraction of outliers

        Returns:
            List of outlier results
        """
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]

        # Upsert all vectors
        await self.upsert(ids, vectors)

        # Calculate distances to neighbors for each vector
        outlier_scores = []

        for i, (vec, vec_id) in enumerate(zip(vectors, ids)):
            results = await self.search(vec, top_k=k_neighbors + 1)

            # Exclude self
            distances = [
                1 - r.score  # Convert similarity to distance
                for r in results
                if r.id != vec_id
            ][:k_neighbors]

            if distances:
                avg_distance = np.mean(distances)
                nearest_distance = min(distances)
            else:
                avg_distance = float("inf")
                nearest_distance = float("inf")

            outlier_scores.append(OutlierResult(
                id=vec_id,
                outlier_score=avg_distance,
                avg_distance_to_neighbors=avg_distance,
                nearest_neighbor_distance=nearest_distance,
            ))

        # Sort by outlier score and return top outliers
        outlier_scores.sort(key=lambda x: x.outlier_score, reverse=True)
        n_outliers = max(1, int(len(outlier_scores) * contamination))

        return outlier_scores[:n_outliers]


class MemoryVectorStore(VectorStore):
    """In-memory vector store for testing and small datasets."""

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        """Initialize in-memory store.

        Args:
            metric: Distance metric to use
        """
        self.metric = metric
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect (no-op for memory store)."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect (no-op for memory store)."""
        self._connected = False

    async def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors."""
        metadata = metadata or [{} for _ in ids]

        for i, (vec_id, vec, meta) in enumerate(zip(ids, vectors, metadata)):
            self._vectors[vec_id] = vec
            self._metadata[vec_id] = meta

        return len(ids)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if not self._vectors:
            return []

        results = []

        for vec_id, vec in self._vectors.items():
            # Apply filter
            if filter:
                meta = self._metadata.get(vec_id, {})
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue

            # Calculate similarity
            similarity = self._calculate_similarity(query_vector, vec)

            results.append(SearchResult(
                id=vec_id,
                score=similarity,
                metadata=self._metadata.get(vec_id, {}),
            ))

        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        deleted = 0
        for vec_id in ids:
            if vec_id in self._vectors:
                del self._vectors[vec_id]
                del self._metadata[vec_id]
                deleted += 1
        return deleted

    def _calculate_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Calculate similarity between two vectors."""
        if self.metric == DistanceMetric.COSINE:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        elif self.metric == DistanceMetric.DOT_PRODUCT:
            return float(np.dot(vec1, vec2))

        elif self.metric == DistanceMetric.EUCLIDEAN:
            distance = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + distance))

        return 0.0


class PineconeStore(VectorStore):
    """Pinecone vector database connector.

    Requires: pip install pinecone-client
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "gcp-starter",
        namespace: str = "",
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """Initialize Pinecone store.

        Args:
            api_key: Pinecone API key
            index_name: Index name
            environment: Pinecone environment
            namespace: Optional namespace
            metric: Distance metric
        """
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.namespace = namespace
        self.metric = metric
        self._index = None

    async def connect(self) -> None:
        """Connect to Pinecone."""
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client is required. Install with: pip install pinecone-client"
            )

        pinecone.init(api_key=self.api_key, environment=self.environment)
        self._index = pinecone.Index(self.index_name)

    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._index = None

    async def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors."""
        if self._index is None:
            raise RuntimeError("Not connected")

        metadata = metadata or [{} for _ in ids]

        # Prepare vectors for upsert
        upsert_data = [
            (id_, vec.tolist(), meta)
            for id_, vec, meta in zip(ids, vectors, metadata)
        ]

        # Batch upsert
        batch_size = 100
        total = 0

        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=self.namespace)
            total += len(batch)

        return total

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if self._index is None:
            raise RuntimeError("Not connected")

        response = self._index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True,
        )

        return [
            SearchResult(
                id=match["id"],
                score=match["score"],
                metadata=match.get("metadata", {}),
            )
            for match in response["matches"]
        ]

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if self._index is None:
            raise RuntimeError("Not connected")

        self._index.delete(ids=ids, namespace=self.namespace)
        return len(ids)


class WeaviateStore(VectorStore):
    """Weaviate vector database connector.

    Requires: pip install weaviate-client
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        class_name: str = "CleanVector",
    ):
        """Initialize Weaviate store.

        Args:
            url: Weaviate server URL
            api_key: Optional API key
            class_name: Weaviate class name
        """
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self._client = None

    async def connect(self) -> None:
        """Connect to Weaviate."""
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "weaviate-client is required. Install with: pip install weaviate-client"
            )

        auth = None
        if self.api_key:
            auth = weaviate.AuthApiKey(api_key=self.api_key)

        self._client = weaviate.Client(url=self.url, auth_client_secret=auth)

        # Create class if not exists
        if not self._client.schema.exists(self.class_name):
            class_obj = {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "external_id", "dataType": ["string"]},
                    {"name": "metadata", "dataType": ["text"]},
                ],
            }
            self._client.schema.create_class(class_obj)

    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        self._client = None

    async def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors."""
        if self._client is None:
            raise RuntimeError("Not connected")

        import json

        metadata = metadata or [{} for _ in ids]

        with self._client.batch as batch:
            for id_, vec, meta in zip(ids, vectors, metadata):
                batch.add_data_object(
                    data_object={
                        "external_id": id_,
                        "metadata": json.dumps(meta),
                    },
                    class_name=self.class_name,
                    vector=vec.tolist(),
                    uuid=id_,
                )

        return len(ids)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if self._client is None:
            raise RuntimeError("Not connected")

        import json

        query = (
            self._client.query
            .get(self.class_name, ["external_id", "metadata"])
            .with_near_vector({"vector": query_vector.tolist()})
            .with_limit(top_k)
            .with_additional(["certainty", "id"])
        )

        result = query.do()

        items = result.get("data", {}).get("Get", {}).get(self.class_name, [])

        return [
            SearchResult(
                id=item["external_id"],
                score=item["_additional"]["certainty"],
                metadata=json.loads(item.get("metadata", "{}")),
            )
            for item in items
        ]

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if self._client is None:
            raise RuntimeError("Not connected")

        for id_ in ids:
            self._client.data_object.delete(uuid=id_, class_name=self.class_name)

        return len(ids)


class MilvusStore(VectorStore):
    """Milvus vector database connector.

    Requires: pip install pymilvus
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "clean_vectors",
        dim: int = 768,
    ):
        """Initialize Milvus store.

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Collection name
            dim: Vector dimension
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self._collection = None

    async def connect(self) -> None:
        """Connect to Milvus."""
        try:
            from pymilvus import Collection, connections, utility
            from pymilvus import CollectionSchema, FieldSchema, DataType
        except ImportError:
            raise ImportError(
                "pymilvus is required. Install with: pip install pymilvus"
            )

        connections.connect(host=self.host, port=self.port)

        # Create collection if not exists
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields=fields)
            self._collection = Collection(name=self.collection_name, schema=schema)

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self._collection = Collection(name=self.collection_name)

        self._collection.load()

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self._collection:
            self._collection.release()
        from pymilvus import connections
        connections.disconnect("default")

    async def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors."""
        if self._collection is None:
            raise RuntimeError("Not connected")

        metadata = metadata or [{} for _ in ids]

        # Delete existing if any
        self._collection.delete(expr=f"id in {ids}")

        # Insert new
        entities = [
            ids,
            vectors.tolist(),
            metadata,
        ]

        self._collection.insert(entities)
        self._collection.flush()

        return len(ids)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if self._collection is None:
            raise RuntimeError("Not connected")

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self._collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["metadata"],
        )

        return [
            SearchResult(
                id=hit.id,
                score=hit.score,
                metadata=hit.entity.get("metadata", {}),
            )
            for hit in results[0]
        ]

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if self._collection is None:
            raise RuntimeError("Not connected")

        self._collection.delete(expr=f"id in {ids}")
        return len(ids)


class QdrantStore(VectorStore):
    """Qdrant vector database connector.

    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "clean_vectors",
        dim: int = 768,
        api_key: str | None = None,
    ):
        """Initialize Qdrant store.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Collection name
            dim: Vector dimension
            api_key: Optional API key
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.api_key = api_key
        self._client = None

    async def connect(self) -> None:
        """Connect to Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self._client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )

        # Create collection if not exists
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            )

    async def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        self._client = None

    async def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors."""
        if self._client is None:
            raise RuntimeError("Not connected")

        from qdrant_client.models import PointStruct

        metadata = metadata or [{} for _ in ids]

        points = [
            PointStruct(
                id=i,  # Qdrant prefers numeric IDs
                vector=vec.tolist(),
                payload={"external_id": id_, **meta},
            )
            for i, (id_, vec, meta) in enumerate(zip(ids, vectors, metadata))
        ]

        self._client.upsert(collection_name=self.collection_name, points=points)

        return len(points)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if self._client is None:
            raise RuntimeError("Not connected")

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                id=hit.payload.get("external_id", str(hit.id)),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "external_id"},
            )
            for hit in results
        ]

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if self._client is None:
            raise RuntimeError("Not connected")

        from qdrant_client.models import Filter, FieldCondition, MatchAny

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="external_id",
                        match=MatchAny(any=ids),
                    ),
                ],
            ),
        )

        return len(ids)


async def create_vector_store(
    backend: VectorDBBackend | str,
    **kwargs: Any,
) -> VectorStore:
    """Create a vector store instance.

    Args:
        backend: Vector database backend
        **kwargs: Backend-specific configuration

    Returns:
        Configured VectorStore instance
    """
    if isinstance(backend, str):
        backend = VectorDBBackend(backend)

    if backend == VectorDBBackend.MEMORY:
        return MemoryVectorStore(**kwargs)
    elif backend == VectorDBBackend.PINECONE:
        return PineconeStore(**kwargs)
    elif backend == VectorDBBackend.WEAVIATE:
        return WeaviateStore(**kwargs)
    elif backend == VectorDBBackend.MILVUS:
        return MilvusStore(**kwargs)
    elif backend == VectorDBBackend.QDRANT:
        return QdrantStore(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


class VectorQualityAnalyzer:
    """Analyze data quality using vector embeddings and vector database."""

    def __init__(
        self,
        store: VectorStore,
        duplicate_threshold: float = 0.95,
        outlier_contamination: float = 0.1,
    ):
        """Initialize analyzer.

        Args:
            store: Vector store to use
            duplicate_threshold: Similarity threshold for duplicates
            outlier_contamination: Expected fraction of outliers
        """
        self.store = store
        self.duplicate_threshold = duplicate_threshold
        self.outlier_contamination = outlier_contamination

    async def analyze(
        self,
        embeddings: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run complete quality analysis on embeddings.

        Args:
            embeddings: Vector embeddings (n_samples, n_dims)
            ids: Optional vector IDs
            metadata: Optional metadata per vector

        Returns:
            Analysis results
        """
        if ids is None:
            ids = [str(i) for i in range(len(embeddings))]

        # Connect if not connected
        await self.store.connect()

        # Find duplicates
        duplicates = await self.store.find_duplicates(
            embeddings, ids, threshold=self.duplicate_threshold
        )

        # Find outliers
        outliers = await self.store.find_outliers(
            embeddings, ids, contamination=self.outlier_contamination
        )

        return {
            "n_samples": len(embeddings),
            "n_duplicates": len(duplicates),
            "duplicate_pairs": [
                {"id1": d.id1, "id2": d.id2, "similarity": d.similarity}
                for d in duplicates
            ],
            "n_outliers": len(outliers),
            "outliers": [
                {"id": o.id, "score": o.outlier_score}
                for o in outliers
            ],
            "duplicate_rate": len(duplicates) * 2 / len(embeddings) if embeddings.size > 0 else 0,
            "outlier_rate": len(outliers) / len(embeddings) if embeddings.size > 0 else 0,
        }
