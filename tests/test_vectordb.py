"""Tests for vectordb module."""

import numpy as np
import pytest

from clean.vectordb import (
    DistanceMetric,
    DuplicatePair,
    MemoryVectorStore,
    OutlierResult,
    SearchResult,
    VectorDBBackend,
    VectorStore,
)


class TestEnums:
    """Test enum definitions."""

    def test_vector_db_backend_values(self):
        """Test VectorDBBackend enum values."""
        assert VectorDBBackend.PINECONE.value == "pinecone"
        assert VectorDBBackend.WEAVIATE.value == "weaviate"
        assert VectorDBBackend.MILVUS.value == "milvus"
        assert VectorDBBackend.QDRANT.value == "qdrant"
        assert VectorDBBackend.CHROMADB.value == "chromadb"
        assert VectorDBBackend.MEMORY.value == "memory"

    def test_distance_metric_values(self):
        """Test DistanceMetric enum values."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"


class TestDataclasses:
    """Test dataclass definitions."""

    def test_search_result(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            id="vec1",
            score=0.95,
            metadata={"label": "test"},
        )
        assert result.id == "vec1"
        assert result.score == 0.95
        assert result.metadata == {"label": "test"}
        assert result.vector is None

    def test_search_result_with_vector(self):
        """Test SearchResult with vector."""
        vec = np.array([1.0, 2.0, 3.0])
        result = SearchResult(
            id="vec1",
            score=0.9,
            vector=vec,
        )
        assert result.vector is not None
        np.testing.assert_array_equal(result.vector, vec)

    def test_duplicate_pair(self):
        """Test DuplicatePair dataclass."""
        pair = DuplicatePair(
            id1="vec1",
            id2="vec2",
            similarity=0.98,
            metadata1={"label": "a"},
            metadata2={"label": "b"},
        )
        assert pair.id1 == "vec1"
        assert pair.id2 == "vec2"
        assert pair.similarity == 0.98
        assert pair.metadata1 == {"label": "a"}
        assert pair.metadata2 == {"label": "b"}

    def test_outlier_result(self):
        """Test OutlierResult dataclass."""
        result = OutlierResult(
            id="vec1",
            outlier_score=0.85,
            avg_distance_to_neighbors=0.5,
            nearest_neighbor_distance=0.3,
            metadata={"index": 0},
        )
        assert result.id == "vec1"
        assert result.outlier_score == 0.85
        assert result.avg_distance_to_neighbors == 0.5
        assert result.nearest_neighbor_distance == 0.3
        assert result.metadata == {"index": 0}


class TestMemoryVectorStore:
    """Test MemoryVectorStore implementation."""

    @pytest.fixture
    def store(self):
        """Create a memory store for testing."""
        return MemoryVectorStore(metric=DistanceMetric.COSINE)

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        return np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, store):
        """Test connect and disconnect."""
        assert not store._connected
        await store.connect()
        assert store._connected
        await store.disconnect()
        assert not store._connected

    @pytest.mark.asyncio
    async def test_upsert(self, store, sample_vectors):
        """Test vector upsert."""
        await store.connect()
        
        ids = ["v0", "v1", "v2", "v3"]
        count = await store.upsert(ids, sample_vectors)
        
        assert count == 4
        assert len(store._vectors) == 4
        assert "v0" in store._vectors

    @pytest.mark.asyncio
    async def test_upsert_with_metadata(self, store, sample_vectors):
        """Test upsert with metadata."""
        await store.connect()
        
        ids = ["v0", "v1", "v2", "v3"]
        metadata = [
            {"label": "a"},
            {"label": "a"},
            {"label": "b"},
            {"label": "c"},
        ]
        
        count = await store.upsert(ids, sample_vectors, metadata=metadata)
        
        assert count == 4
        assert store._metadata["v0"] == {"label": "a"}
        assert store._metadata["v2"] == {"label": "b"}

    @pytest.mark.asyncio
    async def test_search_cosine(self, store, sample_vectors):
        """Test cosine similarity search."""
        await store.connect()
        
        ids = ["v0", "v1", "v2", "v3"]
        await store.upsert(ids, sample_vectors)
        
        # Search for vector similar to [1, 0, 0]
        query = np.array([1.0, 0.0, 0.0])
        results = await store.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "v0"  # Exact match
        assert results[0].score == pytest.approx(1.0, abs=0.01)
        assert results[1].id == "v1"  # Similar

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store, sample_vectors):
        """Test search with metadata filter."""
        await store.connect()
        
        ids = ["v0", "v1", "v2", "v3"]
        metadata = [
            {"label": "a"},
            {"label": "a"},
            {"label": "b"},
            {"label": "c"},
        ]
        await store.upsert(ids, sample_vectors, metadata=metadata)
        
        query = np.array([1.0, 0.0, 0.0])
        results = await store.search(query, top_k=10, filter={"label": "a"})
        
        assert len(results) == 2
        assert all(r.metadata["label"] == "a" for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_store(self, store):
        """Test search on empty store."""
        await store.connect()
        
        query = np.array([1.0, 0.0, 0.0])
        results = await store.search(query)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_vectors):
        """Test vector deletion."""
        await store.connect()
        
        ids = ["v0", "v1", "v2", "v3"]
        await store.upsert(ids, sample_vectors)
        
        deleted = await store.delete(["v0", "v1"])
        
        assert deleted == 2
        assert len(store._vectors) == 2
        assert "v0" not in store._vectors
        assert "v2" in store._vectors

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store, sample_vectors):
        """Test deleting nonexistent vectors."""
        await store.connect()
        
        ids = ["v0", "v1"]
        await store.upsert(ids, sample_vectors[:2])
        
        deleted = await store.delete(["v0", "nonexistent"])
        
        assert deleted == 1

    @pytest.mark.asyncio
    async def test_upsert_update(self, store):
        """Test that upsert updates existing vectors."""
        await store.connect()
        
        vec1 = np.array([[1.0, 0.0, 0.0]])
        await store.upsert(["v0"], vec1)
        
        vec2 = np.array([[0.0, 1.0, 0.0]])
        await store.upsert(["v0"], vec2)
        
        assert len(store._vectors) == 1
        np.testing.assert_array_equal(store._vectors["v0"], vec2[0])


class TestMemoryVectorStoreMetrics:
    """Test different distance metrics."""

    @pytest.mark.asyncio
    async def test_euclidean_metric(self):
        """Test Euclidean distance metric."""
        store = MemoryVectorStore(metric=DistanceMetric.EUCLIDEAN)
        await store.connect()
        
        vectors = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        await store.upsert(["v0", "v1", "v2"], vectors)
        
        query = np.array([0.0, 0.0, 0.0])
        results = await store.search(query, top_k=3)
        
        assert results[0].id == "v0"  # Distance 0
        assert results[1].id == "v1"  # Distance 1
        assert results[2].id == "v2"  # Distance 3

    @pytest.mark.asyncio
    async def test_dot_product_metric(self):
        """Test dot product metric."""
        store = MemoryVectorStore(metric=DistanceMetric.DOT_PRODUCT)
        await store.connect()
        
        vectors = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ])
        await store.upsert(["v0", "v1", "v2"], vectors)
        
        query = np.array([1.0, 0.0])
        results = await store.search(query, top_k=3)
        
        assert results[0].id == "v0"  # Dot product = 1.0
        assert results[0].score == pytest.approx(1.0)


class TestVectorStoreFindDuplicates:
    """Test find_duplicates method."""

    @pytest.fixture
    def store(self):
        """Create a memory store."""
        return MemoryVectorStore()

    @pytest.mark.asyncio
    async def test_find_exact_duplicates(self, store):
        """Test finding exact duplicate vectors."""
        await store.connect()
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Exact duplicate
            [0.0, 1.0, 0.0],
        ])
        
        duplicates = await store.find_duplicates(vectors, threshold=0.99)
        
        assert len(duplicates) == 1
        pair_ids = {duplicates[0].id1, duplicates[0].id2}
        assert pair_ids == {"0", "1"}
        assert duplicates[0].similarity == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_find_near_duplicates(self, store):
        """Test finding near-duplicate vectors."""
        await store.connect()
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],  # Near duplicate
            [0.0, 1.0, 0.0],   # Different
        ])
        
        duplicates = await store.find_duplicates(vectors, threshold=0.95)
        
        assert len(duplicates) == 1
        assert duplicates[0].similarity >= 0.95

    @pytest.mark.asyncio
    async def test_find_duplicates_with_custom_ids(self, store):
        """Test find_duplicates with custom IDs."""
        await store.connect()
        
        vectors = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        ids = ["doc_a", "doc_b"]
        
        duplicates = await store.find_duplicates(vectors, ids=ids, threshold=0.99)
        
        assert len(duplicates) == 1
        assert {duplicates[0].id1, duplicates[0].id2} == {"doc_a", "doc_b"}

    @pytest.mark.asyncio
    async def test_no_duplicates(self, store):
        """Test when there are no duplicates."""
        await store.connect()
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        duplicates = await store.find_duplicates(vectors, threshold=0.9)
        
        assert len(duplicates) == 0


class TestVectorStoreFindOutliers:
    """Test find_outliers method."""

    @pytest.fixture
    def store(self):
        """Create a memory store."""
        return MemoryVectorStore()

    @pytest.mark.asyncio
    async def test_find_outliers_basic(self, store):
        """Test basic outlier detection."""
        await store.connect()
        
        # Create cluster with one outlier - use non-zero vectors for cosine
        vectors = np.array([
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, 0.1],
            [1.0, 0.2],
            [-1.0, 0.0],  # Outlier - opposite direction
        ])
        
        outliers = await store.find_outliers(
            vectors,
            k_neighbors=3,
            contamination=0.2,
        )
        
        assert len(outliers) == 1
        # The outlier should have the highest outlier score (largest distance)
        assert outliers[0].id == "4"  # The outlier in opposite direction

    @pytest.mark.asyncio
    async def test_outlier_scores(self, store):
        """Test that outlier scores are computed correctly."""
        await store.connect()
        
        vectors = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],  # Far from others
        ])
        
        outliers = await store.find_outliers(
            vectors,
            k_neighbors=2,
            contamination=0.5,
        )
        
        # All should have outlier results
        assert len(outliers) >= 1
        
        # The point at [5, 5] should have higher outlier score
        scores = {o.id: o.outlier_score for o in outliers}
        if "2" in scores:
            assert scores["2"] > scores.get("0", 0)

    @pytest.mark.asyncio
    async def test_find_outliers_with_ids(self, store):
        """Test find_outliers with custom IDs."""
        await store.connect()
        
        # Use non-zero vectors for cosine similarity
        vectors = np.array([
            [1.0, 0.0],
            [1.1, 0.1],
            [-1.0, 0.0],  # Opposite direction - outlier
        ])
        ids = ["normal_a", "normal_b", "outlier"]
        
        outliers = await store.find_outliers(
            vectors,
            ids=ids,
            k_neighbors=2,
            contamination=0.4,
        )
        
        assert len(outliers) >= 1
        outlier_ids = [o.id for o in outliers]
        assert "outlier" in outlier_ids


class TestZeroVectorHandling:
    """Test handling of zero vectors."""

    @pytest.mark.asyncio
    async def test_zero_vector_cosine(self):
        """Test that zero vectors return 0 similarity."""
        store = MemoryVectorStore(metric=DistanceMetric.COSINE)
        await store.connect()
        
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 0.0],  # Zero vector
        ])
        await store.upsert(["v0", "v1"], vectors)
        
        query = np.array([1.0, 0.0])
        results = await store.search(query, top_k=2)
        
        # Zero vector should have 0 similarity
        zero_result = next(r for r in results if r.id == "v1")
        assert zero_result.score == 0.0
