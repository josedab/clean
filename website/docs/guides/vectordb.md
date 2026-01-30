---
sidebar_position: 17
title: Vector DB Integration
---

# Vector Database Integration

Scale duplicate and similarity detection to millions of samples using vector databases.

## The Problem

Clean's default duplicate detection loads all embeddings into memory. This works great for datasets up to ~100K samples, but breaks down for larger datasets.

Vector databases solve this by:
- Storing embeddings efficiently on disk
- Using approximate nearest neighbor (ANN) algorithms
- Scaling to billions of vectors

## Supported Backends

| Backend | Type | Best For |
|---------|------|----------|
| **Pinecone** | Managed | Serverless, easy setup |
| **Weaviate** | Managed/Self-hosted | Hybrid search, GraphQL |
| **Milvus** | Self-hosted | High performance, on-prem |
| **Qdrant** | Managed/Self-hosted | Filtering, payloads |

## Quick Start

```python
from clean.vectordb import PineconeConnector

# Connect to Pinecone
connector = PineconeConnector(
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="clean-embeddings",
)

# Index your embeddings
connector.index(
    embeddings=embeddings,      # numpy array (n_samples, dim)
    ids=sample_ids,             # list of string IDs
    metadata={"source": "v1"},  # optional metadata
)

# Search for similar vectors
results = connector.search(
    query_vector=query_embedding,
    top_k=10,
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score:.3f}")
```

## Backend Setup

### Pinecone

```python
from clean.vectordb import PineconeConnector

connector = PineconeConnector(
    api_key="your-api-key",
    environment="us-west1-gcp",  # or your environment
    index_name="clean-embeddings",
)

# Create index (if doesn't exist)
connector.create_index(
    dimension=768,              # embedding dimension
    metric="cosine",            # cosine, euclidean, dotproduct
)
```

### Weaviate

```python
from clean.vectordb import WeaviateConnector

connector = WeaviateConnector(
    url="http://localhost:8080",      # or Weaviate Cloud URL
    api_key=None,                     # optional for self-hosted
    class_name="DataSample",
)

# Index with properties
connector.index(
    embeddings=embeddings,
    ids=sample_ids,
    properties={
        "label": labels,
        "source": sources,
    },
)
```

### Milvus

```python
from clean.vectordb import MilvusConnector

connector = MilvusConnector(
    host="localhost",
    port=19530,
    collection_name="clean_embeddings",
)

# Create collection
connector.create_collection(
    dimension=768,
    metric_type="IP",           # Inner product (cosine)
    index_type="IVF_FLAT",      # Index algorithm
)
```

### Qdrant

```python
from clean.vectordb import QdrantConnector

connector = QdrantConnector(
    url="http://localhost:6333",
    api_key=None,
    collection_name="clean_samples",
)

# Index with payloads
connector.index(
    embeddings=embeddings,
    ids=sample_ids,
    payloads=[
        {"label": label, "confidence": conf}
        for label, conf in zip(labels, confidences)
    ],
)
```

## Integration with Clean

### Duplicate Detection at Scale

```python
from clean import DatasetCleaner
from clean.vectordb import PineconeConnector

# Setup vector store
connector = PineconeConnector(
    api_key="...",
    index_name="dataset-embeddings",
)

# Use with cleaner
cleaner = DatasetCleaner(
    data=df,
    label_column="label",
    vector_store=connector,  # Enable vector-based detection
)

# Duplicates now use vector DB
report = cleaner.analyze()
duplicates = report.duplicates()
```

### Generating Embeddings

```python
from clean.vectordb import generate_embeddings

# Text embeddings
embeddings = generate_embeddings(
    data=df,
    text_column="text",
    model="sentence-transformers/all-MiniLM-L6-v2",
)

# Image embeddings
embeddings = generate_embeddings(
    data=df,
    image_column="image_path",
    model="openai/clip-vit-base-patch32",
)

# Index in vector store
connector.index(embeddings=embeddings, ids=df.index.tolist())
```

## Searching

### Basic Search

```python
results = connector.search(
    query_vector=embedding,
    top_k=10,
)
```

### Filtered Search

```python
# Pinecone
results = connector.search(
    query_vector=embedding,
    top_k=10,
    filter={"source": "training_v1"},
)

# Weaviate
results = connector.search(
    query_vector=embedding,
    top_k=10,
    filters={"label": "positive"},
)

# Qdrant
results = connector.search(
    query_vector=embedding,
    top_k=10,
    filter={
        "must": [
            {"key": "confidence", "range": {"gte": 0.8}}
        ]
    },
)
```

### Batch Search

```python
# Search multiple queries at once
results = connector.batch_search(
    query_vectors=embeddings[:100],
    top_k=10,
)

# results is list of list of SearchResult
for i, query_results in enumerate(results):
    print(f"Query {i}: {len(query_results)} matches")
```

## Search Results

```python
result = results[0]

result.id         # Sample ID
result.score      # Similarity score (0-1 for cosine)
result.metadata   # Associated metadata
result.vector     # The matched vector (if requested)
```

## Factory Function

```python
from clean.vectordb import create_vector_connector

# Create connector for any backend
connector = create_vector_connector(
    backend="pinecone",  # pinecone, weaviate, milvus, qdrant
    api_key="...",
    index_name="my-index",
)
```

## Best Practices

### Batching

Always index and search in batches:

```python
# Good: batch indexing
batch_size = 1000
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    connector.index(embeddings=batch, ids=batch_ids)

# Good: batch search
results = connector.batch_search(query_vectors, top_k=10)
```

### Choosing Metrics

| Metric | When to Use |
|--------|-------------|
| `cosine` | Normalized embeddings (most common) |
| `euclidean` | Raw, unnormalized vectors |
| `dotproduct` | When magnitude matters |

### Index Tuning

For large datasets, tune index parameters:

```python
# Milvus example
connector.create_collection(
    dimension=768,
    metric_type="IP",
    index_type="IVF_PQ",       # Product quantization for compression
    index_params={
        "nlist": 1024,         # Number of clusters
        "m": 16,               # PQ subvectors
    }
)
```

## Installation

```bash
pip install clean-data-quality[vectordb]
```

This installs:
- `pinecone-client`
- `weaviate-client`
- `pymilvus`
- `qdrant-client`

Install only what you need:

```bash
pip install pinecone-client  # Just Pinecone
```

## Performance Tips

1. **Batch operations**: Index/search in batches of 100-1000
2. **Use filters**: Reduce search space when possible
3. **Right metric**: Cosine for normalized embeddings
4. **Index parameters**: Tune for your scale
5. **Local caching**: Cache frequent queries

## Next Steps

- [Intelligent Sampling](/docs/guides/intelligent-sampling) - Use similarity for active learning
- [Slice Discovery](/docs/guides/slice-discovery) - Find similar problem groups
- [API Reference](/docs/guides/vectordb) - Full API documentation
