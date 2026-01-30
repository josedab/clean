# ADR-0005: Optional Dependencies via Feature Extras

## Status

Accepted

## Context

Clean supports a wide range of features with varying dependency requirements:

| Feature | Dependencies | Size Impact |
|---------|--------------|-------------|
| Core analysis | pandas, numpy, scikit-learn, cleanlab | ~50 MB |
| Text embeddings | sentence-transformers | +500 MB |
| Image analysis | torch, torchvision, transformers | +2 GB |
| REST API | FastAPI, uvicorn, pydantic | +20 MB |
| Streaming | aiokafka, pulsar-client, redis | +30 MB |
| Vector DBs | pinecone, weaviate, milvus, qdrant clients | +50 MB |

Problems with bundling everything:
1. **Install time**: 10+ minutes for full install vs 30 seconds for core
2. **Disk space**: 3+ GB vs ~100 MB
3. **Conflicts**: PyTorch versions conflict with other ML tools
4. **CI/CD costs**: Large images increase build times and storage
5. **Security surface**: More dependencies = more CVE exposure

## Decision

We use **pip extras** (optional dependency groups) defined in `pyproject.toml`:

```toml
[project]
dependencies = [
    # Core - always installed
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "cleanlab>=2.0.0",
    "tqdm>=4.60.0",
    "matplotlib>=3.5.0",
]

[project.optional-dependencies]
text = ["sentence-transformers>=2.2.0"]
image = ["Pillow>=9.0.0", "torch>=1.10.0", "torchvision>=0.11.0", "transformers>=4.20.0"]
huggingface = ["datasets>=2.0.0"]
interactive = ["plotly>=5.0.0", "ipywidgets>=7.6.0"]
api = ["fastapi>=0.100.0", "uvicorn>=0.23.0", "pydantic>=2.0.0"]
streaming = ["aiokafka>=0.8.0", "pulsar-client>=3.0.0", "redis>=4.5.0"]
vectordb = ["pinecone-client>=2.2.0", "weaviate-client>=3.15.0", "pymilvus>=2.2.0", "qdrant-client>=1.1.0"]
cloud = ["aiohttp>=3.8.0"]
all = ["clean-data-quality[text,image,huggingface,interactive,api,streaming,vectordb,cloud]"]
dev = ["pytest>=7.0.0", "ruff>=0.1.0", "mypy>=1.0.0", ...]
```

Users install what they need:

```bash
pip install clean-data-quality              # Core only (fast, small)
pip install clean-data-quality[text]        # Add text embeddings
pip install clean-data-quality[api]         # Add REST API
pip install clean-data-quality[all]         # Everything
```

Code handles missing dependencies gracefully:

```python
# __init__.py - Conditional imports
try:
    from clean.loaders import load_huggingface
    __all__.append("load_huggingface")
except ImportError:
    pass  # huggingface extra not installed

# api.py - Clear error message
try:
    from fastapi import FastAPI
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. Install with: pip install clean[api]"
    ) from e
```

## Consequences

### Positive

- **Fast default install**: Core installs in <30 seconds
- **Minimal footprint**: Production deployments only include needed features
- **Reduced conflicts**: PyTorch users control their own torch version
- **Clear feature boundaries**: Extras document what each feature needs
- **CI optimization**: Test matrix can test extras independently

### Negative

- **User confusion**: "Why doesn't `load_huggingface` work?" → need clear error messages
- **Documentation burden**: Must document which features need which extras
- **Test complexity**: CI must test combinations of extras
- **Transitive conflicts**: Extras might conflict with each other (rare)

### Neutral

- **`[all]` convenience**: Power users can still install everything
- **`[dev]` for contributors**: Development dependencies grouped separately

## Implementation Notes

Optional features use lazy imports and clear error messages:

```python
def load_image_folder(path: str, ...):
    try:
        from PIL import Image
        import torch
    except ImportError:
        raise ImportError(
            "Image loading requires the 'image' extra. "
            "Install with: pip install clean-data-quality[image]"
        )
    # ... rest of implementation
```
