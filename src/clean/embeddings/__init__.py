"""Embeddings module for Clean."""

from clean.embeddings.base import BaseEmbedder

__all__ = ["BaseEmbedder"]

# Optional text embeddings
try:
    from clean.embeddings.text import TextEmbedder, get_text_embeddings

    __all__.extend(["TextEmbedder", "get_text_embeddings"])
except ImportError:
    pass

# Optional image embeddings
try:
    from clean.embeddings.image import ImageEmbedder, get_image_embeddings

    __all__.extend(["ImageEmbedder", "get_image_embeddings"])
except ImportError:
    pass
