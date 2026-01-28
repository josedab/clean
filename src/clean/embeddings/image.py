"""Image embedding using various models."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from clean.embeddings.base import BaseEmbedder

# Optional imports
try:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    Image = None  # type: ignore

if TYPE_CHECKING:
    pass


class ImageEmbedder(BaseEmbedder):
    """Generate embeddings for images using CLIP or other models."""

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        """Initialize the image embedder.

        Args:
            model_name: Model name
            device: Device to use

        Raises:
            ImportError: If required packages not installed
        """
        if not HAS_CLIP:
            raise ImportError(
                "transformers and torch required for image embeddings. "
                "Install with: pip install clean-data-quality[image]"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self._embedding_dim = self._model.get_image_features(dummy).shape[-1]

    def encode(
        self,
        data: list,  # list[str | Path | PILImage]
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode images to embeddings.

        Args:
            data: List of image paths, Path objects, or PIL Images
            batch_size: Batch size
            show_progress: Show progress

        Returns:
            Embedding matrix
        """
        from tqdm import tqdm

        embeddings = []

        iterator = range(0, len(data), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")

        for i in iterator:
            batch = data[i:i + batch_size]

            # Load images
            images = []
            for item in batch:
                if isinstance(item, (str, Path)):
                    img = Image.open(item).convert("RGB")
                else:
                    img = item.convert("RGB")
                images.append(img)

            # Process
            inputs = self._processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())

        return np.vstack(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedding_dim

    def encode_single(self, image: Any) -> np.ndarray:
        """Encode a single image.

        Args:
            image: Image path or PIL Image

        Returns:
            Embedding vector
        """
        return self.encode([image])[0]


def get_image_embeddings(
    images: list[str | Path],
    model_name: str | None = None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Get embeddings for a list of images.

    Args:
        images: List of image paths
        model_name: Model name
        batch_size: Batch size
        show_progress: Show progress

    Returns:
        Embedding matrix
    """
    embedder = ImageEmbedder(model_name=model_name)
    return embedder.encode(images, batch_size=batch_size, show_progress=show_progress)
