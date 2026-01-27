"""Image folder loader."""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from clean.core.types import DatasetInfo, DataType, TaskType
from clean.loaders.base import BaseLoader, LoaderConfig

# Optional imports
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageFolderLoader(BaseLoader):
    """Load images from a folder structure.

    Expects folder structure like:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path,
        load_images: bool = False,
        image_size: tuple[int, int] | None = None,
        task_type: TaskType | None = None,
    ):
        """Initialize the image folder loader.

        Args:
            root: Root directory containing class folders
            load_images: Whether to load images into memory (for embeddings)
            image_size: Resize images to this size if loading
            task_type: Type of ML task

        Raises:
            ImportError: If PIL not installed and load_images=True
        """
        self.root = Path(root)
        self.load_images = load_images
        self.image_size = image_size
        self.config = LoaderConfig(task_type=task_type or TaskType.CLASSIFICATION)

        if load_images and not HAS_PIL:
            raise ImportError(
                "Pillow required for loading images. "
                "Install with: pip install clean-data-quality[image]"
            )

        self._features: pd.DataFrame | None = None
        self._labels: np.ndarray | None = None
        self._info: DatasetInfo | None = None
        self._image_paths: list[Path] = []
        self._class_names: list[str] = []

    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Load image paths and labels from folder structure.

        Returns:
            Tuple of (features DataFrame with paths, labels array)
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")

        # Find all class directories
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root}")

        self._class_names = [d.name for d in class_dirs]

        # Collect image paths and labels
        image_paths: list[str] = []
        labels: list[str] = []

        for class_dir in class_dirs:
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(str(img_path))
                    labels.append(class_name)

        if not image_paths:
            raise ValueError(f"No images found in {self.root}")

        # Create features DataFrame
        features_dict: dict[str, Any] = {"image_path": image_paths}

        # Optionally load image data
        if self.load_images:
            image_arrays = []
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                if self.image_size:
                    img = img.resize(self.image_size)
                image_arrays.append(np.array(img))
            features_dict["image_data"] = image_arrays

        self._features = pd.DataFrame(features_dict)
        self._labels = np.array(labels)
        self._image_paths = [Path(p) for p in image_paths]

        return self._features, self._labels

    def get_info(self) -> DatasetInfo:
        """Get information about the dataset.

        Returns:
            DatasetInfo with metadata
        """
        if self._features is None:
            self.load()

        assert self._features is not None
        assert self._labels is not None

        n_classes = len(np.unique(self._labels))

        self._info = DatasetInfo(
            n_samples=len(self._features),
            n_features=len(self._features.columns),
            n_classes=n_classes,
            feature_names=list(self._features.columns),
            label_column="label",
            data_type=DataType.IMAGE,
            task_type=TaskType.CLASSIFICATION,
        )

        return self._info

    @property
    def class_names(self) -> list[str]:
        """Get list of class names."""
        if not self._class_names:
            self.load()
        return self._class_names

    @property
    def image_paths(self) -> list[Path]:
        """Get list of image paths."""
        if not self._image_paths:
            self.load()
        return self._image_paths


def load_image_folder(
    root: str | Path,
    **kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray | None, DatasetInfo]:
    """Convenience function to load images from a folder.

    Args:
        root: Root directory containing class folders
        **kwargs: Additional arguments for ImageFolderLoader

    Returns:
        Tuple of (features with paths, labels, info)
    """
    loader = ImageFolderLoader(root, **kwargs)
    features, labels = loader.load()
    info = loader.get_info()
    return features, labels, info
