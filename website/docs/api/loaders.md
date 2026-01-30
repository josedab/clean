---
sidebar_position: 5
title: Data Loaders
---

# Data Loaders

Load data from various sources.

```python
from clean.loaders import (
    PandasLoader,
    NumpyLoader,
    CSVLoader,
    HuggingFaceLoader,
    ImageFolderLoader,
)
```

## PandasLoader

Load from pandas DataFrames.

```python
PandasLoader(
    data: pd.DataFrame,
    label_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
)
```

### Example

```python
from clean.loaders import PandasLoader

loader = PandasLoader(
    df,
    label_column="target",
    feature_columns=["feature1", "feature2"],
)

features, labels = loader.load()
```

---

## NumpyLoader

Load from NumPy arrays.

```python
NumpyLoader(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
)
```

### Example

```python
from clean.loaders import NumpyLoader

loader = NumpyLoader(X, y)
features, labels = loader.load()
```

---

## CSVLoader

Load from CSV files.

```python
CSVLoader(
    path: Union[str, Path],
    label_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    **read_kwargs,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str or Path | Path to CSV file |
| `label_column` | str | Column containing labels |
| `feature_columns` | list | Columns to use as features |
| `**read_kwargs` | | Passed to `pd.read_csv()` |

### Example

```python
from clean.loaders import CSVLoader

loader = CSVLoader(
    "data.csv",
    label_column="label",
    sep=";",  # Custom separator
)

features, labels = loader.load()
```

---

## HuggingFaceLoader

Load from HuggingFace Datasets.

```python
HuggingFaceLoader(
    dataset: Union[str, Dataset],
    split: str = "train",
    label_column: str = "label",
    feature_columns: Optional[List[str]] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | str or Dataset | Dataset name or object |
| `split` | str | Dataset split to use |
| `label_column` | str | Label column name |
| `feature_columns` | list | Feature columns |

### Example

```python
from clean.loaders import HuggingFaceLoader

# From name
loader = HuggingFaceLoader(
    "imdb",
    split="train",
    label_column="label",
)

# From Dataset object
from datasets import load_dataset
ds = load_dataset("imdb")
loader = HuggingFaceLoader(ds["train"], label_column="label")
```

---

## ImageFolderLoader

Load images from directory structure.

```python
ImageFolderLoader(
    path: Union[str, Path],
    image_size: Tuple[int, int] = (224, 224),
    extensions: Optional[List[str]] = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str or Path | | Root directory |
| `image_size` | tuple | (224, 224) | Resize dimensions |
| `extensions` | list | [".jpg", ".png"] | Valid extensions |

### Directory Structure

```
images/
├── cat/
│   ├── img001.jpg
│   └── img002.jpg
├── dog/
│   ├── img003.jpg
│   └── img004.jpg
```

### Example

```python
from clean.loaders import ImageFolderLoader

loader = ImageFolderLoader(
    "images/",
    image_size=(128, 128),
)

images, labels = loader.load()
# images: (N, 128, 128, 3) array
# labels: ["cat", "cat", "dog", "dog"]
```

---

## Base Loader Interface

Create custom loaders by extending `BaseLoader`:

```python
from clean.loaders.base import BaseLoader
from typing import Tuple
import numpy as np

class CustomLoader(BaseLoader):
    def __init__(self, source):
        self.source = source
    
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        # Load features and labels
        features = ...
        labels = ...
        return features, labels
    
    @property
    def data_type(self) -> str:
        return "tabular"  # or "text", "image"
```

## Auto-Detection

`DatasetCleaner` automatically selects the right loader:

```python
from clean import DatasetCleaner

# DataFrame → PandasLoader
cleaner = DatasetCleaner(df, labels="target")

# Path string → CSVLoader
cleaner = DatasetCleaner("data.csv", labels="target")

# NumPy arrays → NumpyLoader
cleaner = DatasetCleaner(X, labels=y)

# HuggingFace Dataset → HuggingFaceLoader
cleaner = DatasetCleaner(dataset, labels="label")
```
