# Data Loaders

Clean provides several data loaders for different data sources.

## PandasLoader

Load from Pandas DataFrames.

```python
from clean.loaders import PandasLoader

loader = PandasLoader(
    data=df,
    label_column='label',
    feature_columns=None,  # Use all except label
    exclude_columns=None
)

features, labels, info = loader.load()
```

::: clean.loaders.pandas_loader.PandasLoader
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - load

## NumpyLoader

Load from NumPy arrays.

```python
from clean.loaders import NumpyLoader

loader = NumpyLoader(
    features=X,
    labels=y,
    feature_names=None
)

features, labels, info = loader.load()
```

::: clean.loaders.numpy_loader.NumpyLoader
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - load

## CSVLoader

Load from CSV files.

```python
from clean.loaders import CSVLoader

loader = CSVLoader(
    path='data.csv',
    label_column='target',
    **pandas_kwargs
)

features, labels, info = loader.load()
```

::: clean.loaders.csv_loader.CSVLoader
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - load

## HuggingFaceLoader

Load from HuggingFace Datasets.

!!! note
    Requires: `pip install clean-data-quality[huggingface]`

```python
from clean.loaders import HuggingFaceLoader
from datasets import load_dataset

dataset = load_dataset('imdb', split='train')

loader = HuggingFaceLoader(
    dataset=dataset,
    label_column='label',
    text_column='text'
)

features, labels, info = loader.load()
```

::: clean.loaders.huggingface_loader.HuggingFaceLoader
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - load

## ImageFolderLoader

Load images from folder structure.

!!! note
    Requires: `pip install clean-data-quality[image]`

```python
from clean.loaders import ImageFolderLoader

loader = ImageFolderLoader(
    root='data/images/',
    load_images=False,  # Just get paths
    image_size=(224, 224)
)

features, labels, info = loader.load()
```

::: clean.loaders.image_loader.ImageFolderLoader
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - load

## Convenience Functions

```python
from clean.loaders import load_dataframe, load_arrays, load_csv

# Quick loading
features, labels, info = load_dataframe(df, label_column='label')
features, labels, info = load_arrays(X, y)
features, labels, info = load_csv('data.csv', label_column='label')
```

## LoaderConfig

Configuration dataclass for all loaders.

::: clean.loaders.base.LoaderConfig
    options:
      show_root_heading: true
      show_source: false

## BaseLoader

Abstract base class for custom loaders.

::: clean.loaders.base.BaseLoader
    options:
      show_root_heading: true
      show_source: false
      members:
        - load
        - validate
        - infer_task_type
