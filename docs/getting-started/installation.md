# Installation

## Requirements

- Python 3.9 or higher
- pip or conda package manager

## Basic Installation

The simplest way to install Clean is via pip:

```bash
pip install clean-data-quality
```

This installs the core package with support for:

- Pandas DataFrames
- NumPy arrays
- CSV files
- Label error detection
- Duplicate detection
- Outlier detection
- Basic visualization

## Optional Dependencies

Clean has optional dependencies for specific features:

### Text Data Support

For text embedding and semantic similarity:

```bash
pip install clean-data-quality[text]
```

This adds:

- `sentence-transformers` for text embeddings
- Semantic duplicate detection for text data

### Image Data Support

For image datasets:

```bash
pip install clean-data-quality[image]
```

This adds:

- `Pillow` for image loading
- `torch` and `torchvision` for image embeddings
- `transformers` for pre-trained models

### HuggingFace Datasets

For loading HuggingFace datasets:

```bash
pip install clean-data-quality[huggingface]
```

### Interactive Visualization

For interactive Plotly charts and Jupyter widgets:

```bash
pip install clean-data-quality[interactive]
```

This adds:

- `plotly` for interactive charts
- `ipywidgets` for the issue browser

### All Features

To install everything:

```bash
pip install clean-data-quality[all]
```

## Development Installation

For contributing to Clean:

```bash
# Clone the repository
git clone https://github.com/clean-data/clean.git
cd clean

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

## Verifying Installation

```python
import clean
print(clean.__version__)
```

## Troubleshooting

### ImportError for sentence-transformers

If you get an import error for text features:

```bash
pip install clean-data-quality[text]
```

### CUDA/GPU Issues

For GPU acceleration with image embeddings, ensure you have CUDA installed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1/M2)

Clean works on Apple Silicon. For optimal performance:

```bash
pip install torch torchvision
```

PyTorch will automatically use the Metal backend.
