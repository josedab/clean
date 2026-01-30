---
sidebar_position: 12
title: Contributing
---

# Contributing to Clean

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/clean.git
cd clean

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linter
ruff check src tests
```

## Development Setup

### Requirements

- Python 3.9+
- Git
- ~2GB disk space (for dependencies and test data)

### Full Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/clean.git
cd clean

# Add upstream remote
git remote add upstream https://github.com/clean-ai/clean.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest --quick  # Fast subset
pytest          # Full suite
```

## Making Changes

### 1. Create a Branch

```bash
git checkout main
git pull upstream main
git checkout -b feature/my-feature
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring

### 2. Make Your Changes

- Follow existing code style
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_detection/test_label_errors.py

# Run with coverage
pytest --cov=clean --cov-report=html

# Run type checking
mypy src/clean

# Run linter
ruff check src tests
ruff format src tests
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new detector for X"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Refactoring
- `chore:` - Maintenance

### 5. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

We use `ruff` for linting and formatting:

```bash
# Check
ruff check src tests

# Fix automatically
ruff check --fix src tests

# Format
ruff format src tests
```

### Docstrings

Use Google-style docstrings:

```python
def detect(self, features: np.ndarray, labels: np.ndarray) -> List[LabelError]:
    """Detect label errors in the dataset.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Label array of shape (n_samples,).
        
    Returns:
        List of detected label errors with confidence scores.
        
    Raises:
        ValueError: If features and labels have mismatched lengths.
        
    Example:
        >>> detector = LabelErrorDetector()
        >>> errors = detector.detect(X, y)
        >>> print(f"Found {len(errors)} errors")
    """
```

### Type Hints

All public APIs must have type hints:

```python
from typing import List, Optional, Dict, Any

def analyze(
    self,
    detectors: Optional[List[str]] = None,
    show_progress: bool = True,
) -> QualityReport:
    ...
```

## Testing

### Writing Tests

```python
import pytest
import pandas as pd
from clean import DatasetCleaner

class TestDatasetCleaner:
    """Tests for DatasetCleaner class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 0, 1, 1, 1],
        })
    
    def test_analyze_returns_report(self, sample_data):
        """analyze() should return QualityReport."""
        cleaner = DatasetCleaner(sample_data, labels="label")
        report = cleaner.analyze()
        
        assert report is not None
        assert hasattr(report, "quality_score")
    
    def test_analyze_with_invalid_labels_raises(self, sample_data):
        """analyze() should raise ValueError for invalid label column."""
        cleaner = DatasetCleaner(sample_data, labels="nonexistent")
        
        with pytest.raises(ValueError, match="not found"):
            cleaner.analyze()
```

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_core/
│   ├── test_cleaner.py
│   └── test_report.py
├── test_detection/
│   ├── test_label_errors.py
│   ├── test_duplicates.py
│   └── ...
└── test_integration/
    └── test_full_pipeline.py
```

### Running Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific file
pytest tests/test_core/test_cleaner.py

# Run specific test
pytest tests/test_core/test_cleaner.py::TestDatasetCleaner::test_analyze

# With coverage
pytest --cov=clean --cov-report=html
open htmlcov/index.html
```

## Creating a Plugin

### 1. Create Detector

```python
# my_detector.py
from clean.detection.base import BaseDetector
from dataclasses import dataclass
from typing import List

@dataclass
class MyIssue:
    index: int
    score: float

class MyDetector(BaseDetector):
    name = "my_detector"
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def detect(self, features, labels=None, **kwargs) -> List[MyIssue]:
        issues = []
        # Your detection logic
        return issues
```

### 2. Register Plugin

```python
# In your package's __init__.py
from clean.plugins import PluginRegistry
from .my_detector import MyDetector

PluginRegistry.register_detector("my_detector", MyDetector)
```

### 3. Entry Point (Optional)

```toml
# pyproject.toml
[project.entry-points."clean.plugins"]
my_plugin = "my_package:register_plugins"
```

## Documentation

### Building Docs

```bash
cd website
npm install
npm run build
```

### Previewing Locally

```bash
npm run start
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Test all code snippets

## Pull Request Checklist

Before submitting:

- [ ] Tests pass (`pytest`)
- [ ] Linter passes (`ruff check`)
- [ ] Type checker passes (`mypy src/clean`)
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/clean-ai/clean/discussions)
- **Bugs**: Open an [Issue](https://github.com/clean-ai/clean/issues)
- **Chat**: Join our Discord (link in README)

## Recognition

Contributors are recognized in:
- CHANGELOG.md for each release
- README.md contributors section
- GitHub contributors page

Thank you for contributing!
