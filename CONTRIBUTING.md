# Contributing to Clean

Thank you for your interest in contributing to Clean! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/clean-data/clean.git
   cd clean
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=clean --cov-report=html

# Run specific test file
pytest tests/test_detection/test_label_errors.py

# Run tests matching a pattern
pytest -k "test_label"

# Run async tests
pytest tests/test_streaming.py -v
```

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Auto-fix lint issues
ruff check src tests --fix

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

### Building Documentation

```bash
# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write docstrings in Google style
- Keep functions focused and small
- Write tests for all new functionality
- Aim for >80% test coverage

### Example Function

```python
def find_label_errors(
    features: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """Find potential label errors in the dataset.

    Uses confident learning to identify samples where the given label
    likely doesn't match the true label.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Label array of shape (n_samples,).
        confidence_threshold: Minimum confidence to flag as error.

    Returns:
        DataFrame with columns: index, given_label, predicted_label, confidence.

    Raises:
        ValueError: If features and labels have mismatched lengths.

    Example:
        >>> errors = find_label_errors(X, y, confidence_threshold=0.7)
        >>> print(errors.head())
    """
    ...
```

## Adding New Features

### Adding a New Detector

1. Create detector class in `src/clean/detection/`:

```python
from clean.detection.base import BaseDetector, DetectorResult

class MyDetector(BaseDetector):
    """Detect custom quality issues."""
    
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> Self:
        # Fit model if needed
        return self
    
    def detect(self, features: pd.DataFrame, labels: pd.Series) -> DetectorResult:
        issues = []
        # Detection logic
        return DetectorResult(issues=issues, metadata={})
```

2. Add tests in `tests/test_detection/test_my_detector.py`
3. Export in `src/clean/detection/__init__.py`
4. Update documentation

### Adding a Plugin

Use the plugin system for external extensions:

```python
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_plugin")
class MyPluginDetector(DetectorPlugin):
    name = "my_plugin"
    description = "Description of what it does"
    
    def detect(self, data, labels, **kwargs):
        return [indices_with_issues]
```

### Adding CLI Commands

Add commands in `src/clean/cli.py`:

```python
def cmd_mycommand(args: argparse.Namespace) -> int:
    """Run my custom command."""
    # Implementation
    return 0

# Add to parser in create_parser()
mycommand_parser = subparsers.add_parser("mycommand", help="...")
mycommand_parser.add_argument("--option", ...)
```

### Adding API Endpoints

Add endpoints in `src/clean/api.py`:

```python
@app.post("/my-endpoint")
async def my_endpoint(param: str = Query(...)) -> JSONResponse:
    """Endpoint description."""
    # Implementation
    return JSONResponse(content={...})
```

## Pull Request Process

1. **Create a branch** from `main` for your changes
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** with appropriate tests

3. **Run the test suite** to ensure nothing is broken
   ```bash
   pytest
   ruff check src tests
   ```

4. **Update documentation** if needed

5. **Submit a pull request** with a clear description

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code follows style guidelines (`ruff check`)
- [ ] Type hints added for new code
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Commit messages are clear and descriptive

### Commit Message Format

```
type: Short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Reporting Issues

When reporting issues, please include:

- Clean version (`python -c "import clean; print(clean.__version__)"`)
- Python version (`python --version`)
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback if applicable

## Project Structure

```
clean/
├── src/clean/           # Main package
│   ├── core/            # Core classes (Cleaner, Report)
│   ├── detection/       # Issue detectors
│   ├── loaders/         # Data loaders
│   ├── scoring/         # Quality scoring
│   ├── visualization/   # Plots and browsers
│   ├── fixes.py         # Auto-fix engine
│   ├── plugins.py       # Plugin system
│   ├── streaming.py     # Streaming analysis
│   ├── llm.py           # LLM data quality
│   ├── api.py           # REST API
│   └── cli.py           # CLI tool
├── tests/               # Test files
├── docs/                # Documentation
├── examples/            # Example notebooks
└── benchmarks/          # Performance benchmarks
```

## Getting Help

- 📖 [Documentation](https://clean-data.github.io/clean)
- 💬 [GitHub Discussions](https://github.com/clean-data/clean/discussions)
- 🐛 [Issue Tracker](https://github.com/clean-data/clean/issues)

Thank you for contributing! 🎉
