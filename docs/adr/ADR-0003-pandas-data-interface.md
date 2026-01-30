# ADR-0003: Pandas DataFrame as Primary Data Interface

## Status

Accepted

## Context

Clean processes tabular ML datasets from various sources: CSV files, NumPy arrays, HuggingFace datasets, database queries, and more. We needed a consistent internal representation that:

1. **Supports mixed types**: ML datasets have numeric features, categorical columns, text, and metadata
2. **Familiar to users**: Data scientists should feel productive immediately
3. **Rich ecosystem**: Integrates with visualization, analysis, and export tools
4. **Efficient for medium data**: Handles datasets up to ~10M rows in memory
5. **Metadata preservation**: Column names, dtypes, and indices matter for interpretability

Alternatives considered:

1. **NumPy arrays only**: Fast but loses column names and requires separate label handling
2. **Polars**: Faster than Pandas but smaller ecosystem and learning curve
3. **PyArrow tables**: Great for I/O but less ergonomic for manipulation
4. **Custom dataclass**: Full control but massive implementation effort
5. **Dask DataFrame**: Distributed but adds complexity for single-machine use

## Decision

We standardized on **Pandas DataFrame** as the internal data representation, with a **Loader abstraction** to convert various input formats.

```python
# loaders/base.py
class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Load data, returning (features_df, labels_array)."""
        pass

# loaders/pandas_loader.py
class PandasLoader(BaseLoader):
    def __init__(self, data: pd.DataFrame, label_column: str = None, ...):
        self.data = data
        self.label_column = label_column

    def load(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        if self.label_column:
            labels = self.data[self.label_column].values
            features = self.data.drop(columns=[self.label_column])
        else:
            labels = None
            features = self.data
        return features, labels
```

Loaders exist for: CSV, NumPy arrays, HuggingFace datasets, and image folders.

Detectors receive DataFrames and can access column names:

```python
class OutlierDetector(BaseDetector):
    def detect(self, features: pd.DataFrame, labels=None):
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # IQR outlier detection per column
            q1, q3 = features[col].quantile([0.25, 0.75])
            # ... detection logic using column names for interpretability
```

## Consequences

### Positive

- **Zero learning curve**: 90%+ of data scientists already know Pandas
- **Rich column operations**: `select_dtypes()`, `groupby()`, `apply()` simplify detector implementations
- **Interpretable results**: Issue reports reference column names, not indices
- **Ecosystem integration**: Direct compatibility with matplotlib, seaborn, scikit-learn, and export formats
- **Missing value handling**: Built-in `isna()`, `fillna()` for data quality workflows

### Negative

- **Memory overhead**: DataFrames use more memory than raw NumPy for numeric-only data
- **Performance ceiling**: Not suitable for datasets larger than memory (addressed in ADR-0007)
- **Copy semantics**: Pandas copy-on-write behavior can cause subtle bugs
- **API instability**: Pandas deprecations occasionally require updates

### Neutral

- **Label separation**: Labels stored as NumPy array separately from features DataFrame
- **Index handling**: We generally reset indices to avoid iloc/loc confusion

## Related Decisions

- ADR-0007 (Async Streaming): For datasets exceeding memory, streaming processes chunks
- ADR-0006 (Dataclass Types): Results converted to dataclasses for type safety
