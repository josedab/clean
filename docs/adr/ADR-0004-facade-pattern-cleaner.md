# ADR-0004: Facade Pattern for DatasetCleaner Entry Point

## Status

Accepted

## Context

Clean has significant internal complexity:
- Multiple loaders (CSV, NumPy, HuggingFace, images)
- Five core detectors (labels, duplicates, outliers, imbalance, bias)
- Quality scoring with weighted aggregation
- Report generation with multiple export formats
- Optional features (embeddings, streaming, API)

Users shouldn't need to understand this architecture to analyze their data. We needed a simple, memorable API that:

1. **Single import**: One class to remember
2. **Sensible defaults**: Works out of the box for common cases
3. **Progressive disclosure**: Advanced options available but not required
4. **Consistent interface**: Same pattern for different data sources

Alternatives considered:

1. **Functional API**: `analyze(df, label_column="label")` - simple but limited customization
2. **Builder pattern**: Fluent chaining - verbose for simple cases
3. **Multiple entry points**: Separate classes per use case - confusing
4. **Facade pattern**: Single orchestrating class - chosen approach

## Decision

We implemented `DatasetCleaner` as a **Facade** that orchestrates all internal components.

```python
# The entire user-facing API for basic usage
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='target')
report = cleaner.analyze()
print(report.summary())
```

Internally, `DatasetCleaner` coordinates:

```python
class DatasetCleaner:
    def __init__(self, data, label_column=None, task="classification", ...):
        # 1. Select and configure loader
        if isinstance(data, pd.DataFrame):
            self._loader = PandasLoader(data, label_column=label_column)
        elif isinstance(data, np.ndarray):
            self._loader = NumpyLoader(data, labels=labels)
        # ... other loaders
        
        # 2. Load and validate data
        self._features, self._labels = self._loader.load()
        self._info = self._loader.get_info()
    
    def analyze(self, detectors=None, include_bias=True, ...):
        results = {}
        
        # 3. Run configured detectors
        if "label_errors" in detectors and self._labels is not None:
            results["label_errors"] = LabelErrorDetector().fit_detect(...)
        if "duplicates" in detectors:
            results["duplicates"] = DuplicateDetector().fit_detect(...)
        # ... other detectors
        
        # 4. Calculate quality scores
        scorer = QualityScorer()
        quality_score = scorer.score(results, self._info)
        
        # 5. Build report
        return QualityReport(results=results, score=quality_score, info=self._info)
```

The facade also provides convenience methods:

```python
cleaner.get_clean_data(remove_duplicates=True)  # Returns cleaned DataFrame
cleaner.get_review_queue(max_items=100)          # Priority-sorted issues
cleaner.export_report("report.html")             # Multiple formats
```

## Consequences

### Positive

- **Minimal API surface**: Users learn one class, not ten
- **Sensible defaults**: `analyze()` with no arguments runs all detectors
- **Encapsulated complexity**: Internal refactoring doesn't break user code
- **Consistent patterns**: Same interface whether data is CSV, NumPy, or HuggingFace
- **Documentation focus**: README examples are short and memorable

### Negative

- **God class risk**: DatasetCleaner could accumulate too many responsibilities
- **Hidden magic**: Users may not understand what's happening internally
- **Configuration explosion**: Many optional parameters can be overwhelming
- **Testing complexity**: Facade tests require mocking many components

### Neutral

- **Escape hatches exist**: Power users can instantiate detectors directly
- **Report as output**: Analysis returns a `QualityReport` object, not raw data

## Related Decisions

- ADR-0002 (Plugin Architecture): Plugins integrate through the facade automatically
- ADR-0003 (Pandas Interface): Facade accepts DataFrames as primary input
