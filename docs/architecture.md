# Architecture Overview

This document describes the architecture and design of the Clean data quality platform.

## System Overview

Clean is designed as a modular, extensible platform with clear separation of concerns:

```mermaid
graph TB
    subgraph "User Interfaces"
        PY[Python API]
        CLI[CLI Tool]
        REST[REST API]
    end
    
    subgraph "Core Engine"
        DC[DatasetCleaner]
        QR[QualityReport]
        FE[FixEngine]
    end
    
    subgraph "Detection Layer"
        LE[LabelErrorDetector]
        DD[DuplicateDetector]
        OD[OutlierDetector]
        ID[ImbalanceDetector]
        BD[BiasDetector]
    end
    
    subgraph "Strategy Layer"
        OS[OutlierStrategies]
        DS[DuplicateStrategies]
        DF[DetectorFactory]
    end
    
    subgraph "Extensions"
        LLM[LLMDataCleaner]
        SC[StreamingCleaner]
        CP[ChunkProcessor]
        LT[LineageTracker]
    end
    
    subgraph "Plugin System"
        PR[PluginRegistry]
        DP[DetectorPlugins]
        LP[LoaderPlugins]
        EP[ExporterPlugins]
        FP[FixerPlugins]
    end
    
    PY --> DC
    CLI --> DC
    REST --> DC
    
    DC --> DF
    DF --> LE
    DF --> DD
    DF --> OD
    DF --> ID
    DF --> BD
    
    OD --> OS
    DD --> DS
    
    DC --> QR
    QR --> FE
    
    LLM --> DC
    SC --> CP
    SC --> DC
    DC --> LT
    
    PR --> DP
    PR --> LP
    PR --> EP
    PR --> FP
```

## Core Components

### DatasetCleaner

The main entry point for data quality analysis. Orchestrates detection, scoring, and reporting.

```mermaid
sequenceDiagram
    participant User
    participant DC as DatasetCleaner
    participant DF as DetectorFactory
    participant Det as Detectors
    participant QS as QualityScorer
    participant QR as QualityReport
    
    User->>DC: DatasetCleaner(data, label_column)
    User->>DC: analyze()
    
    DC->>DF: create_detectors()
    DF-->>DC: detector instances
    
    loop For each detector
        DC->>Det: detect(features, labels)
        Det-->>DC: DetectorResult
    end
    
    DC->>QS: calculate_score(results)
    QS-->>DC: QualityScore
    
    DC->>QR: QualityReport(results, score)
    QR-->>User: report
    
    User->>QR: summary()
    User->>QR: label_errors()
    User->>QR: duplicates()
```

### Detection Pipeline with Strategy Pattern

Each detector uses the Strategy pattern for flexible algorithm selection:

```mermaid
classDiagram
    class OutlierDetector {
        -strategies: List[OutlierStrategy]
        +detect(features, labels) DetectorResult
    }
    
    class OutlierStrategy {
        <<abstract>>
        +name: str
        +fit(features, labels) Self
        +detect(features, labels) list[int]
    }
    
    class IsolationForestStrategy {
        -contamination: float
        +fit(features, labels) Self
        +detect(features, labels) list[int]
    }
    
    class LOFStrategy {
        -n_neighbors: int
        +fit(features, labels) Self
        +detect(features, labels) list[int]
    }
    
    class ZScoreStrategy {
        -threshold: float
        +detect(features, labels) list[int]
    }
    
    class IQRStrategy {
        -multiplier: float
        +detect(features, labels) list[int]
    }
    
    class MADStrategy {
        -threshold: float
        +detect(features, labels) list[int]
    }
    
    OutlierDetector --> OutlierStrategy
    OutlierStrategy <|-- IsolationForestStrategy
    OutlierStrategy <|-- LOFStrategy
    OutlierStrategy <|-- ZScoreStrategy
    OutlierStrategy <|-- IQRStrategy
    OutlierStrategy <|-- MADStrategy
```

### Duplicate Detection Strategies

```mermaid
classDiagram
    class DuplicateDetector {
        -strategies: List[DuplicateStrategy]
        +detect(data) DetectorResult
    }
    
    class DuplicateStrategy {
        <<abstract>>
        +name: str
        +fit(data) Self
        +detect(data, threshold, seen_pairs) list[DuplicateCandidate]
    }
    
    class HashStrategy {
        -hash_columns: list[str]
        +fit(data) Self
        +detect(data, threshold, seen_pairs) list
    }
    
    class FuzzyStrategy {
        -max_samples: int
        +fit(data) Self
        +detect(data, threshold, seen_pairs) list
    }
    
    class EmbeddingStrategy {
        -model_name: str
        +fit(data) Self
        +detect(data, threshold, seen_pairs) list
    }
    
    DuplicateDetector --> DuplicateStrategy
    DuplicateStrategy <|-- HashStrategy
    DuplicateStrategy <|-- FuzzyStrategy
    DuplicateStrategy <|-- EmbeddingStrategy
```

### Dependency Injection with DetectorFactory

```mermaid
classDiagram
    class DetectorFactoryProtocol {
        <<protocol>>
        +create_label_detector() LabelErrorDetector
        +create_duplicate_detector() DuplicateDetector
        +create_outlier_detector() OutlierDetector
        +create_imbalance_detector() ImbalanceDetector
        +create_bias_detector() BiasDetector
    }
    
    class DetectorFactory {
        +create_label_detector() LabelErrorDetector
        +create_duplicate_detector() DuplicateDetector
        +create_outlier_detector() OutlierDetector
        +create_imbalance_detector() ImbalanceDetector
        +create_bias_detector() BiasDetector
    }
    
    class ConfigurableDetectorFactory {
        -outlier_methods: list[str]
        -duplicate_threshold: float
        +high_precision() ConfigurableDetectorFactory
        +high_recall() ConfigurableDetectorFactory
        +fast_scan() ConfigurableDetectorFactory
    }
    
    class DatasetCleaner {
        -detector_factory: DetectorFactoryProtocol
        +analyze() QualityReport
    }
    
    DetectorFactoryProtocol <|.. DetectorFactory
    DetectorFactoryProtocol <|.. ConfigurableDetectorFactory
    DatasetCleaner --> DetectorFactoryProtocol
```

## Exception Hierarchy

Clean uses a domain-specific exception hierarchy:

```mermaid
classDiagram
    class CleanError {
        +message: str
        +details: dict
    }
    
    class ValidationError {
        +field: str
        +expected: str
        +actual: str
    }
    
    class DetectionError {
        +detector: str
        +phase: str
    }
    
    class DependencyError {
        +package: str
        +install_extra: str
        +feature: str
    }
    
    class ConfigurationError {
        +parameter: str
        +reason: str
    }
    
    class DataError {
        +column: str
        +issue: str
    }
    
    CleanError <|-- ValidationError
    CleanError <|-- DetectionError
    CleanError <|-- DependencyError
    CleanError <|-- ConfigurationError
    CleanError <|-- DataError
```

## Chunk Processing Architecture

For large datasets, Clean provides streaming/chunked processing:

```mermaid
flowchart TB
    subgraph Input
        CSV[CSV File]
        DF[DataFrame]
    end
    
    subgraph ChunkProcessor
        BCP[BaseChunkProcessor]
        SCP[SyncChunkProcessor]
        ACP[AsyncChunkProcessor]
        CA[ChunkAnalyzer]
    end
    
    subgraph Processing
        CI[ChunkInfo]
        CR[ChunkResult]
        PS[ProcessingSummary]
    end
    
    CSV --> BCP
    DF --> BCP
    
    BCP --> SCP
    BCP --> ACP
    
    SCP --> CA
    ACP --> CA
    
    CA --> CI
    CA --> CR
    CR --> PS
```

```mermaid
sequenceDiagram
    participant U as User
    participant CP as ChunkProcessor
    participant CA as ChunkAnalyzer
    participant D as Detectors
    
    U->>CP: process_dataframe(df)
    
    loop For each chunk
        CP->>CP: _iter_dataframe_chunks()
        CP->>CA: analyze_chunk(chunk, info)
        
        CA->>D: _detect_outliers(chunk)
        D-->>CA: outlier_indices
        
        CA->>D: _detect_duplicates(chunk)
        D-->>CA: duplicate_indices
        
        CA-->>CP: ChunkResult
        CP-->>U: yield ChunkResult
    end
    
    U->>CP: get_summary()
    CP-->>U: ProcessingSummary
```

## AutoML Optimization Architecture

The QualityTuner uses pluggable optimization strategies:

```mermaid
classDiagram
    class QualityTuner {
        -optimizer: OptimizationStrategy
        +tune(X, y, validation_labels) TuningResult
    }
    
    class OptimizationStrategy {
        <<abstract>>
        +name: str
        +optimize(objective, space, n_trials) dict
    }
    
    class GridSearchStrategy {
        +optimize(objective, space, n_trials) dict
    }
    
    class RandomSearchStrategy {
        +optimize(objective, space, n_trials) dict
    }
    
    class BayesianStrategy {
        -n_initial_points: int
        +optimize(objective, space, n_trials) dict
    }
    
    class EvolutionaryStrategy {
        -population_size: int
        -mutation_rate: float
        +optimize(objective, space, n_trials) dict
    }
    
    QualityTuner --> OptimizationStrategy
    OptimizationStrategy <|-- GridSearchStrategy
    OptimizationStrategy <|-- RandomSearchStrategy
    OptimizationStrategy <|-- BayesianStrategy
    OptimizationStrategy <|-- EvolutionaryStrategy
```

## Module Structure

```
src/clean/
├── __init__.py              # Public API exports
├── __version__.py           # Version info
├── exceptions.py            # Domain exception hierarchy
├── constants.py             # Configuration constants
│
├── core/                    # Core functionality
│   ├── cleaner.py           # DatasetCleaner (with DI support)
│   ├── report.py            # QualityReport
│   └── types.py             # Type definitions, DetectionResults
│
├── detection/               # Issue detectors
│   ├── base.py              # BaseDetector, DetectorResult
│   ├── label_errors.py      # LabelErrorDetector
│   ├── duplicates.py        # DuplicateDetector
│   ├── outliers.py          # OutlierDetector
│   ├── imbalance.py         # ImbalanceDetector
│   ├── bias.py              # BiasDetector
│   ├── strategies.py        # Outlier detection strategies (NEW)
│   ├── duplicate_strategies.py  # Duplicate detection strategies (NEW)
│   └── factory.py           # DetectorFactory for DI (NEW)
│
├── automl/                  # AutoML package (NEW)
│   ├── __init__.py          # Package exports
│   ├── optimizers.py        # Optimization strategies
│   └── tuner.py             # Refactored QualityTuner
│
├── processing/              # Chunk processing (NEW)
│   └── __init__.py          # ChunkProcessor classes
│
├── loaders/                 # Data loaders
│   ├── base.py              # BaseLoader
│   ├── pandas_loader.py     # DataFrame loading
│   ├── csv_loader.py        # CSV file loading
│   ├── huggingface_loader.py
│   └── image_loader.py
│
├── scoring/                 # Quality scoring
│   ├── quality_scorer.py    # Accepts DetectionResults
│   └── metrics.py
│
├── visualization/           # Plots and browsers
│   ├── plots.py             # Matplotlib plots
│   ├── interactive.py       # Plotly plots
│   └── browser.py           # Issue browser widget
│
├── fixes.py                 # Auto-fix engine
├── plugins.py               # Plugin registry
├── lineage.py               # Data lineage tracking
├── llm.py                   # LLM data quality
├── streaming.py             # Streaming analysis
├── api.py                   # REST API (FastAPI)
└── cli.py                   # Command-line interface
```

## Key Design Decisions

### 1. Strategy Pattern for Detection Algorithms

Detection algorithms are encapsulated in strategy classes, enabling:
- Independent testing of each algorithm
- Easy addition of new algorithms without modifying detectors
- Runtime algorithm selection
- Reduced cyclomatic complexity

```python
# Create strategy directly
strategy = create_strategy("isolation_forest", contamination=0.05)
outliers = strategy.fit(features, labels).detect(features, labels)

# Or via detector with multiple strategies
detector = OutlierDetector(methods=["zscore", "isolation_forest"])
result = detector.fit_detect(features, labels)
```

### 2. Dependency Injection for Testability

The `DetectorFactory` pattern enables:
- Easy mocking in unit tests
- Custom detector configurations
- Preset configurations (high_precision, fast_scan)

```python
# Use custom factory
factory = ConfigurableDetectorFactory.high_precision()
cleaner = DatasetCleaner(data=df, detector_factory=factory)

# Or inject mock for testing
mock_factory = MockDetectorFactory()
cleaner = DatasetCleaner(data=df, detector_factory=mock_factory)
```

### 3. Domain Exception Hierarchy

All exceptions inherit from `CleanError`, providing:
- Consistent error handling
- Detailed error information via `details` dict
- Clear installation instructions for optional dependencies

```python
try:
    embedder = TextEmbedder()
except DependencyError as e:
    print(f"Install: {e.install_command}")
    # Output: Install: pip install clean-data-quality[text]
```

### 4. Parameter Objects for Complex APIs

The `DetectionResults` dataclass aggregates detector outputs:

```python
@dataclass
class DetectionResults:
    label_errors: DetectorResult | None
    duplicates: DetectorResult | None
    outliers: DetectorResult | None
    imbalance: DetectorResult | None
    bias: DetectorResult | None
    
# Simplifies API from 9 parameters to 4
def compute_score(n_samples, results: DetectionResults, labels=None, features=None):
    ...
```

### 5. Lazy Loading of Optional Dependencies

Heavy dependencies are lazily imported with helpful error messages:

```python
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

def get_embeddings(texts):
    if not HAS_SENTENCE_TRANSFORMERS:
        raise DependencyError(
            "sentence-transformers", 
            install_extra="text",
            feature="text embeddings"
        )
    ...
```

## Performance Benchmarks

Strategy pattern overhead is negligible:

| Component | Performance |
|-----------|-------------|
| Strategy factory creation | 0.31 µs/call |
| Z-Score detection | 11M samples/sec |
| LOF detection | 20K samples/sec |
| Hash duplicate detection | 40K samples/sec |
| Chunk processing (2K/chunk) | 368K samples/sec |

## Security Considerations

- **No Eval**: User data is never passed to `eval()` or similar
- **Sandboxed Plugins**: Plugins are validated before registration
- **API Authentication**: REST API supports standard auth middleware
- **Data Privacy**: No data is transmitted externally
- **Exception Safety**: Error messages don't leak sensitive data

## Future Considerations

1. **GPU Acceleration**: cuDF support for faster DataFrame operations
2. **Distributed Processing**: Enhanced Dask/Spark integration
3. **Parallel Detection**: Concurrent detector execution
4. **Caching Layer**: Redis-based caching for expensive computations
