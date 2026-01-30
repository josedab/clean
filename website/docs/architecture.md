---
sidebar_position: 10
title: Architecture
---

# Architecture

How Clean is designed and how components interact.

## High-Level Overview

```mermaid
graph TB
    subgraph Input["Data Input"]
        CSV[CSV Files]
        DF[DataFrames]
        HF[HuggingFace]
        IMG[Image Folders]
    end
    
    subgraph Loaders["Data Loaders"]
        PL[PandasLoader]
        NL[NumpyLoader]
        HL[HuggingFaceLoader]
        IL[ImageLoader]
    end
    
    subgraph Core["Core Engine"]
        DC[DatasetCleaner]
        QS[QualityScorer]
        QR[QualityReport]
    end
    
    subgraph Detection["Detection Suite"]
        LE[LabelErrorDetector]
        DD[DuplicateDetector]
        OD[OutlierDetector]
        ID[ImbalanceDetector]
        BD[BiasDetector]
    end
    
    subgraph Output["Output"]
        JSON[JSON Report]
        HTML[HTML Report]
        VIZ[Visualizations]
        CLEAN[Clean Data]
    end
    
    CSV --> PL
    DF --> PL
    HF --> HL
    IMG --> IL
    
    PL --> DC
    NL --> DC
    HL --> DC
    IL --> DC
    
    DC --> LE
    DC --> DD
    DC --> OD
    DC --> ID
    DC --> BD
    
    LE --> QS
    DD --> QS
    OD --> QS
    ID --> QS
    BD --> QS
    
    QS --> QR
    
    QR --> JSON
    QR --> HTML
    QR --> VIZ
    QR --> CLEAN
```

## Component Details

### Data Loaders

Loaders normalize different data sources into a common format:

```python
# All loaders implement this interface
class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (features, labels)"""
        pass
    
    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return 'tabular', 'text', or 'image'"""
        pass
```

Auto-detection logic:
1. If input is `pd.DataFrame` → `PandasLoader`
2. If input is `np.ndarray` → `NumpyLoader`
3. If input is `str`/`Path` ending in `.csv` → `CSVLoader`
4. If input is HuggingFace `Dataset` → `HuggingFaceLoader`
5. If input is directory path → `ImageFolderLoader`

### Detection Pipeline

```mermaid
sequenceDiagram
    participant C as DatasetCleaner
    participant D as Detector
    participant S as QualityScorer
    participant R as QualityReport
    
    C->>C: Validate input
    C->>C: Detect data type
    
    loop For each detector
        C->>D: detect(features, labels)
        D->>D: Run algorithm
        D-->>C: Return issues
    end
    
    C->>S: Calculate scores
    S->>S: Aggregate by category
    S-->>C: Return QualityScore
    
    C->>R: Create report
    R-->>C: QualityReport
```

### Quality Scoring

Scores are calculated as:

```
overall_score = 100 - penalties

penalties:
  - label_errors: count / total * 100 * weight
  - duplicates: pairs / total * 100 * weight  
  - outliers: count / total * 100 * weight
  - imbalance: max(0, ratio - threshold) * weight
```

Default weights:
| Issue Type | Weight |
|------------|--------|
| Label Errors | 1.0 |
| Duplicates | 0.5 |
| Outliers | 0.3 |
| Imbalance | 0.2 |

### Plugin System

```mermaid
graph LR
    subgraph Registry["Plugin Registry"]
        REG[PluginRegistry]
    end
    
    subgraph Plugins["Plugins"]
        P1[Custom Detector]
        P2[Custom Scorer]
        P3[Custom Loader]
    end
    
    subgraph Discovery["Discovery"]
        EP[Entry Points]
        MAN[Manual Registration]
    end
    
    EP --> REG
    MAN --> REG
    
    REG --> P1
    REG --> P2
    REG --> P3
    
    P1 --> DC[DatasetCleaner]
    P2 --> DC
    P3 --> DC
```

Plugins are discovered via:
1. **Entry points** in `pyproject.toml`
2. **Manual registration** via `PluginRegistry.register()`

## Data Flow

### Batch Analysis

```mermaid
flowchart LR
    A[Load Data] --> B[Validate]
    B --> C[Detect Issues]
    C --> D[Score Quality]
    D --> E[Generate Report]
    E --> F[Export/Visualize]
```

### Streaming Analysis

```mermaid
flowchart LR
    A[Read Chunk] --> B[Partial Analysis]
    B --> C{More Chunks?}
    C -->|Yes| A
    C -->|No| D[Aggregate Results]
    D --> E[Final Report]
```

### Fix Engine

```mermaid
flowchart TD
    A[QualityReport] --> B{Strategy?}
    B -->|Conservative| C[Filter: confidence > 0.95]
    B -->|Moderate| D[Filter: confidence > 0.85]
    B -->|Aggressive| E[Filter: confidence > 0.7]
    
    C --> F[Apply Fixes]
    D --> F
    E --> F
    
    F --> G{Fix Type}
    G -->|Duplicate| H[Remove rows]
    G -->|Label Error| I[Relabel]
    G -->|Outlier| J[Remove rows]
    
    H --> K[FixResult]
    I --> K
    J --> K
```

## Key Design Decisions

### 1. Lazy Evaluation

Detection algorithms only run when `analyze()` is called, not during initialization.

### 2. Composable Detectors

Each detector is independent and can be run separately:

```python
from clean.detection import LabelErrorDetector

detector = LabelErrorDetector()
errors = detector.detect(X, y)
```

### 3. Immutable Reports

`QualityReport` is immutable after creation. Modifications create new instances.

### 4. Optional Dependencies

Heavy dependencies (torch, transformers) are optional:

```python
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
```

### 5. Type Safety

Full type hints throughout:

```python
def analyze(
    self,
    detectors: Optional[List[str]] = None,
    show_progress: bool = True,
) -> QualityReport:
```

## Directory Structure

```
src/clean/
├── __init__.py          # Public API exports
├── core/
│   ├── cleaner.py       # DatasetCleaner
│   ├── report.py        # QualityReport
│   └── types.py         # Enums, dataclasses
├── detection/
│   ├── base.py          # BaseDetector
│   ├── label_errors.py
│   ├── duplicates.py
│   ├── outliers.py
│   ├── imbalance.py
│   └── bias.py
├── loaders/
│   ├── base.py          # BaseLoader
│   ├── pandas_loader.py
│   ├── numpy_loader.py
│   ├── csv_loader.py
│   ├── huggingface_loader.py
│   └── image_loader.py
├── scoring/
│   ├── quality_scorer.py
│   └── metrics.py
├── fix/
│   └── engine.py        # FixEngine
├── streaming/
│   └── cleaner.py       # StreamingCleaner
├── llm/
│   └── cleaner.py       # LLMDataCleaner
├── lineage/
│   └── tracker.py       # LineageTracker
├── plugins/
│   └── registry.py      # PluginRegistry
├── visualization/
│   ├── plots.py
│   └── interactive.py
├── utils/
│   ├── validation.py
│   └── export.py
│
# Enterprise modules
├── realtime.py          # Real-time streaming pipeline
├── automl.py            # AutoML threshold tuning
├── cloud.py             # Multi-tenant SaaS
├── root_cause.py        # Root cause analysis
├── vectordb.py          # Vector DB connectors
├── model_aware.py       # Model-aware scoring
├── slice_discovery.py   # Data slice discovery
├── privacy.py           # PII detection & anonymization
└── collaboration.py     # Collaborative review
```

## Enterprise Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        BATCH[Batch Data]
        STREAM[Kafka/Pulsar/Redis]
        VECDB[Vector DB]
    end
    
    subgraph "Processing"
        RT[RealtimePipeline]
        DC[DatasetCleaner]
        TUNE[QualityTuner]
    end
    
    subgraph "Analysis"
        REPORT[QualityReport]
        RCA[RootCauseAnalyzer]
        SLICE[SliceDiscovery]
        MODEL[ModelAwareScorer]
    end
    
    subgraph "Privacy & Review"
        PRIV[PrivacyVault]
        COLLAB[ReviewWorkspace]
    end
    
    subgraph "Deployment"
        CLOUD[CloudService]
        API[REST API]
    end
    
    BATCH --> DC
    STREAM --> RT
    VECDB --> DC
    
    RT --> REPORT
    DC --> REPORT
    TUNE --> DC
    
    REPORT --> RCA
    REPORT --> SLICE
    REPORT --> MODEL
    
    BATCH --> PRIV
    REPORT --> COLLAB
    
    DC --> CLOUD
    CLOUD --> API
```

## Extension Points

| Extension | How to Extend |
|-----------|---------------|
| New detector | Extend `BaseDetector`, register with `PluginRegistry` |
| New loader | Extend `BaseLoader`, register with `PluginRegistry` |
| New scorer | Extend `BaseScorer`, register with `PluginRegistry` |
| New export format | Extend `QualityReport.export()` |
| New vector backend | Extend `VectorDBConnector` |
| New stream source | Extend `StreamSource` |
