# Clean 🧹

[![PyPI version](https://badge.fury.io/py/clean-data-quality.svg)](https://badge.fury.io/py/clean-data-quality)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/clean-data/clean/workflows/CI/badge.svg)](https://github.com/clean-data/clean/actions)
[![Coverage](https://codecov.io/gh/clean-data/clean/branch/main/graph/badge.svg)](https://codecov.io/gh/clean-data/clean)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://clean-data.github.io/clean)

**AI-powered data quality platform for ML datasets. Find and fix label errors, duplicates, outliers, and biases.**

Clean automatically detects the issues that make ML models fail. The saying "garbage in, garbage out" has never been more true—yet most data quality tools focus on data validation, not ML-specific issues. Clean fills this gap.

## ✨ Features

### Core Detection

| Feature | Description |
|---------|-------------|
| 🏷️ **Label Error Detection** | Find mislabeled samples using confident learning |
| 🔍 **Duplicate Detection** | Exact and near-duplicate finding with semantic similarity |
| 📊 **Outlier Detection** | Statistical (IQR, z-score) and ML-based (Isolation Forest, LOF) |
| ⚖️ **Bias Detection** | Demographic parity, equalized odds, and fairness metrics |
| 📈 **Class Imbalance** | Distribution analysis with resampling recommendations |

### Advanced Capabilities

| Feature | Description |
|---------|-------------|
| 🤖 **LLM Data Quality** | Specialized analysis for instruction-tuning and RAG datasets |
| 🔧 **Auto-Fix Engine** | Automatic fix suggestions with confidence scores |
| 🔌 **Plugin System** | Extensible architecture for custom detectors and fixers |
| 📡 **Streaming Analysis** | Process large datasets that don't fit in memory |
| 🌐 **REST API** | HTTP endpoints for dashboard integration |
| 💻 **CLI Tool** | Command-line interface for quick analysis |
| 📜 **Data Lineage** | Audit trail for analysis runs and review decisions |

### Next-Gen Features (New!)

| Feature | Description |
|---------|-------------|
| 📉 **Data Drift Monitor** | Detect distribution shifts with KS test, PSI, Wasserstein distance |
| 👥 **Annotation Quality** | Inter-annotator agreement (Krippendorff's α, Fleiss' κ, Cohen's κ) |
| 🛡️ **LLM Evaluation Suite** | Toxicity, PII, prompt injection detection for LLM datasets |
| 🧬 **Synthetic Data Validator** | Mode collapse, memorization, distribution gap detection |
| 📋 **Compliance Reports** | EU AI Act and NIST AI RMF compliance documentation |
| 🎯 **Active Learning** | Uncertainty sampling with Label Studio/CVAT/Prodigy export |
| 🖼️ **Multi-Modal Analysis** | Image-text alignment and cross-modal consistency |
| ⚡ **Distributed Processing** | Dask and Spark backends for large-scale analysis |
| 🖥️ **Web Dashboard** | Interactive browser-based quality monitoring UI |
| 🔄 **CI/CD Integration** | GitHub Action for automated quality gates |

### Enterprise Features (Latest!)

| Feature | Description |
|---------|-------------|
| 📡 **Real-Time Streaming** | Kafka, Pulsar, Redis Streams with quality alerting |
| 🤖 **AutoML Tuning** | Bayesian/evolutionary optimization for quality thresholds |
| ☁️ **Multi-Tenant Cloud** | RBAC, workspaces, API keys, billing for SaaS deployment |
| 🔍 **Root Cause Analysis** | Automated drill-down into quality issue causes |
| 🗄️ **Vector DB Integration** | Pinecone, Weaviate, Milvus, Qdrant for scale |
| 📊 **Model-Aware Scoring** | Quality metrics tailored to your target model |
| 🎓 **Intelligent Sampling** | Advanced active learning with committee/EMC strategies |
| 🍕 **Slice Discovery** | Find underperforming data subgroups automatically |
| 🔐 **Privacy Vault** | PII detection, anonymization, encryption, audit logging |
| 👥 **Collaborative Review** | Multi-user annotation review with voting and consensus |

## 🚀 Quick Start

### Installation

```bash
pip install clean-data-quality
```

Optional features:

```bash
pip install clean-data-quality[text]        # Text embeddings (sentence-transformers)
pip install clean-data-quality[image]       # Image analysis (CLIP, torchvision)
pip install clean-data-quality[interactive] # Interactive plots (plotly, ipywidgets)
pip install clean-data-quality[api]         # REST API (FastAPI, uvicorn)
pip install clean-data-quality[streaming]   # Kafka, Pulsar, Redis Streams
pip install clean-data-quality[vectordb]    # Pinecone, Weaviate, Milvus, Qdrant
pip install clean-data-quality[all]         # Everything
```

### Basic Usage

```python
from clean import DatasetCleaner

# Initialize with your data
cleaner = DatasetCleaner(data=df, label_column='label')

# Run comprehensive analysis
report = cleaner.analyze()

# View summary
print(report.summary())
```

Output:
```
Data Quality Report
==================
Samples analyzed: 10,000
Quality Score: 82.5/100

Issues Found:
  - Label errors: 347 (3.5%) - HIGH PRIORITY
  - Near-duplicates: 234 pairs (4.7%)
  - Outliers: 156 (1.6%)
  - Class imbalance: 15:1 ratio
```

### Get Issue Details

```python
# Get label errors with suggested corrections
errors = report.label_errors()
print(errors.head())
#   index  given_label  predicted_label  confidence
#   42     cat          dog              0.94
#   187    cat          bird             0.89

# Get duplicate pairs
duplicates = report.duplicates()

# Get outlier indices
outliers = report.outliers()
```

### Apply Fixes

```python
from clean import FixEngine, FixConfig

# Configure fix strategy
config = FixConfig(
    auto_relabel=False,           # Don't auto-correct labels
    label_error_threshold=0.95,   # High confidence only
    outlier_action="flag",        # Flag, don't remove
)

# Create engine and apply
engine = FixEngine(report=report, features=X, labels=y, config=config)
fixes = engine.suggest_fixes()
result = engine.apply_fixes(fixes)

print(f"Applied {result.n_applied} fixes")
```

## 🤖 LLM Data Quality

Specialized analysis for instruction-tuning and RAG datasets:

```python
from clean import LLMDataCleaner

cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
    min_response_length=10,
)

report = cleaner.analyze(df)
print(report.summary())
# Issues: 45 duplicate instructions, 12 refusals, 8 short responses

# Get clean data
clean_df = report.get_clean_data(df, remove_duplicates=True, remove_refusals=True)
```

## 📡 Streaming for Large Datasets

Process datasets that don't fit in memory:

```python
from clean import StreamingCleaner
import asyncio

async def analyze_large_file():
    cleaner = StreamingCleaner(label_column="label", chunk_size=50000)
    
    async for result in cleaner.analyze_file("large_dataset.csv"):
        print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
    
    summary = cleaner.get_summary()
    print(f"Total: {summary.total_issues} issues in {summary.total_rows:,} rows")

asyncio.run(analyze_large_file())
```

## 💻 Command-Line Interface

```bash
# Analyze a dataset
clean analyze data.csv --label-column target --output report.json

# Apply fixes
clean fix data.csv --output cleaned.csv --strategy conservative

# Get dataset info
clean info data.csv

# Start REST API server
clean serve --port 8000
```

## 🌐 REST API

```bash
# Start server
clean serve --port 8000

# Analyze via API
curl -X POST http://localhost:8000/analyze \
  -F "file=@data.csv" \
  -F "label_column=target"
```

API endpoints: `/analyze`, `/analyze/streaming`, `/analyze/llm`, `/fix`, `/lineage/runs`

📖 [Full API documentation](docs/api/rest-api.md)

## 🆕 Next-Gen Features

### Data Drift Monitoring

```python
from clean import DriftDetector, DriftMonitor

# One-time drift detection
detector = DriftDetector(method="ks")  # ks, psi, wasserstein
report = detector.detect(reference_df, current_df)
print(f"Drift detected: {report.drift_detected}")

# Continuous monitoring with alerts
monitor = DriftMonitor(reference_df, alert_threshold=0.05)
monitor.add_alert_handler(lambda alert: print(f"ALERT: {alert}"))
report = monitor.check(new_data)
```

### Annotation Quality Analysis

```python
from clean import AnnotationAnalyzer

# Analyze inter-annotator agreement
analyzer = AnnotationAnalyzer()
report = analyzer.analyze(annotations_df, annotator_column="annotator", label_column="label")

print(f"Krippendorff's Alpha: {report.overall_agreement:.3f}")
print(f"Fleiss' Kappa: {report.fleiss_kappa:.3f}")
for annotator, metrics in report.per_annotator_metrics.items():
    print(f"  {annotator}: agreement={metrics['agreement']:.2%}")
```

### LLM Safety Evaluation

```python
from clean import LLMEvaluator

evaluator = LLMEvaluator()
report = evaluator.evaluate(df, prompt_column="prompt", response_column="response")

print(f"Safety Score: {report.safety_score:.1f}/100")
print(f"Toxicity issues: {report.toxicity_count}")
print(f"PII detected: {report.pii_count}")
print(f"Prompt injections: {report.injection_count}")
```

### Synthetic Data Validation

```python
from clean import validate_synthetic_data

report = validate_synthetic_data(
    real_data=original_df,
    synthetic_data=generated_df,
)

print(f"Fidelity Score: {report.fidelity_score:.1f}")
print(f"Mode Collapse: {report.has_mode_collapse}")
print(f"Memorization Risk: {report.memorization_risk}")
```

### Compliance Reports

```python
from clean import generate_compliance_report, ComplianceFramework

report = generate_compliance_report(
    df,
    label_column="label",
    frameworks=[ComplianceFramework.EU_AI_ACT, ComplianceFramework.NIST_AI_RMF],
)

print(report.summary())
report.export_pdf("compliance_report.pdf")
```

### Active Learning Integration

```python
from clean import ActiveLearner, LabelStudioExporter

learner = ActiveLearner(strategy="uncertainty")
learner.fit(X_labeled, y_labeled)

# Select samples for labeling
indices = learner.select(X_unlabeled, n_samples=100)

# Export to labeling tool
exporter = LabelStudioExporter()
exporter.export(X_unlabeled[indices], "label_studio_tasks.json")
```

### Distributed Processing

```python
from clean import DaskCleaner, analyze_distributed

# Process large datasets with Dask
cleaner = DaskCleaner(n_workers=8)
report = cleaner.analyze("large_dataset.csv", label_column="label", chunk_size=100000)

# Or use convenience function
report = analyze_distributed(df, backend="dask", n_workers=4)
```

### Web Dashboard

```bash
# Start interactive dashboard
clean dashboard --port 8080
```

```python
# Or programmatically
from clean import run_dashboard
run_dashboard(host="0.0.0.0", port=8080, title="My Data Quality Dashboard")
```

### CI/CD Quality Gates

```yaml
# .github/workflows/data-quality.yml
- uses: clean-data/clean-action@v1
  with:
    file: data/training.csv
    label-column: label
    fail-below: 80
```

```bash
# CLI quality check
clean check data.csv --label-column label --fail-below 80 --github-output
```

## 🏢 Enterprise Features

### Real-Time Streaming Pipeline

```python
import asyncio
from clean.realtime import RealtimePipeline, KafkaSource, PipelineConfig

async def monitor_stream():
    source = KafkaSource(
        bootstrap_servers="localhost:9092",
        topic="ml-training-data",
    )
    
    config = PipelineConfig(window_size=1000, quality_threshold=0.8)
    pipeline = RealtimePipeline(source=source, config=config)
    
    # Alert on quality degradation
    pipeline.add_alert_handler(lambda alert: print(f"🚨 {alert}"))
    
    await pipeline.start()

asyncio.run(monitor_stream())
```

### AutoML Quality Threshold Tuning

```python
from clean.automl import QualityTuner, TuningConfig

config = TuningConfig(method="bayesian", n_trials=50, metric="f1")
tuner = QualityTuner(config=config)

result = tuner.tune(X=features, y=labels, validation_labels=ground_truth)
print(f"Best params: {result.best_params}")  # Optimized thresholds
```

### Root Cause Analysis

```python
from clean.root_cause import RootCauseAnalyzer

analyzer = RootCauseAnalyzer()
causes = analyzer.analyze(data=df, quality_report=report, issue_type="label_errors")

for cause in causes.top_causes[:3]:
    print(f"• {cause.description} (impact: {cause.impact_score:.2f})")
    print(f"  Fix: {cause.suggested_fix}")
```

### Data Slice Discovery

```python
from clean.slice_discovery import SliceDiscovery

discoverer = SliceDiscovery(method="decision_tree")
result = discoverer.discover(data=df, predictions=y_pred, targets=y_true)

for slice in result.top_slices[:5]:
    print(f"{slice.description}: accuracy={slice.metric_value:.3f} (gap: {slice.gap:+.3f})")
```

### Privacy Vault

```python
from clean.privacy import PrivacyVault

vault = PrivacyVault(pii_types=["email", "phone", "ssn"])

# Scan for PII
scan = vault.scan(df)
print(f"PII found: {scan.total_pii_count}")

# Anonymize
safe_df = vault.anonymize(df)
```

### Collaborative Review

```python
from clean.collaboration import ReviewWorkspace

workspace = ReviewWorkspace(storage_backend="sqlite")
session = workspace.create_session(
    name="Q4 Data Review",
    data=df,
    reviewers=["alice@co.com", "bob@co.com"],
)

# Reviewers submit votes, system builds consensus
consensus = session.get_consensus(min_votes=2)
```

📖 [Full Enterprise Features Documentation](docs/guides/next-gen-architecture.md)

## 🔌 Plugin System

Extend Clean with custom detectors:

```python
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    name = "my_detector"
    description = "Custom quality detector"
    
    def detect(self, data, labels, **kwargs):
        issues = []
        # Your detection logic
        return issues

# Use your plugin
detector = registry.get_detector("my_detector")
issues = detector.detect(df, labels)
```

📖 [Plugin migration guide](docs/guides/plugin-migration.md)

## 🔧 Supported Data Types

| Data Type | Loader | Features |
|-----------|--------|----------|
| Pandas DataFrame | `load_dataframe()` | Full support |
| NumPy Arrays | `load_arrays()` | Full support |
| CSV Files | `load_csv()` | Full support |
| HuggingFace Datasets | `load_huggingface()` | Full support |
| Image Folders | `load_image_folder()` | Label errors, duplicates |

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| [📚 Full Documentation](https://clean-data.github.io/clean) | Complete guides and API reference |
| [🚀 Quickstart](docs/getting-started/quickstart.md) | Get started in 5 minutes |
| [📓 Examples](examples/) | Jupyter notebooks with use cases |
| [🔌 Plugin Guide](docs/guides/plugin-migration.md) | Create custom plugins |
| [🌐 REST API](docs/api/rest-api.md) | HTTP endpoint reference |

## 🏗️ Architecture

Clean uses modern design patterns for extensibility and testability:

```
┌─────────────────────────────────────────────────────────────┐
│                      Clean Platform                          │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Loaders   │  Detectors  │   Fixers    │   Exporters      │
├─────────────┼─────────────┼─────────────┼──────────────────┤
│ • DataFrame │ • Labels    │ • Relabel   │ • JSON           │
│ • CSV       │ • Duplicates│ • Remove    │ • HTML           │
│ • HuggingFace│ • Outliers  │ • Impute    │ • DataFrame      │
│ • Images    │ • Imbalance │ • Merge     │                  │
│             │ • Bias      │             │                  │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│  Detection Strategies: IsolationForest│LOF│ZScore│IQR│MAD  │
├─────────────────────────────────────────────────────────────┤
│  Factory & DI: DetectorFactory │ ConfigurableDetectorFactory │
├─────────────────────────────────────────────────────────────┤
│                     Plugin Registry                          │
├─────────────────────────────────────────────────────────────┤
│  Core: DatasetCleaner │ QualityReport │ FixEngine          │
├─────────────────────────────────────────────────────────────┤
│  Streaming: ChunkProcessor │ AsyncChunkProcessor            │
├─────────────────────────────────────────────────────────────┤
│  Extensions: LLMDataCleaner │ StreamingCleaner │ LineageTracker│
├─────────────────────────────────────────────────────────────┤
│  Interfaces: Python API │ CLI │ REST API                    │
└─────────────────────────────────────────────────────────────┘
```

### Design Patterns

| Pattern | Usage | Benefit |
|---------|-------|---------|
| **Strategy** | Outlier/duplicate detection algorithms | Swap algorithms at runtime |
| **Factory** | `DetectorFactory` for DI | Easy testing and customization |
| **Parameter Object** | `DetectionResults` | Clean APIs with fewer params |

### Configuration Presets

```python
from clean import DatasetCleaner
from clean.detection.factory import ConfigurableDetectorFactory

# High precision (fewer false positives)
factory = ConfigurableDetectorFactory.high_precision()
cleaner = DatasetCleaner(data=df, detector_factory=factory)

# Fast scan (quick exploratory analysis)
factory = ConfigurableDetectorFactory.fast_scan()
cleaner = DatasetCleaner(data=df, detector_factory=factory)
```

📖 [Full Architecture Documentation](docs/architecture.md)

## 🧪 Development

```bash
# Clone and install
git clone https://github.com/clean-data/clean.git
cd clean
pip install -e ".[dev,all]"

# Run tests (900+ tests, 76% coverage)
pytest

# Run linter
ruff check src tests

# Build docs
mkdocs serve

# Run benchmarks
python benchmarks/strategy_benchmark.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- 🐛 [Report bugs](https://github.com/clean-data/clean/issues/new?template=bug_report.yml)
- 💡 [Request features](https://github.com/clean-data/clean/issues/new?template=feature_request.yml)
- 📖 [Improve docs](https://github.com/clean-data/clean/edit/main/docs/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Clean builds on excellent open-source projects:

- [cleanlab](https://github.com/cleanlab/cleanlab) - Confident learning for label errors
- [sentence-transformers](https://www.sbert.net/) - Text embeddings
- [scikit-learn](https://scikit-learn.org/) - ML algorithms
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework
