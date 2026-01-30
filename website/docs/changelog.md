---
sidebar_position: 14
title: Changelog
---

# Changelog

All notable changes to Clean are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the full changelog, see [CHANGELOG.md on GitHub](https://github.com/clean-data/clean/blob/main/CHANGELOG.md).

## [1.0.0] - 2026-01-29

### 🎉 Initial Public Release

Clean is now available on PyPI!

```bash
pip install clean-data-quality
```

### Core Features

| Feature | Description |
|---------|-------------|
| **Label Error Detection** | Find mislabeled samples using confident learning |
| **Duplicate Detection** | Exact and near-duplicate finding with semantic similarity |
| **Outlier Detection** | Statistical (IQR, z-score) and ML-based (Isolation Forest, LOF) |
| **Bias Detection** | Demographic parity, equalized odds, and fairness metrics |
| **Class Imbalance** | Distribution analysis with resampling recommendations |

### LLM & Advanced Features

- **LLM Data Quality** - Specialized analysis for instruction-tuning and RAG datasets
- **Auto-Fix Engine** - Automatic fix suggestions with confidence scores
- **Plugin System** - Extensible architecture for custom detectors and fixers
- **Streaming Analysis** - Process large datasets that don't fit in memory
- **REST API** - HTTP endpoints for dashboard integration
- **CLI Tool** - Command-line interface for quick analysis
- **Data Lineage** - Audit trail for analysis runs and review decisions

### Enterprise Features (Preview)

The following enterprise features are included in this release:

- **Real-Time Streaming** - Kafka, Pulsar, Redis Streams with quality alerting
- **AutoML Tuning** - Bayesian/evolutionary optimization for quality thresholds
- **Root Cause Analysis** - Automated drill-down into quality issue causes
- **Slice Discovery** - Find underperforming data subgroups automatically
- **Privacy Vault** - PII detection, anonymization, encryption, audit logging
- **Collaborative Review** - Multi-user annotation review with voting and consensus
- **Vector DB Integration** - Pinecone, Weaviate, Milvus, Qdrant support
- **Multi-Tenant Cloud** - RBAC, workspaces, API keys for SaaS deployment

### Installation Options

```bash
# Core
pip install clean-data-quality

# With text embeddings
pip install clean-data-quality[text]

# With image support
pip install clean-data-quality[image]

# Everything
pip install clean-data-quality[all]
```

---

## Upgrade Guide

### From Pre-release Versions

If you were using a pre-release version of Clean:

1. Uninstall the old version:
   ```bash
   pip uninstall clean
   ```

2. Install the new package:
   ```bash
   pip install clean-data-quality
   ```

3. Update imports:
   ```python
   # Old
   from clean import DatasetCleaner
   
   # New (unchanged)
   from clean import DatasetCleaner
   ```

The API is backwards compatible. Your existing code should work without changes.

---

## Release Schedule

Clean follows semantic versioning:

- **Major versions** (2.0, 3.0): Breaking API changes
- **Minor versions** (1.1, 1.2): New features, backwards compatible
- **Patch versions** (1.0.1, 1.0.2): Bug fixes only

We aim for quarterly minor releases and immediate patch releases for critical bugs.

---

## Contributing

Found a bug? Have a feature request?

- 🐛 [Report bugs](https://github.com/clean-data/clean/issues/new?template=bug_report.yml)
- 💡 [Request features](https://github.com/clean-data/clean/issues/new?template=feature_request.yml)
- 📖 [Improve docs](https://github.com/clean-data/clean/edit/main/docs/)

See the [Contributing Guide](/docs/contributing) for details.
