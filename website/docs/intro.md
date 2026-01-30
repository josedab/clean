---
sidebar_position: 1
slug: /
title: Introduction
---

# Clean: AI-Powered Data Quality for ML

**Clean** is a Python library that automatically detects and helps fix the issues that make ML models fail: label errors, duplicates, outliers, and biases.

## The Problem

> "Garbage in, garbage out"

Most ML debugging focuses on model architecture, hyperparameters, and training procedures. But research shows that **data quality issues** cause the majority of ML failures:

- **Mislabeled training samples** teach your model the wrong patterns
- **Duplicate data** inflates metrics and causes data leakage
- **Outliers** skew learned representations
- **Class imbalance** leads to poor minority-class performance
- **Bias in training data** gets amplified by models

Traditional data validation tools check schemas and types. Clean goes deeper—it understands ML.

## What Clean Does

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='target')
report = cleaner.analyze()

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

## Key Features

| Feature | Description |
|---------|-------------|
| **Label Error Detection** | Find mislabeled samples using confident learning |
| **Duplicate Detection** | Exact and near-duplicates with semantic similarity |
| **Outlier Detection** | Statistical and ML-based methods |
| **Bias Detection** | Demographic parity and fairness metrics |
| **LLM Data Quality** | Specialized for instruction-tuning datasets |
| **Auto-Fix Engine** | Suggested corrections with confidence scores |
| **Streaming** | Process datasets too large for memory |
| **Plugin System** | Extend with custom detectors |

## Enterprise Features

| Feature | Description |
|---------|-------------|
| **Real-Time Streaming** | Monitor Kafka/Pulsar/Redis with quality alerting |
| **AutoML Tuning** | Optimize thresholds automatically |
| **Root Cause Analysis** | Understand why issues occur |
| **Slice Discovery** | Find underperforming data subgroups |
| **Model-Aware Scoring** | Quality metrics tailored to your ML model |
| **Privacy Vault** | PII detection, anonymization, encryption |
| **Collaborative Review** | Team-based annotation review with voting |
| **Vector DB Integration** | Scale to millions with Pinecone/Milvus/Qdrant |
| **Multi-Tenant Cloud** | Deploy as SaaS with RBAC and billing |

## When to Use Clean

✅ **Use Clean when:**
- Training a new model and want clean data from the start
- Model performance plateaued and you suspect data issues
- Auditing a dataset before using it
- Building data quality checks into your MLOps pipeline
- Cleaning LLM fine-tuning data

❌ **Clean is not for:**
- Schema validation (use Pydantic, Pandera)
- Data profiling only (use ydata-profiling)
- Feature engineering (use Featuretools)

## Next Steps

- [**Getting Started**](/docs/getting-started) - Install and run your first analysis in 5 minutes
- [**Core Concepts**](/docs/concepts/overview) - Understand how Clean thinks about data quality
- [**Enterprise Features**](/docs/guides/realtime) - Real-time streaming, AutoML, and more
- [**API Reference**](/docs/api/cleaner) - Full API documentation
