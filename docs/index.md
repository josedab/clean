# Clean 🧹

**AI-powered data debugging. Turn your messy dataset into a clean, high-quality training set.**

<div class="badges">
<a href="https://pypi.org/project/clean-data-quality/"><img src="https://badge.fury.io/py/clean-data-quality.svg" alt="PyPI version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://github.com/clean-data/clean/actions"><img src="https://github.com/clean-data/clean/workflows/CI/badge.svg" alt="Tests"></a>
</div>

---

## What is Clean?

Clean automatically detects and helps fix the issues that make ML models fail: **label errors**, **duplicates**, **outliers**, and **biases**. The saying "garbage in, garbage out" has never been more true—yet most data quality tools focus on data validation, not ML-specific issues.

## ✨ Features

<div class="grid">
<div class="card">
<h3>🏷️ Label Errors</h3>
<p>Find mislabeled data using confident learning algorithms.</p>
</div>
<div class="card">
<h3>🔍 Duplicates</h3>
<p>Detect exact and near-duplicate samples with semantic similarity.</p>
</div>
<div class="card">
<h3>📊 Outliers</h3>
<p>Identify anomalies using statistical and ML-based methods.</p>
</div>
<div class="card">
<h3>⚖️ Bias Detection</h3>
<p>Analyze demographic parity and fairness across groups.</p>
</div>
<div class="card">
<h3>📈 Quality Scoring</h3>
<p>Get overall and per-component quality scores for your dataset.</p>
</div>
<div class="card">
<h3>🎨 Visualization</h3>
<p>Interactive issue browser and distribution plots.</p>
</div>
</div>

## Quick Start

### Installation

```bash
pip install clean-data-quality
```

For additional features:

=== "Text Data"
    ```bash
    pip install clean-data-quality[text]
    ```

=== "Image Data"
    ```bash
    pip install clean-data-quality[image]
    ```

=== "Interactive Viz"
    ```bash
    pip install clean-data-quality[interactive]
    ```

=== "Everything"
    ```bash
    pip install clean-data-quality[all]
    ```

### Basic Usage

```python
from clean import DatasetCleaner

# Initialize with your data
cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    task='classification'
)

# Run comprehensive analysis
report = cleaner.analyze()

# View summary
print(report.summary())
```

**Output:**
```
Data Quality Report
==================
Samples analyzed: 10,000

Issues Found:
- Label errors: 347 (3.5%) - HIGH PRIORITY
- Near-duplicates: 234 pairs (4.7%)
- Outliers: 156 (1.6%)
- Class imbalance: 15:1 ratio
```

### Get Specific Issues

```python
# Get label errors as DataFrame
label_errors = report.label_errors()
print(label_errors.head())
```

```
   index  current_label  suggested_label  confidence
0     42           cat              dog        0.94
1    187           cat             bird        0.89
```

### Export Clean Dataset

```python
clean_df = cleaner.get_clean_data(
    remove_duplicates=True,
    remove_outliers='conservative'
)
```

## Why Clean?

| Challenge | Traditional Approach | Clean's Solution |
|-----------|---------------------|------------------|
| Mislabeled data | Manual review | Confident learning detection |
| Hidden duplicates | Hash matching only | Semantic similarity |
| Outliers | Statistical thresholds | ML-based ensemble |
| Bias | Post-hoc auditing | Integrated fairness checks |

## Next-Gen Features

<div class="grid">
<div class="card">
<h3>📉 Data Drift Monitor</h3>
<p>Detect distribution shifts with KS test, PSI, Wasserstein distance.</p>
</div>
<div class="card">
<h3>👥 Annotation Quality</h3>
<p>Inter-annotator agreement metrics (Krippendorff's α, Fleiss' κ).</p>
</div>
<div class="card">
<h3>🛡️ LLM Evaluation Suite</h3>
<p>Toxicity, PII, prompt injection detection for LLM datasets.</p>
</div>
<div class="card">
<h3>🧬 Synthetic Data Validator</h3>
<p>Mode collapse, memorization, distribution gap detection.</p>
</div>
<div class="card">
<h3>📋 Compliance Reports</h3>
<p>EU AI Act and NIST AI RMF compliance documentation.</p>
</div>
<div class="card">
<h3>🎯 Active Learning</h3>
<p>Uncertainty sampling with Label Studio/CVAT/Prodigy export.</p>
</div>
</div>

## Enterprise Features

<div class="grid">
<div class="card">
<h3>📡 Real-Time Streaming</h3>
<p>Kafka, Pulsar, Redis Streams with quality alerting.</p>
</div>
<div class="card">
<h3>🤖 AutoML Tuning</h3>
<p>Bayesian/evolutionary optimization for quality thresholds.</p>
</div>
<div class="card">
<h3>☁️ Multi-Tenant Cloud</h3>
<p>RBAC, workspaces, API keys for SaaS deployment.</p>
</div>
<div class="card">
<h3>🔍 Root Cause Analysis</h3>
<p>Automated drill-down into quality issue causes.</p>
</div>
<div class="card">
<h3>🔐 Privacy Vault</h3>
<p>PII detection, anonymization, encryption, audit logging.</p>
</div>
<div class="card">
<h3>👥 Collaborative Review</h3>
<p>Multi-user annotation review with voting and consensus.</p>
</div>
</div>

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation options
- [Quickstart Tutorial](getting-started/quickstart.md) - Step-by-step walkthrough
- [API Reference](api/cleaner.md) - Complete API documentation

## License

MIT License - see [LICENSE](https://github.com/clean-data/clean/blob/main/LICENSE) for details.
