# Next-Gen Features Guide

This guide covers the advanced features added to Clean for enterprise-grade data quality management.

## Overview

Clean's next-gen features extend beyond basic quality detection to provide:

- **Production monitoring** with drift detection and alerting
- **Team collaboration** with annotation quality analysis
- **LLM-specific** safety and quality evaluation
- **Synthetic data** validation for generated datasets
- **Regulatory compliance** documentation
- **Efficient labeling** with active learning
- **Multi-modal** data consistency checks
- **Scale** with distributed processing
- **Visibility** with the web dashboard
- **Automation** with CI/CD integration

## Quick Comparison

| Feature | Use Case | Key Benefit |
|---------|----------|-------------|
| Drift Monitor | Production data changes | Early warning of issues |
| Annotation Quality | Multi-annotator projects | Ensure consistent labels |
| LLM Eval Suite | LLM training data | Safety and compliance |
| Synthetic Validator | Generated data | Quality assurance |
| Compliance Reports | Regulated industries | Audit-ready documentation |
| Active Learning | Limited labeling budget | Efficient annotation |
| Multi-Modal | Image-text datasets | Cross-modal consistency |
| Distributed | Large datasets | Scalability |
| Dashboard | Stakeholder visibility | Easy monitoring |
| CI/CD | Automated pipelines | Quality gates |

## Feature Guides

### 1. Data Drift Monitoring

Detect when your production data distribution changes.

```python
from clean import DriftDetector, DriftMonitor

# Setup reference from training data
reference_data = pd.read_csv("training_data.csv")

# One-time drift check
detector = DriftDetector(method="psi")
report = detector.detect(reference_data, new_batch)

if report.drift_detected:
    print(f"⚠️ Drift detected in: {report.drifted_features}")

# Continuous monitoring with alerts
monitor = DriftMonitor(reference_data, alert_threshold=0.1)

def send_alert(alert):
    # Send to Slack, PagerDuty, etc.
    slack.post(f"🚨 Data Drift Alert: {alert.message}")

monitor.add_alert_handler(send_alert)

# Check each new batch
for batch in production_stream:
    monitor.check(batch)
```

**When to use:** Model performance degradation, production data pipelines

---

### 2. Annotation Quality Analysis

Ensure your annotators agree on labels.

```python
from clean import AnnotationAnalyzer

# Load annotations from multiple annotators
annotations = pd.DataFrame({
    "sample_id": [1, 1, 1, 2, 2, 2],
    "annotator": ["alice", "bob", "carol", "alice", "bob", "carol"],
    "label": ["spam", "spam", "ham", "ham", "ham", "ham"],
})

analyzer = AnnotationAnalyzer()
report = analyzer.analyze(
    annotations,
    sample_column="sample_id",
    annotator_column="annotator",
    label_column="label",
)

print(f"Agreement: {report.overall_agreement:.2f}")

# Find problem annotators
for annotator, metrics in report.per_annotator_metrics.items():
    if metrics["agreement"] < 0.8:
        print(f"⚠️ {annotator} needs review (agreement: {metrics['agreement']:.1%})")
```

**When to use:** Crowdsourced labeling, quality assurance for annotation teams

---

### 3. LLM Safety Evaluation

Check LLM training data for safety issues.

```python
from clean import LLMEvaluator

evaluator = LLMEvaluator(
    check_toxicity=True,
    check_pii=True,
    check_injection=True,
)

# Evaluate instruction-tuning dataset
report = evaluator.evaluate(
    df,
    prompt_column="instruction",
    response_column="output",
)

print(f"Safety Score: {report.safety_score:.0f}/100")

# Get samples needing review
if report.toxicity_count > 0:
    toxic = report.get_flagged_samples(issue_type="toxicity")
    print(f"Review {len(toxic)} samples for toxicity")

if report.pii_count > 0:
    pii = report.get_flagged_samples(issue_type="pii")
    print(f"Remove PII from {len(pii)} samples")
```

**When to use:** Fine-tuning LLMs, RAG dataset curation, safety audits

---

### 4. Synthetic Data Validation

Validate quality of generated datasets.

```python
from clean import validate_synthetic_data

# Compare generated data to real data
report = validate_synthetic_data(
    real_data=original_df,
    synthetic_data=generated_df,
)

print(f"Quality: {report.quality_score:.0f}/100")
print(f"Fidelity: {report.fidelity_score:.0f}/100")

# Check for common issues
if report.has_mode_collapse:
    print("⚠️ Generator producing limited variety")
    
if report.memorization_risk > 0.05:
    print(f"⚠️ {report.memorization_risk:.1%} samples may be memorized")
```

**When to use:** GANs, VAEs, diffusion models, data augmentation

---

### 5. Compliance Reporting

Generate regulatory compliance documentation.

```python
from clean import generate_compliance_report, ComplianceFramework

report = generate_compliance_report(
    df,
    label_column="label",
    frameworks=[
        ComplianceFramework.EU_AI_ACT,
        ComplianceFramework.NIST_AI_RMF,
    ],
    metadata={
        "dataset_name": "Customer Prediction Model Training Data",
        "version": "2.1.0",
        "owner": "ML Team",
    }
)

# Check compliance status
for req in report.requirements:
    status = "✓" if req.status == "compliant" else "✗"
    print(f"{status} {req.name}")

# Export for auditors
report.export_pdf("compliance_audit_q1_2024.pdf")
```

**When to use:** EU AI Act compliance, NIST frameworks, internal audits

---

### 6. Active Learning for Efficient Labeling

Select the most valuable samples to label.

```python
from clean import ActiveLearner, LabelStudioExporter

# Start with small labeled set
learner = ActiveLearner(strategy="uncertainty")
learner.fit(X_labeled, y_labeled)

# Select samples for labeling
indices = learner.select(X_unlabeled, n_samples=100)

# Export to labeling tool
exporter = LabelStudioExporter(project_name="Batch 1")
exporter.export(
    data=X_unlabeled[indices],
    output_path="batch_1_tasks.json",
)

# After labeling, update model and repeat
new_labels = load_labels("batch_1_labeled.json")
learner.update(X_unlabeled[indices], new_labels)
```

**When to use:** Limited annotation budget, iterative model improvement

---

### 7. Multi-Modal Consistency

Check image-text alignment in multi-modal datasets.

```python
from clean import MultiModalAnalyzer

analyzer = MultiModalAnalyzer(
    modalities=["image", "text"],
    alignment_threshold=0.4,
)

report = analyzer.analyze(
    df,
    image_column="image_path",
    text_column="caption",
)

print(f"Alignment Score: {report.alignment_score:.2f}")
print(f"Misaligned pairs: {report.n_misaligned}")

# Review misaligned samples
for idx in report.get_misaligned()[:10]:
    print(f"Sample {idx}: Review image-caption match")
```

**When to use:** Image captioning, VQA, multi-modal training data

---

### 8. Distributed Processing

Scale to large datasets.

```python
from clean import analyze_distributed

# Process large file with Dask
report = analyze_distributed(
    "large_dataset.csv",
    label_column="label",
    backend="dask",
    n_workers=8,
    chunk_size=100000,
)

print(f"Analyzed {report.total_samples:,} samples in {report.processing_time_seconds:.0f}s")
```

**When to use:** Datasets > 1GB, production data pipelines

---

### 9. Web Dashboard

Visual monitoring interface.

```bash
# Start dashboard
clean dashboard --port 8080
```

Then open http://localhost:8080 to:
- Upload and analyze CSV files
- View quality scores and trends
- Explore issue breakdowns
- Monitor feature-level quality

**When to use:** Stakeholder visibility, ad-hoc analysis, demos

---

### 10. CI/CD Quality Gates

Automate quality checks in pipelines.

```yaml
# .github/workflows/data-quality.yml
- uses: clean-data/clean-action@v1
  with:
    file: data/training.csv
    label-column: label
    fail-below: 80
```

```bash
# CLI alternative
clean check data.csv -l label --fail-below 80 --github-output
```

**When to use:** Automated pipelines, PR checks, deployment gates

## Combining Features

### Production ML Pipeline

```python
from clean import (
    DriftMonitor,
    LLMEvaluator,
    generate_compliance_report,
    analyze_distributed,
)

# 1. Check incoming data for drift
monitor = DriftMonitor(reference_data)
drift_report = monitor.check(new_data)

if drift_report.drift_detected:
    alert_team(drift_report)

# 2. Analyze at scale
quality_report = analyze_distributed(new_data, backend="dask")

if quality_report.overall_quality_score < 80:
    block_pipeline()

# 3. LLM safety check (if applicable)
safety_report = LLMEvaluator().evaluate(new_data)

# 4. Generate compliance documentation
compliance = generate_compliance_report(new_data)
compliance.export_pdf(f"compliance_{date.today()}.pdf")
```

## Next Steps

- [API Reference](../api/) - Detailed API documentation
- [Examples](../../examples/) - Jupyter notebooks with full examples
- [CLI Reference](../api/cli.md) - Command-line interface details
