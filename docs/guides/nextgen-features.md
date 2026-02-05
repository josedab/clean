# Next-Generation Features

Clean 1.0 introduces 10 powerful next-generation features for advanced data quality workflows.

## Overview

| Feature | Description | Use Case |
|---------|-------------|----------|
| [Quality Predictor](#quality-prediction-model) | Predict quality scores instantly | CI/CD quality gates |
| [Natural Language Query](#natural-language-query) | Query reports in plain English | Interactive exploration |
| [Quality Augmentation](#quality-aware-augmentation) | Smart data augmentation | Fix class imbalance |
| [Contamination Detector](#contamination-detector) | Detect train/test leakage | Ensure dataset integrity |
| [Curriculum Learning](#curriculum-learning) | Optimize training order | Improve model learning |
| [Quality Regression](#quality-regression-testing) | Track quality over time | Prevent degradation |
| [Embedding Visualizer](#embedding-visualizer) | Visualize data in 2D/3D | Understand data structure |
| [Synthetic Certification](#synthetic-certification) | Certify synthetic data | Validate generated data |
| [Data Marketplace](#data-marketplace) | Industry benchmarking | Compare to peers |
| [Labeler Scoring](#labeler-scoring) | Evaluate labeler performance | Smart task routing |

---

## Quality Prediction Model

Instantly predict data quality scores without running full analysis.

```python
from clean.quality_predictor import QualityPredictor, QualityGate

# Train predictor on historical data
predictor = QualityPredictor()
predictor.fit(historical_datasets, historical_scores)

# Predict quality for new data
prediction = predictor.predict(new_data)
print(f"Predicted score: {prediction.quality_score:.1f}")
print(f"Confidence: {prediction.confidence:.1%}")

# Use as CI/CD gate
gate = QualityGate(predictor, threshold=80.0)
result = gate.check(data)
if not result.passed:
    raise ValueError("Quality below threshold!")
```

---

## Natural Language Query

Explore quality reports using natural language.

```python
from clean.nl_query import NLQueryEngine, create_query_engine

# Create engine with a quality report
engine = create_query_engine(report)

# Ask questions naturally
result = engine.query("What are the top label errors?")
print(result.response)

result = engine.query("Show me duplicates in category A")
print(result.data)

# Interactive session
engine.query("What's the overall quality?")
engine.query("Why is it low?")  # Maintains context
```

---

## Quality-Aware Augmentation

Augment data to fix quality gaps like class imbalance.

```python
from clean.quality_augmentation import QualityAwareAugmenter, augment_for_quality

# Analyze and augment in one step
result = augment_for_quality(X, y)
X_improved = result.X_augmented
y_improved = result.y_augmented

print(f"Original samples: {result.n_original}")
print(f"Augmented samples: {result.n_augmented}")

# Or use the augmenter with more control
augmenter = QualityAwareAugmenter()
result = augmenter.augment(X, y)
```

---

## Contamination Detector

Detect data leakage between train/test splits.

```python
from clean.contamination import ContaminationDetector, detect_contamination

# Quick detection
report = detect_contamination(train_df, test_df)
if report.has_contamination:
    print(f"Found {len(report.contaminated_pairs)} leaking samples!")

# Detailed analysis
detector = ContaminationDetector()
detector.register_dataset("train", train_data)
detector.register_dataset("validation", val_data)
detector.register_dataset("test", test_data)

report = detector.detect()
print(report.summary())
```

---

## Curriculum Learning

Optimize training data order for better model learning.

```python
from clean.curriculum import CurriculumOptimizer, create_curriculum

# Create curriculum (easy to hard ordering)
schedule = create_curriculum(X, y, strategy="easy_to_hard")

# Get ordered data for training
optimizer = CurriculumOptimizer()
indices = optimizer.optimize(X, y, quality_scores)

# Train with curriculum
for epoch in range(n_epochs):
    epoch_indices = schedule.get_indices(epoch)
    train_on_batch(X[epoch_indices], y[epoch_indices])
```

---

## Quality Regression Testing

Track quality metrics over time and detect regressions.

```python
from clean.quality_regression import (
    QualityRegressionTester,
    QualitySnapshot,
    QualityHistoryStore,
)

# Store quality snapshots
store = QualityHistoryStore(path="quality_history.db")

snapshot = QualitySnapshot(
    version="v1.2.0",
    quality_score=report.quality_score,
    metrics={"error_rate": 0.05},
)
store.save_snapshot(snapshot)

# Test for regression
tester = QualityRegressionTester(store=store)
result = tester.test(current_snapshot)

if not result.passed:
    print(f"Quality regression detected!")
    print(result.summary())
```

---

## Embedding Visualizer

Visualize high-dimensional data with quality overlays.

```python
from clean.embedding_viz import (
    EmbeddingVisualizer,
    VisualizationConfig,
    ReductionMethod,
    visualize_embeddings,
)

# Quick visualization
result = visualize_embeddings(
    embeddings,
    labels=labels,
    quality_scores=quality_scores,
)

# Save or display
result.save("embeddings.html")  # Interactive HTML

# Custom configuration
config = VisualizationConfig(
    method=ReductionMethod.UMAP,
    n_components=3,  # 3D visualization
)
visualizer = EmbeddingVisualizer(config=config)
result = visualizer.visualize(embeddings, labels=labels)
```

---

## Synthetic Certification

Certify synthetic data quality before use.

```python
from clean.synthetic_certification import (
    SyntheticCertifier,
    CertificationConfig,
    certify_synthetic_data,
)

# Quick certification
certificate = certify_synthetic_data(synthetic_data, real_data)

print(f"Status: {certificate.status.value}")
print(f"Score: {certificate.overall_score:.1%}")
print(certificate.summary())

# Detailed certification
config = CertificationConfig(
    min_fidelity_score=0.85,
    min_privacy_score=0.90,
)
certifier = SyntheticCertifier(config=config)
certificate = certifier.certify(synthetic_data, real_data)

# Check specific dimensions
for score in certificate.dimension_scores:
    print(f"{score.dimension.value}: {score.score:.1%}")
```

---

## Data Marketplace

Compare quality against industry benchmarks.

```python
from clean.marketplace import (
    QualityMarketplace,
    Domain,
    create_marketplace,
)

# Create marketplace connection
marketplace = create_marketplace(
    store_path="benchmarks.db",
    org_id="my_company",
)

# Contribute your benchmark (anonymized)
marketplace.contribute_benchmark(
    quality_score=85.0,
    n_samples=10000,
    domain=Domain.HEALTHCARE,
)

# See how you compare
result = marketplace.get_percentile(85.0, Domain.HEALTHCARE)
print(f"Your score is at the {result.percentile}th percentile")
print(result.recommendation)

# Get industry benchmark
benchmark = marketplace.get_industry_benchmark(Domain.HEALTHCARE)
print(f"Industry median: {benchmark.quality_score_median:.1f}")
```

---

## Labeler Scoring

Evaluate labeler performance and route tasks smartly.

```python
from clean.labeler_scoring import (
    LabelerEvaluator,
    SmartRouter,
    evaluate_labelers,
)

# Evaluate labelers
evaluator = evaluate_labelers(labels, labeler_ids, ground_truth)

# Get individual reports
report = evaluator.get_labeler_report("labeler_001")
print(report.summary())
print(f"Accuracy: {report.metrics.accuracy:.1%}")
print(f"Expertise: {report.metrics.expertise_level.value}")

# Smart task routing
router = SmartRouter(evaluator, min_accuracy=0.85)
recommendations = router.recommend_labelers(
    task_category="sentiment",
    n_labelers=3,
)

for rec in recommendations:
    print(f"{rec.labeler_id}: {rec.reason}")

# Batch assignment
assignments = router.create_task_assignments(
    task_ids=[1, 2, 3, 4, 5],
    n_labelers_per_task=2,
)
```

---

## Integration Example

Combine multiple features for a complete workflow:

```python
from clean import DatasetCleaner
from clean.quality_predictor import QualityPredictor, QualityGate
from clean.contamination import detect_contamination
from clean.quality_regression import QualityRegressionTester, QualitySnapshot
from clean.marketplace import QualityMarketplace, Domain

# Step 1: Check for contamination
report = detect_contamination(train_df, test_df)
if report.has_contamination:
    print("Warning: Data leakage detected!")

# Step 2: Analyze quality
cleaner = DatasetCleaner(data=train_df, label_column="label")
quality_report = cleaner.analyze()

# Step 3: Track regression
snapshot = QualitySnapshot(
    version="v1.0",
    quality_score=quality_report.quality_score,
    metrics=quality_report.metrics,
)
tester = QualityRegressionTester()
result = tester.test(snapshot, baseline)

# Step 4: Compare to industry
marketplace = QualityMarketplace(org_id="my_org")
percentile = marketplace.get_percentile(
    quality_report.quality_score,
    Domain.GENERAL,
)
print(f"Industry percentile: {percentile.percentile}th")

# Step 5: Quality gate for deployment
gate = QualityGate(predictor, threshold=80.0)
if gate.check(train_df).passed:
    print("Ready for production!")
```
