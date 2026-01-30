---
sidebar_position: 6
title: Bias Detection
---

# Bias Detection

Bias in training data leads to biased models—which can cause real-world harm.

## Types of Bias

### Representation Bias

Certain groups are underrepresented:

```
Dataset: 80% male, 20% female → Model worse for women
```

### Label Bias

Labels are systematically different across groups:

```
Same resume, different name → Different hiring decisions
```

### Measurement Bias

Features are measured differently across groups:

```
Credit scores using zip code → Proxy for race
```

## Why Bias Detection Matters

- **Legal requirements**: GDPR, ECOA, Fair Housing Act
- **Reputational risk**: Biased AI in headlines
- **Business impact**: Worse performance for customer segments
- **Ethical responsibility**: AI should be fair

## How Clean Detects Bias

Clean checks several fairness metrics:

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

bias_result = report.bias_result
print(bias_result.metadata)
```

### Specifying Protected Attributes

```python
from clean.detection import BiasDetector

detector = BiasDetector(
    protected_attributes=['gender', 'age_group'],
    label_column='hired',
)
result = detector.detect(df)
```

## Fairness Metrics

### Demographic Parity

Same positive outcome rate across groups:

```
P(Y=1 | A=0) = P(Y=1 | A=1)
```

```python
# Demographic parity difference
dp_diff = result.metadata['demographic_parity_diff']
# 0 = perfectly fair, >0.1 typically concerning
```

### Equalized Odds

Same TPR and FPR across groups:

```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  # Equal TPR
P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  # Equal FPR
```

### Predictive Parity

Same precision across groups:

```
P(Y=1 | Ŷ=1, A=0) = P(Y=1 | Ŷ=1, A=1)
```

### Disparate Impact Ratio

Ratio of positive outcomes:

```
DI = P(Y=1 | A=0) / P(Y=1 | A=1)
```

`DI > 0.8` is often the legal threshold (80% rule).

## Understanding Results

```python
bias_result = report.bias_result

for attr, metrics in bias_result.metadata['metrics'].items():
    print(f"\n=== {attr} ===")
    print(f"Demographic Parity Diff: {metrics['dp_diff']:.3f}")
    print(f"Disparate Impact Ratio: {metrics['di_ratio']:.3f}")
    print(f"Group counts: {metrics['group_counts']}")
```

Example output:
```
=== gender ===
Demographic Parity Diff: 0.15
Disparate Impact Ratio: 0.72
Group counts: {'male': 8000, 'female': 2000}

⚠️ Potential bias detected: DI ratio < 0.8
```

## Proxy Variable Detection

Features that encode protected attributes:

```python
from clean.detection import BiasDetector

detector = BiasDetector(
    protected_attributes=['race'],
    detect_proxies=True,
)
result = detector.detect(df)

# Potential proxies
proxies = result.metadata['proxy_variables']
# [{'feature': 'zip_code', 'correlation': 0.72}]
```

## Mitigating Bias

### Pre-processing: Rebalancing

```python
# Oversample underrepresented groups
from imblearn.over_sampling import SMOTE

# Create group-aware sampling
def group_aware_resample(X, y, protected):
    # Resample each subgroup
    pass
```

### In-processing: Constrained Learning

```python
# Use fairness-aware algorithms
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

constraint = DemographicParity()
mitigator = ExponentiatedGradient(estimator, constraint)
mitigator.fit(X, y, sensitive_features=protected)
```

### Post-processing: Threshold Adjustment

```python
# Different thresholds per group
thresholds = {
    'male': 0.5,
    'female': 0.35,  # Lower threshold to equalize rates
}

def fair_predict(probs, group):
    return probs > thresholds[group]
```

## Visualization

```python
from clean.visualization import plot_bias

# Fairness metrics by group
plot_bias(report, protected_attribute='gender')
```

Manual visualization:

```python
import matplotlib.pyplot as plt

groups = ['male', 'female']
positive_rates = [0.65, 0.50]

plt.bar(groups, positive_rates)
plt.axhline(y=0.8 * max(positive_rates), color='r', linestyle='--', label='80% rule')
plt.ylabel('Positive Outcome Rate')
plt.title('Outcome Rate by Gender')
plt.legend()
```

## Best Practices

### 1. Define Protected Attributes Upfront

```python
PROTECTED_ATTRIBUTES = [
    'gender',
    'age_group', 
    'race_ethnicity',
    'disability_status',
]
```

### 2. Check Multiple Metrics

No single metric captures all fairness concerns:

```python
metrics = ['demographic_parity', 'equalized_odds', 'predictive_parity']
for metric in metrics:
    check_fairness(model, X_test, y_test, protected, metric=metric)
```

### 3. Document Trade-offs

Sometimes fairness metrics conflict:

```
Optimizing for demographic parity may reduce accuracy
Optimizing for equalized odds may reduce demographic parity
```

Document your choices and reasoning.

### 4. Monitor in Production

Bias can emerge or change over time:

```python
from clean.lineage import LineageTracker

tracker = LineageTracker(project='hiring_model')

# Log fairness metrics with each analysis
tracker.log_analysis(
    report=report,
    metadata={'fairness_checked': True}
)
```

### 5. Involve Domain Experts

- What does "fair" mean in your context?
- Who could be harmed?
- What are the legal requirements?

## Common Questions

### Which metric should I use?

| Scenario | Recommended Metric |
|----------|-------------------|
| Hiring decisions | Demographic parity |
| Criminal justice | Equalized odds |
| Medical diagnosis | Predictive parity |
| Loan approvals | Disparate impact ratio |

### What threshold is "fair enough"?

Common guidelines:
- Disparate Impact Ratio > 0.8 (80% rule)
- Demographic Parity Diff < 0.1
- But context matters—consult legal/ethics experts

### Can I remove protected attributes?

Usually not enough—proxies exist:
- Zip code correlates with race
- Height correlates with gender
- Name correlates with ethnicity

## Next Steps

- [LLM Data Quality](/docs/guides/llm-data) - Bias in AI training data
- [Auto-Fix Engine](/docs/guides/auto-fix) - Automated bias mitigation
- [API Reference](/docs/api/detectors) - BiasDetector API
