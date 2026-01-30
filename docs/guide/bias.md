# Bias Detection

Clean analyzes your dataset for potential fairness issues across sensitive features.

## What is Bias?

In ML, bias refers to systematic differences in model behavior across groups defined by sensitive attributes (e.g., gender, race, age). Clean detects:

- **Demographic parity**: Different positive rates across groups
- **Representation bias**: Unequal group sizes in the dataset
- **Feature correlation**: Sensitive features correlated with the target

## Basic Usage

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(
    data=df,
    label_column='hired',
    sensitive_features=['gender', 'age_group']  # Features to check
)
report = cleaner.analyze()

# Get bias issues
bias_issues = report.bias_issues()
print(bias_issues)
```

## Understanding the Output

```
   feature           metric      value  threshold  severity
0   gender  demographic_parity   0.25       0.1      high
1   gender  representation_ratio  0.65       0.8    medium
2   age_group  demographic_parity   0.15       0.1    medium
```

| Column | Description |
|--------|-------------|
| `feature` | The sensitive feature |
| `metric` | Type of bias detected |
| `value` | Measured disparity |
| `threshold` | Acceptable threshold |
| `severity` | low, medium, high |

## Metrics Explained

### Demographic Parity

Measures whether the positive outcome rate is equal across groups.

$$DP = |P(Y=1|A=a) - P(Y=1|A=b)|$$

- **0**: Perfect parity
- **1**: Complete disparity

Example: If 60% of men are hired but only 40% of women, DP = 0.2.

### Representation Ratio

Compares group sizes to detect underrepresentation.

$$RR = \frac{\min(n_a, n_b)}{\max(n_a, n_b)}$$

- **1.0**: Equal representation
- **0.5**: One group is half the size
- **< 0.2**: Severe underrepresentation

### Feature Correlation

Correlation between sensitive feature and target.

!!! warning
    High correlation doesn't always indicate unfairness—it depends on context.

## Configuration

```python
from clean import BiasDetector

detector = BiasDetector(
    sensitive_features=['gender', 'race'],
    demographic_parity_threshold=0.1,  # Max acceptable DP difference
    representation_threshold=0.8,       # Min acceptable representation ratio
)

cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    bias_detector=detector
)
```

## Automated Sensitive Feature Detection

If you don't specify sensitive features, Clean will check common patterns:

```python
# Auto-detect potential sensitive features
cleaner = DatasetCleaner(data=df, label_column='label')
# Clean checks for: gender, sex, race, ethnicity, age, religion, nationality
```

## Handling Bias Issues

### Option 1: Analyze and Report

```python
# Generate fairness report
bias_issues = report.bias_issues()

# Summarize by feature
for feature in bias_issues['feature'].unique():
    issues = bias_issues[bias_issues['feature'] == feature]
    print(f"\n{feature}:")
    for _, row in issues.iterrows():
        print(f"  {row['metric']}: {row['value']:.3f} ({row['severity']})")
```

### Option 2: Rebalancing

```python
# Stratified sampling to balance groups
from sklearn.model_selection import train_test_split

# Combine class and sensitive feature for stratification
df['strat_key'] = df['label'].astype(str) + '_' + df['gender'].astype(str)

train, test = train_test_split(df, stratify=df['strat_key'])
```

### Option 3: Fairness-Aware Training

Use fairness-aware ML libraries:

```python
# Example with fairlearn
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

constraint = DemographicParity()
mitigator = ExponentiatedGradient(base_estimator, constraint)
mitigator.fit(X, y, sensitive_features=df['gender'])
```

## Visualization

```python
from clean.visualization import plot_bias_analysis

# Visualize demographic parity across groups
fig = plot_bias_analysis(report, feature='gender')
fig.savefig('bias_analysis.png')
```

## Best Practices

1. **Domain expertise matters**: Statistical parity isn't always the right goal
2. **Multiple metrics**: Check multiple fairness metrics
3. **Intersectionality**: Check combinations of sensitive features
4. **Document decisions**: Record fairness considerations and trade-offs
5. **Ongoing monitoring**: Bias can emerge over time

## Legal and Ethical Considerations

!!! warning "Important"
    Bias detection is a tool for awareness, not a complete solution. Consider:
    
    - Legal requirements in your jurisdiction
    - Ethical implications of your use case
    - Consulting with domain experts and stakeholders
    - The difference between statistical and legal fairness

## Example: Hiring Dataset Analysis

```python
import pandas as pd
import numpy as np
from clean import DatasetCleaner

# Sample hiring data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'experience': np.random.uniform(0, 20, n),
    'education': np.random.choice(['HS', 'BA', 'MA', 'PhD'], n),
    'gender': np.random.choice(['M', 'F'], n, p=[0.6, 0.4]),  # Imbalanced
    'age': np.random.normal(35, 10, n).clip(18, 65),
})

# Biased hiring: higher rate for one gender
df['hired'] = (
    (df['experience'] > 5) & 
    (df['education'].isin(['MA', 'PhD'])) |
    ((df['gender'] == 'M') & (np.random.random(n) > 0.3))  # Bias
).astype(int)

# Detect
cleaner = DatasetCleaner(
    data=df,
    label_column='hired',
    sensitive_features=['gender', 'age']
)
report = cleaner.analyze()

# Check bias
print(report.bias_issues())
```
