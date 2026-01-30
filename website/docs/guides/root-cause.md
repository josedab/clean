---
sidebar_position: 12
title: Root Cause Analysis
---

# Automated Root Cause Analysis

When Clean finds quality issues, the next question is: *why?* Root cause analysis automatically drills down to identify what's causing problems in your data.

## Why Root Cause Analysis?

Finding 500 label errors is useful. Understanding that 80% of them come from a single annotator, or correlate with a specific feature value, is actionable.

Root cause analysis transforms a list of issues into a diagnosis you can fix at the source.

## Quick Start

```python
from clean import DatasetCleaner
from clean.root_cause import RootCauseAnalyzer

# First, run quality analysis
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Then analyze root causes
analyzer = RootCauseAnalyzer()
causes = analyzer.analyze(
    data=df,
    quality_report=report,
    issue_type="label_errors",
)

# View top causes
for cause in causes.top_causes[:5]:
    print(f"\n{cause.description}")
    print(f"  Impact: {cause.impact_score:.0%} of issues explained")
    print(f"  Affected: {cause.affected_count} samples")
    print(f"  Suggestion: {cause.suggested_fix}")
```

Example output:
```
Feature 'source' = 'vendor_b' correlates with label errors
  Impact: 45% of issues explained
  Affected: 225 samples
  Suggestion: Audit data pipeline from vendor_b

Feature 'text_length' < 10 correlates with label errors
  Impact: 23% of issues explained
  Affected: 115 samples
  Suggestion: Review labeling guidelines for short texts

Annotator 'user_47' has 3x higher error rate
  Impact: 18% of issues explained
  Affected: 90 samples
  Suggestion: Retrain or replace annotator user_47
```

## Analysis Methods

### Statistical Correlation

Finds features that correlate with quality issues:

```python
analyzer = RootCauseAnalyzer(methods=["statistical"])
causes = analyzer.analyze(df, report, "label_errors")

for cause in causes.top_causes:
    if cause.cause_type == "feature_correlation":
        print(f"{cause.feature}: correlation={cause.correlation:.3f}")
```

### Feature Importance

Uses ML to identify predictive features:

```python
analyzer = RootCauseAnalyzer(methods=["feature_importance"])
causes = analyzer.analyze(df, report, "outliers")

# Shows which features predict outliers
for cause in causes.top_causes:
    print(f"{cause.feature}: importance={cause.importance:.3f}")
```

### Cluster Analysis

Groups issues to find common patterns:

```python
analyzer = RootCauseAnalyzer(methods=["clustering"])
causes = analyzer.analyze(df, report, "duplicates")

for cause in causes.top_causes:
    if cause.cause_type == "cluster":
        print(f"Cluster of {cause.cluster_size} similar issues")
        print(f"  Common traits: {cause.common_features}")
```

### Combined Analysis

Use all methods for comprehensive analysis:

```python
analyzer = RootCauseAnalyzer(
    methods=["statistical", "feature_importance", "clustering"]
)
```

## Issue Types

Analyze causes for any detected issue:

```python
# Label errors
causes = analyzer.analyze(df, report, "label_errors")

# Outliers
causes = analyzer.analyze(df, report, "outliers")

# Duplicates
causes = analyzer.analyze(df, report, "duplicates")

# Bias
causes = analyzer.analyze(df, report, "bias")
```

## Cause Types

The analyzer identifies several types of root causes:

| Cause Type | Example | Action |
|------------|---------|--------|
| `feature_value` | "High price items have more errors" | Review labeling for edge cases |
| `data_source` | "Vendor B data has 3x error rate" | Audit vendor B pipeline |
| `temporal` | "Weekend data has more issues" | Check weekend processes |
| `annotator` | "Annotator X underperforms" | Retrain or remove |
| `pattern` | "Short texts are mislabeled" | Update labeling guidelines |

## Root Cause Object

```python
cause = causes.top_causes[0]

# Description
cause.description       # Human-readable explanation
cause.cause_type        # "feature_value", "data_source", etc.

# Impact
cause.impact_score      # 0-1, fraction of issues explained
cause.confidence        # Statistical confidence
cause.affected_count    # Number of samples affected
cause.affected_indices  # List of sample indices

# Action
cause.suggested_fix     # Recommended remediation

# Details
cause.feature           # Feature name (if applicable)
cause.feature_value     # Feature value (if applicable)
cause.correlation       # Correlation coefficient
cause.p_value           # Statistical significance
```

## Filtering and Ranking

```python
# Get top N causes
top_5 = causes.top_causes[:5]

# Filter by impact
significant = [c for c in causes.top_causes if c.impact_score > 0.1]

# Filter by type
feature_causes = [c for c in causes.top_causes if c.cause_type == "feature_value"]

# Filter by confidence
confident = [c for c in causes.top_causes if c.confidence > 0.95]
```

## Exporting Results

```python
# To JSON
causes.to_json("root_causes.json")

# To DataFrame
df_causes = causes.to_dataframe()

# To HTML report
causes.to_html("root_cause_report.html")
```

## Integration Example

Full workflow from detection to remediation:

```python
from clean import DatasetCleaner
from clean.root_cause import RootCauseAnalyzer

# 1. Detect issues
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

print(f"Found {len(report.label_errors())} label errors")

# 2. Analyze root causes
if len(report.label_errors()) > 50:  # Worth investigating
    analyzer = RootCauseAnalyzer()
    causes = analyzer.analyze(df, report, "label_errors")
    
    # 3. Take action based on top cause
    top_cause = causes.top_causes[0]
    
    if top_cause.cause_type == "data_source":
        print(f"Action: Audit data from {top_cause.feature_value}")
        # Filter out problematic source
        clean_df = df[df[top_cause.feature] != top_cause.feature_value]
        
    elif top_cause.cause_type == "annotator":
        print(f"Action: Review labels from {top_cause.feature_value}")
        # Re-label affected samples
        to_relabel = df.iloc[top_cause.affected_indices]
        
    elif top_cause.cause_type == "feature_value":
        print(f"Action: Update guidelines for {top_cause.description}")
```

## Convenience Function

```python
from clean.root_cause import analyze_root_causes

causes = analyze_root_causes(
    data=df,
    issue_indices=report.label_errors().index,
    methods=["statistical", "feature_importance"],
    max_causes=10,
)
```

## Configuration

```python
analyzer = RootCauseAnalyzer(
    methods=["statistical", "feature_importance", "clustering"],
    max_causes=20,              # Maximum causes to return
    min_impact=0.05,            # Minimum impact score
    min_confidence=0.9,         # Minimum statistical confidence
    feature_columns=None,       # Auto-detect, or specify list
)
```

## Best Practices

1. **Have enough issues**: Need 50+ issues for reliable analysis
2. **Include metadata**: Source, annotator, timestamp columns help identify causes
3. **Act on high-impact causes**: Fix causes explaining >10% of issues first
4. **Validate fixes**: Re-run analysis after remediation to confirm improvement
5. **Document findings**: Root causes often reveal systematic problems worth tracking

## Next Steps

- [Slice Discovery](/docs/guides/slice-discovery) - Find problematic data subgroups
- [Collaborative Review](/docs/guides/collaboration) - Team-based issue resolution
- [API Reference](/docs/guides/root-cause) - Full API documentation
