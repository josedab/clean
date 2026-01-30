# Auto-Fix Engine

The Auto-Fix Engine automatically suggests and applies fixes for detected
quality issues.

## Overview

The `FixEngine` analyzes quality reports and generates targeted fixes:

```python
from clean import DatasetCleaner, FixEngine, FixConfig

# Analyze dataset
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Create fix engine
engine = FixEngine(
    report=report,
    features=df.drop(columns=["label"]),
    labels=df["label"].values,
)

# Get suggestions
fixes = engine.suggest_fixes()
print(f"Found {len(fixes)} suggested fixes")

# Apply fixes
result = engine.apply_fixes(fixes)
print(result.summary())
```

## Configuration

Use `FixConfig` to control fix behavior:

```python
from clean import FixConfig

# Conservative (default)
config = FixConfig(
    label_error_threshold=0.95,
    duplicate_similarity_threshold=0.99,
    auto_relabel=False,
)

# Aggressive
config = FixConfig(
    label_error_threshold=0.7,
    duplicate_similarity_threshold=0.95,
    auto_relabel=True,
    outlier_action="remove",
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `label_error_threshold` | float | 0.9 | Min confidence for relabeling |
| `auto_relabel` | bool | False | Auto-apply label corrections |
| `duplicate_similarity_threshold` | float | 0.98 | Min similarity for duplicates |
| `keep_strategy` | str | "first" | Which duplicate to keep |
| `outlier_score_threshold` | float | 0.9 | Min score for outlier removal |
| `outlier_action` | str | "flag" | 'remove', 'flag', or 'impute' |
| `max_fixes` | int | None | Limit number of fixes |

## Preset Strategies

Use `from_strategy()` for preset configurations:

```python
from clean import FixConfig, FixStrategy

# Conservative - only high-confidence fixes
config = FixConfig.from_strategy(FixStrategy.CONSERVATIVE)

# Aggressive - apply more fixes
config = FixConfig.from_strategy(FixStrategy.AGGRESSIVE)
```

## Suggested Fixes

Each suggestion is a `SuggestedFix` object:

```python
for fix in fixes:
    print(f"Issue: {fix.issue_type}")
    print(f"Index: {fix.issue_index}")
    print(f"Action: {fix.fix_type}")
    print(f"Confidence: {fix.confidence:.2f}")
    print(f"Description: {fix.description}")
    if fix.old_value is not None:
        print(f"Change: {fix.old_value} -> {fix.new_value}")
```

### Fix Types

| Type | Description |
|------|-------------|
| `relabel` | Change sample label |
| `remove` | Remove sample from dataset |
| `merge` | Merge duplicate samples |
| `flag` | Mark for manual review |
| `impute` | Impute missing/outlier value |

## Applying Fixes

### Preview (Dry Run)

```python
result = engine.apply_fixes(fixes, dry_run=True)
print(result.summary())
# No changes actually made
```

### Apply All

```python
result = engine.apply_fixes(fixes)
clean_features = result.features
clean_labels = result.labels
```

### Selective Application

```python
# Only high-confidence fixes
high_conf = [f for f in fixes if f.confidence > 0.95]
result = engine.apply_fixes(high_conf)

# Only label error fixes
label_fixes = [f for f in fixes if f.issue_type == "label_error"]
result = engine.apply_fixes(label_fixes)
```

## Fix Results

The `FixApplicationResult` contains:

```python
@dataclass
class FixApplicationResult:
    features: pd.DataFrame      # Cleaned features
    labels: np.ndarray | None   # Cleaned labels
    applied_fixes: list         # Successfully applied
    skipped_fixes: list         # Skipped fixes
    errors: list                # Failed fixes with errors
```

### Summary

```python
print(result.summary())
# Output:
# Fix Application Summary
# ========================================
# Applied: 45 fixes
# Skipped: 5 fixes
# Errors:  0 fixes
#
# Applied fixes by type:
#   - relabel: 20
#   - remove: 25
```

## API Reference

::: clean.fixes.FixEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - suggest_fixes
        - apply_fixes

::: clean.fixes.FixConfig
    options:
      show_root_heading: true
