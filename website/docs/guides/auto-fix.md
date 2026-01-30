---
sidebar_position: 3
title: Auto-Fix Engine
---

# Auto-Fix Engine

The FixEngine generates and applies corrections for detected data quality issues.

## Quick Start

```python
from clean import DatasetCleaner, FixEngine, FixConfig

# Analyze
cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Create fix engine
features = df.drop(columns=['label'])
labels = df['label'].values

engine = FixEngine(
    report=report,
    features=features,
    labels=labels,
)

# Get suggestions
fixes = engine.suggest_fixes()
print(f"Found {len(fixes)} suggested fixes")

# Apply
result = engine.apply_fixes(fixes)
print(result.summary())
```

## Configuration

### FixConfig Options

```python
config = FixConfig(
    # Label errors
    label_error_threshold=0.9,   # Min confidence to suggest relabeling
    auto_relabel=False,          # Auto-apply or just suggest
    
    # Duplicates
    duplicate_similarity_threshold=0.98,
    keep_strategy='first',       # 'first', 'last', 'random'
    
    # Outliers
    outlier_score_threshold=0.9,
    outlier_action='flag',       # 'remove', 'flag', 'impute'
    
    # General
    max_fixes=None,              # Limit number of fixes
    require_confirmation=True,   # Safety check
)

engine = FixEngine(report=report, features=features, config=config)
```

### Preset Strategies

```python
from clean import FixStrategy

# Conservative: Only high-confidence fixes
config = FixConfig.from_strategy(FixStrategy.CONSERVATIVE)

# Aggressive: More fixes applied
config = FixConfig.from_strategy(FixStrategy.AGGRESSIVE)

# Moderate: Balanced (default)
config = FixConfig.from_strategy(FixStrategy.MODERATE)
```

## Suggested Fixes

Each suggestion is a `SuggestedFix`:

```python
from clean import SuggestedFix

@dataclass
class SuggestedFix:
    issue_type: str          # 'label_error', 'duplicate', 'outlier'
    issue_index: int         # Row index
    fix_type: str            # 'relabel', 'remove', 'impute', 'flag'
    confidence: float        # 0-1
    description: str         # Human-readable
    old_value: Any           # Current value
    new_value: Any           # Proposed value
    metadata: dict           # Additional info
```

### Exploring Fixes

```python
fixes = engine.suggest_fixes()

# View all
for fix in fixes[:10]:
    print(f"[{fix.confidence:.2f}] {fix.issue_type}: {fix.description}")

# Filter by type
label_fixes = [f for f in fixes if f.issue_type == 'label_error']
dup_fixes = [f for f in fixes if f.issue_type == 'duplicate']

# Filter by confidence
high_conf = [f for f in fixes if f.confidence > 0.95]
print(f"High-confidence fixes: {len(high_conf)}")
```

### Selective Suggestions

```python
fixes = engine.suggest_fixes(
    include_label_errors=True,
    include_duplicates=True,
    include_outliers=False,  # Skip outlier fixes
)
```

## Applying Fixes

### Dry Run (Preview)

```python
result = engine.apply_fixes(fixes, dry_run=True)

print(result.summary())
# Fix Application Summary
# ========================================
# Applied: 45 fixes (DRY RUN)
# Skipped: 5 fixes
# Errors:  0 fixes
```

### Apply All

```python
result = engine.apply_fixes(fixes, dry_run=False)

# Get cleaned data
clean_features = result.features
clean_labels = result.labels
```

### Selective Application

```python
# Only apply high-confidence fixes
high_conf_fixes = [f for f in fixes if f.confidence > 0.95]
result = engine.apply_fixes(high_conf_fixes)

# Only label fixes
label_only = [f for f in fixes if f.issue_type == 'label_error']
result = engine.apply_fixes(label_only)
```

## Fix Types

### Relabel

Changes the label of a sample:

```python
# Manual relabeling
fix = SuggestedFix(
    issue_type='label_error',
    issue_index=42,
    fix_type='relabel',
    old_value='cat',
    new_value='dog',
    confidence=0.95,
    description="Change label from 'cat' to 'dog'",
)
```

### Remove

Removes a sample from the dataset:

```python
# Duplicate removal
fix = SuggestedFix(
    issue_type='duplicate',
    issue_index=187,
    fix_type='remove',
    confidence=1.0,
    description="Remove duplicate of row 42",
)
```

### Flag

Marks a sample for review without removing:

```python
# Flag outlier
fix = SuggestedFix(
    issue_type='outlier',
    issue_index=523,
    fix_type='flag',
    confidence=0.8,
    description="Flagged as potential outlier",
)
```

### Impute

Replaces outlier values:

```python
# Impute outlier value
fix = SuggestedFix(
    issue_type='outlier',
    issue_index=789,
    fix_type='impute',
    old_value=1000.0,
    new_value=50.0,  # Median
    confidence=0.7,
    description="Impute outlier with median",
)
```

## Results

### FixApplicationResult

```python
@dataclass
class FixApplicationResult:
    features: pd.DataFrame      # Cleaned features
    labels: np.ndarray          # Cleaned labels
    applied_fixes: list         # Successfully applied
    skipped_fixes: list         # Skipped (low confidence, etc.)
    errors: list                # Failed with exceptions
```

### Summary

```python
result = engine.apply_fixes(fixes)

print(f"Applied: {result.n_applied}")
print(f"Skipped: {result.n_skipped}")
print(f"Errors: {result.n_errors}")

# Detailed summary
print(result.summary())
```

### Reconstruction

```python
# Combine back into DataFrame
clean_df = result.features.copy()
clean_df['label'] = result.labels

# Compare sizes
print(f"Original: {len(df)}, Clean: {len(clean_df)}")
print(f"Removed: {len(df) - len(clean_df)} rows")
```

## Audit Trail

### With LineageTracker

```python
from clean import LineageTracker

tracker = LineageTracker(project="my_project")

# Log the analysis
run_id = tracker.log_analysis(
    dataset_name="training_data",
    report=report,
)

# Log fixes applied
for fix in result.applied_fixes:
    tracker.log_review(
        run_id=run_id,
        sample_index=fix.issue_index,
        decision="fixed",
        notes=fix.description,
    )
```

### Manual Logging

```python
import json
from datetime import datetime

audit_log = {
    "timestamp": datetime.now().isoformat(),
    "dataset": "training_data.csv",
    "original_rows": len(df),
    "fixes_applied": len(result.applied_fixes),
    "final_rows": len(result.features),
    "fixes": [
        {
            "index": f.issue_index,
            "type": f.fix_type,
            "old": str(f.old_value),
            "new": str(f.new_value),
        }
        for f in result.applied_fixes
    ],
}

with open("fix_audit.json", "w") as f:
    json.dump(audit_log, f, indent=2)
```

## Best Practices

### 1. Always Preview First

```python
# Step 1: Preview
preview = engine.apply_fixes(fixes, dry_run=True)
print(preview.summary())

# Step 2: Review and confirm
if input("Apply fixes? (y/n): ").lower() == 'y':
    result = engine.apply_fixes(fixes, dry_run=False)
```

### 2. Apply in Stages

```python
# Stage 1: High-confidence label errors
stage1 = [f for f in fixes if f.issue_type == 'label_error' and f.confidence > 0.95]
result1 = engine.apply_fixes(stage1)

# Stage 2: Duplicates
engine2 = FixEngine(report=report, features=result1.features, labels=result1.labels)
stage2 = [f for f in engine2.suggest_fixes() if f.issue_type == 'duplicate']
result2 = engine2.apply_fixes(stage2)
```

### 3. Validate After Fixing

```python
# Re-analyze cleaned data
clean_df = result.features.copy()
clean_df['label'] = result.labels

cleaner2 = DatasetCleaner(data=clean_df, label_column='label')
report2 = cleaner2.analyze()

print(f"Before: {report.quality_score.overall:.1f}")
print(f"After: {report2.quality_score.overall:.1f}")
```

### 4. Keep Original Data

```python
# Save original
df.to_csv("data_original.csv", index=False)

# Apply fixes
result = engine.apply_fixes(fixes)

# Save cleaned
clean_df.to_csv("data_cleaned.csv", index=False)
```

## Next Steps

- [Plugins](/docs/guides/plugins) - Custom fix strategies
- [CLI](/docs/guides/cli) - Apply fixes from command line
- [API Reference](/docs/api/fix-engine) - FixEngine API
