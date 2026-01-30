---
sidebar_position: 4
title: FixEngine
---

# FixEngine

Automatically fix detected data quality issues.

```python
from clean.fix import FixEngine, FixConfig, FixStrategy
```

## FixConfig

Configuration for fix behavior.

```python
@dataclass
class FixConfig:
    strategy: FixStrategy = FixStrategy.CONSERVATIVE
    min_confidence: float = 0.9
    remove_duplicates: bool = True
    remove_outliers: bool = False
    relabel_errors: bool = False
    max_removals_pct: float = 0.1
    dry_run: bool = False
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | FixStrategy | CONSERVATIVE | Overall fix aggressiveness |
| `min_confidence` | float | 0.9 | Minimum confidence for fixes |
| `remove_duplicates` | bool | True | Remove duplicate rows |
| `remove_outliers` | bool | False | Remove outlier rows |
| `relabel_errors` | bool | False | Apply label corrections |
| `max_removals_pct` | float | 0.1 | Max % of data to remove |
| `dry_run` | bool | False | Preview without applying |

## FixStrategy

```python
class FixStrategy(Enum):
    CONSERVATIVE = "conservative"  # Only high-confidence fixes
    MODERATE = "moderate"          # Balanced approach
    AGGRESSIVE = "aggressive"      # Apply all suggested fixes
```

## FixEngine

### Constructor

```python
FixEngine(config: Optional[FixConfig] = None)
```

### Methods

#### apply_fixes()

Apply fixes to a dataset based on quality report.

```python
apply_fixes(
    data: pd.DataFrame,
    report: QualityReport,
) -> FixResult
```

#### Returns

`FixResult` with:
- `data`: Fixed DataFrame
- `fixes_applied`: List of applied fixes
- `fixes_skipped`: List of skipped fixes
- `summary`: Text summary

### Example

```python
from clean import DatasetCleaner
from clean.fix import FixEngine, FixConfig, FixStrategy

# Analyze
cleaner = DatasetCleaner(df, labels="target")
report = cleaner.analyze()

# Configure fixes
config = FixConfig(
    strategy=FixStrategy.CONSERVATIVE,
    min_confidence=0.95,
    remove_duplicates=True,
    relabel_errors=True,
)

# Apply
engine = FixEngine(config)
result = engine.apply_fixes(df, report)

print(f"Applied {len(result.fixes_applied)} fixes")
print(result.summary)

# Use fixed data
fixed_df = result.data
```

## Dry Run

Preview fixes before applying:

```python
config = FixConfig(dry_run=True)
engine = FixEngine(config)
result = engine.apply_fixes(df, report)

print("Fixes that would be applied:")
for fix in result.fixes_applied:
    print(f"  - {fix}")
```

## Strategies Compared

| Aspect | Conservative | Moderate | Aggressive |
|--------|--------------|----------|------------|
| Confidence threshold | 0.95 | 0.85 | 0.7 |
| Duplicates | Exact only | Near + Exact | All matches |
| Outliers | Never | Extreme only | All detected |
| Max removal | 5% | 10% | 20% |

## Fix Types

### LabelFix

```python
@dataclass
class LabelFix:
    index: int
    old_label: Any
    new_label: Any
    confidence: float
```

### RemovalFix

```python
@dataclass
class RemovalFix:
    indices: List[int]
    reason: str  # "duplicate", "outlier", "label_error"
```

## Full Example

```python
import pandas as pd
from clean import DatasetCleaner
from clean.fix import FixEngine, FixConfig, FixStrategy

# Load data
df = pd.read_csv("training.csv")

# Analyze
cleaner = DatasetCleaner(df, labels="label")
report = cleaner.analyze()

print(f"Quality: {report.quality_score.overall}/100")
print(f"Issues: {report.n_issues}")

# Dry run first
config = FixConfig(
    strategy=FixStrategy.MODERATE,
    dry_run=True,
)
preview = FixEngine(config).apply_fixes(df, report)
print(f"\nWould apply {len(preview.fixes_applied)} fixes")

# Apply for real
config.dry_run = False
result = FixEngine(config).apply_fixes(df, report)

print(f"\nApplied {len(result.fixes_applied)} fixes:")
print(result.summary)

# Re-analyze to verify
cleaner2 = DatasetCleaner(result.data, labels="label")
report2 = cleaner2.analyze()
print(f"\nNew quality: {report2.quality_score.overall}/100")

# Save
result.data.to_csv("training_fixed.csv", index=False)
```
