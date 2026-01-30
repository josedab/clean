# ADR-0009: Strategy Pattern for Fix Application

## Status

Accepted

## Context

After detecting data quality issues, users need to fix them. But "fixing" means different things:

- **Label errors**: Relabel, remove, or flag for review?
- **Duplicates**: Remove all but first, last, or random?
- **Outliers**: Remove, clip, impute, or just flag?

Different contexts demand different aggressiveness:

| Context | Risk Tolerance | Recommended Approach |
|---------|---------------|---------------------|
| Exploratory analysis | Low | Flag everything, manual review |
| Production pipeline | Medium | Conservative auto-fix |
| Data cleaning sprint | High | Aggressive auto-fix |
| Regulated industry | Very low | Flag only, full audit trail |

We needed to:
1. Support multiple fix strategies without code duplication
2. Allow fine-grained configuration
3. Provide sensible presets for common cases
4. Track what was fixed and why

## Decision

We implemented the **Strategy Pattern** with configurable presets via `FixStrategy` enum and `FixConfig` dataclass.

```python
# fixes.py
class FixStrategy(Enum):
    """Strategy presets for fix aggressiveness."""
    CONSERVATIVE = "conservative"  # Only high-confidence fixes
    MODERATE = "moderate"          # Balanced approach
    AGGRESSIVE = "aggressive"      # Apply more fixes

@dataclass
class FixConfig:
    """Configuration for fix suggestions and application."""
    # Label error fixes
    label_error_threshold: float = 0.9   # Min confidence to suggest relabeling
    auto_relabel: bool = False           # Auto-apply relabeling
    
    # Duplicate fixes
    duplicate_similarity_threshold: float = 0.98
    keep_strategy: str = "first"         # 'first', 'last', 'random'
    
    # Outlier fixes
    outlier_score_threshold: float = 0.9
    outlier_action: str = "flag"         # 'remove', 'flag', 'impute'
    
    # General
    max_fixes: int | None = None
    require_confirmation: bool = True

    @classmethod
    def from_strategy(cls, strategy: FixStrategy) -> "FixConfig":
        """Create config from a preset strategy."""
        if strategy == FixStrategy.CONSERVATIVE:
            return cls(
                label_error_threshold=0.95,
                duplicate_similarity_threshold=0.99,
                outlier_score_threshold=0.95,
                auto_relabel=False,
            )
        elif strategy == FixStrategy.AGGRESSIVE:
            return cls(
                label_error_threshold=0.7,
                duplicate_similarity_threshold=0.9,
                outlier_score_threshold=0.7,
                auto_relabel=True,
            )
        else:  # MODERATE
            return cls()
```

The `FixEngine` applies configurations:

```python
class FixEngine:
    def __init__(self, report: QualityReport, features: pd.DataFrame, 
                 labels: np.ndarray, config: FixConfig = None):
        self.config = config or FixConfig()
    
    def suggest_fixes(self) -> list[SuggestedFix]:
        """Generate fix suggestions based on config."""
        fixes = []
        
        # Label errors → relabel suggestions
        for error in self.report.label_errors():
            if error.confidence >= self.config.label_error_threshold:
                fixes.append(SuggestedFix(
                    issue_type="label_error",
                    fix_type="relabel",
                    confidence=error.confidence,
                    old_value=error.given_label,
                    new_value=error.predicted_label,
                ))
        
        # Duplicates → remove suggestions
        for dup in self.report.duplicates():
            if dup.similarity >= self.config.duplicate_similarity_threshold:
                fixes.append(SuggestedFix(
                    issue_type="duplicate",
                    fix_type="remove",
                    issue_index=dup.index2,  # Keep index1, remove index2
                    confidence=dup.similarity,
                ))
        
        return fixes
    
    def apply_fixes(self, fixes: list[SuggestedFix], dry_run=False) -> FixResult:
        """Apply fixes to data, returning modified data and audit trail."""
        ...
```

## Consequences

### Positive

- **Preset simplicity**: `FixConfig.from_strategy(FixStrategy.CONSERVATIVE)` for common cases
- **Fine-grained control**: Override individual thresholds when needed
- **Audit trail**: `FixResult` tracks exactly what was changed
- **Dry run support**: Preview fixes before applying
- **Separation of concerns**: Detection and fixing are independent

### Negative

- **Configuration complexity**: Many knobs to understand
- **Strategy proliferation**: May need more presets over time
- **Testing burden**: Must test each strategy and edge cases

### Neutral

- **No undo**: Fixes are applied to copies, originals unchanged
- **Confidence-based**: All fixes have confidence scores for filtering

## Usage Examples

```python
# Simple: use a preset
config = FixConfig.from_strategy(FixStrategy.CONSERVATIVE)
engine = FixEngine(report, features, labels, config)
fixes = engine.suggest_fixes()
result = engine.apply_fixes(fixes)

# Advanced: custom configuration
config = FixConfig(
    label_error_threshold=0.85,
    auto_relabel=False,
    duplicate_similarity_threshold=0.95,
    outlier_action="impute",
)
engine = FixEngine(report, features, labels, config)
```

## Related Decisions

- ADR-0010 (Lineage Tracking): Fixes are logged to audit trail
- ADR-0002 (Plugin Architecture): Custom fixers can be registered
