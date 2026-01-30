# ADR-0006: Dataclass-Based Type System

## Status

Accepted

## Context

Clean produces structured results: label errors have indices, given labels, predicted labels, and confidence scores. Duplicates have pairs of indices and similarity scores. These results need to be:

1. **Type-safe**: IDE autocompletion and static analysis
2. **Serializable**: JSON export for APIs and reports
3. **Documented**: Self-describing field names
4. **Immutable-ish**: Results shouldn't be accidentally modified
5. **Lightweight**: No heavy runtime dependencies

Options considered:

1. **Plain dicts**: Flexible but no type safety or IDE support
2. **NamedTuples**: Immutable but awkward for optional fields
3. **Pydantic models**: Great validation but adds dependency and overhead
4. **attrs**: Powerful but another dependency to learn
5. **Dataclasses**: Built-in, typed, simple - chosen approach

## Decision

We use **Python dataclasses** for all domain objects with consistent `to_dict()` methods for serialization.

```python
# core/types.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class LabelError:
    """Represents a detected label error."""
    index: int
    given_label: Any
    predicted_label: Any
    confidence: float
    self_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "given_label": self.given_label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "self_confidence": self.self_confidence,
        }

@dataclass
class DuplicatePair:
    """Represents a pair of duplicate samples."""
    index1: int
    index2: int
    similarity: float
    is_exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index1": self.index1,
            "index2": self.index2,
            "similarity": self.similarity,
            "is_exact": self.is_exact,
        }

@dataclass
class QualityScore:
    """Quality score for the dataset."""
    overall: float
    label_quality: float
    duplicate_quality: float
    outlier_quality: float
    imbalance_quality: float
    bias_quality: float
    per_class: dict[Any, float] = field(default_factory=dict)
    per_feature: dict[str, float] = field(default_factory=dict)
```

Enums define valid values for categorical fields:

```python
class IssueType(Enum):
    LABEL_ERROR = "label_error"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    OUTLIER = "outlier"
    CLASS_IMBALANCE = "class_imbalance"
    BIAS = "bias"

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_LABEL = "multi_label"
```

Utility functions handle DataFrame conversion:

```python
def issues_to_dataframe(issues: list[LabelError | DuplicatePair | Outlier]) -> pd.DataFrame:
    """Convert a list of issues to a DataFrame."""
    if not issues:
        return pd.DataFrame()
    return pd.DataFrame([issue.to_dict() for issue in issues])
```

## Consequences

### Positive

- **Zero dependencies**: Dataclasses are built into Python 3.7+
- **IDE support**: Full autocompletion and type checking in VS Code, PyCharm
- **Self-documenting**: Field names and docstrings describe the data
- **Immutability option**: `frozen=True` available when needed
- **Default values**: Optional fields with sensible defaults
- **Equality**: Automatic `__eq__` for testing

### Negative

- **Manual serialization**: Must implement `to_dict()` on each class (could use `asdict()` but less control)
- **No validation**: Dataclasses don't validate types at runtime (Pydantic would)
- **Verbose for many fields**: Large dataclasses get unwieldy
- **NumPy compatibility**: Must handle numpy type conversion in `to_dict()`

### Neutral

- **Not frozen by default**: We chose mutability for ease of construction
- **Inheritance supported**: Can subclass dataclasses when needed

## NumPy Type Handling

Special care for JSON serialization of numpy types:

```python
def numpy_to_list(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj
```
