---
sidebar_position: 3
title: Detectors
---

# Detector Classes

Individual detectors for specific quality issues.

```python
from clean.detection import (
    LabelErrorDetector,
    DuplicateDetector,
    OutlierDetector,
    ImbalanceDetector,
    BiasDetector,
)
```

## LabelErrorDetector

Finds mislabeled samples using confident learning.

```python
LabelErrorDetector(
    classifier: Optional[Any] = None,
    cv_folds: int = 5,
    threshold: float = 0.5,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifier` | sklearn estimator | LogisticRegression | Model for predictions |
| `cv_folds` | int | 5 | Cross-validation folds |
| `threshold` | float | 0.5 | Confidence threshold |

### Methods

#### detect()

```python
detect(
    features: np.ndarray,
    labels: np.ndarray,
) -> List[LabelError]
```

### Example

```python
from clean.detection import LabelErrorDetector
from sklearn.ensemble import RandomForestClassifier

detector = LabelErrorDetector(
    classifier=RandomForestClassifier(n_estimators=100),
    cv_folds=10,
)

errors = detector.detect(X, y)
for e in errors[:5]:
    print(f"{e.index}: {e.given_label} → {e.predicted_label} ({e.confidence:.2f})")
```

---

## DuplicateDetector

Finds exact and near-duplicate samples.

```python
DuplicateDetector(
    method: str = "auto",
    similarity_threshold: float = 0.9,
    hash_algorithm: str = "md5",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | "auto" | "hash", "embedding", or "auto" |
| `similarity_threshold` | float | 0.9 | Near-duplicate threshold |
| `hash_algorithm` | str | "md5" | Hash for exact matching |

### Methods

#### detect()

```python
detect(
    data: Union[pd.DataFrame, np.ndarray],
    text_column: Optional[str] = None,
) -> List[DuplicatePair]
```

### Example

```python
from clean.detection import DuplicateDetector

detector = DuplicateDetector(
    method="embedding",
    similarity_threshold=0.95,
)

duplicates = detector.detect(df, text_column="description")
print(f"Found {len(duplicates)} duplicate pairs")
```

---

## OutlierDetector

Detects anomalous samples.

```python
OutlierDetector(
    method: str = "isolation_forest",
    contamination: float = 0.05,
    n_estimators: int = 100,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | "isolation_forest" | "isolation_forest", "lof", "ensemble" |
| `contamination` | float | 0.05 | Expected outlier fraction |
| `n_estimators` | int | 100 | Trees for isolation forest |

### Methods

#### detect()

```python
detect(features: np.ndarray) -> List[Outlier]
```

### Example

```python
from clean.detection import OutlierDetector

detector = OutlierDetector(
    method="ensemble",
    contamination=0.01,
)

outliers = detector.detect(X)
```

---

## ImbalanceDetector

Analyzes class distribution.

```python
ImbalanceDetector(
    imbalance_threshold: float = 5.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imbalance_threshold` | float | 5.0 | Ratio to flag as imbalanced |

### Methods

#### detect()

```python
detect(labels: np.ndarray) -> Optional[ImbalanceInfo]
```

### Example

```python
from clean.detection import ImbalanceDetector

detector = ImbalanceDetector(imbalance_threshold=10.0)
result = detector.detect(y)

if result:
    print(f"Imbalance ratio: {result.imbalance_ratio:.1f}")
    print(f"Minority class: {result.minority_class}")
```

---

## BiasDetector

Detects potential biases in data.

```python
BiasDetector(
    sensitive_features: Optional[List[str]] = None,
    fairness_threshold: float = 0.8,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitive_features` | list | None | Features to check for bias |
| `fairness_threshold` | float | 0.8 | Demographic parity threshold |

### Methods

#### detect()

```python
detect(
    data: pd.DataFrame,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
) -> List[BiasIssue]
```

### Example

```python
from clean.detection import BiasDetector

detector = BiasDetector(
    sensitive_features=["gender", "age_group"],
)

issues = detector.detect(df, y, predictions=y_pred)
for issue in issues:
    print(f"{issue.feature}: {issue.metric}={issue.value:.2f}")
```

---

## Custom Detectors

Extend `BaseDetector` to create custom detectors:

```python
from clean.detection.base import BaseDetector
from dataclasses import dataclass
from typing import List

@dataclass
class CustomIssue:
    index: int
    severity: float
    message: str

class CustomDetector(BaseDetector):
    name = "custom"
    
    def detect(self, data, **kwargs) -> List[CustomIssue]:
        issues = []
        # Your detection logic
        return issues
```

Register for automatic discovery:

```python
from clean.plugins import PluginRegistry

PluginRegistry.register_detector("custom", CustomDetector)
```
