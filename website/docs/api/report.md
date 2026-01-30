---
sidebar_position: 2
title: QualityReport
---

# QualityReport

Container for analysis results with methods for inspection and export.

```python
from clean import QualityReport
```

## Properties

### quality_score

```python
@property
def quality_score(self) -> QualityScore
```

Overall and component quality scores.

```python
report = cleaner.analyze()
print(report.quality_score.overall)      # 85.5
print(report.quality_score.label_quality) # 92.0
```

### n_issues

```python
@property
def n_issues(self) -> int
```

Total number of issues found.

### issue_counts

```python
@property
def issue_counts(self) -> Dict[str, int]
```

Count of issues by type:

```python
{
    "label_errors": 150,
    "duplicates": 50,
    "outliers": 75,
    "imbalance": 1
}
```

## Methods

### summary()

Get text summary of the report.

```python
summary(verbose: bool = False) -> str
```

#### Example

```python
print(report.summary())
```

Output:
```
Data Quality Report
==================
Samples analyzed: 10,000
Quality Score: 85.5/100

Issues Found:
  - Label errors: 150 (1.5%)
  - Duplicates: 50 pairs
  - Outliers: 75 (0.75%)

Recommendations:
  1. Review label errors with confidence > 0.9
  2. Consider removing exact duplicates
```

### label_errors()

Get detected label errors.

```python
label_errors(
    min_confidence: float = 0.0,
    limit: Optional[int] = None,
) -> List[LabelError]
```

#### Returns

List of `LabelError` with:
- `index`: Row index
- `given_label`: Current label
- `predicted_label`: Suggested label
- `confidence`: Detection confidence

#### Example

```python
for error in report.label_errors(min_confidence=0.9)[:10]:
    print(f"Row {error.index}: {error.given_label} → {error.predicted_label}")
```

### duplicates()

Get detected duplicates.

```python
duplicates(
    min_similarity: float = 0.0,
    limit: Optional[int] = None,
) -> List[DuplicatePair]
```

#### Returns

List of `DuplicatePair` with:
- `index1`: First row index
- `index2`: Second row index
- `similarity`: Similarity score (0-1)
- `match_type`: "exact" or "near"

### outliers()

Get detected outliers.

```python
outliers(
    min_score: float = 0.0,
    limit: Optional[int] = None,
) -> List[Outlier]
```

#### Returns

List of `Outlier` with:
- `index`: Row index
- `outlier_score`: Anomaly score
- `features`: Contributing features

### imbalance()

Get class imbalance information.

```python
imbalance() -> Optional[ImbalanceInfo]
```

#### Returns

`ImbalanceInfo` with:
- `class_distribution`: Dict[label, count]
- `imbalance_ratio`: max_count / min_count
- `minority_class`: Smallest class label

### to_dict()

Export to dictionary.

```python
to_dict() -> Dict[str, Any]
```

### to_json()

Export to JSON string.

```python
to_json(indent: int = 2) -> str
```

### to_html()

Export to HTML report.

```python
to_html(output_path: Optional[str] = None) -> str
```

#### Example

```python
# Get HTML string
html = report.to_html()

# Save to file
report.to_html("report.html")
```

## Example: Processing Results

```python
report = cleaner.analyze()

# Check if quality is acceptable
if report.quality_score.overall < 80:
    print("Warning: Low data quality")
    
    # Get high-confidence label errors
    errors = report.label_errors(min_confidence=0.95)
    print(f"Found {len(errors)} confident label errors")
    
    # Get exact duplicates
    dupes = report.duplicates(min_similarity=1.0)
    print(f"Found {len(dupes)} exact duplicates")

# Export for review
report.to_html("quality_report.html")
```
