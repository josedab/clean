---
sidebar_position: 3
title: Duplicates
---

# Duplicate Detection

Duplicates are redundant samples that waste compute and can cause data leakage.

## Types of Duplicates

### Exact Duplicates

Rows that are byte-for-byte identical:

```python
# These are exact duplicates
row1: ["Hello world", 1, 0.5, "positive"]
row2: ["Hello world", 1, 0.5, "positive"]
```

Detected via hash comparison (fast, O(n)).

### Near-Duplicates

Rows that are semantically similar but not identical:

```python
# These are near-duplicates
row1: ["Hello world!", 1, 0.5, "positive"]
row2: ["Hello, world", 1, 0.5, "positive"]  # Different punctuation

row3: ["The quick brown fox", ...]
row4: ["A fast brown fox", ...]  # Similar meaning
```

Detected via embedding similarity (requires `[text]` extra).

## Why Duplicates Matter

1. **Inflated metrics**: Same sample in train and test = cheating
2. **Wasted compute**: Training on redundant data
3. **Biased learning**: Over-represented patterns
4. **Data leakage**: If duplicates span train/test splits

## How to Detect Duplicates

### Basic Usage

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get duplicate pairs
duplicates = report.duplicates()
```

### Understanding Results

```python
for dup in duplicates[:5]:
    print(f"Indices: {dup['indices']}, Similarity: {dup['similarity']:.2f}")
```

Output:
```
Indices: (42, 187), Similarity: 1.00
Indices: (523, 891), Similarity: 0.98
Indices: (12, 45), Similarity: 0.95
```

- **indices**: Tuple of duplicate row indices
- **similarity**: 1.0 = exact match, under 1.0 = near-duplicate
- **type**: 'exact' or 'near'

## Configuration

### Similarity Threshold

```python
from clean.detection import DuplicateDetector

detector = DuplicateDetector(
    similarity_threshold=0.95,  # 0.0-1.0, higher = stricter
    method='embedding',         # 'hash', 'embedding', or 'both'
)
```

### Text Columns

For text data, specify which columns to embed:

```python
detector = DuplicateDetector(
    text_columns=['title', 'description'],
    embedding_model='all-MiniLM-L6-v2',  # sentence-transformers model
)
```

### Blocking for Large Datasets

For datasets over 100K rows, use blocking to reduce comparisons:

```python
detector = DuplicateDetector(
    blocking_columns=['category'],  # Only compare within same category
    similarity_threshold=0.9,
)
```

## Handling Duplicates

### Option 1: Remove All Duplicates

```python
clean_df = cleaner.get_clean_data(remove_duplicates=True)
```

### Option 2: Keep First/Last

```python
from clean import FixEngine, FixConfig

config = FixConfig(keep_strategy='first')  # or 'last', 'random'
engine = FixEngine(report=report, features=df, config=config)
```

### Option 3: Merge Duplicates

For duplicates with different labels, merge them:

```python
# Find conflicting duplicates
conflicts = [d for d in duplicates if d['label_conflict']]
print(f"Conflicting labels: {len(conflicts)}")
```

## Best Practices

### 1. Check Before/After Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.2)

# Check for duplicates ACROSS splits (data leakage!)
from clean.detection import find_duplicates

cross_dups = find_duplicates(
    pd.concat([X_train, X_test]),
    return_pairs=True
)

# Filter to pairs that span train/test
train_indices = set(X_train.index)
test_indices = set(X_test.index)

leakage = [
    d for d in cross_dups
    if (d['indices'][0] in train_indices and d['indices'][1] in test_indices)
    or (d['indices'][1] in train_indices and d['indices'][0] in test_indices)
]

print(f"Data leakage: {len(leakage)} duplicate pairs span train/test")
```

### 2. Use Appropriate Similarity

| Data Type | Recommended Method |
|-----------|-------------------|
| Tabular (numeric) | Hash + cosine similarity |
| Text | Sentence embeddings |
| Images | CLIP or ResNet embeddings |
| Mixed | Combine methods |

### 3. Investigate Near-Duplicates

Near-duplicates might be:
- Data entry variations (legitimate)
- Copy-paste errors (remove)
- Augmented samples (expected)

```python
# Sample near-duplicates for manual review
near_dups = [d for d in duplicates if 0.9 < d['similarity'] < 1.0]
for d in near_dups[:5]:
    i, j = d['indices']
    print(f"--- Pair ({i}, {j}) ---")
    print(df.iloc[i])
    print(df.iloc[j])
```

## Performance

| Dataset Size | Exact (Hash) | Embedding | With Blocking |
|--------------|--------------|-----------|---------------|
| 10K | under 1s | ~5s | N/A |
| 100K | under 5s | ~60s | ~15s |
| 1M | ~30s | OOM | ~120s |

For large datasets, use:
- `blocking_columns` to reduce comparisons
- `StreamingCleaner` for chunk-by-chunk analysis

## Next Steps

- [Outliers](/docs/concepts/outliers) - Find anomalous samples
- [Streaming](/docs/guides/streaming) - Handle large datasets
- [API Reference](/docs/api/detectors) - DuplicateDetector API
