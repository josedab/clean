# Duplicate Detection

Duplicates in training data can cause:

- **Overfitting**: Model memorizes duplicated samples
- **Data leakage**: Same sample in train/test splits
- **Wasted compute**: Training on redundant data

## Types of Duplicates

Clean detects two types of duplicates:

### Exact Duplicates

Samples with identical feature values. Detected using content hashing.

### Near-Duplicates

Samples that are semantically similar but not identical. Examples:

- Text with minor typos or rephrasing
- Images with slight crops or color adjustments
- Numeric data with small perturbations

## Basic Usage

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

# Get duplicates
duplicates = report.duplicates()
print(duplicates.head())
```

**Output:**
```
   index1  index2  similarity  is_exact
0      12      45       1.000      True
1      23     156       0.982     False
2      78     234       0.956     False
```

## Understanding the Output

| Column | Description |
|--------|-------------|
| `index1` | First sample in the pair |
| `index2` | Second sample in the pair |
| `similarity` | Cosine similarity (1.0 = identical) |
| `is_exact` | Whether this is an exact duplicate |

## Configuration

```python
from clean import DuplicateDetector

detector = DuplicateDetector(
    similarity_threshold=0.9,    # Minimum similarity for near-duplicates
    hash_columns=None,           # Columns for exact matching (None = all)
    embedding_column=None,       # Column with pre-computed embeddings
)

cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    duplicate_detector=detector
)
```

## Text Data

For text data, Clean uses sentence embeddings for semantic similarity:

```python
cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    text_column='review_text'  # Enable text embeddings
)
```

!!! note
    Requires the text extra: `pip install clean-data-quality[text]`

## Image Data

For image datasets loaded with `ImageFolderLoader`:

```python
from clean.loaders import ImageFolderLoader

loader = ImageFolderLoader('data/images/', compute_embeddings=True)
features, labels, info = loader.load()

cleaner = DatasetCleaner(
    data=features,
    labels=labels,
    embedding_column='embedding'
)
```

!!! note
    Requires the image extra: `pip install clean-data-quality[image]`

## Handling Duplicates

### Option 1: Remove All Duplicates

```python
clean_df = cleaner.get_clean_data(
    remove_duplicates=True  # Keeps first occurrence
)
```

### Option 2: Review and Decide

```python
# Get duplicate pairs for review
duplicates = report.duplicates()

# Check if labels match
for _, row in duplicates.iterrows():
    label1 = df.loc[row['index1'], 'label']
    label2 = df.loc[row['index2'], 'label']
    if label1 != label2:
        print(f"Conflict: {row['index1']} ({label1}) vs {row['index2']} ({label2})")
```

### Option 3: Keep Representative Sample

```python
# Get indices to remove (keeps first of each duplicate group)
indices_to_remove = set()
for _, row in duplicates.iterrows():
    indices_to_remove.add(row['index2'])  # Keep index1, remove index2

clean_df = df.drop(indices_to_remove)
```

## Clustering Duplicates

Find groups of similar samples:

```python
from collections import defaultdict

# Build duplicate graph
graph = defaultdict(set)
for _, row in duplicates.iterrows():
    graph[row['index1']].add(row['index2'])
    graph[row['index2']].add(row['index1'])

# Find connected components (duplicate clusters)
def find_cluster(start, graph, visited):
    cluster = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                cluster.add(neighbor)
                stack.append(neighbor)
    return cluster

visited = set()
clusters = []
for node in graph:
    if node not in visited:
        visited.add(node)
        clusters.append(find_cluster(node, graph, visited))

print(f"Found {len(clusters)} duplicate clusters")
```

## Performance Considerations

For large datasets, near-duplicate detection can be slow. Options:

1. **Increase threshold**: Higher similarity threshold = faster
2. **Use blocking**: Only compare samples within the same class
3. **Sample-based**: Detect duplicates on a sample, then extrapolate

```python
# Fast mode: higher threshold, class-based blocking
detector = DuplicateDetector(
    similarity_threshold=0.95,
    use_class_blocking=True  # Only compare within same class
)
```

## Best Practices

1. **Check label consistency**: Duplicates with different labels indicate errors
2. **Consider context**: Some duplicates are intentional (e.g., data augmentation)
3. **Validate removal**: Ensure you're not removing important samples
4. **Document decisions**: Record why duplicates were kept or removed
