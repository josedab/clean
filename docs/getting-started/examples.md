# Examples

This page provides additional examples for common use cases.

## Classification Dataset

```python
from clean import DatasetCleaner
import pandas as pd
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                           n_informative=15, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['label'] = y

# Analyze
cleaner = DatasetCleaner(data=df, label_column='label')
report = cleaner.analyze()

print(report.summary())
```

## Text Classification

```python
from clean import DatasetCleaner
import pandas as pd

# Sample text data
data = {
    'text': [
        'This movie is great!',
        'Terrible film, waste of time',
        'This movie is great!',  # duplicate
        'Amazing cinematography',
        'Teh moive was grat!',  # near-duplicate with typos
    ],
    'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative']  # mislabeled
}
df = pd.DataFrame(data)

cleaner = DatasetCleaner(
    data=df,
    label_column='sentiment',
    text_column='text'  # Enable text embeddings
)
report = cleaner.analyze()
```

## Loading from CSV

```python
from clean import DatasetCleaner

cleaner = DatasetCleaner.from_csv(
    'data/training_data.csv',
    label_column='target'
)
report = cleaner.analyze()
```

## HuggingFace Datasets

```python
from clean import DatasetCleaner
from datasets import load_dataset

# Load dataset
dataset = load_dataset('imdb', split='train[:1000]')

cleaner = DatasetCleaner.from_huggingface(
    dataset,
    label_column='label',
    text_column='text'
)
report = cleaner.analyze()
```

## Getting Clean Data

```python
# Remove detected issues
clean_df = cleaner.get_clean_data(
    remove_duplicates=True,
    remove_outliers='conservative',  # 'aggressive' removes more
    exclude_label_errors=False  # Keep for manual review
)

print(f"Original: {len(df)} rows")
print(f"Cleaned: {len(clean_df)} rows")
```

## Review Queue

```python
# Get prioritized list of samples to review
review_queue = cleaner.get_review_queue(max_items=100)
print(review_queue.head())
```

## Saving Reports

```python
# Save as JSON
report.save_json('quality_report.json')

# Save as HTML
report.save_html('quality_report.html')

# Export to pandas
issues_df = report.all_issues()
issues_df.to_csv('all_issues.csv', index=False)
```

## Visualization

```python
from clean.visualization import (
    plot_quality_scores,
    plot_class_distribution,
    plot_label_error_confusion
)

# Plot quality scores
fig = plot_quality_scores(report)
fig.savefig('quality_scores.png')

# Plot class distribution
fig = plot_class_distribution(labels)
fig.savefig('class_distribution.png')
```

## Interactive Visualization (Jupyter)

```python
from clean.visualization import plot_report_dashboard

# Interactive dashboard
plot_report_dashboard(report, features=df, labels=labels)
```

## Batch Processing

```python
import os
from clean import DatasetCleaner

# Process multiple files
for filename in os.listdir('data/'):
    if filename.endswith('.csv'):
        cleaner = DatasetCleaner.from_csv(
            f'data/{filename}',
            label_column='label'
        )
        report = cleaner.analyze(show_progress=False)
        report.save_json(f'reports/{filename.replace(".csv", ".json")}')
        print(f"{filename}: Quality Score = {report.quality_score.overall:.2f}")
```

## Custom Detector Configuration

```python
from clean import DatasetCleaner, LabelErrorDetector, OutlierDetector

# Custom label error detection
label_detector = LabelErrorDetector(
    n_folds=10,  # More cross-validation folds
    confidence_threshold=0.7  # Higher threshold
)

# Custom outlier detection
outlier_detector = OutlierDetector(
    methods=['isolation_forest', 'lof'],
    contamination=0.05  # Expect 5% outliers
)

cleaner = DatasetCleaner(
    data=df,
    label_column='label',
    label_error_detector=label_detector,
    outlier_detector=outlier_detector
)
```
