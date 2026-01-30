# Visualization

Clean provides both static (matplotlib) and interactive (plotly) visualizations.

## Static Plots

### Quality Scores

```python
from clean.visualization import plot_quality_scores

fig = plot_quality_scores(report)
fig.savefig('quality_scores.png', dpi=150, bbox_inches='tight')
```

### Class Distribution

```python
from clean.visualization import plot_class_distribution

fig = plot_class_distribution(labels)
fig.savefig('class_distribution.png')
```

### Label Error Confusion

```python
from clean.visualization import plot_label_error_confusion

fig = plot_label_error_confusion(report)
fig.savefig('label_errors.png')
```

### Outlier Distribution

```python
from clean.visualization import plot_outlier_distribution

fig = plot_outlier_distribution(report, features=df)
fig.savefig('outliers.png')
```

### Duplicate Similarity

```python
from clean.visualization import plot_duplicate_similarity

fig = plot_duplicate_similarity(report)
fig.savefig('duplicates.png')
```

### Report Summary

```python
from clean.visualization import plot_report_summary

fig = plot_report_summary(report)
fig.savefig('summary.png')
```

## Save All Plots

```python
from clean.visualization import save_all_plots

save_all_plots(
    report,
    output_dir='plots/',
    features=df,
    labels=labels,
    format='png',
    dpi=150
)
# Saves: quality_scores.png, summary.png, label_errors.png, etc.
```

## Interactive Plots

Requires: `pip install clean-data-quality[interactive]`

### Interactive Dashboard

```python
from clean.visualization import plot_report_dashboard

# In Jupyter notebook
plot_report_dashboard(report, features=df, labels=labels)
```

### Individual Interactive Plots

```python
from clean.visualization import (
    plot_quality_scores_interactive,
    plot_class_distribution_interactive,
    plot_label_errors_interactive,
    plot_outliers_interactive,
)

# Quality scores as bar chart
fig = plot_quality_scores_interactive(report)
fig.show()

# Interactive class distribution
fig = plot_class_distribution_interactive(labels)
fig.show()

# Label errors table with hover
fig = plot_label_errors_interactive(report)
fig.show()

# Outlier scatter with zoom/pan
fig = plot_outliers_interactive(report, features=df)
fig.show()
```

### Save Interactive HTML

```python
from clean.visualization import save_interactive_html

save_interactive_html(
    report,
    path='interactive_report.html',
    features=df,
    labels=labels
)
```

## Issue Browser Widget

An interactive Jupyter widget for reviewing issues:

```python
from clean.visualization import IssueBrowser

browser = IssueBrowser(report, features=df)
browser.display()
```

Features:
- Navigate through issues with next/previous buttons
- See sample details and context
- Filter by issue type
- Mark samples for review

## Review Queue Widget

For systematic issue review:

```python
from clean.visualization import ReviewQueue

queue = ReviewQueue(
    report,
    features=df,
    max_items=100
)
queue.display()

# After review
decisions = queue.get_decisions()
# Returns: {index: 'keep'|'remove'|'skip', ...}
```

## Customizing Plots

### Color Schemes

```python
from clean.visualization import plot_quality_scores

fig = plot_quality_scores(
    report,
    colors=['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
)
```

### Figure Size

```python
fig = plot_class_distribution(labels, figsize=(12, 6))
```

### Adding Titles

```python
fig = plot_quality_scores(report)
fig.suptitle('My Dataset Quality Report', fontsize=14)
fig.savefig('report.png')
```

## Integration with Other Tools

### Weights & Biases

```python
import wandb
from clean.visualization import plot_quality_scores

wandb.init(project='data-quality')

fig = plot_quality_scores(report)
wandb.log({'quality_scores': wandb.Image(fig)})
```

### MLflow

```python
import mlflow
from clean.visualization import save_all_plots
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    save_all_plots(report, output_dir=tmpdir, features=df)
    for filename in os.listdir(tmpdir):
        mlflow.log_artifact(os.path.join(tmpdir, filename))
```

### Streamlit

```python
import streamlit as st
from clean.visualization import plot_quality_scores

st.title('Data Quality Dashboard')

fig = plot_quality_scores(report)
st.pyplot(fig)

st.dataframe(report.label_errors())
```

## Best Practices

1. **Start with summary**: Use `plot_report_summary()` for overview
2. **Drill down**: Use specific plots for detailed analysis
3. **Interactive for exploration**: Use plotly for interactive analysis
4. **Static for reports**: Use matplotlib for documentation/reports
5. **Save high DPI**: Use `dpi=150` or higher for publications
