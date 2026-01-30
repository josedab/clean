# Visualization

Clean provides both static (matplotlib) and interactive (plotly) visualization functions.

## Static Plots (matplotlib)

### Quality Scores

```python
from clean.visualization import plot_quality_scores

fig = plot_quality_scores(report)
fig.savefig('quality.png')
```

### Class Distribution

```python
from clean.visualization import plot_class_distribution

fig = plot_class_distribution(labels)
fig.savefig('distribution.png')
```

### Label Error Confusion

```python
from clean.visualization import plot_label_error_confusion

fig = plot_label_error_confusion(report)
fig.savefig('confusion.png')
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

### Save All Plots

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
```

## Interactive Plots (plotly)

!!! note
    Requires: `pip install clean-data-quality[interactive]`

### Interactive Dashboard

```python
from clean.visualization import plot_report_dashboard

plot_report_dashboard(report, features=df, labels=labels)
```

### Individual Interactive Plots

```python
from clean.visualization import (
    plot_quality_scores_interactive,
    plot_class_distribution_interactive,
    plot_label_errors_interactive,
    plot_outliers_interactive,
    plot_duplicates_interactive,
)

# Each returns a plotly Figure
fig = plot_quality_scores_interactive(report)
fig.show()
```

### Save Interactive HTML

```python
from clean.visualization import save_interactive_html

save_interactive_html(
    report,
    path='dashboard.html',
    features=df,
    labels=labels
)
```

## Jupyter Widgets

!!! note
    Requires: `pip install clean-data-quality[interactive]`

### Issue Browser

Interactive widget for browsing detected issues.

```python
from clean.visualization import IssueBrowser

browser = IssueBrowser(report, features=df)
browser.display()
```

::: clean.visualization.browser.IssueBrowser
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - display

### Review Queue

Widget for systematic issue review with keep/remove decisions.

```python
from clean.visualization import ReviewQueue

queue = ReviewQueue(report, features=df, max_items=100)
queue.display()

# After review
decisions = queue.get_decisions()
```

::: clean.visualization.browser.ReviewQueue
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - display
        - get_decisions

## Convenience Functions

```python
from clean.visualization import browse_issues, review_issues

# Quick browsing
browse_issues(report, features=df)

# Quick review
decisions = review_issues(report, features=df)
```
