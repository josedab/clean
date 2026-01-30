---
sidebar_position: 4
title: Plugin System
---

# Plugin System

Extend Clean with custom detectors, loaders, exporters, and fixers.

## Overview

The plugin system uses a registry pattern:

```python
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    name = "my_detector"
    description = "My custom detector"
    
    def detect(self, data, labels, **kwargs):
        issues = []
        # Your detection logic
        return issues
```

## Plugin Types

### DetectorPlugin

Detect new types of data quality issues:

```python
from clean.plugins import registry, DetectorPlugin
from clean.detection.base import DetectorResult, Issue
import pandas as pd

@registry.detector("null_checker")
class NullChecker(DetectorPlugin):
    """Detect rows with null values."""
    
    name = "null_checker"
    description = "Finds rows with missing values"
    
    def detect(
        self,
        data: pd.DataFrame,
        labels: pd.Series = None,
        threshold: float = 0.5,
        **kwargs
    ) -> list[int]:
        """Return indices of rows with too many nulls."""
        # Calculate null ratio per row
        null_ratio = data.isnull().sum(axis=1) / len(data.columns)
        
        # Return indices above threshold
        return list(data.index[null_ratio > threshold])
```

### LoaderPlugin

Load data from custom sources:

```python
from clean.plugins import registry, LoaderPlugin
import pandas as pd

@registry.loader("jsonl")
class JSONLLoader(LoaderPlugin):
    """Load JSONL files."""
    
    name = "jsonl"
    description = "Loads JSON Lines files"
    supported_extensions = [".jsonl", ".jl"]
    
    def load(
        self,
        source: str,
        label_column: str = "label",
        **kwargs
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_json(source, lines=True)
        labels = df.pop(label_column)
        return df, labels
    
    def can_load(self, source: str) -> bool:
        return str(source).endswith(tuple(self.supported_extensions))
```

### ExporterPlugin

Export reports in custom formats:

```python
from clean.plugins import registry, ExporterPlugin
from clean.core.report import QualityReport
from pathlib import Path

@registry.exporter("markdown")
class MarkdownExporter(ExporterPlugin):
    """Export reports as Markdown."""
    
    name = "markdown"
    description = "Exports reports as Markdown files"
    
    def export(
        self,
        report: QualityReport,
        path: str | Path,
        **kwargs
    ) -> None:
        path = Path(path)
        
        md = f"""# Data Quality Report

## Summary
- **Samples**: {report.dataset_info.n_samples:,}
- **Quality Score**: {report.quality_score.overall:.1f}/100

## Issues Found
- Label Errors: {len(report.label_errors())}
- Duplicates: {len(report.duplicates())}
- Outliers: {len(report.outliers())}

## Recommendations
{report.summary()}
"""
        path.write_text(md)
```

### FixerPlugin

Custom fix strategies:

```python
from clean.plugins import registry, FixerPlugin, SuggestedFix
from clean.core.report import QualityReport
import pandas as pd

@registry.fixer("smart_imputer")
class SmartImputer(FixerPlugin):
    """Intelligent imputation for outliers."""
    
    name = "smart_imputer"
    description = "Uses ML-based imputation for outlier values"
    
    def suggest_fixes(
        self,
        data: pd.DataFrame,
        report: QualityReport,
        **kwargs
    ) -> list[SuggestedFix]:
        fixes = []
        
        for idx in report.outliers():
            for col in data.columns:
                val = data.loc[idx, col]
                median = data[col].median()
                
                # Suggest imputation if value is extreme
                if abs(val - median) > 3 * data[col].std():
                    fixes.append(SuggestedFix(
                        issue_type="outlier",
                        issue_index=idx,
                        fix_type="impute",
                        confidence=0.75,
                        description=f"Impute {col} with median",
                        old_value=val,
                        new_value=median,
                        metadata={"column": col},
                    ))
        
        return fixes
    
    def apply_fix(
        self,
        data: pd.DataFrame,
        fix: SuggestedFix,
        **kwargs
    ) -> pd.DataFrame:
        col = fix.metadata["column"]
        data.loc[fix.issue_index, col] = fix.new_value
        return data
```

## Using Plugins

### List Registered Plugins

```python
from clean.plugins import registry

print("Detectors:", registry.list_detectors())
print("Loaders:", registry.list_loaders())
print("Exporters:", registry.list_exporters())
print("Fixers:", registry.list_fixers())
```

### Get a Plugin

```python
# Get by name
detector = registry.get_detector("null_checker")
loader = registry.get_loader("jsonl")
exporter = registry.get_exporter("markdown")
fixer = registry.get_fixer("smart_imputer")
```

### Use a Plugin

```python
# Use detector
detector = registry.get_detector("null_checker")
issues = detector.detect(df, threshold=0.3)
print(f"Found {len(issues)} rows with >30% nulls")

# Use loader
loader = registry.get_loader("jsonl")
features, labels = loader.load("data.jsonl")

# Use exporter
exporter = registry.get_exporter("markdown")
exporter.export(report, "report.md")
```

## Plugin Discovery

Plugins are registered when their module is imported:

```python
# my_plugins.py
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    ...

# main.py
import my_plugins  # Registers the plugin

from clean.plugins import registry
detector = registry.get_detector("my_detector")  # Works!
```

### Auto-Discovery (Optional)

For packages, use entry points:

```toml
# pyproject.toml
[project.entry-points."clean.plugins"]
my_detector = "my_package.plugins:MyDetector"
```

## Plugin Metadata

Add metadata for better discovery:

```python
@registry.detector(
    "advanced_null_checker",
    description="Advanced null detection with patterns",
    version="2.0.0",
    author="Your Name",
    tags=["nulls", "missing", "data-quality"],
)
class AdvancedNullChecker(DetectorPlugin):
    ...
```

Access metadata:

```python
info = registry.get_plugin_info("advanced_null_checker", "detector")
print(f"Version: {info.version}")
print(f"Author: {info.author}")
print(f"Tags: {info.tags}")
```

## Testing Plugins

```python
import pytest
from clean.plugins import registry, DetectorPlugin
import pandas as pd

class TestNullChecker:
    @pytest.fixture
    def detector(self):
        return registry.get_detector("null_checker")
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, None, 3],
            "c": [1, 2, 3],
        })
    
    def test_detects_nulls(self, detector, sample_data):
        issues = detector.detect(sample_data, threshold=0.5)
        assert 1 in issues  # Row 1 has 2/3 nulls
    
    def test_threshold(self, detector, sample_data):
        issues = detector.detect(sample_data, threshold=0.9)
        assert len(issues) == 0  # No row has >90% nulls
```

## Best Practices

### 1. Follow the Interface

```python
# Detectors must return list of indices
def detect(self, data, labels, **kwargs) -> list[int]:
    return [indices]

# Loaders must return (features, labels) tuple
def load(self, source, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
    return features, labels
```

### 2. Handle Edge Cases

```python
def detect(self, data, labels, **kwargs):
    if data.empty:
        return []
    
    if labels is None:
        # Handle unsupervised case
        pass
```

### 3. Document Parameters

```python
def detect(
    self,
    data: pd.DataFrame,
    labels: pd.Series = None,
    threshold: float = 0.5,
    **kwargs
) -> list[int]:
    """Detect rows with missing values.
    
    Args:
        data: Feature DataFrame
        labels: Labels (optional)
        threshold: Null ratio threshold (0-1)
    
    Returns:
        List of row indices above threshold
    """
```

### 4. Use Type Hints

```python
from typing import Any
import pandas as pd
import numpy as np

def detect(
    self,
    data: pd.DataFrame,
    labels: pd.Series | np.ndarray | None = None,
    **kwargs: Any,
) -> list[int]:
    ...
```

## Next Steps

- [CLI](/docs/guides/cli) - Use plugins from command line
- [REST API](/docs/guides/rest-api) - Expose plugins via API
- [API Reference](/docs/api/fix-engine) - Plugin base classes
