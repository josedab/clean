# Plugin System Migration Guide

This guide helps you migrate custom detectors and loaders to Clean's plugin system.

## Overview

Clean's plugin system allows you to extend the platform with custom:
- **Detectors** - Custom quality issue detectors
- **Loaders** - Custom data source loaders
- **Exporters** - Custom report exporters
- **Fixers** - Custom fix strategies

## Quick Start

### Registering a Custom Detector

```python
from clean.plugins import registry, DetectorPlugin
from clean.core.report import QualityReport
import pandas as pd

@registry.detector("my_detector")
class MyCustomDetector(DetectorPlugin):
    """Detect custom quality issues."""

    name = "my_detector"
    description = "Detects custom quality issues in data"

    def detect(self, data: pd.DataFrame, labels: pd.Series, **kwargs) -> list[int]:
        """Return indices of problematic samples."""
        issues = []
        # Your detection logic here
        for i, row in data.iterrows():
            if self._is_problematic(row):
                issues.append(i)
        return issues

    def _is_problematic(self, row: pd.Series) -> bool:
        # Custom logic
        return False
```

### Registering a Custom Loader

```python
from clean.plugins import registry, LoaderPlugin
import pandas as pd

@registry.loader("my_format")
class MyFormatLoader(LoaderPlugin):
    """Load data from custom format."""

    name = "my_format"
    description = "Loads data from .myformat files"
    supported_extensions = [".myformat", ".mf"]

    def load(self, source: str, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
        """Load data and labels from source."""
        # Your loading logic here
        data = pd.DataFrame(...)
        labels = pd.Series(...)
        return data, labels

    def can_load(self, source: str) -> bool:
        """Check if this loader can handle the source."""
        return str(source).endswith(tuple(self.supported_extensions))
```

### Registering a Custom Exporter

```python
from clean.plugins import registry, ExporterPlugin
from clean.core.report import QualityReport
from pathlib import Path

@registry.exporter("my_format")
class MyFormatExporter(ExporterPlugin):
    """Export reports to custom format."""

    name = "my_format"
    description = "Exports reports to .myformat files"

    def export(self, report: QualityReport, path: str | Path, **kwargs) -> None:
        """Export the report to the specified path."""
        path = Path(path)
        # Your export logic here
        with open(path, 'w') as f:
            f.write(self._format_report(report))

    def _format_report(self, report: QualityReport) -> str:
        # Custom formatting
        return ""
```

### Registering a Custom Fixer

```python
from clean.plugins import registry, FixerPlugin, SuggestedFix
from clean.core.report import QualityReport
import pandas as pd

@registry.fixer("my_fixer")
class MyCustomFixer(FixerPlugin):
    """Custom fix strategy."""

    name = "my_fixer"
    description = "Applies custom fixes to detected issues"

    def suggest_fixes(
        self,
        data: pd.DataFrame,
        report: QualityReport,
        **kwargs,
    ) -> list[SuggestedFix]:
        """Generate fix suggestions."""
        fixes = []
        # Your fix suggestion logic
        return fixes

    def apply_fix(
        self,
        data: pd.DataFrame,
        fix: SuggestedFix,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply a single fix to the data."""
        # Your fix application logic
        return data
```

## Migration from Legacy Code

### Before (Hardcoded Detector)

```python
# Old approach - directly in codebase
class MyDetector:
    def find_issues(self, df, labels):
        issues = []
        for i, row in df.iterrows():
            if some_condition(row):
                issues.append(i)
        return issues

# Usage
detector = MyDetector()
issues = detector.find_issues(df, labels)
```

### After (Plugin-based)

```python
# New approach - as a plugin
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    name = "my_detector"
    description = "My custom detector"

    def detect(self, data, labels, **kwargs):
        issues = []
        for i, row in data.iterrows():
            if some_condition(row):
                issues.append(i)
        return issues

# Usage - automatic discovery
from clean.plugins import registry
detector = registry.get_detector("my_detector")
issues = detector.detect(df, labels)
```

## Plugin Discovery

Plugins are automatically discovered when registered using decorators. To list available plugins:

```python
from clean.plugins import registry

# List all registered plugins
print("Detectors:", registry.list_detectors())
print("Loaders:", registry.list_loaders())
print("Exporters:", registry.list_exporters())
print("Fixers:", registry.list_fixers())

# Get a specific plugin
detector = registry.get_detector("my_detector")
loader = registry.get_loader("my_format")
```

## Best Practices

1. **Use descriptive names**: Plugin names should be lowercase with underscores
2. **Document thoroughly**: Include docstrings explaining what your plugin does
3. **Handle errors gracefully**: Catch and re-raise with context
4. **Test independently**: Write unit tests for your plugins
5. **Follow type hints**: Use proper type annotations for IDE support

## Example: Complete Custom Detector

```python
from clean.plugins import registry, DetectorPlugin
import pandas as pd
import numpy as np

@registry.detector("pattern_detector")
class PatternDetector(DetectorPlugin):
    """Detect samples matching suspicious patterns."""

    name = "pattern_detector"
    description = "Finds samples with suspicious data patterns"

    def __init__(self, patterns: list[str] | None = None):
        self.patterns = patterns or ["N/A", "null", "undefined", "-999"]

    def detect(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        columns: list[str] | None = None,
        **kwargs,
    ) -> list[int]:
        """Find samples containing suspicious patterns.

        Args:
            data: Feature DataFrame
            labels: Label series
            columns: Columns to check (default: all string columns)

        Returns:
            List of indices with suspicious patterns
        """
        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()

        suspicious_indices = set()

        for col in columns:
            if col not in data.columns:
                continue

            for pattern in self.patterns:
                mask = data[col].astype(str).str.contains(pattern, case=False, na=False)
                suspicious_indices.update(data.index[mask].tolist())

        return sorted(suspicious_indices)
```

## Integrating Plugins with DatasetCleaner

```python
from clean import DatasetCleaner
from clean.plugins import registry

# Register your custom detector first
@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    # ... implementation

# Then use it in analysis (future feature)
cleaner = DatasetCleaner(data=df, label_column="label")
# cleaner.add_detector("my_detector")  # Coming soon
report = cleaner.analyze()
```

## Troubleshooting

### Plugin Not Found

```python
# Error: KeyError: 'my_detector' not registered
# Solution: Ensure the module containing your plugin is imported

import my_plugins  # Import the module with your plugin definitions
from clean.plugins import registry
detector = registry.get_detector("my_detector")
```

### Type Errors

```python
# Error: TypeError: detect() must return list[int]
# Solution: Ensure your detect method returns a list of integer indices

def detect(self, data, labels, **kwargs) -> list[int]:
    # Return integer indices, not boolean mask
    mask = some_condition(data)
    return data.index[mask].tolist()  # Convert to list of ints
```

## Next Steps

- See the [API Reference](../api/plugins.md) for complete plugin API documentation
- Check out [examples/custom_detector.py](../../examples/custom_detector.py) for a working example
- Read about [Auto-Fix Engine](../api/fixes.md) to create custom fixers
