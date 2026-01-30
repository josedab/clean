# Plugin System

Clean's plugin architecture allows extending functionality with custom
detectors, loaders, exporters, and fixers.

## Overview

The plugin system uses a registry pattern with decorators for registration:

```python
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    name = "my_detector"
    description = "My custom detector"

    def detect(self, data, labels, **kwargs):
        # Detection logic
        return [indices_with_issues]
```

## Plugin Types

### DetectorPlugin

Detects quality issues in data:

```python
from clean.plugins import registry, DetectorPlugin
import pandas as pd

@registry.detector("pattern_checker")
class PatternChecker(DetectorPlugin):
    name = "pattern_checker"
    description = "Checks for suspicious patterns"

    def detect(self, data: pd.DataFrame, labels: pd.Series, **kwargs) -> list[int]:
        issues = []
        for i, row in data.iterrows():
            if self._is_suspicious(row):
                issues.append(i)
        return issues

    def _is_suspicious(self, row):
        return any(pd.isna(row))
```

### LoaderPlugin

Loads data from custom sources:

```python
from clean.plugins import registry, LoaderPlugin
import pandas as pd

@registry.loader("parquet")
class ParquetLoader(LoaderPlugin):
    name = "parquet"
    description = "Loads Parquet files"
    supported_extensions = [".parquet", ".pq"]

    def load(self, source: str, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_parquet(source)
        label_col = kwargs.get("label_column", "label")
        return df.drop(columns=[label_col]), df[label_col]

    def can_load(self, source: str) -> bool:
        return str(source).endswith(tuple(self.supported_extensions))
```

### ExporterPlugin

Exports reports to custom formats:

```python
from clean.plugins import registry, ExporterPlugin
from clean.core.report import QualityReport

@registry.exporter("markdown")
class MarkdownExporter(ExporterPlugin):
    name = "markdown"
    description = "Export to Markdown"

    def export(self, report: QualityReport, path, **kwargs) -> None:
        with open(path, "w") as f:
            f.write(f"# Quality Report\\n\\n")
            f.write(f"Score: {report.quality_score.overall}\\n")
```

### FixerPlugin

Applies custom fixes:

```python
from clean.plugins import registry, FixerPlugin, SuggestedFix
from clean.core.report import QualityReport

@registry.fixer("imputer")
class ImputerFixer(FixerPlugin):
    name = "imputer"
    description = "Imputes missing values"

    def suggest_fixes(self, data, report, **kwargs) -> list[SuggestedFix]:
        fixes = []
        for col in data.columns:
            if data[col].isna().any():
                fixes.append(SuggestedFix(
                    issue_type="missing_value",
                    issue_index=col,
                    fix_type="impute",
                    confidence=0.9,
                    description=f"Impute {col} with median",
                ))
        return fixes
```

## Using Plugins

### Listing Registered Plugins

```python
from clean.plugins import registry

print("Detectors:", registry.list_detectors())
print("Loaders:", registry.list_loaders())
print("Exporters:", registry.list_exporters())
print("Fixers:", registry.list_fixers())
```

### Getting a Plugin

```python
detector = registry.get_detector("pattern_checker")
results = detector.detect(df, labels)
```

### Plugin Discovery

Import your plugin module to register it:

```python
# my_plugins.py
from clean.plugins import registry, DetectorPlugin

@registry.detector("my_detector")
class MyDetector(DetectorPlugin):
    ...

# main.py
import my_plugins  # Registers the plugin
from clean.plugins import registry

detector = registry.get_detector("my_detector")
```

## SuggestedFix

The `SuggestedFix` dataclass represents a recommended fix:

```python
@dataclass
class SuggestedFix:
    issue_type: str                    # Type of issue
    issue_index: int | tuple[int, int] # Affected index/indices
    fix_type: str                      # 'relabel', 'remove', 'merge', etc.
    confidence: float                  # 0-1 confidence score
    description: str                   # Human-readable description
    old_value: Any = None             # Current value
    new_value: Any = None             # Proposed new value
    metadata: dict = field(default_factory=dict)
```

## API Reference

::: clean.plugins.PluginRegistry
    options:
      show_root_heading: true
      members:
        - detector
        - loader
        - exporter
        - fixer
        - get_detector
        - get_loader
        - get_exporter
        - get_fixer
        - list_detectors
        - list_loaders
        - list_exporters
        - list_fixers
