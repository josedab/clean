# ADR-0002: Plugin Architecture for Extensibility

## Status

Accepted

## Context

Clean needed to support extensibility for several reasons:

1. **Domain-specific detectors**: Different industries have unique data quality concerns (healthcare PII, financial fraud patterns, etc.)
2. **Custom data sources**: Enterprise users often have proprietary data formats or internal data lakes
3. **Export integrations**: Teams want to export reports to their existing tools (Slack, JIRA, custom dashboards)
4. **Fix strategies**: Auto-fix logic varies by domain and risk tolerance

We considered several extensibility patterns:

1. **Inheritance-based**: Users subclass core classes
2. **Configuration-driven**: YAML/JSON files define custom behavior
3. **Plugin registry**: Decorator-based registration with discovery
4. **Monkey patching**: Allow runtime modifications (discouraged)

Requirements:
- Minimal boilerplate for plugin authors
- Type safety and IDE support
- Discoverable plugins (list what's available)
- Isolated failures (bad plugin doesn't crash core)
- No core code modifications needed

## Decision

We implemented a **decorator-based plugin registry** with four plugin types: Detectors, Loaders, Exporters, and Fixers.

```python
# plugins.py - Core registry
class PluginRegistry:
    def detector(self, name: str, **metadata):
        def decorator(cls):
            self._detectors[name] = PluginInfo(name=name, cls=cls, ...)
            return cls
        return decorator

# Global singleton
registry = PluginRegistry()
```

Plugin authors use decorators:

```python
from clean.plugins import registry, DetectorPlugin

@registry.detector("drift", description="Detect data drift")
class DriftDetector(DetectorPlugin):
    def fit(self, features, labels=None):
        self._baseline = features.mean()
        return self
    
    def detect(self, features, labels=None):
        # Custom detection logic
        return DetectorResult(issues=[...])
```

Plugin consumers discover and use plugins:

```python
from clean.plugins import registry

# List available
for info in registry.list_detectors():
    print(f"{info.name}: {info.description}")

# Instantiate and use
detector_cls = registry.get_detector("drift")
detector = detector_cls(threshold=0.1)
result = detector.fit_detect(X, y)
```

## Consequences

### Positive

- **Zero boilerplate registration**: Single decorator registers plugin with metadata
- **Type safety**: Abstract base classes (`DetectorPlugin`, etc.) provide interface contracts
- **Discoverability**: `registry.list_*()` methods enable plugin browsers and documentation
- **Namespace isolation**: Plugins are identified by string names, avoiding class name conflicts
- **Testability**: `registry.clear()` enables clean test isolation
- **Metadata support**: Version, author, tags enable plugin marketplaces

### Negative

- **Global state**: Singleton registry requires care in testing and multi-tenant scenarios
- **Import side effects**: Plugins register on import, which can surprise users
- **No lazy loading**: All plugins load at import time (could impact startup)
- **String-based lookup**: Typos in plugin names fail at runtime, not compile time

### Neutral

- **Four plugin types**: Covers current needs but may need expansion (e.g., Transformers, Validators)
- **No dependency injection**: Plugins instantiate their own dependencies

## Related Decisions

- ADR-0005 (Optional Dependencies): Plugins requiring heavy dependencies should be in separate packages
- ADR-0004 (Facade Pattern): DatasetCleaner can use registered plugins automatically
