# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) documenting the significant architectural decisions made in the Clean data quality platform.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences. ADRs help new team members understand why the system is built the way it is.

## ADR Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-0001](ADR-0001-cleanlab-label-error-detection.md) | Cleanlab as Core Label Error Detection Engine | Accepted |
| [ADR-0002](ADR-0002-plugin-architecture.md) | Plugin Architecture for Extensibility | Accepted |
| [ADR-0003](ADR-0003-pandas-data-interface.md) | Pandas DataFrame as Primary Data Interface | Accepted |
| [ADR-0004](ADR-0004-facade-pattern-cleaner.md) | Facade Pattern for DatasetCleaner Entry Point | Accepted |
| [ADR-0005](ADR-0005-optional-dependencies.md) | Optional Dependencies via Feature Extras | Accepted |
| [ADR-0006](ADR-0006-dataclass-type-system.md) | Dataclass-Based Type System | Accepted |
| [ADR-0007](ADR-0007-async-streaming.md) | Async Streaming for Large Dataset Processing | Accepted |
| [ADR-0008](ADR-0008-fastapi-rest-api.md) | FastAPI for REST API Layer | Accepted |
| [ADR-0009](ADR-0009-fix-strategy-pattern.md) | Strategy Pattern for Fix Application | Accepted |
| [ADR-0010](ADR-0010-lineage-tracking.md) | Lineage Tracking with Append-Only Audit Log | Accepted |
| [ADR-0011](ADR-0011-multi-backend-streaming.md) | Multi-Backend Streaming Architecture | Accepted |
| [ADR-0012](ADR-0012-github-actions-integration.md) | GitHub Actions Integration for CI/CD Quality Gates | Accepted |

## ADR Template

New ADRs should follow this structure:

```markdown
# ADR-NNNN: Title

## Status
Accepted | Superseded | Deprecated

## Context
What prompted this decision?

## Decision
What was decided?

## Consequences
Tradeoffs, implications, what this enabled/prevented.
```

## Contributing

When making significant architectural changes, please create a new ADR documenting the decision. Use the next available number and follow the template above.
