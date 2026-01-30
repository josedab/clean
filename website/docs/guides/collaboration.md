---
sidebar_position: 16
title: Collaborative Review
---

# Collaborative Review Workspace

Enable multiple team members to review and resolve data quality issues together.

## Why Collaborative Review?

When Clean finds thousands of potential issues, you need a systematic way to:
- Distribute review work across team members
- Track who reviewed what and when
- Build consensus on ambiguous cases
- Resolve disagreements fairly
- Audit the review process

The Collaborative Review Workspace provides all of this.

## Quick Start

```python
from clean.collaboration import ReviewWorkspace

# Initialize workspace
workspace = ReviewWorkspace(storage_backend="sqlite")

# Create a review session
session = workspace.create_session(
    name="Q4 Training Data Review",
    data=df,
    quality_report=report,
    reviewers=["alice@company.com", "bob@company.com", "carol@company.com"],
)

# Add items for review (e.g., suspected label errors)
session.add_items(
    indices=report.label_errors().head(200).index,
    issue_type="label_error",
    priority="high",
)

print(f"Session ID: {session.id}")
print(f"Items to review: {session.pending_count}")
print(f"Reviewers: {session.reviewers}")
```

## Review Workflow

### 1. Reviewers Submit Assessments

```python
# Alice reviews an item
session.submit_review(
    reviewer="alice@company.com",
    item_id=42,
    decision="relabel",        # keep, relabel, remove, uncertain
    new_label="cat",           # if relabeling
    confidence=0.9,            # how sure they are
    notes="Clearly a cat, mislabeled as dog",
)

# Batch reviews
session.submit_reviews(
    reviewer="bob@company.com",
    reviews=[
        {"item_id": 42, "decision": "relabel", "new_label": "cat"},
        {"item_id": 43, "decision": "keep"},
        {"item_id": 44, "decision": "remove", "notes": "Corrupted image"},
    ]
)
```

### 2. Build Consensus

```python
from clean.collaboration import VotingStrategy

# Get consensus using majority voting
consensus = session.get_consensus(
    strategy=VotingStrategy.MAJORITY,
    min_votes=2,  # Require at least 2 votes
)

for item_id, result in consensus.items():
    print(f"Item {item_id}:")
    print(f"  Decision: {result.decision}")
    print(f"  Agreement: {result.agreement_score:.0%}")
    print(f"  Has conflict: {result.has_conflict}")
```

### 3. Resolve Conflicts

```python
# Find items with disagreement
conflicts = session.get_conflicts(min_disagreement=0.5)

for conflict in conflicts:
    print(f"\nItem {conflict.item_id}: {conflict.description}")
    for reviewer, vote in conflict.votes.items():
        print(f"  {reviewer}: {vote.decision} - {vote.notes}")

# Escalate to expert
session.escalate(
    item_ids=[conflict.item_id],
    expert="senior_reviewer@company.com",
    reason="Ambiguous edge case",
)

# Or admin resolves directly
session.resolve_conflict(
    item_id=conflict.item_id,
    resolver="admin@company.com",
    final_decision="relabel",
    final_label="cat",
    resolution_note="Expert confirmed this is a cat",
)
```

### 4. Finalize and Apply

```python
# Check if complete
if session.is_complete:
    # Finalize session
    final_result = session.finalize()
    
    # Apply decisions to dataset
    clean_df = final_result.apply_to_dataframe(df)
    
    print(f"Relabeled: {final_result.relabeled_count}")
    print(f"Removed: {final_result.removed_count}")
    print(f"Kept: {final_result.kept_count}")
```

## Voting Strategies

### Majority Voting

```python
consensus = session.get_consensus(strategy=VotingStrategy.MAJORITY)
# Decision = most common vote
```

### Unanimous

```python
consensus = session.get_consensus(strategy=VotingStrategy.UNANIMOUS)
# Requires all reviewers to agree
```

### Weighted Voting

```python
consensus = session.get_consensus(
    strategy=VotingStrategy.WEIGHTED,
    weights={
        "alice@company.com": 1.0,   # Senior reviewer
        "bob@company.com": 0.8,
        "carol@company.com": 0.8,
    }
)
```

### Dawid-Skene Model

Statistical model that accounts for annotator reliability:

```python
consensus = session.get_consensus(strategy=VotingStrategy.DAWID_SKENE)
# Learns annotator quality from agreement patterns
```

## Reviewer Management

### Adding Reviewers

```python
# Add a reviewer
session.add_reviewer(
    email="dave@company.com",
    role="reviewer",  # admin, reviewer, expert
)

# Assign specific items
session.assign_items(
    reviewer="dave@company.com",
    item_ids=[100, 101, 102, 103, 104],
)

# Auto-assign (round-robin)
session.auto_assign(items_per_reviewer=50)
```

### Reviewer Analytics

```python
from clean.collaboration import ReviewerAnalytics

analytics = ReviewerAnalytics(session)

for reviewer in session.reviewers:
    stats = analytics.get_reviewer_stats(reviewer)
    print(f"\n{reviewer}:")
    print(f"  Progress: {stats.completed}/{stats.assigned}")
    print(f"  Agreement with consensus: {stats.consensus_agreement:.0%}")
    print(f"  Avg confidence: {stats.avg_confidence:.2f}")
    print(f"  Speed: {stats.avg_time_per_item:.1f}s/item")
```

## Session Management

### Session Lifecycle

```
Created → Started → In Progress → Complete → Finalized
              ↓           ↓
          Paused ←→ Conflicts → Resolution
```

### Session Operations

```python
# Start accepting reviews
session.start()

# Pause (no new reviews accepted)
session.pause()

# Resume
session.resume()

# Get status
print(f"Status: {session.status}")
print(f"Progress: {session.completed_count}/{session.total_count}")
print(f"Conflicts: {len(session.get_conflicts())}")
```

## Storage Backends

### SQLite (Default)

```python
workspace = ReviewWorkspace(storage_backend="sqlite")
# Stores in ~/.clean/reviews.db
```

### PostgreSQL

```python
workspace = ReviewWorkspace(
    storage_backend="postgresql",
    connection_string="postgresql://user:pass@localhost/clean",
)
```

### MongoDB

```python
workspace = ReviewWorkspace(
    storage_backend="mongodb",
    connection_string="mongodb://localhost:27017",
    database="clean_reviews",
)
```

## Export and Reports

### Export Reviewed Data

```python
# Export with metadata
session.export(
    output_path="reviewed_data.csv",
    include_metadata=True,  # Includes reviewer, decision, notes
)

# Export decisions only
decisions_df = session.export_decisions(format="dataframe")
```

### Generate Reports

```python
report = session.generate_report()

print(report.summary())
# Review Session: Q4 Training Data Review
# Status: Complete
# Total items: 200
# Reviewed: 200
# Relabeled: 45
# Removed: 12
# Agreement rate: 87%

report.to_html("review_report.html")
```

### Audit Trail

```python
# Export for compliance
audit = session.export_audit_trail()
audit.to_json("audit_trail.json")

# Includes:
# - Every review submitted
# - Timestamps
# - Reviewer identities
# - Conflict resolutions
# - Admin overrides
```

## Notifications

```python
session.configure_notifications(
    on_assignment=True,           # Notify when assigned items
    on_conflict=["admin@co.com"], # Notify specific people on conflicts
    on_complete=["pm@co.com"],    # Notify when session complete
    slack_webhook="https://hooks.slack.com/...",
)
```

## Integration with Labeling Tools

### Export to Label Studio

```python
from clean.active_learning import LabelStudioExporter

exporter = LabelStudioExporter()
exporter.export(
    samples=session.get_pending_items(),
    output_path="label_studio_tasks.json",
)
```

### Import from Labeling Tools

```python
# Import completed labels
session.import_labels(
    file_path="label_studio_results.json",
    format="label_studio",
    reviewer="label_studio_import",
)
```

## Convenience Function

```python
from clean.collaboration import create_review_session

session = create_review_session(
    data=df,
    quality_report=report,
    issue_types=["label_errors", "outliers"],
    reviewers=["alice@co.com", "bob@co.com"],
    min_reviews_per_item=2,
)
```

## Best Practices

1. **Set minimum reviewers**: At least 2-3 for reliable consensus
2. **Use weighted voting**: Account for expertise levels
3. **Monitor agreement rates**: Low agreement suggests unclear guidelines
4. **Resolve conflicts promptly**: Don't let them accumulate
5. **Track reviewer quality**: Inter-annotator agreement is a signal
6. **Export audit trails**: Required for compliance and reproducibility
7. **Provide clear guidelines**: Document what "relabel" vs "remove" means

## Next Steps

- [Privacy Vault](/docs/guides/privacy) - Protect sensitive review data
- [Root Cause Analysis](/docs/guides/root-cause) - Understand issue patterns
- [API Reference](/docs/guides/collaboration) - Full API documentation
