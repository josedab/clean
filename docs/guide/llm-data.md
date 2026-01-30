# LLM Data Quality

Clean provides specialized quality analysis for LLM training data, including
instruction-tuning datasets and RAG document collections.

## Overview

The `LLMDataCleaner` detects common issues in LLM datasets:
- Duplicate instructions/prompts
- Short or empty responses
- Refusal patterns ("I cannot", "I'm sorry")
- Incoherent instruction-response pairs

## Quick Start

```python
from clean import LLMDataCleaner
import pandas as pd

# Load instruction-tuning dataset
df = pd.DataFrame({
    "instruction": ["Explain photosynthesis", "What is 2+2?", "Explain photosynthesis"],
    "response": ["Plants convert sunlight...", "4", "Sorry, I can't help with that."]
})

# Analyze
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
)
report = cleaner.analyze(df)

print(report.summary())
```

## Configuration

### Instruction-Tuning Mode

```python
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
    min_response_length=10,  # Flag short responses
)
```

### RAG Document Mode

```python
cleaner = LLMDataCleaner(
    text_column="document",
    mode="rag",
    min_doc_length=100,
)
```

## Issue Types

### Duplicate Instructions

Detects semantically similar prompts:

```python
duplicates = report.get_issues_by_type("duplicate_instruction")
for issue in duplicates:
    print(f"Row {issue.index}: duplicate of row {issue.metadata['duplicate_of']}")
```

### Short Responses

Flags responses below minimum length:

```python
short = report.get_issues_by_type("short_response")
for issue in short:
    print(f"Row {issue.index}: only {issue.metadata['length']} chars")
```

### Refusals

Detects common refusal patterns:

```python
refusals = report.get_issues_by_type("refusal")
# "I cannot", "I'm sorry", "I don't have", etc.
```

### Incoherent Pairs

Flags potentially mismatched instruction-response pairs:

```python
incoherent = report.get_issues_by_type("incoherent")
```

## Get Clean Data

Remove problematic samples:

```python
clean_df = report.get_clean_data(
    df,
    remove_duplicates=True,
    remove_short=True,
    remove_refusals=True,
)
```

## Custom Refusal Patterns

Add domain-specific refusal patterns:

```python
custom_refusals = [
    r"I'm unable to",
    r"This goes against",
    r"I must decline",
]

cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
    refusal_patterns=custom_refusals,
)
```

## Best Practices

1. **Start with default settings** - They catch most common issues
2. **Review flagged samples** - Some may be false positives
3. **Adjust min_response_length** - Depends on your use case
4. **Combine with manual review** - Automated checks + human oversight
