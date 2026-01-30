---
sidebar_position: 1
title: LLM Data Quality
---

# LLM Data Quality

Clean provides specialized analysis for LLM training data, including instruction-tuning datasets and RAG document collections.

## The Problem

LLM training data has unique quality issues:

| Issue | Impact |
|-------|--------|
| Duplicate instructions | Wastes tokens, overfits |
| Short/empty responses | Teaches bad habits |
| Refusals | "I cannot help with that" pollution |
| Incoherent pairs | Instruction doesn't match response |
| Template artifacts | [INST] tags, XML remnants |

## Quick Start

```python
from clean import LLMDataCleaner

# Initialize for instruction-tuning data
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
)

# Analyze
report = cleaner.analyze(df)
print(report.summary())
```

Output:
```
LLM Data Quality Report
=======================
Samples: 50,000

Issues Found:
  - Duplicate instructions: 1,234 (2.5%)
  - Short responses: 567 (under 10 chars)
  - Refusals detected: 89
  - Incoherent pairs: 45

Quality Score: 88.5/100
```

## Modes

### Instruction-Tuning Mode

For instruction-response pairs:

```python
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
    mode="instruction",  # Default
    min_response_length=10,
)
```

### RAG Document Mode

For document collections:

```python
cleaner = LLMDataCleaner(
    text_column="document",
    mode="rag",
    min_doc_length=100,
)
```

## Issue Types

### Duplicate Instructions

Same prompt with different responses can confuse training:

```python
duplicates = report.get_issues_by_type("duplicate_instruction")

for issue in duplicates[:5]:
    print(f"Index {issue.index}: duplicate of {issue.metadata['duplicate_of']}")
    print(f"  Instruction: {df.iloc[issue.index]['instruction'][:50]}...")
```

### Short Responses

Responses below minimum length:

```python
short = report.get_issues_by_type("short_response")

# Adjust threshold
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
    min_response_length=50,  # Require at least 50 chars
)
```

### Refusals

Common refusal patterns detected:

```python
refusals = report.get_issues_by_type("refusal")

# Default patterns include:
# - "I cannot", "I can't"
# - "I'm sorry, but"
# - "I don't have the ability"
# - "As an AI language model"
```

Custom patterns:

```python
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
    refusal_patterns=[
        r"I cannot",
        r"I'm unable to",
        r"I must decline",
        r"against my programming",
    ],
)
```

### Incoherent Pairs

Instruction-response pairs that don't match:

```python
incoherent = report.get_issues_by_type("incoherent")

# These are detected via:
# - Empty response for non-empty instruction
# - Response that doesn't address the instruction
# - Mismatched language
```

## Getting Clean Data

### Remove Issues

```python
clean_df = report.get_clean_data(
    df,
    remove_duplicates=True,
    remove_short=True,
    remove_refusals=True,
    remove_incoherent=True,
)

print(f"Original: {len(df)}, Clean: {len(clean_df)}")
```

### Flag Issues

```python
# Add columns indicating issues
df['has_issue'] = df.index.isin(report.get_all_issue_indices())
df['issue_type'] = df.index.map(report.get_issue_type_map())
```

## Advanced Analysis

### Instruction Length Distribution

```python
import matplotlib.pyplot as plt

lengths = df['instruction'].str.len()
plt.hist(lengths, bins=50)
plt.xlabel('Instruction Length')
plt.ylabel('Count')
plt.title('Instruction Length Distribution')
```

### Response Quality Metrics

```python
# Average response length
avg_length = df['response'].str.len().mean()

# Unique instruction ratio
unique_ratio = df['instruction'].nunique() / len(df)

# Refusal rate
refusal_rate = len(report.get_issues_by_type('refusal')) / len(df)

print(f"Avg response length: {avg_length:.0f} chars")
print(f"Unique instructions: {unique_ratio:.1%}")
print(f"Refusal rate: {refusal_rate:.1%}")
```

### Topic Distribution

```python
# Check if topics are balanced
from collections import Counter

# Assuming a 'topic' column exists
topics = Counter(df['topic'])
print(topics.most_common(10))
```

## Best Practices

### 1. Clean Before Fine-Tuning

```python
# Standard pipeline
raw_df = load_instruction_data()

# Step 1: Analyze
cleaner = LLMDataCleaner(
    instruction_column="instruction",
    response_column="response",
)
report = cleaner.analyze(raw_df)

# Step 2: Review high-confidence issues
issues = report.to_dataframe()
high_conf = issues[issues['confidence'] > 0.9]
print(f"Review {len(high_conf)} high-confidence issues")

# Step 3: Clean
clean_df = report.get_clean_data(
    raw_df,
    remove_duplicates=True,
    remove_refusals=True,
)

# Step 4: Fine-tune on clean data
fine_tune(model, clean_df)
```

### 2. Balance Removal vs Keeping Data

Too aggressive cleaning → not enough training data
Too lenient → noisy training

```python
# Check removal rate
removal_rate = 1 - len(clean_df) / len(raw_df)
print(f"Removal rate: {removal_rate:.1%}")

# Target: under 10% removal for most datasets
```

### 3. Sample Review

Always manually review a sample:

```python
# Random sample of flagged issues
sample = report.to_dataframe().sample(n=20)

for _, row in sample.iterrows():
    print(f"=== Issue: {row['issue_type']} ===")
    print(f"Instruction: {raw_df.iloc[row['index']]['instruction']}")
    print(f"Response: {raw_df.iloc[row['index']]['response'][:200]}...")
    print()
```

### 4. Track Lineage

```python
from clean import LineageTracker

tracker = LineageTracker(project="my_llm")

# Log the cleaning run
run_id = tracker.log_analysis(
    dataset_name="instruction_data_v2",
    report=report,
)

# Log review decisions
tracker.log_review(run_id, sample_index=42, decision="keep")
tracker.log_review(run_id, sample_index=187, decision="remove")
```

## Integration with Training

### HuggingFace Datasets

```python
from datasets import Dataset

# Clean first
clean_df = report.get_clean_data(df, remove_duplicates=True)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(clean_df)

# Train
trainer.train(dataset)
```

### Axolotl Config

```yaml
# In your axolotl config
datasets:
  - path: ./clean_data.json  # Use cleaned data
    type: sharegpt
```

## Next Steps

- [Streaming](/docs/guides/streaming) - Process large LLM datasets
- [Auto-Fix Engine](/docs/guides/auto-fix) - Automated cleaning
- [API Reference](/docs/api/llm) - LLMDataCleaner API
