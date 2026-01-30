---
sidebar_position: 7
title: LLMDataCleaner
---

# LLMDataCleaner

Specialized quality analysis for LLM training data.

```python
from clean.llm import LLMDataCleaner, LLMConfig, LLMMode
```

## Overview

`LLMDataCleaner` detects quality issues specific to LLM training data:
- Instruction-response inconsistencies
- Low-quality or incomplete responses
- Semantic duplicates
- Format violations
- Toxicity and safety issues

## LLMConfig

```python
@dataclass
class LLMConfig:
    mode: LLMMode = LLMMode.INSTRUCTION
    instruction_column: str = "instruction"
    response_column: str = "response"
    context_column: Optional[str] = None
    min_response_length: int = 10
    max_response_length: int = 10000
    quality_threshold: float = 0.7
    check_format: bool = True
    check_consistency: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | LLMMode | INSTRUCTION | Analysis mode |
| `instruction_column` | str | "instruction" | Instruction column |
| `response_column` | str | "response" | Response column |
| `context_column` | str | None | Context column (RAG) |
| `min_response_length` | int | 10 | Minimum response length |
| `max_response_length` | int | 10000 | Maximum response length |
| `quality_threshold` | float | 0.7 | Quality score threshold |
| `check_format` | bool | True | Check formatting issues |
| `check_consistency` | bool | True | Check instruction-response alignment |

## LLMMode

```python
class LLMMode(Enum):
    INSTRUCTION = "instruction"   # Instruction-tuning data
    RAG = "rag"                   # RAG training data
    CHAT = "chat"                 # Multi-turn conversations
    COMPLETION = "completion"     # Text completion data
```

## Constructor

```python
LLMDataCleaner(config: Optional[LLMConfig] = None)
```

## Methods

### analyze()

Analyze LLM training dataset.

```python
analyze(data: pd.DataFrame) -> LLMQualityReport
```

#### Returns

`LLMQualityReport` with:
- `quality_score`: Overall and component scores
- `issues`: List of detected issues
- `summary()`: Text summary
- `low_quality_samples()`: Samples below threshold
- `inconsistent_pairs()`: Instruction-response mismatches

## Example: Instruction Data

```python
import pandas as pd
from clean.llm import LLMDataCleaner, LLMConfig, LLMMode

# Load data
df = pd.DataFrame({
    "instruction": [
        "Explain photosynthesis",
        "Write a poem about cats",
        "What is 2+2?",
    ],
    "response": [
        "Photosynthesis is the process...",
        "Whiskers soft and paws so light...",
        "4",
    ],
})

# Analyze
config = LLMConfig(
    mode=LLMMode.INSTRUCTION,
    instruction_column="instruction",
    response_column="response",
)

cleaner = LLMDataCleaner(config)
report = cleaner.analyze(df)

print(report.summary())
```

## Example: RAG Data

```python
from clean.llm import LLMDataCleaner, LLMConfig, LLMMode

config = LLMConfig(
    mode=LLMMode.RAG,
    instruction_column="query",
    response_column="answer",
    context_column="context",
)

cleaner = LLMDataCleaner(config)
report = cleaner.analyze(df)

# Get samples where response contradicts context
contradictions = report.context_contradictions()
for item in contradictions[:5]:
    print(f"Index {item.index}: {item.reason}")
```

## Issue Types

### InstructionResponseMismatch

Response doesn't address the instruction.

```python
@dataclass
class InstructionResponseMismatch:
    index: int
    instruction: str
    response: str
    similarity_score: float
    reason: str
```

### LowQualityResponse

Response quality below threshold.

```python
@dataclass
class LowQualityResponse:
    index: int
    response: str
    quality_score: float
    issues: List[str]  # "too_short", "repetitive", "incomplete"
```

### FormatViolation

Response format issues.

```python
@dataclass
class FormatViolation:
    index: int
    violation_type: str  # "truncated", "encoding", "structure"
    details: str
```

### SemanticDuplicate

Semantically similar instruction-response pairs.

```python
@dataclass
class SemanticDuplicate:
    index1: int
    index2: int
    similarity: float
    duplicate_type: str  # "instruction", "response", "both"
```

## Full Example

```python
import pandas as pd
from clean.llm import LLMDataCleaner, LLMConfig, LLMMode

# Load training data
df = pd.read_json("training_data.jsonl", lines=True)
print(f"Loaded {len(df)} samples")

# Configure
config = LLMConfig(
    mode=LLMMode.INSTRUCTION,
    instruction_column="prompt",
    response_column="completion",
    min_response_length=20,
    quality_threshold=0.8,
)

# Analyze
cleaner = LLMDataCleaner(config)
report = cleaner.analyze(df)

print(report.summary())

# Review issues
print("\nLow quality responses:")
for item in report.low_quality_samples()[:10]:
    print(f"  [{item.index}] Score: {item.quality_score:.2f}")
    print(f"      Issues: {', '.join(item.issues)}")

print("\nInstruction-response mismatches:")
for item in report.inconsistent_pairs()[:10]:
    print(f"  [{item.index}] Similarity: {item.similarity_score:.2f}")

# Get clean data
clean_indices = report.get_clean_indices(
    min_quality=0.8,
    exclude_duplicates=True,
)
clean_df = df.iloc[clean_indices]
print(f"\nClean samples: {len(clean_df)}/{len(df)}")

# Export
clean_df.to_json("training_data_clean.jsonl", orient="records", lines=True)
```

## Best Practices

1. **Start with conservative thresholds** - Adjust based on data quality
2. **Review edge cases** - Low scores may still be valid
3. **Check for domain-specific patterns** - Some issues are context-dependent
4. **Use semantic deduplication** - Similar questions with different phrasing
5. **Validate context alignment** - For RAG data, ensure responses use context
