# Natural Language Query

Query quality reports using natural language questions.

## Quick Example

```python
from clean import DatasetCleaner
from clean.nl_query import NLQueryEngine, query_report

# Analyze data
cleaner = DatasetCleaner(data=df, label_column="label")
report = cleaner.analyze()

# Query with natural language
engine = NLQueryEngine(report=report, data=df)
result = engine.query("What percentage of labels might be incorrect?")
print(result.answer)

# Or use convenience function
result = query_report(report, "Show me the class distribution")
```

## API Reference

### NLQueryEngine

Main query engine class.

#### `__init__(report, data=None, llm_provider=None)`

Initialize the query engine.

| Parameter | Type | Description |
|-----------|------|-------------|
| `report` | `QualityReport` | Quality report to query |
| `data` | `pd.DataFrame \| None` | Original dataset for detailed queries |
| `llm_provider` | `Any \| None` | Optional custom LLM provider |

#### `query(text: str) -> QueryResult`

Execute a natural language query.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Natural language question |

### QueryResult

Query result dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Original query text |
| `answer` | str | Natural language answer |
| `data` | Any \| None | Structured data (if applicable) |
| `statistics` | dict | Relevant statistics |
| `suggestions` | list[str] | Follow-up query suggestions |
| `execution_time_ms` | float | Query execution time |
| `confidence` | float | Answer confidence (0-1) |

#### `to_dict() -> dict`

Convert result to dictionary.

### Convenience Functions

#### `create_query_engine(report, data=None, **kwargs) -> NLQueryEngine`

Create a query engine instance.

#### `query_report(report, query, **kwargs) -> QueryResult`

One-liner query without creating engine explicitly.

```python
from clean.nl_query import query_report

result = query_report(report, "What are the main quality issues?")
print(result.answer)
```

## Supported Query Types

The NL Query engine supports various question patterns:

### Descriptive Queries
- "How many samples are there?"
- "What is the class distribution?"
- "Show me the data summary"

### Quality-Focused Queries
- "What are the main quality issues?"
- "Which labels might be incorrect?"
- "Are there any duplicate samples?"
- "What percentage of data are outliers?"

### Statistical Queries
- "What is the quality score?"
- "How severe are the issues?"
- "Compare label error rates by class"

### Recommendation Queries
- "What should I fix first?"
- "How can I improve data quality?"
- "Which samples need review?"

## Example Workflows

### Interactive Exploration

```python
from clean.nl_query import NLQueryEngine

engine = NLQueryEngine(report=report, data=df)

# Ask a series of questions
questions = [
    "What is the overall quality score?",
    "Which class has the most label errors?",
    "Show me the top 5 suspicious samples",
    "What are your recommendations?"
]

for q in questions:
    result = engine.query(q)
    print(f"Q: {q}")
    print(f"A: {result.answer}\n")
```

### Using Structured Data

```python
result = engine.query("Show me samples with low confidence labels")

if result.data is not None:
    # Access structured results
    suspicious_indices = result.data
    print(f"Found {len(suspicious_indices)} suspicious samples")
```
