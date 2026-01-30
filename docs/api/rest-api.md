# REST API

Clean provides a REST API for remote data quality analysis, enabling
dashboard integration and web-based access.

## Installation

Install with API dependencies:

```bash
pip install clean-data-quality[api]
```

## Quick Start

Start the server:

```bash
clean serve --port 8000
```

Or programmatically:

```python
from clean.api import run_server
run_server(host="0.0.0.0", port=8000)
```

## Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00"
}
```

### Analyze Dataset

```http
POST /analyze
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file (required)
- `label_column`: Label column name (default: "label")
- `task_type`: Task type (optional)

Response:
```json
{
    "dataset_name": "data.csv",
    "total_samples": 10000,
    "quality_score": 85.5,
    "issues": {
        "label_errors": 150,
        "duplicates": 50,
        "outliers": 75
    },
    "summary": "Dataset Quality Report..."
}
```

### Streaming Analysis

```http
POST /analyze/streaming
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file (required)
- `label_column`: Label column name
- `chunk_size`: Rows per chunk (100-1000000)

Response:
```json
{
    "dataset_name": "large_data.csv",
    "streaming_summary": {
        "total_rows": 1000000,
        "total_chunks": 100,
        "total_issues": 5000,
        "average_quality_score": 92.5
    },
    "chunk_results": [...]
}
```

### LLM Data Analysis

```http
POST /analyze/llm
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file (required)
- `instruction_column`: Instruction column name
- `response_column`: Response column name
- `mode`: "instruction" or "rag"
- `min_response_length`: Minimum response length

### Apply Fixes

```http
POST /fix
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file (required)
- `label_column`: Label column name
- `strategy`: "conservative" or "aggressive"
- `remove_duplicates`: Boolean
- `remove_outliers`: Boolean
- `min_confidence`: Float (0-1)

Response:
```json
{
    "original_rows": 10000,
    "final_rows": 9850,
    "rows_removed": 150,
    "rows_relabeled": 25,
    "fixes_applied": [...],
    "cleaned_data_preview": [...]
}
```

### List Analysis Runs

```http
GET /lineage/runs?limit=50
```

### Get Run Details

```http
GET /lineage/runs/{run_id}
```

### Submit Review

```http
POST /review
Content-Type: application/json

{
    "sample_index": 42,
    "decision": "correct",
    "new_label": "positive",
    "notes": "Verified manually"
}
```

### List Detectors

```http
GET /detectors
```

### List Plugins

```http
GET /plugins
```

## Python Client

```python
import requests

# Upload and analyze
with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        params={"label_column": "target"}
    )

result = response.json()
print(f"Quality Score: {result['quality_score']}")
```

## Docker

```dockerfile
FROM python:3.11-slim

RUN pip install clean-data-quality[api]

EXPOSE 8000

CMD ["clean", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

## CORS

The API enables CORS by default for dashboard integration.
Configure allowed origins in production:

```python
from clean.api import app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-dashboard.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
