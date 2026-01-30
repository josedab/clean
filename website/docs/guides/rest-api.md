---
sidebar_position: 6
title: REST API
---

# REST API

Clean provides HTTP endpoints for remote analysis and dashboard integration.

## Installation

```bash
pip install clean-data-quality[api]
```

## Starting the Server

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

Example:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@data.csv" \
  -F "label_column=target"
```

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
    "summary": "Data Quality Report..."
}
```

### Streaming Analysis

```http
POST /analyze/streaming
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file
- `label_column`: Label column name
- `chunk_size`: Rows per chunk (100-1000000)

### LLM Data Analysis

```http
POST /analyze/llm
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file
- `instruction_column`: Instruction column
- `response_column`: Response column
- `mode`: "instruction" or "rag"

### Apply Fixes

```http
POST /fix
Content-Type: multipart/form-data
```

Parameters:
- `file`: CSV file
- `label_column`: Label column
- `strategy`: "conservative" or "aggressive"

### List Detectors

```http
GET /detectors
```

### List Plugins

```http
GET /plugins
```

## API Documentation

Interactive docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Python Client

```python
import requests

with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        params={"label_column": "target"}
    )

result = response.json()
print(f"Quality Score: {result['quality_score']}")
```

## CORS

CORS is enabled by default. Configure for production:

```python
from clean.api import app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
)
```
