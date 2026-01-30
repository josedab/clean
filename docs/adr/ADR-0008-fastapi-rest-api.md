# ADR-0008: FastAPI for REST API Layer

## Status

Accepted

## Context

Clean needed an HTTP API for:

1. **Dashboard integration**: Web UIs need to call analysis endpoints
2. **Microservice deployment**: Teams want Clean as a service
3. **Language-agnostic access**: R, Julia, and JavaScript users
4. **Webhook triggers**: CI/CD pipelines calling analysis remotely

Requirements:
- Modern async support for streaming endpoints
- Automatic API documentation
- Request validation
- File upload handling
- Easy deployment (single command)

Frameworks considered:

| Framework | Pros | Cons |
|-----------|------|------|
| Flask | Simple, well-known | Sync only, manual docs |
| Django REST | Full-featured | Heavy, complex |
| FastAPI | Async, auto-docs, Pydantic | Newer, smaller community |
| Starlette | Lightweight async | Too low-level |

## Decision

We chose **FastAPI** for the REST API layer, isolated as an optional dependency.

```python
# api.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Clean Data Quality API",
    description="AI-powered data quality analysis for ML datasets",
    version="1.0.0",
)

# CORS for dashboard integration
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    label_column: str = Query(default="label"),
) -> JSONResponse:
    """Analyze a CSV dataset for quality issues."""
    df = pd.read_csv(io.BytesIO(await file.read()))
    cleaner = DatasetCleaner(data=df, label_column=label_column)
    report = cleaner.analyze()
    return JSONResponse(content={
        "quality_score": report.quality_score.overall,
        "issues": {...},
        "summary": report.summary(),
    })

@app.post("/analyze/streaming")
async def analyze_streaming(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=10000),
) -> JSONResponse:
    """Analyze large datasets with streaming."""
    # Uses StreamingCleaner internally
    ...
```

Key endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/analyze` | POST | Full dataset analysis |
| `/analyze/streaming` | POST | Chunked analysis for large files |
| `/analyze/llm` | POST | LLM data quality analysis |
| `/fix` | POST | Apply fixes and return cleaned data |
| `/lineage/runs` | GET | List analysis history |
| `/detectors` | GET | List available detectors |
| `/plugins` | GET | List registered plugins |

The API is optional and fails gracefully:

```python
try:
    from fastapi import FastAPI
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. Install with: pip install clean[api]"
    ) from e
```

## Consequences

### Positive

- **Automatic OpenAPI docs**: `/docs` provides interactive Swagger UI
- **Pydantic validation**: Request/response models validated automatically
- **Async native**: Streaming endpoints work naturally
- **Type hints → schema**: Python types become JSON schema
- **CORS built-in**: Dashboard integration works out of the box

### Negative

- **Additional dependency**: FastAPI + uvicorn + pydantic (~20 MB)
- **Learning curve**: Teams unfamiliar with async may struggle
- **Deployment complexity**: Need ASGI server (uvicorn) not just WSGI

### Neutral

- **Separate from core**: API is in `api.py`, not mixed with core logic
- **Stateless design**: Each request is independent (lineage tracker uses file storage)

## Running the API

```bash
# Via CLI
clean serve --port 8000

# Via uvicorn directly
uvicorn clean.api:app --host 0.0.0.0 --port 8000

# Programmatically
from clean.api import run_server
run_server(port=8000)
```

## Security Considerations

- CORS allows all origins by default (configure for production)
- No authentication built-in (add via middleware for production)
- File uploads go to temp directory (cleaned after processing)
- Rate limiting not included (use reverse proxy)

## Related Decisions

- ADR-0005 (Optional Dependencies): API is in `[api]` extra
- ADR-0007 (Async Streaming): `/analyze/streaming` uses `StreamingCleaner`
