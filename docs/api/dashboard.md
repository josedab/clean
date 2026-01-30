# Web Dashboard

Interactive browser-based interface for data quality monitoring.

## Overview

The Clean dashboard provides a visual interface for uploading, analyzing, and monitoring data quality. Built with FastAPI and Chart.js, it offers real-time analysis without writing code.

## Quick Start

### Command Line

```bash
# Start the dashboard
clean dashboard --port 8080

# With custom title
clean dashboard --port 8080 --title "My Data Quality Dashboard"
```

### Python

```python
from clean import run_dashboard

run_dashboard(
    host="0.0.0.0",
    port=8080,
    title="Production Data Monitor",
)
```

## DashboardApp

The main dashboard application class.

::: clean.dashboard.DashboardApp
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - run

### Programmatic Usage

```python
from clean import DashboardApp, DashboardConfig

# Create with custom config
config = DashboardConfig(
    host="127.0.0.1",
    port=3000,
    title="Data Quality Dashboard",
    enable_upload=True,
    max_upload_size_mb=100,
)

app = DashboardApp(config)
app.run()
```

## DashboardConfig

Configuration options for the dashboard.

::: clean.dashboard.DashboardConfig
    options:
      show_root_heading: true
      show_source: false

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `host` | `0.0.0.0` | Host to bind server to |
| `port` | `8080` | Port to bind server to |
| `title` | `Clean - Data Quality Dashboard` | Dashboard title |
| `enable_upload` | `True` | Allow file uploads |
| `max_upload_size_mb` | `100` | Maximum upload size in MB |

## Features

### File Upload

Drag-and-drop or click to upload CSV files for instant analysis.

- Supported formats: CSV
- Auto-detects label column (`label`, `target`, `y`, `class`)
- Progress indicator during analysis

### Quality Score

Visual gauge showing overall data quality (0-100):

- **Green (80-100)**: Good quality
- **Yellow (60-79)**: Needs attention
- **Red (0-59)**: Significant issues

### Issue Breakdown

Bar chart showing issue counts by type:

- Label errors
- Duplicates
- Outliers
- Missing values
- Class imbalance

### Feature Analysis

Table showing per-feature metrics:

- Data type
- Missing percentage
- Quality score with progress bar

## API Endpoints

The dashboard exposes a REST API:

### `GET /`

Returns the dashboard HTML page.

### `GET /api/health`

Health check endpoint.

```json
{"status": "healthy", "title": "Clean - Data Quality Dashboard"}
```

### `POST /api/analyze`

Analyze an uploaded file.

**Request:**
```bash
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@data.csv" \
  -F "label_column=target"
```

**Response:**
```json
{
  "total_samples": 10000,
  "total_features": 15,
  "total_issues": 234,
  "quality_score": 82.5,
  "issue_counts": {
    "label_errors": 45,
    "duplicates": 89,
    "outliers": 67,
    "missing_values": 33,
    "class_imbalance": 0
  },
  "top_issues": [
    {"severity": "error", "message": "Found 45 potential label errors"},
    {"severity": "warning", "message": "Found 89 duplicate rows"}
  ],
  "features": [
    {"name": "age", "dtype": "float64", "missing_pct": 0.01, "quality": 99},
    {"name": "income", "dtype": "float64", "missing_pct": 0.05, "quality": 95}
  ]
}
```

### `GET /api/config`

Get dashboard configuration.

## Embedding in Applications

### FastAPI Integration

```python
from fastapi import FastAPI
from clean.dashboard import DashboardApp

# Create your main app
main_app = FastAPI()

# Mount dashboard at /dashboard
dashboard = DashboardApp()
main_app.mount("/dashboard", dashboard.app)
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
RUN pip install clean-data-quality[api]

EXPOSE 8080
CMD ["clean", "dashboard", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t clean-dashboard .
docker run -p 8080:8080 clean-dashboard
```

## Customization

### Custom Analysis

```python
from clean.dashboard import DashboardApp

class CustomDashboard(DashboardApp):
    def _analyze_dataframe(self, df, label_column):
        # Call parent analysis
        result = super()._analyze_dataframe(df, label_column)
        
        # Add custom metrics
        result["custom_metric"] = calculate_custom_metric(df)
        
        return result

app = CustomDashboard()
app.run()
```

### Adding Endpoints

```python
from clean import create_dashboard_app

app = create_dashboard_app()

@app.app.get("/api/custom")
async def custom_endpoint():
    return {"custom": "data"}

app.run()
```

## Architecture

```mermaid
graph TD
    A[Browser] --> B[FastAPI Server]
    B --> C[Static HTML/JS]
    B --> D[/api/analyze]
    B --> E[/api/health]
    C --> F[Chart.js]
    C --> G[File Upload Handler]
    D --> H[CSV Parser]
    H --> I[DatasetCleaner]
    I --> J[Analysis Results]
    J --> K[JSON Response]
    K --> A
    G --> D
```

## Best Practices

1. **Secure deployment**: Use HTTPS in production
2. **Limit upload size**: Set appropriate `max_upload_size_mb`
3. **Monitor resources**: Large files consume memory during analysis
4. **Use behind proxy**: Deploy behind nginx/traefik for production
5. **Enable authentication**: Add auth middleware for sensitive data
