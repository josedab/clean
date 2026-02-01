"""Web Dashboard for Clean data quality monitoring.

This module provides a web-based dashboard for visualizing data quality metrics.
It uses FastAPI to serve a single-page application with real-time analysis capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import uvicorn
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from clean.core import DatasetCleaner


__all__ = [
    "DashboardConfig",
    "DashboardApp",
    "create_dashboard_app",
    "run_dashboard",
]


@dataclass
class DashboardConfig:
    """Configuration for the web dashboard.

    Attributes:
        host: Host to bind the server to (default: 127.0.0.1 for local-only access)
        port: Port to bind the server to
        title: Dashboard title
        enable_upload: Whether to allow file uploads
        max_upload_size_mb: Maximum upload file size in MB
    """

    host: str = "127.0.0.1"
    port: int = 8080
    title: str = "Clean - Data Quality Dashboard"
    enable_upload: bool = True
    max_upload_size_mb: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "title": self.title,
            "enable_upload": self.enable_upload,
            "max_upload_size_mb": self.max_upload_size_mb,
        }


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" integrity="sha384-O95ozGPUOKXoF1Kn3rl5IxPNGpJTQs9aLcpFbHuHj8C5LcOHAyYcZq0YL9PX4c5" crossorigin="anonymous"></script>
    <style>
        :root {{
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 1rem;
        }}
        header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        header p {{ opacity: 0.9; }}
        .grid {{ display: grid; gap: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
        .card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }}
        .card h2 {{
            font-size: 1rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }}
        .metric {{ font-size: 2.5rem; font-weight: 700; }}
        .metric.good {{ color: var(--success); }}
        .metric.warning {{ color: var(--warning); }}
        .metric.danger {{ color: var(--danger); }}
        .score-ring {{
            width: 200px;
            height: 200px;
            margin: 0 auto;
            position: relative;
        }}
        .score-ring canvas {{ width: 100%; height: 100%; }}
        .score-value {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}
        .score-value .number {{ font-size: 3rem; font-weight: 700; }}
        .score-value .label {{ color: var(--text-muted); font-size: 0.875rem; }}
        .upload-area {{
            border: 2px dashed var(--border);
            border-radius: 0.75rem;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .upload-area:hover {{ border-color: var(--primary); background: rgba(59, 130, 246, 0.05); }}
        .upload-area.dragging {{ border-color: var(--primary); background: rgba(59, 130, 246, 0.1); }}
        .btn {{
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .btn:hover {{ background: var(--primary-dark); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .issues-list {{ list-style: none; }}
        .issues-list li {{
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .issues-list li:last-child {{ border-bottom: none; }}
        .issue-badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .issue-badge.error {{ background: #fee2e2; color: var(--danger); }}
        .issue-badge.warning {{ background: #fef3c7; color: var(--warning); }}
        .issue-badge.info {{ background: #dbeafe; color: var(--primary); }}
        .chart-container {{ height: 300px; }}
        .loading {{
            display: none;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        .loading.active {{ display: flex; }}
        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .hidden {{ display: none; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ font-weight: 600; color: var(--text-muted); font-size: 0.875rem; }}
        .progress-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-bar .fill {{
            height: 100%;
            transition: width 0.5s;
        }}
        .progress-bar .fill.good {{ background: var(--success); }}
        .progress-bar .fill.warning {{ background: var(--warning); }}
        .progress-bar .fill.danger {{ background: var(--danger); }}
        .tab-buttons {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; }}
        .tab-btn {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            background: white;
            border-radius: 0.5rem;
            cursor: pointer;
        }}
        .tab-btn.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧹 {title}</h1>
            <p>AI-Powered Data Quality Analysis</p>
        </header>

        <div class="grid">
            <!-- Upload Section -->
            <div class="card" id="upload-card">
                <h2>📁 Upload Dataset</h2>
                <div class="upload-area" id="upload-area">
                    <p>Drop CSV file here or click to browse</p>
                    <input type="file" id="file-input" accept=".csv" style="display: none;">
                    <button class="btn" style="margin-top: 1rem;">Select File</button>
                </div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <span style="margin-left: 1rem;">Analyzing data...</span>
                </div>
            </div>

            <!-- Quality Score -->
            <div class="card" id="score-card">
                <h2>📊 Overall Quality Score</h2>
                <div class="score-ring">
                    <canvas id="score-chart"></canvas>
                    <div class="score-value">
                        <div class="number" id="score-number">--</div>
                        <div class="label">out of 100</div>
                    </div>
                </div>
            </div>

            <!-- Quick Stats -->
            <div class="card" id="stats-card">
                <h2>📈 Dataset Statistics</h2>
                <table>
                    <tr><td>Total Samples</td><td id="stat-samples">--</td></tr>
                    <tr><td>Total Features</td><td id="stat-features">--</td></tr>
                    <tr><td>Issues Found</td><td id="stat-issues">--</td></tr>
                    <tr><td>Clean Samples</td><td id="stat-clean">--</td></tr>
                </table>
            </div>
        </div>

        <div class="grid" style="margin-top: 1.5rem;">
            <!-- Issue Breakdown -->
            <div class="card" style="grid-column: span 2;">
                <h2>⚠️ Issue Breakdown</h2>
                <div class="chart-container">
                    <canvas id="issues-chart"></canvas>
                </div>
            </div>

            <!-- Issues List -->
            <div class="card">
                <h2>🔍 Top Issues</h2>
                <ul class="issues-list" id="issues-list">
                    <li><span class="issue-badge info">Info</span> Upload a dataset to see issues</li>
                </ul>
            </div>
        </div>

        <div class="grid" style="margin-top: 1.5rem;">
            <!-- Feature Quality -->
            <div class="card" style="grid-column: span 3;">
                <h2>📋 Feature-Level Analysis</h2>
                <table id="feature-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Type</th>
                            <th>Missing %</th>
                            <th>Quality</th>
                        </tr>
                    </thead>
                    <tbody id="feature-tbody">
                        <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No data</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Dashboard Application
        const Dashboard = {{
            charts: {{}},
            data: null,

            init() {{
                this.setupUpload();
                this.initCharts();
            }},

            setupUpload() {{
                const area = document.getElementById('upload-area');
                const input = document.getElementById('file-input');

                area.addEventListener('click', () => input.click());
                area.addEventListener('dragover', (e) => {{
                    e.preventDefault();
                    area.classList.add('dragging');
                }});
                area.addEventListener('dragleave', () => {{
                    area.classList.remove('dragging');
                }});
                area.addEventListener('drop', (e) => {{
                    e.preventDefault();
                    area.classList.remove('dragging');
                    if (e.dataTransfer.files.length) {{
                        this.uploadFile(e.dataTransfer.files[0]);
                    }}
                }});
                input.addEventListener('change', () => {{
                    if (input.files.length) {{
                        this.uploadFile(input.files[0]);
                    }}
                }});
            }},

            async uploadFile(file) {{
                const loading = document.getElementById('loading');
                const uploadArea = document.getElementById('upload-area');

                uploadArea.classList.add('hidden');
                loading.classList.add('active');

                const formData = new FormData();
                formData.append('file', file);

                try {{
                    const resp = await fetch('/api/analyze', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await resp.json();

                    if (resp.ok) {{
                        this.data = data;
                        this.updateDashboard(data);
                    }} else {{
                        alert('Error: ' + (data.detail || 'Analysis failed'));
                    }}
                }} catch (err) {{
                    alert('Error: ' + err.message);
                }} finally {{
                    loading.classList.remove('active');
                    uploadArea.classList.remove('hidden');
                }}
            }},

            initCharts() {{
                // Score donut chart
                const scoreCtx = document.getElementById('score-chart').getContext('2d');
                this.charts.score = new Chart(scoreCtx, {{
                    type: 'doughnut',
                    data: {{
                        datasets: [{{
                            data: [0, 100],
                            backgroundColor: ['#e2e8f0', '#e2e8f0'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        cutout: '80%',
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{ legend: {{ display: false }} }}
                    }}
                }});

                // Issues bar chart
                const issuesCtx = document.getElementById('issues-chart').getContext('2d');
                this.charts.issues = new Chart(issuesCtx, {{
                    type: 'bar',
                    data: {{
                        labels: ['Label Errors', 'Duplicates', 'Outliers', 'Missing Values', 'Class Imbalance'],
                        datasets: [{{
                            label: 'Issue Count',
                            data: [0, 0, 0, 0, 0],
                            backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#22c55e']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            y: {{ beginAtZero: true }}
                        }}
                    }}
                }});
            }},

            updateDashboard(data) {{
                // Update score
                const score = Math.round(data.quality_score || 0);
                document.getElementById('score-number').textContent = score;

                const color = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';
                this.charts.score.data.datasets[0].data = [score, 100 - score];
                this.charts.score.data.datasets[0].backgroundColor = [color, '#e2e8f0'];
                this.charts.score.update();

                // Update stats
                document.getElementById('stat-samples').textContent = data.total_samples?.toLocaleString() || '--';
                document.getElementById('stat-features').textContent = data.total_features || '--';
                document.getElementById('stat-issues').textContent = data.total_issues?.toLocaleString() || '--';
                document.getElementById('stat-clean').textContent =
                    ((data.total_samples || 0) - (data.total_issues || 0)).toLocaleString();

                // Update issues chart
                const issueTypes = data.issue_counts || {{}};
                this.charts.issues.data.datasets[0].data = [
                    issueTypes.label_errors || 0,
                    issueTypes.duplicates || 0,
                    issueTypes.outliers || 0,
                    issueTypes.missing_values || 0,
                    issueTypes.class_imbalance || 0
                ];
                this.charts.issues.update();

                // Update issues list
                const list = document.getElementById('issues-list');
                const issues = data.top_issues || [];
                list.innerHTML = issues.length ? issues.map(issue => `
                    <li>
                        <span class="issue-badge ${{issue.severity}}">${{issue.severity}}</span>
                        ${{issue.message}}
                    </li>
                `).join('') : '<li><span class="issue-badge info">Info</span> No significant issues found!</li>';

                // Update feature table
                const tbody = document.getElementById('feature-tbody');
                const features = data.features || [];
                tbody.innerHTML = features.length ? features.map(f => `
                    <tr>
                        <td>${{f.name}}</td>
                        <td>${{f.dtype}}</td>
                        <td>${{(f.missing_pct * 100).toFixed(1)}}%</td>
                        <td>
                            <div class="progress-bar">
                                <div class="fill ${{f.quality >= 80 ? 'good' : f.quality >= 60 ? 'warning' : 'danger'}}"
                                     style="width: ${{f.quality}}%"></div>
                            </div>
                        </td>
                    </tr>
                `).join('') : '<tr><td colspan="4" style="text-align: center;">No feature data</td></tr>';
            }}
        }};

        document.addEventListener('DOMContentLoaded', () => Dashboard.init());
    </script>
</body>
</html>
"""


class DashboardApp:
    """Web dashboard application for data quality monitoring.

    This class creates a FastAPI application that serves an interactive
    dashboard for analyzing data quality.

    Attributes:
        config: Dashboard configuration
        app: FastAPI application instance
    """

    def __init__(self, config: DashboardConfig | None = None):
        """Initialize the dashboard application.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI and uvicorn are required for the dashboard. "
                "Install with: pip install fastapi uvicorn python-multipart"
            )

        self.config = config or DashboardConfig()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title=self.config.title,
            description="AI-Powered Data Quality Dashboard",
            version="1.0.0",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register API routes."""

        @app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the dashboard HTML."""
            return DASHBOARD_HTML.format(title=self.config.title)

        @app.get("/api/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "title": self.config.title}

        @app.post("/api/analyze")
        async def analyze_file(
            file: UploadFile = File(...),
            label_column: str | None = Query(None, description="Label column name"),
        ):
            """Analyze an uploaded CSV file."""
            if not self.config.enable_upload:
                raise HTTPException(status_code=403, detail="File upload is disabled")

            # Check file size
            content = await file.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > self.config.max_upload_size_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {self.config.max_upload_size_mb}MB",
                )

            try:
                # Parse CSV
                import io

                df = pd.read_csv(io.BytesIO(content))

                # Detect label column if not specified
                if label_column is None:
                    for col in ["label", "target", "y", "class"]:
                        if col in df.columns:
                            label_column = col
                            break

                # Run analysis
                result = self._analyze_dataframe(df, label_column)
                return JSONResponse(content=result)

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @app.get("/api/config")
        async def get_config():
            """Get dashboard configuration."""
            return self.config.to_dict()

    def _analyze_dataframe(
        self,
        df: pd.DataFrame,
        label_column: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a DataFrame and return results."""
        result = {
            "total_samples": len(df),
            "total_features": len(df.columns),
            "total_issues": 0,
            "quality_score": 100.0,
            "issue_counts": {
                "label_errors": 0,
                "duplicates": 0,
                "outliers": 0,
                "missing_values": 0,
                "class_imbalance": 0,
            },
            "top_issues": [],
            "features": [],
        }

        # Separate features and labels
        feature_cols = [c for c in df.columns if c != label_column]
        features = df[feature_cols] if feature_cols else df

        # Analyze features
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            dtype = str(df[col].dtype)
            quality = max(0, 100 - (missing_pct * 100))

            result["features"].append({
                "name": col,
                "dtype": dtype,
                "missing_pct": missing_pct,
                "quality": quality,
            })

            if missing_pct > 0:
                result["issue_counts"]["missing_values"] += int(df[col].isna().sum())

        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            result["issue_counts"]["duplicates"] = int(duplicates)
            result["top_issues"].append({
                "severity": "warning",
                "message": f"Found {duplicates} duplicate rows",
            })

        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            if col == label_column:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            outlier_count += int(outliers)

        result["issue_counts"]["outliers"] = outlier_count
        if outlier_count > 0:
            result["top_issues"].append({
                "severity": "info",
                "message": f"Found {outlier_count} potential outliers",
            })

        # Check class imbalance
        if label_column and label_column in df.columns:
            class_counts = df[label_column].value_counts()
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.min() / class_counts.max()
                if imbalance_ratio < 0.5:
                    result["issue_counts"]["class_imbalance"] = 1
                    result["top_issues"].append({
                        "severity": "warning",
                        "message": f"Class imbalance detected (ratio: {imbalance_ratio:.2f})",
                    })

        # Run core detection if labels available
        if label_column and label_column in df.columns:
            try:
                cleaner = DatasetCleaner(
                    data=df,
                    label_column=label_column,
                )
                report = cleaner.analyze(show_progress=False)

                # Count label errors
                label_errors = len(report.label_errors())
                result["issue_counts"]["label_errors"] = label_errors
                if label_errors > 0:
                    result["top_issues"].insert(0, {
                        "severity": "error",
                        "message": f"Found {label_errors} potential label errors",
                    })

            except Exception:
                pass  # Silently handle analysis errors

        # Calculate total issues and quality score
        result["total_issues"] = sum(result["issue_counts"].values())
        issue_ratio = result["total_issues"] / max(1, result["total_samples"])
        result["quality_score"] = round(max(0, 100 * (1 - min(1, issue_ratio * 2))), 1)

        return result

    def run(self) -> None:
        """Run the dashboard server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
        )


def create_dashboard_app(
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "Clean - Data Quality Dashboard",
    **kwargs: Any,
) -> DashboardApp:
    """Create a dashboard application.

    Args:
        host: Host to bind to (default: 127.0.0.1 for local-only access)
        port: Port to bind to
        title: Dashboard title
        **kwargs: Additional configuration options

    Returns:
        Configured DashboardApp instance
    """
    config = DashboardConfig(host=host, port=port, title=title, **kwargs)
    return DashboardApp(config)


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "Clean - Data Quality Dashboard",
    **kwargs: Any,
) -> None:
    """Start the dashboard server.

    Args:
        host: Host to bind to (default: 127.0.0.1 for local-only access)
        port: Port to bind to
        title: Dashboard title
        **kwargs: Additional configuration options
    """
    app = create_dashboard_app(host=host, port=port, title=title, **kwargs)
    app.run()
