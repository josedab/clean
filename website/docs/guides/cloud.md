---
sidebar_position: 19
title: Cloud Deployment
---

# Multi-Tenant Cloud Service

Deploy Clean as a hosted SaaS platform with authentication, workspaces, and billing.

## Overview

The cloud module provides everything needed to run Clean as a service:

- **Authentication**: JWT tokens, API keys
- **Multi-tenancy**: Isolated workspaces per customer
- **RBAC**: Role-based access control
- **Billing**: Usage tracking and Stripe integration
- **API**: REST endpoints for all features

## Quick Start

```python
from clean.cloud import CloudService, CloudConfig

# Initialize service
config = CloudConfig(
    database_url="postgresql://user:pass@localhost/clean",
    jwt_secret="your-secret-key",
    stripe_api_key="sk_...",  # Optional
)

service = CloudService(config=config)

# Create a workspace
workspace = service.create_workspace(
    name="Acme Corp",
    plan="professional",
    owner_email="admin@acme.com",
)

# Create a user
user = service.create_user(
    email="analyst@acme.com",
    workspace_id=workspace.id,
    role="analyst",
)

# Generate API key
api_key = service.create_api_key(
    user_id=user.id,
    name="Production Key",
    scopes=["analyze", "fix"],
)

print(f"API Key: {api_key.key}")
```

## Authentication

### User Login

```python
from clean.cloud import UserManager

manager = UserManager(service)

# Create user
user = manager.create_user(
    email="user@company.com",
    password="secure_password",
    role="analyst",
    workspace_id=workspace.id,
)

# Authenticate
token = manager.authenticate(email="user@company.com", password="secure_password")

# Verify token
user = manager.verify_token(token)
print(f"Authenticated: {user.email}")
```

### API Keys

```python
from clean.cloud import APIKeyManager

key_manager = APIKeyManager(service)

# Create key with scopes
api_key = key_manager.create(
    user_id=user.id,
    name="CI/CD Pipeline",
    scopes=["analyze"],           # Limited permissions
    expires_in_days=365,
)

# Validate key
user, scopes = key_manager.validate(api_key.key)
print(f"Key valid for: {scopes}")
```

## Workspaces

Workspaces provide tenant isolation:

```python
from clean.cloud import WorkspaceManager

ws_manager = WorkspaceManager(service)

# Create workspace
workspace = ws_manager.create(
    name="ML Team",
    plan="enterprise",
    settings={
        "max_users": 50,
        "max_datasets": 1000,
        "retention_days": 365,
    }
)

# Get workspace usage
usage = ws_manager.get_usage(workspace.id)
print(f"Datasets: {usage.datasets_count}")
print(f"API calls: {usage.api_calls_count}")
print(f"Storage: {usage.storage_mb} MB")
```

## Role-Based Access Control

### Roles

| Role | Permissions |
|------|-------------|
| `owner` | Full access, billing, user management |
| `admin` | User management, all analysis features |
| `analyst` | Run analysis, view reports |
| `viewer` | View reports only |

### Permission Checking

```python
from clean.cloud import check_permission, Permission

# Check before operations
if check_permission(user, Permission.ANALYZE):
    result = service.analyze(data, workspace_id=user.workspace_id)
else:
    raise PermissionError("Insufficient permissions")

# Available permissions
# Permission.ANALYZE - Run quality analysis
# Permission.FIX - Apply fixes
# Permission.MANAGE_USERS - Add/remove users
# Permission.MANAGE_BILLING - Access billing
# Permission.VIEW_REPORTS - View analysis reports
```

## API Endpoints

### Running the Server

```bash
# CLI
clean cloud serve --port 8000 --workers 4

# Or programmatically
from clean.cloud import create_cloud_app

app = create_cloud_app(service)
# Run with uvicorn: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | User login |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/workspaces` | GET | List workspaces |
| `/api/v1/analyze` | POST | Run analysis |
| `/api/v1/fix` | POST | Apply fixes |
| `/api/v1/usage` | GET | Get usage stats |
| `/api/v1/api-keys` | POST | Create API key |

### Example: Analyze via API

```bash
curl -X POST https://api.clean.example.com/api/v1/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@training_data.csv" \
  -F "label_column=label"
```

## Billing Integration

### Usage Tracking

```python
from clean.cloud import UsageTracker

tracker = UsageTracker(service)

# Record analysis usage
tracker.record_analysis(
    workspace_id=workspace.id,
    rows_analyzed=10000,
    features_used=["labels", "duplicates", "outliers"],
)

# Get usage summary
usage = tracker.get_usage(workspace.id, period="month")
print(f"Rows analyzed: {usage.rows_analyzed:,}")
print(f"API calls: {usage.api_calls}")
```

### Stripe Integration

```python
from clean.cloud import BillingManager

billing = BillingManager(service, stripe_api_key="sk_...")

# Get current invoice
invoice = billing.get_current_invoice(workspace.id)
print(f"Current charges: ${invoice.total:.2f}")

# Get billing history
history = billing.get_history(workspace.id)
```

### Plans and Quotas

```python
from clean.cloud import Plan, QuotaManager

# Define plans
PLANS = {
    "free": Plan(
        name="free",
        max_rows_per_month=10_000,
        max_api_calls_per_day=100,
        max_users=1,
        features=["basic_analysis"],
        price_per_month=0,
    ),
    "professional": Plan(
        name="professional",
        max_rows_per_month=1_000_000,
        max_api_calls_per_day=10_000,
        max_users=10,
        features=["all"],
        price_per_month=99,
    ),
    "enterprise": Plan(
        name="enterprise",
        max_rows_per_month=None,  # Unlimited
        max_api_calls_per_day=None,
        max_users=None,
        features=["all", "sla", "support"],
        price_per_month=499,
    ),
}

# Check quotas
quota_manager = QuotaManager(service)

if quota_manager.check_quota(workspace.id, "rows", count=5000):
    # Proceed with analysis
    pass
else:
    raise QuotaExceededError("Monthly row limit reached")
```

## Configuration

```python
from clean.cloud import CloudConfig

config = CloudConfig(
    # Database
    database_url="postgresql://user:pass@localhost/clean",
    
    # Authentication
    jwt_secret="your-secret-key",
    jwt_expiry_hours=24,
    
    # Billing (optional)
    stripe_api_key="sk_...",
    
    # Storage (optional)
    storage_backend="s3",
    storage_bucket="clean-uploads",
    
    # Rate limiting
    rate_limit_per_minute=60,
)
```

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["clean", "cloud", "serve", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  clean:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db/clean
      - JWT_SECRET=your-secret
    depends_on:
      - db
      
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=clean
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clean-cloud
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: clean
        image: clean-data-quality:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: clean-secrets
              key: database-url
```

## Monitoring

### Health Checks

```bash
curl https://api.clean.example.com/health
# {"status": "healthy", "database": "connected", "version": "1.0.0"}
```

### Metrics

```python
from clean.cloud import MetricsExporter

# Prometheus metrics at /metrics
exporter = MetricsExporter(port=9090)
service.add_metrics_exporter(exporter)
```

## Security Best Practices

1. **Use HTTPS**: Always deploy behind TLS
2. **Rotate secrets**: Change JWT secrets periodically
3. **Limit API key scopes**: Give minimum required permissions
4. **Enable audit logging**: Track all operations
5. **Set rate limits**: Prevent abuse
6. **Use database encryption**: Encrypt sensitive data at rest

## Next Steps

- [Real-Time Pipeline](/docs/guides/realtime) - Stream processing
- [Privacy Vault](/docs/guides/privacy) - Data protection
- [API Reference](/docs/guides/cloud) - Full API documentation
