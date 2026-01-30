---
sidebar_position: 10
title: Real-Time Streaming
---

# Real-Time Streaming Pipeline

Monitor data quality continuously on streaming data from Kafka, Pulsar, or Redis.

## Why Real-Time Monitoring?

In production ML systems, data quality can degrade over time:
- Upstream data sources change format
- New edge cases appear in production traffic
- Data drift causes model performance to drop

Clean's real-time pipeline catches these issues as they happen, not days later when your model accuracy tanks.

## Quick Start

```python
import asyncio
from clean.realtime import RealtimePipeline, KafkaSource, PipelineConfig

async def monitor():
    # Connect to Kafka
    source = KafkaSource(
        bootstrap_servers="localhost:9092",
        topic="ml-training-data",
        group_id="quality-monitor",
    )
    
    # Configure pipeline
    config = PipelineConfig(
        window_size=1000,          # Analyze every 1000 messages
        quality_threshold=0.8,     # Alert if score drops below 80
        alert_on_degradation=True,
    )
    
    pipeline = RealtimePipeline(source=source, config=config)
    
    # Add alert handler
    pipeline.add_alert_handler(
        lambda alert: print(f"🚨 Quality Alert: {alert.message}")
    )
    
    # Start monitoring
    await pipeline.start()

asyncio.run(monitor())
```

## Supported Data Sources

### Apache Kafka

```python
from clean.realtime import KafkaSource

source = KafkaSource(
    bootstrap_servers="localhost:9092",
    topic="training-data",
    group_id="clean-monitor",
    auto_offset_reset="latest",  # or "earliest"
)
```

### Apache Pulsar

```python
from clean.realtime import PulsarSource

source = PulsarSource(
    service_url="pulsar://localhost:6650",
    topic="persistent://public/default/ml-data",
    subscription="quality-check",
)
```

### Redis Streams

```python
from clean.realtime import RedisSource

source = RedisSource(
    host="localhost",
    port=6379,
    stream="ml:data:stream",
    consumer_group="quality-monitors",
)
```

## Pipeline Configuration

```python
from clean.realtime import PipelineConfig

config = PipelineConfig(
    # Window settings
    window_size=1000,           # Samples per analysis window
    batch_timeout=30.0,         # Max seconds to wait for window
    
    # Quality thresholds
    quality_threshold=0.8,      # Minimum acceptable score
    alert_on_degradation=True,  # Send alerts when quality drops
    
    # Detectors to run
    detectors=["labels", "duplicates", "outliers"],  # or "all"
    
    # Performance
    max_workers=4,              # Parallel workers
)
```

## Alerting

### Slack Notifications

```python
from clean.realtime import SlackAlertHandler

slack = SlackAlertHandler(
    webhook_url="https://hooks.slack.com/services/T.../B.../xxx",
    channel="#data-quality-alerts",
)

pipeline.add_alert_handler(slack)
```

### Webhook Integration

```python
from clean.realtime import WebhookAlertHandler

webhook = WebhookAlertHandler(
    url="https://api.your-service.com/alerts",
    headers={"Authorization": "Bearer xxx"},
)

pipeline.add_alert_handler(webhook)
```

### Custom Handlers

```python
def my_alert_handler(alert):
    # alert.severity: "warning" | "critical"
    # alert.metric: "quality_score" | "label_errors" | etc
    # alert.value: current value
    # alert.threshold: configured threshold
    # alert.message: human-readable message
    
    if alert.severity == "critical":
        page_on_call_engineer(alert)
    else:
        log_to_datadog(alert)

pipeline.add_alert_handler(my_alert_handler)
```

## Metrics and Monitoring

```python
# Get current metrics
metrics = pipeline.get_metrics()

print(f"Messages processed: {metrics.total_messages}")
print(f"Windows analyzed: {metrics.windows_analyzed}")
print(f"Current quality score: {metrics.current_quality_score}")
print(f"Issues detected: {metrics.total_issues}")
print(f"Alerts sent: {metrics.alerts_sent}")
```

### Prometheus Integration

```python
from clean.realtime import PrometheusMetrics

# Expose metrics on /metrics endpoint
prometheus = PrometheusMetrics(port=9090)
pipeline.add_metrics_handler(prometheus)
```

## Message Format

Clean expects JSON messages with a predictable structure:

```json
{
  "features": {"col1": 1.5, "col2": "text", "col3": 0.8},
  "label": "positive",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {"source": "api", "user_id": "123"}
}
```

### Custom Parsers

```python
def my_parser(raw_message: bytes) -> dict:
    """Parse your custom message format."""
    data = json.loads(raw_message)
    return {
        "features": data["input"],
        "label": data["output"]["class"],
    }

pipeline = RealtimePipeline(
    source=source,
    config=config,
    message_parser=my_parser,
)
```

## Windowing Strategies

### Count-Based Windows

```python
config = PipelineConfig(
    window_size=1000,  # Every 1000 messages
)
```

### Time-Based Windows

```python
config = PipelineConfig(
    window_duration=60.0,  # Every 60 seconds
    min_window_size=100,   # But at least 100 messages
)
```

### Sliding Windows

```python
config = PipelineConfig(
    window_size=1000,
    slide_size=200,  # Slide by 200, overlap 800
)
```

## Graceful Shutdown

```python
import signal

async def main():
    pipeline = RealtimePipeline(source=source, config=config)
    
    # Handle shutdown signals
    def shutdown(sig, frame):
        asyncio.create_task(pipeline.stop())
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    await pipeline.start()
    
    # Wait for shutdown
    await pipeline.wait_closed()
    print(f"Final metrics: {pipeline.get_metrics()}")
```

## Installation

```bash
pip install clean-data-quality[streaming]
```

This installs:
- `aiokafka` - Async Kafka client
- `pulsar-client` - Pulsar client
- `redis` - Redis client

## Best Practices

1. **Start with larger windows**: Begin with 1000+ samples for stable statistics
2. **Set appropriate thresholds**: Don't alert on minor fluctuations
3. **Use multiple handlers**: Slack for warnings, PagerDuty for critical
4. **Monitor the monitor**: Track pipeline health alongside data quality
5. **Test with historical data**: Replay old data to validate alerting

## Next Steps

- [AutoML Tuning](/docs/guides/automl) - Optimize thresholds automatically
- [Root Cause Analysis](/docs/guides/root-cause) - Understand why quality drops
- [API Reference](/docs/guides/realtime) - Full API documentation
