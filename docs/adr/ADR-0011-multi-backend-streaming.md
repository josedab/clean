# ADR-0011: Multi-Backend Streaming Architecture

## Status

Accepted

## Context

Enterprise deployments require real-time data quality monitoring on streaming data:

- **Kafka**: Most common in data engineering (LinkedIn, Uber, Netflix)
- **Apache Pulsar**: Growing adoption, multi-tenancy built-in
- **Redis Streams**: Lightweight, good for smaller scale
- **Cloud services**: AWS Kinesis, GCP Pub/Sub, Azure Event Hubs

Locking into a single streaming platform would limit adoption. We needed:

1. Common interface across backends
2. Quality metrics with sliding windows
3. Alerting when quality degrades
4. Minimal latency impact on data flow
5. Easy testing without running Kafka

## Decision

We implemented an **abstracted streaming architecture** with pluggable backends.

```python
# realtime.py
class StreamBackend(Enum):
    """Supported streaming backends."""
    KAFKA = "kafka"
    PULSAR = "pulsar"
    REDIS = "redis"
    MEMORY = "memory"  # For testing

class StreamSource(ABC):
    """Abstract base for stream sources."""
    @abstractmethod
    async def connect(self) -> None: ...
    
    @abstractmethod
    async def consume(self) -> AsyncIterator[dict[str, Any]]: ...
    
    @abstractmethod
    async def close(self) -> None: ...

class KafkaSource(StreamSource):
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = None):
        self.servers = bootstrap_servers
        self.topic = topic
    
    async def connect(self):
        from aiokafka import AIOKafkaConsumer
        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.servers,
        )
        await self._consumer.start()
    
    async def consume(self) -> AsyncIterator[dict]:
        async for msg in self._consumer:
            yield json.loads(msg.value)

class PulsarSource(StreamSource):
    def __init__(self, service_url: str, topic: str): ...

class RedisSource(StreamSource):
    def __init__(self, url: str, stream_name: str): ...

class MemorySource(StreamSource):
    """In-memory source for testing."""
    def __init__(self, data: list[dict]):
        self._data = data
        self._index = 0
    
    async def consume(self) -> AsyncIterator[dict]:
        for item in self._data:
            yield item
            await asyncio.sleep(0.01)  # Simulate network delay
```

The `RealtimePipeline` orchestrates quality monitoring:

```python
@dataclass
class PipelineConfig:
    window_size: int = 1000           # Samples in rolling window
    quality_threshold: float = 0.8    # Alert below this score
    check_interval: float = 1.0       # Seconds between quality checks
    detectors: list[str] = field(default_factory=lambda: ["outliers", "duplicates"])

class RealtimePipeline:
    def __init__(self, source: StreamSource, config: PipelineConfig = None):
        self.source = source
        self.config = config or PipelineConfig()
        self._metrics = RealtimeMetrics(window_size=config.window_size)
        self._alert_handlers: list[Callable[[QualityAlert], None]] = []
    
    def add_alert_handler(self, handler: Callable[[QualityAlert], None]):
        """Register callback for quality alerts."""
        self._alert_handlers.append(handler)
    
    async def start(self):
        """Start consuming and analyzing."""
        await self.source.connect()
        
        async for record in self.source.consume():
            # Update rolling metrics
            self._metrics.record(record)
            
            # Check quality periodically
            if self._should_check():
                quality = self._metrics.current_quality_score()
                if quality < self.config.quality_threshold:
                    alert = QualityAlert(
                        severity=AlertSeverity.WARNING,
                        message=f"Quality dropped to {quality:.2f}",
                        metric_value=quality,
                        threshold=self.config.quality_threshold,
                    )
                    for handler in self._alert_handlers:
                        handler(alert)
```

## Consequences

### Positive

- **Backend flexibility**: Switch from Kafka to Pulsar without code changes
- **Testability**: `MemorySource` enables unit tests without infrastructure
- **Consistent interface**: Same pipeline code regardless of source
- **Async-native**: Non-blocking I/O for high throughput
- **Alert system**: Proactive quality degradation notifications

### Negative

- **Lowest common denominator**: Can't use backend-specific features easily
- **Multiple dependencies**: Each backend adds optional dependencies
- **Testing coverage**: Must test each backend integration
- **Latency overhead**: Abstraction adds small overhead

### Neutral

- **No exactly-once**: At-least-once semantics (quality metrics tolerate duplicates)
- **Single consumer**: Not designed for consumer groups (use backend features for that)

## Usage Example

```python
import asyncio
from clean.realtime import RealtimePipeline, KafkaSource, PipelineConfig

async def monitor():
    source = KafkaSource(
        bootstrap_servers="localhost:9092",
        topic="ml-training-data",
    )
    
    config = PipelineConfig(
        window_size=1000,
        quality_threshold=0.85,
    )
    
    pipeline = RealtimePipeline(source=source, config=config)
    
    # Alert to Slack, PagerDuty, etc.
    pipeline.add_alert_handler(lambda alert: send_to_slack(alert))
    
    await pipeline.start()

asyncio.run(monitor())
```

## Related Decisions

- ADR-0005 (Optional Dependencies): Streaming clients in `[streaming]` extra
- ADR-0007 (Async Streaming): Shares async patterns with batch streaming
