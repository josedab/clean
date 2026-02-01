"""Real-time streaming pipeline for continuous data quality monitoring.

This module provides Kafka/Pulsar/Redis Streams integration for real-time
data quality analysis with sub-second latency.

Example:
    >>> from clean.realtime import RealtimePipeline, KafkaSource
    >>>
    >>> pipeline = RealtimePipeline(
    ...     source=KafkaSource(bootstrap_servers="localhost:9092", topic="data"),
    ...     label_column="label",
    ... )
    >>> pipeline.add_alert_handler(lambda alert: print(f"Alert: {alert}"))
    >>> await pipeline.start()
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.core.types import IssueType

if TYPE_CHECKING:
    pass


class StreamBackend(Enum):
    """Supported streaming backends."""

    KAFKA = "kafka"
    PULSAR = "pulsar"
    REDIS = "redis"
    MEMORY = "memory"  # For testing


class AlertSeverity(Enum):
    """Severity levels for quality alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """Alert generated when quality threshold is breached."""

    timestamp: float
    severity: AlertSeverity
    issue_type: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "issue_type": self.issue_type,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }


@dataclass
class RealtimeMetrics:
    """Real-time quality metrics with rolling window statistics."""

    window_size: int = 1000
    _outlier_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    _duplicate_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    _label_error_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    _quality_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    _timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_processed: int = 0
    total_issues: int = 0
    start_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Initialize with correct window size."""
        self._outlier_counts = deque(maxlen=self.window_size)
        self._duplicate_counts = deque(maxlen=self.window_size)
        self._label_error_scores = deque(maxlen=self.window_size)
        self._quality_scores = deque(maxlen=self.window_size)
        self._timestamps = deque(maxlen=self.window_size)

    def record(
        self,
        is_outlier: bool = False,
        is_duplicate: bool = False,
        label_error_score: float = 0.0,
        quality_score: float = 100.0,
    ) -> None:
        """Record metrics for a single sample."""
        self._outlier_counts.append(1 if is_outlier else 0)
        self._duplicate_counts.append(1 if is_duplicate else 0)
        self._label_error_scores.append(label_error_score)
        self._quality_scores.append(quality_score)
        self._timestamps.append(time.time())
        self.total_processed += 1
        if is_outlier or is_duplicate or label_error_score > 0.5:
            self.total_issues += 1

    @property
    def outlier_rate(self) -> float:
        """Rolling outlier rate."""
        if not self._outlier_counts:
            return 0.0
        return sum(self._outlier_counts) / len(self._outlier_counts)

    @property
    def duplicate_rate(self) -> float:
        """Rolling duplicate rate."""
        if not self._duplicate_counts:
            return 0.0
        return sum(self._duplicate_counts) / len(self._duplicate_counts)

    @property
    def avg_label_error_score(self) -> float:
        """Average label error score in window."""
        if not self._label_error_scores:
            return 0.0
        return float(np.mean(list(self._label_error_scores)))

    @property
    def avg_quality_score(self) -> float:
        """Average quality score in window."""
        if not self._quality_scores:
            return 100.0
        return float(np.mean(list(self._quality_scores)))

    @property
    def throughput(self) -> float:
        """Samples processed per second."""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.total_processed / elapsed

    def get_summary(self) -> dict[str, Any]:
        """Get current metrics summary."""
        return {
            "total_processed": self.total_processed,
            "total_issues": self.total_issues,
            "outlier_rate": self.outlier_rate,
            "duplicate_rate": self.duplicate_rate,
            "avg_label_error_score": self.avg_label_error_score,
            "avg_quality_score": self.avg_quality_score,
            "throughput_per_sec": self.throughput,
            "window_size": len(self._quality_scores),
        }


class StreamSource(ABC):
    """Abstract base class for streaming data sources."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def consume(self) -> AsyncIterator[dict[str, Any]]:
        """Consume messages from the source."""
        pass

    @abstractmethod
    async def commit(self, offset: Any) -> None:
        """Commit the processed offset."""
        pass


class KafkaSource(StreamSource):
    """Kafka streaming source.

    Requires: pip install aiokafka
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        topic: str,
        group_id: str = "clean-quality-monitor",
        auto_offset_reset: str = "latest",
        **kwargs: Any,
    ):
        """Initialize Kafka source.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to consume from
            group_id: Consumer group ID
            auto_offset_reset: Offset reset policy
            **kwargs: Additional consumer config
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.kwargs = kwargs
        self._consumer = None

    async def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka support. "
                "Install with: pip install aiokafka"
            )

        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            **self.kwargs,
        )
        await self._consumer.start()

    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

    async def consume(self):
        """Consume messages from Kafka."""
        if not self._consumer:
            raise RuntimeError("Not connected. Call connect() first.")

        async for msg in self._consumer:
            yield {
                "data": msg.value,
                "offset": msg.offset,
                "partition": msg.partition,
                "timestamp": msg.timestamp,
            }

    async def commit(self, offset: Any = None) -> None:
        """Commit offsets."""
        if self._consumer:
            await self._consumer.commit()


class PulsarSource(StreamSource):
    """Apache Pulsar streaming source.

    Requires: pip install pulsar-client
    """

    def __init__(
        self,
        service_url: str,
        topic: str,
        subscription: str = "clean-quality-monitor",
        **kwargs: Any,
    ):
        """Initialize Pulsar source.

        Args:
            service_url: Pulsar service URL
            topic: Topic to consume from
            subscription: Subscription name
            **kwargs: Additional consumer config
        """
        self.service_url = service_url
        self.topic = topic
        self.subscription = subscription
        self.kwargs = kwargs
        self._client = None
        self._consumer = None

    async def connect(self) -> None:
        """Connect to Pulsar."""
        try:
            import pulsar
        except ImportError:
            raise ImportError(
                "pulsar-client is required for Pulsar support. "
                "Install with: pip install pulsar-client"
            )

        self._client = pulsar.Client(self.service_url)
        self._consumer = self._client.subscribe(
            self.topic,
            self.subscription,
            **self.kwargs,
        )

    async def disconnect(self) -> None:
        """Disconnect from Pulsar."""
        if self._consumer:
            self._consumer.close()
            self._consumer = None
        if self._client:
            self._client.close()
            self._client = None

    async def consume(self):
        """Consume messages from Pulsar."""
        if not self._consumer:
            raise RuntimeError("Not connected. Call connect() first.")

        while True:
            try:
                msg = self._consumer.receive(timeout_millis=100)
                data = json.loads(msg.data().decode("utf-8"))
                yield {
                    "data": data,
                    "message_id": msg.message_id(),
                    "timestamp": msg.publish_timestamp(),
                }
            except Exception:
                await asyncio.sleep(0.01)

    async def commit(self, offset: Any = None) -> None:
        """Acknowledge message."""
        if self._consumer and offset:
            self._consumer.acknowledge(offset)


class RedisSource(StreamSource):
    """Redis Streams source.

    Requires: pip install redis
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        stream: str = "clean-data-stream",
        group: str = "clean-quality-monitor",
        consumer: str = "consumer-1",
        **kwargs: Any,
    ):
        """Initialize Redis Streams source.

        Args:
            url: Redis connection URL
            stream: Stream name
            group: Consumer group name
            consumer: Consumer name
            **kwargs: Additional connection config
        """
        self.url = url
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.kwargs = kwargs
        self._redis = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for Redis Streams support. "
                "Install with: pip install redis"
            )

        self._redis = redis.from_url(self.url, **self.kwargs)

        # Create consumer group if it doesn't exist
        try:
            await self._redis.xgroup_create(
                self.stream, self.group, id="0", mkstream=True
            )
        except Exception:
            pass  # Group already exists

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def consume(self):
        """Consume messages from Redis Streams."""
        if not self._redis:
            raise RuntimeError("Not connected. Call connect() first.")

        while True:
            try:
                messages = await self._redis.xreadgroup(
                    self.group,
                    self.consumer,
                    {self.stream: ">"},
                    count=10,
                    block=100,
                )

                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        # Decode bytes to string
                        decoded_data = {
                            k.decode() if isinstance(k, bytes) else k: (
                                v.decode() if isinstance(v, bytes) else v
                            )
                            for k, v in data.items()
                        }
                        yield {
                            "data": decoded_data,
                            "message_id": message_id,
                            "stream": stream_name,
                        }
            except Exception:
                await asyncio.sleep(0.01)

    async def commit(self, offset: Any = None) -> None:
        """Acknowledge message."""
        if self._redis and offset:
            await self._redis.xack(self.stream, self.group, offset)


class MemorySource(StreamSource):
    """In-memory source for testing."""

    def __init__(self, data: list[dict[str, Any]] | None = None):
        """Initialize memory source.

        Args:
            data: List of records to stream
        """
        self.data = data or []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._connected = False

    async def connect(self) -> None:
        """Initialize the queue."""
        self._connected = True
        for item in self.data:
            await self._queue.put(item)

    async def disconnect(self) -> None:
        """Disconnect."""
        self._connected = False

    async def consume(self):
        """Consume from queue."""
        while self._connected:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield {"data": item, "offset": None}
            except asyncio.TimeoutError:
                continue

    async def commit(self, offset: Any = None) -> None:
        """No-op for memory source."""
        pass

    async def push(self, data: dict[str, Any]) -> None:
        """Push data to the queue."""
        await self._queue.put(data)


@dataclass
class PipelineConfig:
    """Configuration for the real-time pipeline."""

    label_column: str = "label"
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    outlier_threshold: float = 0.1
    duplicate_threshold: float = 0.05
    quality_threshold: float = 70.0
    alert_cooldown_seconds: float = 60.0
    enable_deduplication: bool = True
    dedup_window_size: int = 10000


class RealtimePipeline:
    """Real-time data quality monitoring pipeline.

    Connects to streaming sources (Kafka, Pulsar, Redis) and performs
    continuous quality analysis with alerting capabilities.
    """

    def __init__(
        self,
        source: StreamSource,
        config: PipelineConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize the pipeline.

        Args:
            source: Streaming data source
            config: Pipeline configuration
            **kwargs: Override config options
        """
        self.source = source
        self.config = config or PipelineConfig(**kwargs)

        self._metrics = RealtimeMetrics(window_size=self.config.batch_size * 10)
        self._alert_handlers: list[Callable[[QualityAlert], None]] = []
        self._last_alerts: dict[str, float] = {}
        self._running = False
        self._dedup_hashes: deque = deque(maxlen=self.config.dedup_window_size)

        # Batch accumulator
        self._batch: list[dict[str, Any]] = []
        self._batch_start_time: float = 0

    def add_alert_handler(
        self, handler: Callable[[QualityAlert], None]
    ) -> None:
        """Add an alert handler.

        Args:
            handler: Function to call when alert is triggered
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(
        self, handler: Callable[[QualityAlert], None]
    ) -> None:
        """Remove an alert handler."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    async def start(self) -> None:
        """Start the pipeline."""
        self._running = True
        await self.source.connect()

        try:
            async for message in self.source.consume():
                if not self._running:
                    break

                await self._process_message(message)
                await self.source.commit(message.get("offset"))

        finally:
            await self.source.disconnect()

    async def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False

    async def _process_message(self, message: dict[str, Any]) -> None:
        """Process a single message."""
        data = message.get("data", {})

        # Add to batch
        if not self._batch:
            self._batch_start_time = time.time()
        self._batch.append(data)

        # Process batch if full or timeout
        batch_age_ms = (time.time() - self._batch_start_time) * 1000
        if (
            len(self._batch) >= self.config.batch_size
            or batch_age_ms >= self.config.batch_timeout_ms
        ):
            await self._process_batch()

    async def _process_batch(self) -> None:
        """Process accumulated batch."""
        if not self._batch:
            return

        batch = self._batch
        self._batch = []

        try:
            df = pd.DataFrame(batch)
        except Exception:
            return

        # Analyze batch
        for idx, row in df.iterrows():
            is_outlier = self._detect_outlier(row, df)
            is_duplicate = self._detect_duplicate(row)
            label_error_score = self._estimate_label_error(row, df)
            quality_score = self._calculate_quality_score(
                is_outlier, is_duplicate, label_error_score
            )

            self._metrics.record(
                is_outlier=is_outlier,
                is_duplicate=is_duplicate,
                label_error_score=label_error_score,
                quality_score=quality_score,
            )

        # Check thresholds and generate alerts
        await self._check_alerts()

    def _detect_outlier(
        self, row: pd.Series, batch: pd.DataFrame
    ) -> bool:
        """Detect if row is an outlier using IQR."""
        numeric_cols = batch.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.config.label_column]

        if not numeric_cols:
            return False

        for col in numeric_cols:
            if col not in row.index:
                continue

            value = row[col]
            if pd.isna(value):
                continue

            q1 = batch[col].quantile(0.25)
            q3 = batch[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if value < lower or value > upper:
                return True

        return False

    def _detect_duplicate(self, row: pd.Series) -> bool:
        """Detect if row is a duplicate."""
        if not self.config.enable_deduplication:
            return False

        # Create hash of row
        row_hash = hash(tuple(sorted(row.items())))

        if row_hash in self._dedup_hashes:
            return True

        self._dedup_hashes.append(row_hash)
        return False

    def _estimate_label_error(
        self, row: pd.Series, batch: pd.DataFrame
    ) -> float:
        """Estimate probability of label error.

        Uses a simple heuristic based on feature distance to class centroids.
        """
        if self.config.label_column not in row.index:
            return 0.0

        label = row.get(self.config.label_column)
        if pd.isna(label):
            return 0.5  # Missing label is suspicious

        # Get numeric features
        numeric_cols = batch.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.config.label_column]

        if not numeric_cols:
            return 0.0

        # Calculate class centroids
        if self.config.label_column not in batch.columns:
            return 0.0

        class_data = batch[batch[self.config.label_column] == label][numeric_cols]
        if len(class_data) < 2:
            return 0.0

        centroid = class_data.mean()

        # Calculate distance to centroid
        row_features = row[numeric_cols]
        if row_features.isna().any():
            return 0.0

        distance = np.sqrt(((row_features - centroid) ** 2).sum())

        # Normalize by class std
        class_std = class_data.std().mean()
        if class_std == 0:
            return 0.0

        normalized_distance = distance / (class_std + 1e-10)

        # Convert to probability (samples far from centroid are suspicious)
        # Use sigmoid-like function
        return float(1 / (1 + np.exp(-0.5 * (normalized_distance - 2))))

    def _calculate_quality_score(
        self,
        is_outlier: bool,
        is_duplicate: bool,
        label_error_score: float,
    ) -> float:
        """Calculate overall quality score for a sample."""
        score = 100.0

        if is_outlier:
            score -= 30.0
        if is_duplicate:
            score -= 20.0

        score -= label_error_score * 50.0

        return max(0.0, score)

    async def _check_alerts(self) -> None:
        """Check metrics against thresholds and generate alerts."""
        now = time.time()

        # Check outlier rate
        if self._metrics.outlier_rate > self.config.outlier_threshold:
            await self._trigger_alert(
                severity=AlertSeverity.WARNING,
                issue_type=IssueType.OUTLIER.value,
                message=f"Outlier rate ({self._metrics.outlier_rate:.1%}) exceeds threshold",
                metric_name="outlier_rate",
                metric_value=self._metrics.outlier_rate,
                threshold=self.config.outlier_threshold,
            )

        # Check duplicate rate
        if self._metrics.duplicate_rate > self.config.duplicate_threshold:
            await self._trigger_alert(
                severity=AlertSeverity.WARNING,
                issue_type=IssueType.DUPLICATE.value,
                message=f"Duplicate rate ({self._metrics.duplicate_rate:.1%}) exceeds threshold",
                metric_name="duplicate_rate",
                metric_value=self._metrics.duplicate_rate,
                threshold=self.config.duplicate_threshold,
            )

        # Check quality score
        if self._metrics.avg_quality_score < self.config.quality_threshold:
            await self._trigger_alert(
                severity=AlertSeverity.CRITICAL,
                issue_type="overall_quality",
                message=f"Average quality score ({self._metrics.avg_quality_score:.1f}) below threshold",
                metric_name="avg_quality_score",
                metric_value=self._metrics.avg_quality_score,
                threshold=self.config.quality_threshold,
            )

    async def _trigger_alert(
        self,
        severity: AlertSeverity,
        issue_type: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Trigger an alert if not in cooldown."""
        now = time.time()

        # Check cooldown
        last_alert = self._last_alerts.get(issue_type, 0)
        if now - last_alert < self.config.alert_cooldown_seconds:
            return

        self._last_alerts[issue_type] = now

        alert = QualityAlert(
            timestamp=now,
            severity=severity,
            issue_type=issue_type,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            sample_count=self._metrics.total_processed,
            metadata=self._metrics.get_summary(),
        )

        # Call handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception:
                pass  # Don't let handler errors stop processing

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return self._metrics.get_summary()

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running


class AlertWebhook:
    """Webhook alert handler for external integrations."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        """Initialize webhook handler.

        Args:
            url: Webhook URL
            headers: HTTP headers
            timeout: Request timeout
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    async def __call__(self, alert: QualityAlert) -> None:
        """Send alert to webhook."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for webhook support. "
                "Install with: pip install aiohttp"
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                json=alert.to_dict(),
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()


class SlackAlertHandler:
    """Slack alert handler."""

    def __init__(self, webhook_url: str):
        """Initialize Slack handler.

        Args:
            webhook_url: Slack incoming webhook URL
        """
        self.webhook_url = webhook_url

    async def __call__(self, alert: QualityAlert) -> None:
        """Send alert to Slack."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for Slack alerts. "
                "Install with: pip install aiohttp"
            )

        emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }

        payload = {
            "text": f"{emoji.get(alert.severity, ':bell:')} *Data Quality Alert*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🔔 {alert.severity.value.upper()}: {alert.issue_type}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.message,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Metric:* {alert.metric_name}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Value:* {alert.metric_value:.3f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Threshold:* {alert.threshold:.3f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Samples:* {alert.sample_count:,}",
                        },
                    ],
                },
            ],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as response:
                response.raise_for_status()


class PrometheusExporter:
    """Export metrics to Prometheus format."""

    def __init__(self, pipeline: RealtimePipeline):
        """Initialize exporter.

        Args:
            pipeline: Pipeline to export metrics from
        """
        self.pipeline = pipeline

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        metrics = self.pipeline.get_metrics()

        lines = [
            "# HELP clean_total_processed Total samples processed",
            "# TYPE clean_total_processed counter",
            f"clean_total_processed {metrics['total_processed']}",
            "",
            "# HELP clean_total_issues Total issues detected",
            "# TYPE clean_total_issues counter",
            f"clean_total_issues {metrics['total_issues']}",
            "",
            "# HELP clean_outlier_rate Rolling outlier rate",
            "# TYPE clean_outlier_rate gauge",
            f"clean_outlier_rate {metrics['outlier_rate']:.4f}",
            "",
            "# HELP clean_duplicate_rate Rolling duplicate rate",
            "# TYPE clean_duplicate_rate gauge",
            f"clean_duplicate_rate {metrics['duplicate_rate']:.4f}",
            "",
            "# HELP clean_quality_score Average quality score",
            "# TYPE clean_quality_score gauge",
            f"clean_quality_score {metrics['avg_quality_score']:.2f}",
            "",
            "# HELP clean_throughput Samples per second",
            "# TYPE clean_throughput gauge",
            f"clean_throughput {metrics['throughput_per_sec']:.2f}",
        ]

        return "\n".join(lines)


async def create_pipeline(
    backend: StreamBackend | str,
    config: PipelineConfig | None = None,
    **source_kwargs: Any,
) -> RealtimePipeline:
    """Create a real-time pipeline with the specified backend.

    Args:
        backend: Streaming backend to use
        config: Pipeline configuration
        **source_kwargs: Arguments for the source

    Returns:
        Configured RealtimePipeline
    """
    if isinstance(backend, str):
        backend = StreamBackend(backend)

    if backend == StreamBackend.KAFKA:
        source = KafkaSource(**source_kwargs)
    elif backend == StreamBackend.PULSAR:
        source = PulsarSource(**source_kwargs)
    elif backend == StreamBackend.REDIS:
        source = RedisSource(**source_kwargs)
    elif backend == StreamBackend.MEMORY:
        source = MemorySource(**source_kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return RealtimePipeline(source=source, config=config)
