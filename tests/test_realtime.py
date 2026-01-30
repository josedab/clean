"""Tests for real-time streaming pipeline."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from clean.realtime import (
    AlertSeverity,
    MemorySource,
    PipelineConfig,
    PrometheusExporter,
    QualityAlert,
    RealtimeMetrics,
    RealtimePipeline,
    StreamBackend,
    create_pipeline,
)


class TestRealtimeMetrics:
    """Tests for RealtimeMetrics."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = RealtimeMetrics()
        assert metrics.total_processed == 0
        assert metrics.total_issues == 0
        assert metrics.outlier_rate == 0.0
        assert metrics.duplicate_rate == 0.0

    def test_record_metrics(self):
        """Test recording metrics."""
        metrics = RealtimeMetrics(window_size=100)

        metrics.record(is_outlier=True, quality_score=70.0)
        assert metrics.total_processed == 1
        assert metrics.total_issues == 1
        assert metrics.outlier_rate == 1.0

        metrics.record(is_outlier=False, quality_score=90.0)
        assert metrics.total_processed == 2
        assert metrics.outlier_rate == 0.5

    def test_rolling_window(self):
        """Test rolling window calculations."""
        metrics = RealtimeMetrics(window_size=10)

        # Fill window
        for _ in range(10):
            metrics.record(is_outlier=False)

        # Add outliers
        for _ in range(5):
            metrics.record(is_outlier=True)

        # Window should show last 10 samples (5 outliers)
        assert metrics.outlier_rate == 0.5

    def test_get_summary(self):
        """Test summary generation."""
        metrics = RealtimeMetrics()
        metrics.record(is_outlier=True, is_duplicate=True, quality_score=50.0)

        summary = metrics.get_summary()
        assert "total_processed" in summary
        assert "outlier_rate" in summary
        assert "throughput_per_sec" in summary


class TestMemorySource:
    """Tests for MemorySource."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        source = MemorySource(data=[{"a": 1}])
        await source.connect()
        assert source._connected
        await source.disconnect()
        assert not source._connected

    @pytest.mark.asyncio
    async def test_consume_data(self):
        """Test consuming data."""
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        source = MemorySource(data=data)
        await source.connect()

        received = []
        async def collect():
            async for msg in source.consume():
                received.append(msg["data"])
                if len(received) >= 3:
                    await source.disconnect()
                    break

        await asyncio.wait_for(collect(), timeout=1.0)
        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_push_data(self):
        """Test pushing data dynamically."""
        source = MemorySource()
        await source.connect()

        # Push data
        await source.push({"value": 42})

        # Consume
        async for msg in source.consume():
            assert msg["data"]["value"] == 42
            await source.disconnect()
            break


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        assert config.label_column == "label"
        assert config.batch_size == 100
        assert config.outlier_threshold == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            label_column="target",
            batch_size=50,
            outlier_threshold=0.2,
        )
        assert config.label_column == "target"
        assert config.batch_size == 50


class TestRealtimePipeline:
    """Tests for RealtimePipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        source = MemorySource()
        pipeline = RealtimePipeline(source=source)
        assert not pipeline.is_running
        assert pipeline.config.label_column == "label"

    @pytest.mark.asyncio
    async def test_add_alert_handler(self):
        """Test adding alert handlers."""
        source = MemorySource()
        pipeline = RealtimePipeline(source=source)

        handler = MagicMock()
        pipeline.add_alert_handler(handler)
        assert handler in pipeline._alert_handlers

        pipeline.remove_alert_handler(handler)
        assert handler not in pipeline._alert_handlers

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing."""
        data = [
            {"feature": i, "label": i % 2}
            for i in range(10)
        ]
        source = MemorySource(data=data)
        config = PipelineConfig(batch_size=5)
        pipeline = RealtimePipeline(source=source, config=config)

        await source.connect()

        # Process messages manually
        for item in data:
            await pipeline._process_message({"data": item})

        metrics = pipeline.get_metrics()
        assert metrics["total_processed"] > 0

    @pytest.mark.asyncio
    async def test_outlier_detection(self):
        """Test outlier detection in pipeline."""
        source = MemorySource()
        pipeline = RealtimePipeline(source=source)

        # Create batch with outlier
        batch = pd.DataFrame({
            "feature": [1, 2, 3, 4, 100],  # 100 is outlier
            "label": [0, 0, 0, 0, 0],
        })

        # Test outlier detection
        for idx, row in batch.iterrows():
            is_outlier = pipeline._detect_outlier(row, batch)
            if row["feature"] == 100:
                assert is_outlier

    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test duplicate detection."""
        source = MemorySource()
        pipeline = RealtimePipeline(source=source)

        row1 = pd.Series({"a": 1, "b": 2})
        row2 = pd.Series({"a": 1, "b": 2})

        is_dup1 = pipeline._detect_duplicate(row1)
        is_dup2 = pipeline._detect_duplicate(row2)

        assert not is_dup1  # First occurrence
        assert is_dup2  # Duplicate

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self):
        """Test quality score calculation."""
        source = MemorySource()
        pipeline = RealtimePipeline(source=source)

        # Perfect sample
        score1 = pipeline._calculate_quality_score(
            is_outlier=False, is_duplicate=False, label_error_score=0.0
        )
        assert score1 == 100.0

        # Outlier sample
        score2 = pipeline._calculate_quality_score(
            is_outlier=True, is_duplicate=False, label_error_score=0.0
        )
        assert score2 == 70.0

        # Multiple issues
        score3 = pipeline._calculate_quality_score(
            is_outlier=True, is_duplicate=True, label_error_score=0.5
        )
        assert score3 < 50.0


class TestQualityAlert:
    """Tests for QualityAlert."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = QualityAlert(
            timestamp=1234567890.0,
            severity=AlertSeverity.WARNING,
            issue_type="outlier",
            message="Test alert",
            metric_name="outlier_rate",
            metric_value=0.15,
            threshold=0.10,
            sample_count=1000,
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_value == 0.15

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = QualityAlert(
            timestamp=1234567890.0,
            severity=AlertSeverity.CRITICAL,
            issue_type="quality",
            message="Quality below threshold",
            metric_name="quality_score",
            metric_value=60.0,
            threshold=70.0,
            sample_count=500,
        )
        data = alert.to_dict()
        assert data["severity"] == "critical"
        assert data["metric_value"] == 60.0


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    @pytest.mark.asyncio
    async def test_export_metrics(self):
        """Test Prometheus metrics export."""
        source = MemorySource()
        pipeline = RealtimePipeline(source=source)

        # Record some metrics
        pipeline._metrics.record(is_outlier=True, quality_score=80.0)
        pipeline._metrics.record(is_duplicate=True, quality_score=90.0)

        exporter = PrometheusExporter(pipeline)
        output = exporter.get_metrics()

        assert "clean_total_processed" in output
        assert "clean_outlier_rate" in output
        assert "clean_quality_score" in output


class TestCreatePipeline:
    """Tests for pipeline factory function."""

    @pytest.mark.asyncio
    async def test_create_memory_pipeline(self):
        """Test creating memory pipeline."""
        pipeline = await create_pipeline(
            backend=StreamBackend.MEMORY,
            data=[{"a": 1}],
        )
        assert isinstance(pipeline, RealtimePipeline)
        assert isinstance(pipeline.source, MemorySource)

    @pytest.mark.asyncio
    async def test_create_pipeline_string_backend(self):
        """Test creating pipeline with string backend."""
        pipeline = await create_pipeline(backend="memory")
        assert isinstance(pipeline, RealtimePipeline)

    @pytest.mark.asyncio
    async def test_unsupported_backend(self):
        """Test unsupported backend error."""
        with pytest.raises(ValueError):
            await create_pipeline(backend="unsupported")


class TestIntegration:
    """Integration tests for the streaming pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test full pipeline flow with alerts."""
        # Create data with issues
        data = []
        for i in range(50):
            data.append({"feature": i, "label": 0})
        # Add outliers
        for i in range(10):
            data.append({"feature": 1000 + i, "label": 0})

        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        source = MemorySource(data=data)
        config = PipelineConfig(
            batch_size=20,
            outlier_threshold=0.05,
            alert_cooldown_seconds=0,  # Disable cooldown for testing
        )
        pipeline = RealtimePipeline(source=source, config=config)
        pipeline.add_alert_handler(alert_handler)

        # Process all data
        await source.connect()
        for item in data:
            await pipeline._process_message({"data": item})
        await source.disconnect()

        metrics = pipeline.get_metrics()
        assert metrics["total_processed"] == 60
