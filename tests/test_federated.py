"""Tests for federated analysis module."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from clean.federated import (
    AggregationType,
    FederatedAnalyzer,
    FederatedQualityReport,
    LocalNode,
    LocalStatistics,
    PrivacyConfig,
    PrivacyLevel,
    SecureAggregator,
    create_federated_analyzer,
    federated_analyze,
)


class TestPrivacyConfig:
    """Tests for PrivacyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PrivacyConfig()

        assert config.privacy_level == PrivacyLevel.MEDIUM
        assert config.epsilon > 0
        assert config.min_samples_per_node > 0

    def test_custom_values(self):
        """Test custom configuration."""
        config = PrivacyConfig(
            privacy_level=PrivacyLevel.HIGH,
            epsilon=0.5,
        )

        assert config.privacy_level == PrivacyLevel.HIGH
        assert config.epsilon == 0.5


class TestLocalStatistics:
    """Tests for LocalStatistics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LocalStatistics(
            node_id="test_node",
            n_samples=1000,
            n_features=10,
            n_classes=2,
            feature_stats={"f1": {"mean": 0.5}},
            label_distribution={"0": 500, "1": 500},
            quality_indicators={"completeness": 0.95},
        )

        result = stats.to_dict()

        assert result["node_id"] == "test_node"
        assert result["n_samples"] == 1000
        assert "timestamp" in result


class TestFederatedQualityReport:
    """Tests for FederatedQualityReport dataclass."""

    def test_summary(self):
        """Test summary generation."""
        report = FederatedQualityReport(
            timestamp=datetime.now(),
            n_nodes=3,
            total_samples=3000,
            privacy_level=PrivacyLevel.MEDIUM,
            aggregated_quality_score=85.0,
            per_node_quality={"node1": 80.0, "node2": 90.0},
            global_statistics={},
            detected_issues={"label_errors": 100},
            feature_quality={},
            recommendations=["Fix labels"],
            privacy_guarantees={"epsilon": 1.0},
        )

        summary = report.summary()

        assert "Federated" in summary
        assert "3" in summary  # n_nodes
        assert "3,000" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = FederatedQualityReport(
            timestamp=datetime.now(),
            n_nodes=2,
            total_samples=2000,
            privacy_level=PrivacyLevel.LOW,
            aggregated_quality_score=80.0,
            per_node_quality={},
            global_statistics={},
            detected_issues={},
            feature_quality={},
            recommendations=[],
            privacy_guarantees={},
        )

        result = report.to_dict()

        assert result["n_nodes"] == 2
        assert result["total_samples"] == 2000


class TestLocalNode:
    """Tests for LocalNode class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "label": np.random.choice([0, 1], 200),
        })

    def test_init(self, sample_data):
        """Test node initialization."""
        node = LocalNode(data=sample_data, label_column="label")

        assert node.data is not None
        assert node.label_column == "label"

    def test_compute_local_statistics(self, sample_data):
        """Test computing local statistics."""
        node = LocalNode(data=sample_data, label_column="label")

        stats = node.compute_local_statistics()

        assert isinstance(stats, LocalStatistics)
        assert stats.n_samples == 200
        assert stats.n_features == 2  # feature1 and feature2

    def test_node_id_generated(self, sample_data):
        """Test that node ID is generated."""
        node = LocalNode(data=sample_data, label_column="label")

        assert node.node_id is not None
        assert len(node.node_id) > 0

    def test_custom_node_id(self, sample_data):
        """Test custom node ID."""
        node = LocalNode(data=sample_data, label_column="label", node_id="custom_id")

        assert node.node_id == "custom_id"


class TestSecureAggregator:
    """Tests for SecureAggregator class."""

    def test_init(self):
        """Test aggregator initialization."""
        config = PrivacyConfig()
        aggregator = SecureAggregator(config=config)

        assert aggregator is not None

    def test_aggregate_sums(self):
        """Test aggregating sums."""
        config = PrivacyConfig(privacy_level=PrivacyLevel.LOW)  # Low privacy for exact sum
        aggregator = SecureAggregator(config=config)

        values = [100, 200, 150]
        result = aggregator.aggregate_sums(values)

        # With LOW privacy, should be close to exact sum
        assert 440 <= result <= 460

    def test_aggregate_means(self):
        """Test aggregating means."""
        config = PrivacyConfig()
        aggregator = SecureAggregator(config=config)

        values = [80.0, 90.0, 85.0]
        weights = [100, 200, 100]
        result = aggregator.aggregate_means(values, weights)

        # Weighted mean
        assert 80 <= result <= 90


class TestFederatedAnalyzer:
    """Tests for FederatedAnalyzer class."""

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes."""
        np.random.seed(42)

        data1 = pd.DataFrame({
            "feature": np.random.randn(200),
            "label": np.random.choice([0, 1], 200),
        })
        data2 = pd.DataFrame({
            "feature": np.random.randn(200),
            "label": np.random.choice([0, 1], 200),
        })

        return [
            LocalNode(data=data1, label_column="label", node_id="node1"),
            LocalNode(data=data2, label_column="label", node_id="node2"),
        ]

    def test_init_with_nodes(self, sample_nodes):
        """Test initialization with nodes."""
        analyzer = FederatedAnalyzer(nodes=sample_nodes)

        assert len(analyzer.nodes) == 2

    def test_add_node(self, sample_nodes):
        """Test adding a node."""
        analyzer = FederatedAnalyzer(nodes=[sample_nodes[0]])
        analyzer.add_node(sample_nodes[1])

        assert len(analyzer.nodes) == 2

    def test_analyze_returns_report(self, sample_nodes):
        """Test that analyze returns a report."""
        analyzer = FederatedAnalyzer(nodes=sample_nodes)

        report = analyzer.analyze()

        assert isinstance(report, FederatedQualityReport)
        assert report.n_nodes == 2
        assert report.total_samples == 400

    def test_analyze_aggregates_scores(self, sample_nodes):
        """Test that scores are aggregated."""
        analyzer = FederatedAnalyzer(nodes=sample_nodes)

        report = analyzer.analyze()

        assert report.aggregated_quality_score > 0
        # Allow slight overflow due to privacy noise
        assert report.aggregated_quality_score <= 110


class TestCreateFederatedAnalyzer:
    """Tests for create_federated_analyzer function."""

    def test_create_basic(self):
        """Test creating analyzer."""
        np.random.seed(42)

        datasets = [
            pd.DataFrame({
                "feature": np.random.randn(200),
                "label": np.random.choice([0, 1], 200),
            }),
            pd.DataFrame({
                "feature": np.random.randn(200),
                "label": np.random.choice([0, 1], 200),
            }),
        ]

        analyzer = create_federated_analyzer(datasets=datasets, label_column="label")

        assert len(analyzer.nodes) == 2


class TestFederatedAnalyze:
    """Tests for federated_analyze convenience function."""

    def test_federated_analyze_basic(self):
        """Test basic federated analysis."""
        np.random.seed(42)

        datasets = [
            pd.DataFrame({
                "feature": np.random.randn(200),
                "label": np.random.choice([0, 1], 200),
            }),
            pd.DataFrame({
                "feature": np.random.randn(200),
                "label": np.random.choice([0, 1], 200),
            }),
        ]

        report = federated_analyze(datasets=datasets, label_column="label")

        assert isinstance(report, FederatedQualityReport)
        assert report.n_nodes == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_nodes_list(self):
        """Test with empty nodes list."""
        analyzer = FederatedAnalyzer(nodes=[])

        with pytest.raises(Exception):
            analyzer.analyze()

    def test_single_node(self):
        """Test with single node."""
        np.random.seed(42)
        data = pd.DataFrame({
            "feature": np.random.randn(200),
            "label": np.random.choice([0, 1], 200),
        })

        analyzer = FederatedAnalyzer(nodes=[
            LocalNode(data=data, label_column="label")
        ])

        report = analyzer.analyze()

        assert report.n_nodes == 1
        assert report.total_samples == 200

    def test_different_feature_counts(self):
        """Test nodes with different feature counts."""
        np.random.seed(42)

        data1 = pd.DataFrame({
            "feature1": np.random.randn(200),
            "label": np.random.choice([0, 1], 200),
        })
        data2 = pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),  # Extra feature
            "label": np.random.choice([0, 1], 200),
        })

        analyzer = FederatedAnalyzer(nodes=[
            LocalNode(data=data1, label_column="label"),
            LocalNode(data=data2, label_column="label"),
        ])

        # Should still work
        report = analyzer.analyze()
        assert report.n_nodes == 2
