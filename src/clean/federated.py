"""Federated Data Quality Analysis.

Enable privacy-preserving quality analysis across distributed datasets without
centralizing data. Supports secure aggregation protocols for multi-party analysis.

Example:
    >>> from clean.federated import FederatedAnalyzer, LocalNode
    >>>
    >>> # Create local nodes (data stays on each node)
    >>> node1 = LocalNode(data=df1, label_column="label")
    >>> node2 = LocalNode(data=df2, label_column="label")
    >>>
    >>> # Create federated analyzer
    >>> analyzer = FederatedAnalyzer(nodes=[node1, node2])
    >>>
    >>> # Run privacy-preserving analysis
    >>> report = analyzer.analyze()
    >>> print(report.summary())
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy protection level."""

    LOW = "low"  # Basic aggregation only
    MEDIUM = "medium"  # Add noise to aggregates
    HIGH = "high"  # Differential privacy
    MAXIMUM = "maximum"  # Secure multi-party computation


class AggregationType(Enum):
    """Type of federated aggregation."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    HISTOGRAM = "histogram"
    VARIANCE = "variance"
    QUANTILE = "quantile"


@dataclass
class PrivacyConfig:
    """Configuration for privacy-preserving computation."""

    privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
    epsilon: float = 1.0  # Differential privacy parameter
    delta: float = 1e-5  # Differential privacy parameter
    noise_scale: float = 0.1  # Scale for Gaussian noise
    min_samples_per_node: int = 100  # Minimum samples for privacy
    secure_aggregation: bool = True


@dataclass
class LocalStatistics:
    """Statistics computed locally on a node."""

    node_id: str
    n_samples: int
    n_features: int
    n_classes: int | None
    feature_stats: dict[str, dict[str, float]]  # {feature: {stat: value}}
    label_distribution: dict[str, int] | None
    quality_indicators: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_stats": self.feature_stats,
            "label_distribution": self.label_distribution,
            "quality_indicators": self.quality_indicators,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FederatedQualityReport:
    """Aggregated quality report from federated analysis."""

    timestamp: datetime
    n_nodes: int
    total_samples: int
    privacy_level: PrivacyLevel
    aggregated_quality_score: float
    per_node_quality: dict[str, float]
    global_statistics: dict[str, Any]
    detected_issues: dict[str, int]
    feature_quality: dict[str, float]
    recommendations: list[str]
    privacy_guarantees: dict[str, Any]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Federated Data Quality Report",
            "=" * 50,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Nodes Analyzed: {self.n_nodes}",
            f"Total Samples: {self.total_samples:,}",
            f"Privacy Level: {self.privacy_level.value}",
            "",
            f"Aggregated Quality Score: {self.aggregated_quality_score:.1f}/100",
            "",
            "Per-Node Quality:",
        ]

        for node_id, score in self.per_node_quality.items():
            lines.append(f"  • {node_id}: {score:.1f}/100")

        if self.detected_issues:
            lines.append("")
            lines.append("Detected Issues (Aggregate):")
            for issue, count in self.detected_issues.items():
                lines.append(f"  • {issue}: {count:,}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        lines.append("")
        lines.append("Privacy Guarantees:")
        for key, value in self.privacy_guarantees.items():
            lines.append(f"  • {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "n_nodes": self.n_nodes,
            "total_samples": self.total_samples,
            "privacy_level": self.privacy_level.value,
            "aggregated_quality_score": self.aggregated_quality_score,
            "per_node_quality": self.per_node_quality,
            "global_statistics": self.global_statistics,
            "detected_issues": self.detected_issues,
            "feature_quality": self.feature_quality,
            "recommendations": self.recommendations,
            "privacy_guarantees": self.privacy_guarantees,
        }


class SecureAggregator:
    """Implements secure aggregation protocols."""

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self._noise_generator = np.random.default_rng(secrets.randbits(64))

    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add differentially private noise to a value."""
        if self.config.privacy_level == PrivacyLevel.LOW:
            return value

        if self.config.privacy_level == PrivacyLevel.HIGH:
            # Laplace mechanism for differential privacy
            scale = sensitivity / self.config.epsilon
            noise = self._noise_generator.laplace(0, scale)
            return value + noise

        # Medium privacy - Gaussian noise
        noise = self._noise_generator.normal(0, self.config.noise_scale * sensitivity)
        return value + noise

    def aggregate_sums(
        self, local_values: list[float], sensitivities: list[float] | None = None
    ) -> float:
        """Securely aggregate sums from multiple nodes."""
        if not local_values:
            return 0.0

        if sensitivities is None:
            sensitivities = [1.0] * len(local_values)

        total = sum(local_values)

        # Add noise based on combined sensitivity
        combined_sensitivity = sum(sensitivities)
        return self.add_noise(total, combined_sensitivity)

    def aggregate_means(
        self, local_means: list[float], local_counts: list[int]
    ) -> float:
        """Securely aggregate means from multiple nodes."""
        if not local_means or not local_counts:
            return 0.0

        total_count = sum(local_counts)
        if total_count == 0:
            return 0.0

        weighted_sum = sum(m * c for m, c in zip(local_means, local_counts))
        global_mean = weighted_sum / total_count

        # Sensitivity scales with 1/n for means
        sensitivity = 1.0 / max(total_count, 1)
        return self.add_noise(global_mean, sensitivity)

    def aggregate_histograms(
        self, local_histograms: list[dict[str, int]]
    ) -> dict[str, int]:
        """Securely aggregate histograms from multiple nodes."""
        if not local_histograms:
            return {}

        aggregated: dict[str, float] = {}
        for hist in local_histograms:
            for key, count in hist.items():
                aggregated[key] = aggregated.get(key, 0) + count

        # Add noise to each bin
        noisy_result: dict[str, int] = {}
        for key, count in aggregated.items():
            noisy_count = self.add_noise(count, sensitivity=1.0)
            noisy_result[key] = max(0, int(round(noisy_count)))

        return noisy_result


class LocalNode:
    """Represents a local data node that computes statistics locally."""

    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        node_id: str | None = None,
    ):
        self.data = data
        self.label_column = label_column
        self.node_id = node_id or self._generate_node_id()
        self._local_stats: LocalStatistics | None = None

    def _generate_node_id(self) -> str:
        """Generate unique node identifier."""
        hash_input = f"{id(self.data)}_{secrets.token_hex(4)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    def compute_local_statistics(self) -> LocalStatistics:
        """Compute statistics locally without sharing raw data."""
        n_samples = len(self.data)
        feature_cols = [c for c in self.data.columns if c != self.label_column]
        n_features = len(feature_cols)

        # Compute per-feature statistics
        feature_stats: dict[str, dict[str, float]] = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                col_data = self.data[col].dropna()
                feature_stats[col] = {
                    "mean": float(col_data.mean()) if len(col_data) > 0 else 0.0,
                    "std": float(col_data.std()) if len(col_data) > 0 else 0.0,
                    "min": float(col_data.min()) if len(col_data) > 0 else 0.0,
                    "max": float(col_data.max()) if len(col_data) > 0 else 0.0,
                    "null_rate": float(self.data[col].isna().mean()),
                }

        # Compute label distribution
        label_distribution = None
        n_classes = None
        if self.label_column and self.label_column in self.data.columns:
            label_counts = self.data[self.label_column].value_counts()
            label_distribution = {str(k): int(v) for k, v in label_counts.items()}
            n_classes = len(label_distribution)

        # Compute quality indicators
        quality_indicators = self._compute_quality_indicators(feature_cols)

        self._local_stats = LocalStatistics(
            node_id=self.node_id,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            feature_stats=feature_stats,
            label_distribution=label_distribution,
            quality_indicators=quality_indicators,
        )

        return self._local_stats

    def _compute_quality_indicators(self, feature_cols: list[str]) -> dict[str, float]:
        """Compute local quality indicators."""
        indicators: dict[str, float] = {}

        # Missing value rate
        total_cells = len(self.data) * len(feature_cols)
        if total_cells > 0:
            missing_cells = self.data[feature_cols].isna().sum().sum()
            indicators["missing_rate"] = float(missing_cells / total_cells)
        else:
            indicators["missing_rate"] = 0.0

        # Duplicate rate (exact)
        n_duplicates = self.data.duplicated(subset=feature_cols).sum()
        indicators["duplicate_rate"] = float(n_duplicates / max(len(self.data), 1))

        # Feature completeness
        complete_rows = self.data[feature_cols].dropna().shape[0]
        indicators["completeness"] = float(complete_rows / max(len(self.data), 1))

        # Class imbalance (if applicable)
        if self.label_column and self.label_column in self.data.columns:
            label_counts = self.data[self.label_column].value_counts()
            if len(label_counts) > 1:
                imbalance_ratio = label_counts.max() / max(label_counts.min(), 1)
                indicators["imbalance_ratio"] = float(imbalance_ratio)

        return indicators

    def compute_local_outlier_count(
        self, method: str = "iqr", contamination: float = 0.1
    ) -> int:
        """Compute local outlier count."""
        feature_cols = [c for c in self.data.columns if c != self.label_column]
        numeric_cols = self.data[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return 0

        outlier_mask = pd.Series([False] * len(self.data))

        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) < 10:
                continue

            if method == "iqr":
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                col_outliers = (self.data[col] < lower) | (self.data[col] > upper)
                outlier_mask = outlier_mask | col_outliers.fillna(False)
            elif method == "zscore":
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    z_scores = abs((self.data[col] - mean) / std)
                    col_outliers = z_scores > 3
                    outlier_mask = outlier_mask | col_outliers.fillna(False)

        return int(outlier_mask.sum())


class FederatedAnalyzer:
    """Orchestrates federated data quality analysis across nodes."""

    def __init__(
        self,
        nodes: list[LocalNode],
        privacy_config: PrivacyConfig | None = None,
    ):
        self.nodes = nodes
        self.privacy_config = privacy_config or PrivacyConfig()
        self._aggregator = SecureAggregator(self.privacy_config)
        self._local_stats: list[LocalStatistics] = []

    def add_node(self, node: LocalNode) -> None:
        """Add a new node to the federation."""
        self.nodes.append(node)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the federation."""
        original_len = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        return len(self.nodes) < original_len

    def analyze(
        self,
        detect_outliers: bool = True,
        detect_imbalance: bool = True,
        compute_drift: bool = False,
    ) -> FederatedQualityReport:
        """Run federated quality analysis across all nodes."""
        logger.info(
            "Starting federated analysis across %d nodes with privacy level: %s",
            len(self.nodes),
            self.privacy_config.privacy_level.value,
        )

        # Step 1: Each node computes local statistics
        self._local_stats = []
        for node in self.nodes:
            try:
                stats = node.compute_local_statistics()
                self._local_stats.append(stats)
                logger.debug("Collected stats from node %s", node.node_id)
            except Exception as e:
                logger.warning("Failed to collect stats from node %s: %s", node.node_id, e)

        if not self._local_stats:
            raise ValueError("No statistics collected from any node")

        # Step 2: Secure aggregation
        aggregated = self._aggregate_statistics()

        # Step 3: Compute global quality metrics
        per_node_quality = self._compute_per_node_quality()
        global_quality = self._compute_global_quality(per_node_quality)

        # Step 4: Detect issues across federation
        detected_issues = self._detect_federated_issues(detect_outliers, detect_imbalance)

        # Step 5: Compute feature quality
        feature_quality = self._compute_feature_quality()

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            global_quality, detected_issues, feature_quality
        )

        # Step 7: Document privacy guarantees
        privacy_guarantees = self._document_privacy_guarantees()

        return FederatedQualityReport(
            timestamp=datetime.now(),
            n_nodes=len(self.nodes),
            total_samples=sum(s.n_samples for s in self._local_stats),
            privacy_level=self.privacy_config.privacy_level,
            aggregated_quality_score=global_quality,
            per_node_quality=per_node_quality,
            global_statistics=aggregated,
            detected_issues=detected_issues,
            feature_quality=feature_quality,
            recommendations=recommendations,
            privacy_guarantees=privacy_guarantees,
        )

    def _aggregate_statistics(self) -> dict[str, Any]:
        """Securely aggregate statistics from all nodes."""
        # Aggregate sample counts
        sample_counts = [s.n_samples for s in self._local_stats]
        total_samples = self._aggregator.aggregate_sums(
            [float(c) for c in sample_counts]
        )

        # Aggregate feature statistics
        all_features = set()
        for stats in self._local_stats:
            all_features.update(stats.feature_stats.keys())

        aggregated_feature_stats: dict[str, dict[str, float]] = {}
        for feature in all_features:
            local_means = []
            local_counts = []
            for stats in self._local_stats:
                if feature in stats.feature_stats:
                    local_means.append(stats.feature_stats[feature].get("mean", 0))
                    local_counts.append(stats.n_samples)

            if local_means:
                global_mean = self._aggregator.aggregate_means(local_means, local_counts)
                aggregated_feature_stats[feature] = {"mean": global_mean}

        # Aggregate label distribution
        label_histograms = [
            s.label_distribution for s in self._local_stats if s.label_distribution
        ]
        aggregated_labels = self._aggregator.aggregate_histograms(label_histograms)

        return {
            "total_samples": int(total_samples),
            "feature_stats": aggregated_feature_stats,
            "label_distribution": aggregated_labels,
        }

    def _compute_per_node_quality(self) -> dict[str, float]:
        """Compute quality score for each node."""
        per_node_quality: dict[str, float] = {}

        for stats in self._local_stats:
            # Base score
            score = 100.0

            # Penalize missing data
            missing_rate = stats.quality_indicators.get("missing_rate", 0)
            score -= missing_rate * 30

            # Penalize duplicates
            dup_rate = stats.quality_indicators.get("duplicate_rate", 0)
            score -= dup_rate * 20

            # Penalize imbalance
            imbalance = stats.quality_indicators.get("imbalance_ratio", 1)
            if imbalance > 10:
                score -= 15
            elif imbalance > 5:
                score -= 10
            elif imbalance > 2:
                score -= 5

            per_node_quality[stats.node_id] = max(0, min(100, score))

        return per_node_quality

    def _compute_global_quality(self, per_node_quality: dict[str, float]) -> float:
        """Compute global quality score from per-node scores."""
        if not per_node_quality:
            return 0.0

        # Weighted average by sample count
        weighted_sum = 0.0
        total_weight = 0

        for stats in self._local_stats:
            if stats.node_id in per_node_quality:
                weight = stats.n_samples
                weighted_sum += per_node_quality[stats.node_id] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        global_score = weighted_sum / total_weight

        # Add noise for privacy
        return self._aggregator.add_noise(global_score, sensitivity=10.0)

    def _detect_federated_issues(
        self, detect_outliers: bool, detect_imbalance: bool
    ) -> dict[str, int]:
        """Detect issues across the federation."""
        issues: dict[str, int] = {}

        # Aggregate outlier counts
        if detect_outliers:
            outlier_counts = []
            for node in self.nodes:
                count = node.compute_local_outlier_count()
                outlier_counts.append(float(count))

            total_outliers = self._aggregator.aggregate_sums(outlier_counts)
            issues["outliers"] = int(max(0, total_outliers))

        # Check for global imbalance
        if detect_imbalance:
            all_labels: dict[str, int] = {}
            for stats in self._local_stats:
                if stats.label_distribution:
                    for label, count in stats.label_distribution.items():
                        all_labels[label] = all_labels.get(label, 0) + count

            if len(all_labels) > 1:
                counts = list(all_labels.values())
                if min(counts) > 0:
                    imbalance = max(counts) / min(counts)
                    if imbalance > 5:
                        issues["class_imbalance"] = 1

        # Aggregate duplicate counts
        dup_counts = []
        for stats in self._local_stats:
            dup_rate = stats.quality_indicators.get("duplicate_rate", 0)
            dup_counts.append(dup_rate * stats.n_samples)

        total_dups = self._aggregator.aggregate_sums(dup_counts)
        issues["duplicates"] = int(max(0, total_dups))

        return issues

    def _compute_feature_quality(self) -> dict[str, float]:
        """Compute quality score for each feature."""
        feature_quality: dict[str, float] = {}

        all_features = set()
        for stats in self._local_stats:
            all_features.update(stats.feature_stats.keys())

        for feature in all_features:
            quality = 100.0

            # Check null rates across nodes
            null_rates = []
            for stats in self._local_stats:
                if feature in stats.feature_stats:
                    null_rate = stats.feature_stats[feature].get("null_rate", 0)
                    null_rates.append(null_rate)

            if null_rates:
                avg_null = sum(null_rates) / len(null_rates)
                quality -= avg_null * 50

            feature_quality[feature] = max(0, min(100, quality))

        return feature_quality

    def _generate_recommendations(
        self,
        global_quality: float,
        detected_issues: dict[str, int],
        feature_quality: dict[str, float],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if global_quality < 70:
            recommendations.append(
                "Global quality score is below 70. Consider comprehensive data review."
            )

        if detected_issues.get("outliers", 0) > 100:
            recommendations.append(
                f"High outlier count ({detected_issues['outliers']}) detected. "
                "Review data collection processes."
            )

        if detected_issues.get("class_imbalance"):
            recommendations.append(
                "Significant class imbalance detected across federation. "
                "Consider stratified sampling or rebalancing."
            )

        if detected_issues.get("duplicates", 0) > 50:
            recommendations.append(
                "Duplicate data detected across nodes. "
                "Implement deduplication before federated training."
            )

        # Feature-specific recommendations
        low_quality_features = [
            f for f, q in feature_quality.items() if q < 60
        ]
        if low_quality_features:
            recommendations.append(
                f"Features with quality issues: {', '.join(low_quality_features[:5])}"
            )

        # Node-specific recommendations
        node_qualities = self._compute_per_node_quality()
        low_quality_nodes = [
            n for n, q in node_qualities.items() if q < 60
        ]
        if low_quality_nodes:
            recommendations.append(
                f"Nodes requiring attention: {len(low_quality_nodes)} nodes below quality threshold"
            )

        return recommendations

    def _document_privacy_guarantees(self) -> dict[str, Any]:
        """Document the privacy guarantees provided."""
        guarantees: dict[str, Any] = {
            "privacy_level": self.privacy_config.privacy_level.value,
            "raw_data_shared": False,
            "aggregation_method": "secure" if self.privacy_config.secure_aggregation else "standard",
        }

        if self.privacy_config.privacy_level in (PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM):
            guarantees["differential_privacy"] = {
                "epsilon": self.privacy_config.epsilon,
                "delta": self.privacy_config.delta,
                "mechanism": "Laplace" if self.privacy_config.privacy_level == PrivacyLevel.HIGH else "Gaussian",
            }

        guarantees["min_samples_per_node"] = self.privacy_config.min_samples_per_node

        return guarantees


class FederatedCoordinator:
    """Coordinates federated analysis across network nodes."""

    def __init__(self, privacy_config: PrivacyConfig | None = None):
        self.privacy_config = privacy_config or PrivacyConfig()
        self._registered_nodes: dict[str, LocalNode] = {}
        self._analysis_history: list[FederatedQualityReport] = []

    def register_node(self, node: LocalNode) -> str:
        """Register a node with the coordinator."""
        self._registered_nodes[node.node_id] = node
        logger.info("Registered node %s", node.node_id)
        return node.node_id

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the coordinator."""
        if node_id in self._registered_nodes:
            del self._registered_nodes[node_id]
            logger.info("Unregistered node %s", node_id)
            return True
        return False

    def list_nodes(self) -> list[str]:
        """List all registered node IDs."""
        return list(self._registered_nodes.keys())

    def run_analysis(self, node_ids: list[str] | None = None) -> FederatedQualityReport:
        """Run federated analysis on specified or all nodes."""
        if node_ids is None:
            nodes = list(self._registered_nodes.values())
        else:
            nodes = [
                self._registered_nodes[nid]
                for nid in node_ids
                if nid in self._registered_nodes
            ]

        if not nodes:
            raise ValueError("No valid nodes for analysis")

        analyzer = FederatedAnalyzer(nodes, self.privacy_config)
        report = analyzer.analyze()

        self._analysis_history.append(report)
        return report

    def get_history(self) -> list[FederatedQualityReport]:
        """Get analysis history."""
        return self._analysis_history.copy()


def create_federated_analyzer(
    datasets: list[pd.DataFrame],
    label_column: str | None = None,
    privacy_level: str = "medium",
    epsilon: float = 1.0,
) -> FederatedAnalyzer:
    """Convenience function to create a federated analyzer.

    Args:
        datasets: List of DataFrames from different sources
        label_column: Name of the label column
        privacy_level: Privacy protection level ('low', 'medium', 'high', 'maximum')
        epsilon: Differential privacy epsilon parameter

    Returns:
        Configured FederatedAnalyzer
    """
    nodes = [
        LocalNode(data=df, label_column=label_column)
        for df in datasets
    ]

    config = PrivacyConfig(
        privacy_level=PrivacyLevel(privacy_level),
        epsilon=epsilon,
    )

    return FederatedAnalyzer(nodes=nodes, privacy_config=config)


def federated_analyze(
    datasets: list[pd.DataFrame],
    label_column: str | None = None,
    privacy_level: str = "medium",
) -> FederatedQualityReport:
    """Run federated analysis on multiple datasets.

    Args:
        datasets: List of DataFrames from different sources
        label_column: Name of the label column
        privacy_level: Privacy protection level

    Returns:
        FederatedQualityReport with aggregated results
    """
    analyzer = create_federated_analyzer(
        datasets=datasets,
        label_column=label_column,
        privacy_level=privacy_level,
    )
    return analyzer.analyze()
