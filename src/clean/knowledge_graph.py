"""Data Quality Knowledge Graph.

Build a knowledge graph connecting data issues to model performance impact,
enabling "what-if" analysis for data cleaning decisions.

Example:
    >>> from clean.knowledge_graph import QualityKnowledgeGraph, ImpactAnalyzer
    >>>
    >>> # Build knowledge graph from quality report
    >>> graph = QualityKnowledgeGraph()
    >>> graph.build_from_report(quality_report)
    >>>
    >>> # Analyze impact of fixing specific issues
    >>> analyzer = ImpactAnalyzer(graph)
    >>> impact = analyzer.predict_impact(fix_indices=[42, 187, 256])
    >>> print(f"Expected accuracy improvement: {impact.accuracy_delta:+.2%}")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""

    DATASET = "dataset"
    SAMPLE = "sample"
    FEATURE = "feature"
    LABEL = "label"
    ISSUE = "issue"
    MODEL = "model"
    METRIC = "metric"
    CLUSTER = "cluster"
    CLASS = "class"


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""

    HAS_FEATURE = "has_feature"
    HAS_LABEL = "has_label"
    HAS_ISSUE = "has_issue"
    AFFECTS = "affects"
    SIMILAR_TO = "similar_to"
    DUPLICATE_OF = "duplicate_of"
    BELONGS_TO = "belongs_to"
    TRAINED_ON = "trained_on"
    MEASURED_BY = "measured_by"
    CAUSES = "causes"
    CORRELATES_WITH = "correlates_with"


class IssueCategory(Enum):
    """Categories of data quality issues."""

    LABEL_ERROR = "label_error"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    MISSING_VALUE = "missing_value"
    BIAS = "bias"
    NOISE = "noise"
    IMBALANCE = "imbalance"
    DRIFT = "drift"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    node_id: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphNode):
            return self.node_id == other.node_id
        return False


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.edge_type))


@dataclass
class ImpactPrediction:
    """Predicted impact of fixing data issues."""

    fix_indices: list[int]
    n_samples_affected: int
    accuracy_delta: float
    precision_delta: float
    recall_delta: float
    f1_delta: float
    confidence: float
    breakdown_by_issue: dict[str, float]
    breakdown_by_class: dict[str, float]
    reasoning: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Impact Prediction",
            "=" * 50,
            f"Samples to Fix: {self.n_samples_affected}",
            f"Confidence: {self.confidence:.1%}",
            "",
            "Predicted Metric Changes:",
            f"  • Accuracy: {self.accuracy_delta:+.2%}",
            f"  • Precision: {self.precision_delta:+.2%}",
            f"  • Recall: {self.recall_delta:+.2%}",
            f"  • F1 Score: {self.f1_delta:+.2%}",
        ]

        if self.breakdown_by_issue:
            lines.append("")
            lines.append("Impact by Issue Type:")
            for issue, impact in sorted(
                self.breakdown_by_issue.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  • {issue}: {impact:+.2%}")

        if self.reasoning:
            lines.append("")
            lines.append("Reasoning:")
            for reason in self.reasoning[:5]:
                lines.append(f"  → {reason}")

        return "\n".join(lines)


@dataclass
class WhatIfScenario:
    """A what-if analysis scenario."""

    scenario_id: str
    description: str
    actions: list[dict[str, Any]]
    predicted_impact: ImpactPrediction | None = None
    actual_impact: dict[str, float] | None = None


class QualityKnowledgeGraph:
    """Knowledge graph for data quality analysis."""

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[str]] = {}  # node_id -> [connected_node_ids]
        self._reverse_adjacency: dict[str, list[str]] = {}
        self._edge_index: dict[tuple[str, str, EdgeType], GraphEdge] = {}

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self._edges.append(edge)
        self._adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        self._reverse_adjacency.setdefault(edge.target_id, []).append(edge.source_id)
        self._edge_index[(edge.source_id, edge.target_id, edge.edge_type)] = edge

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_neighbors(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[GraphNode]:
        """Get neighboring nodes."""
        neighbor_ids = self._adjacency.get(node_id, [])
        neighbors = []

        for nid in neighbor_ids:
            if edge_type is not None:
                if (node_id, nid, edge_type) in self._edge_index:
                    node = self._nodes.get(nid)
                    if node:
                        neighbors.append(node)
            else:
                node = self._nodes.get(nid)
                if node:
                    neighbors.append(node)

        return neighbors

    def get_edges_from(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[GraphEdge]:
        """Get outgoing edges from a node."""
        edges = []
        for edge in self._edges:
            if edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    edges.append(edge)
        return edges

    def get_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[str] | None:
        """Find shortest path between two nodes using BFS."""
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor_id in self._adjacency.get(current, []):
                if neighbor_id == target_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def build_from_report(
        self,
        quality_report: Any,  # QualityReport from clean
        data: pd.DataFrame | None = None,
    ) -> None:
        """Build knowledge graph from a quality report.

        Args:
            quality_report: QualityReport object from clean analysis
            data: Optional DataFrame for additional context
        """
        logger.info("Building knowledge graph from quality report")

        # Create dataset node
        dataset_node = GraphNode(
            node_id="dataset_main",
            node_type=NodeType.DATASET,
            properties={
                "n_samples": quality_report.dataset_info.n_samples,
                "n_features": quality_report.dataset_info.n_features,
                "quality_score": quality_report.quality_score.overall,
            },
        )
        self.add_node(dataset_node)

        # Add feature nodes
        if data is not None:
            for col in data.columns:
                feature_node = GraphNode(
                    node_id=f"feature_{col}",
                    node_type=NodeType.FEATURE,
                    properties={
                        "name": col,
                        "dtype": str(data[col].dtype),
                        "missing_rate": float(data[col].isna().mean()),
                    },
                )
                self.add_node(feature_node)
                self.add_edge(
                    GraphEdge(
                        source_id="dataset_main",
                        target_id=f"feature_{col}",
                        edge_type=EdgeType.HAS_FEATURE,
                    )
                )

        # Add label error issues
        if quality_report.label_errors_result:
            for issue in quality_report.label_errors_result.issues:
                issue_node = GraphNode(
                    node_id=f"issue_label_{issue.index}",
                    node_type=NodeType.ISSUE,
                    properties={
                        "category": IssueCategory.LABEL_ERROR.value,
                        "sample_index": issue.index,
                        "given_label": issue.given_label,
                        "predicted_label": issue.predicted_label,
                        "confidence": issue.confidence,
                    },
                )
                self.add_node(issue_node)

                # Create sample node
                sample_node = GraphNode(
                    node_id=f"sample_{issue.index}",
                    node_type=NodeType.SAMPLE,
                    properties={"index": issue.index},
                )
                self.add_node(sample_node)

                # Link issue to sample
                self.add_edge(
                    GraphEdge(
                        source_id=f"sample_{issue.index}",
                        target_id=f"issue_label_{issue.index}",
                        edge_type=EdgeType.HAS_ISSUE,
                        weight=issue.confidence,
                    )
                )

        # Add duplicate issues
        if quality_report.duplicates_result:
            for i, dup in enumerate(quality_report.duplicates_result.issues):
                issue_node = GraphNode(
                    node_id=f"issue_dup_{i}",
                    node_type=NodeType.ISSUE,
                    properties={
                        "category": IssueCategory.DUPLICATE.value,
                        "index1": dup.index1,
                        "index2": dup.index2,
                        "similarity": dup.similarity,
                    },
                )
                self.add_node(issue_node)

                # Link duplicates
                for idx in [dup.index1, dup.index2]:
                    sample_id = f"sample_{idx}"
                    if sample_id not in self._nodes:
                        self.add_node(
                            GraphNode(
                                node_id=sample_id,
                                node_type=NodeType.SAMPLE,
                                properties={"index": idx},
                            )
                        )
                    self.add_edge(
                        GraphEdge(
                            source_id=sample_id,
                            target_id=f"issue_dup_{i}",
                            edge_type=EdgeType.HAS_ISSUE,
                            weight=dup.similarity,
                        )
                    )

        # Add outlier issues
        if quality_report.outliers_result:
            for outlier in quality_report.outliers_result.issues:
                issue_node = GraphNode(
                    node_id=f"issue_outlier_{outlier.index}",
                    node_type=NodeType.ISSUE,
                    properties={
                        "category": IssueCategory.OUTLIER.value,
                        "sample_index": outlier.index,
                        "score": outlier.score,
                        "method": outlier.method,
                    },
                )
                self.add_node(issue_node)

                sample_id = f"sample_{outlier.index}"
                if sample_id not in self._nodes:
                    self.add_node(
                        GraphNode(
                            node_id=sample_id,
                            node_type=NodeType.SAMPLE,
                            properties={"index": outlier.index},
                        )
                    )

                self.add_edge(
                    GraphEdge(
                        source_id=sample_id,
                        target_id=f"issue_outlier_{outlier.index}",
                        edge_type=EdgeType.HAS_ISSUE,
                        weight=outlier.score,
                    )
                )

        logger.info(
            "Built graph with %d nodes and %d edges",
            len(self._nodes),
            len(self._edges),
        )

    def get_issues_for_sample(self, sample_index: int) -> list[GraphNode]:
        """Get all issues associated with a sample."""
        sample_id = f"sample_{sample_index}"
        return self.get_neighbors(sample_id, EdgeType.HAS_ISSUE)

    def get_samples_with_issue_type(self, issue_category: IssueCategory) -> list[int]:
        """Get all sample indices with a specific issue type."""
        issue_nodes = [
            n
            for n in self.get_nodes_by_type(NodeType.ISSUE)
            if n.properties.get("category") == issue_category.value
        ]

        sample_indices = []
        for issue_node in issue_nodes:
            # Get samples connected to this issue
            for source_id in self._reverse_adjacency.get(issue_node.node_id, []):
                node = self._nodes.get(source_id)
                if node and node.node_type == NodeType.SAMPLE:
                    idx = node.properties.get("index")
                    if idx is not None:
                        sample_indices.append(idx)

        return list(set(sample_indices))

    def export_to_dict(self) -> dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type.value,
                    "properties": n.properties,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                    "properties": e.properties,
                }
                for e in self._edges
            ],
        }

    def summary(self) -> str:
        """Generate summary of the graph."""
        node_counts: dict[str, int] = {}
        for node in self._nodes.values():
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        edge_counts: dict[str, int] = {}
        for edge in self._edges:
            edge_type = edge.edge_type.value
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

        lines = [
            "Quality Knowledge Graph Summary",
            "=" * 50,
            f"Total Nodes: {len(self._nodes)}",
            f"Total Edges: {len(self._edges)}",
            "",
            "Nodes by Type:",
        ]

        for node_type, count in sorted(node_counts.items()):
            lines.append(f"  • {node_type}: {count}")

        lines.append("")
        lines.append("Edges by Type:")
        for edge_type, count in sorted(edge_counts.items()):
            lines.append(f"  • {edge_type}: {count}")

        return "\n".join(lines)


class ImpactAnalyzer:
    """Analyzes the impact of fixing data quality issues."""

    # Empirical impact factors (can be calibrated with actual data)
    IMPACT_FACTORS = {
        IssueCategory.LABEL_ERROR: {
            "accuracy": 0.015,  # Each label error fix improves accuracy by ~1.5%
            "precision": 0.012,
            "recall": 0.012,
            "f1": 0.012,
        },
        IssueCategory.DUPLICATE: {
            "accuracy": 0.002,
            "precision": 0.003,
            "recall": 0.001,
            "f1": 0.002,
        },
        IssueCategory.OUTLIER: {
            "accuracy": 0.005,
            "precision": 0.004,
            "recall": 0.003,
            "f1": 0.004,
        },
        IssueCategory.MISSING_VALUE: {
            "accuracy": 0.001,
            "precision": 0.001,
            "recall": 0.001,
            "f1": 0.001,
        },
    }

    def __init__(self, graph: QualityKnowledgeGraph):
        self.graph = graph
        self._calibration_data: list[dict[str, Any]] = []

    def predict_impact(
        self,
        fix_indices: list[int] | None = None,
        fix_issue_types: list[IssueCategory] | None = None,
    ) -> ImpactPrediction:
        """Predict the impact of fixing specified issues.

        Args:
            fix_indices: Specific sample indices to fix
            fix_issue_types: Fix all issues of these types

        Returns:
            ImpactPrediction with estimated metric changes
        """
        # Collect all issues to be fixed
        issues_to_fix: list[GraphNode] = []

        if fix_indices:
            for idx in fix_indices:
                issues = self.graph.get_issues_for_sample(idx)
                issues_to_fix.extend(issues)

        if fix_issue_types:
            for issue_type in fix_issue_types:
                issue_nodes = [
                    n
                    for n in self.graph.get_nodes_by_type(NodeType.ISSUE)
                    if n.properties.get("category") == issue_type.value
                ]
                issues_to_fix.extend(issue_nodes)

        # Remove duplicates
        issues_to_fix = list({i.node_id: i for i in issues_to_fix}.values())

        if not issues_to_fix:
            return ImpactPrediction(
                fix_indices=fix_indices or [],
                n_samples_affected=0,
                accuracy_delta=0.0,
                precision_delta=0.0,
                recall_delta=0.0,
                f1_delta=0.0,
                confidence=1.0,
                breakdown_by_issue={},
                breakdown_by_class={},
                reasoning=["No issues found to fix"],
            )

        # Calculate impact
        accuracy_delta = 0.0
        precision_delta = 0.0
        recall_delta = 0.0
        f1_delta = 0.0

        breakdown_by_issue: dict[str, float] = {}
        reasoning = []

        for issue_node in issues_to_fix:
            category_str = issue_node.properties.get("category", "unknown")
            try:
                category = IssueCategory(category_str)
            except ValueError:
                continue

            factors = self.IMPACT_FACTORS.get(category, {})
            confidence = issue_node.properties.get("confidence", 0.5)

            # Weight impact by confidence
            acc_impact = factors.get("accuracy", 0) * confidence
            prec_impact = factors.get("precision", 0) * confidence
            rec_impact = factors.get("recall", 0) * confidence
            f1_impact = factors.get("f1", 0) * confidence

            accuracy_delta += acc_impact
            precision_delta += prec_impact
            recall_delta += rec_impact
            f1_delta += f1_impact

            breakdown_by_issue[category_str] = (
                breakdown_by_issue.get(category_str, 0) + acc_impact
            )

        # Cap maximum improvement
        accuracy_delta = min(accuracy_delta, 0.15)  # Max 15% improvement
        precision_delta = min(precision_delta, 0.15)
        recall_delta = min(recall_delta, 0.15)
        f1_delta = min(f1_delta, 0.15)

        # Calculate confidence in prediction
        n_issues = len(issues_to_fix)
        prediction_confidence = min(0.95, 0.5 + n_issues * 0.01)

        # Generate reasoning
        for category, impact in sorted(
            breakdown_by_issue.items(), key=lambda x: -x[1]
        ):
            count = sum(
                1
                for i in issues_to_fix
                if i.properties.get("category") == category
            )
            reasoning.append(
                f"Fixing {count} {category} issues contributes {impact:.2%} accuracy improvement"
            )

        # Count affected samples
        affected_indices = set()
        if fix_indices:
            affected_indices.update(fix_indices)
        for issue in issues_to_fix:
            idx = issue.properties.get("sample_index")
            if idx is not None:
                affected_indices.add(idx)

        return ImpactPrediction(
            fix_indices=fix_indices or [],
            n_samples_affected=len(affected_indices),
            accuracy_delta=accuracy_delta,
            precision_delta=precision_delta,
            recall_delta=recall_delta,
            f1_delta=f1_delta,
            confidence=prediction_confidence,
            breakdown_by_issue=breakdown_by_issue,
            breakdown_by_class={},
            reasoning=reasoning,
        )

    def analyze_what_if(self, scenario: WhatIfScenario) -> WhatIfScenario:
        """Analyze a what-if scenario.

        Args:
            scenario: Scenario to analyze

        Returns:
            Scenario with predicted impact filled in
        """
        # Extract indices from scenario actions
        fix_indices = []
        fix_types = []

        for action in scenario.actions:
            if "indices" in action:
                fix_indices.extend(action["indices"])
            if "issue_type" in action:
                try:
                    fix_types.append(IssueCategory(action["issue_type"]))
                except ValueError:
                    pass

        scenario.predicted_impact = self.predict_impact(
            fix_indices=fix_indices if fix_indices else None,
            fix_issue_types=fix_types if fix_types else None,
        )

        return scenario

    def calibrate(
        self,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        fixed_issues: list[GraphNode],
    ) -> None:
        """Calibrate impact factors with actual results.

        Args:
            before_metrics: Metrics before fixing
            after_metrics: Metrics after fixing
            fixed_issues: Issues that were fixed
        """
        self._calibration_data.append(
            {
                "before": before_metrics,
                "after": after_metrics,
                "issues": [i.properties for i in fixed_issues],
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Recalibrate factors if enough data
        if len(self._calibration_data) >= 5:
            self._recalibrate()

    def _recalibrate(self) -> None:
        """Recalibrate impact factors based on collected data."""
        # Simple averaging of observed impacts
        observed_impacts: dict[str, list[float]] = {}

        for record in self._calibration_data:
            accuracy_change = (
                record["after"].get("accuracy", 0) - record["before"].get("accuracy", 0)
            )

            for issue in record["issues"]:
                category = issue.get("category")
                if category:
                    observed_impacts.setdefault(category, []).append(accuracy_change)

        # Update impact factors
        for category_str, impacts in observed_impacts.items():
            if impacts:
                avg_impact = sum(impacts) / len(impacts)
                try:
                    category = IssueCategory(category_str)
                    if category in self.IMPACT_FACTORS:
                        # Blend with existing factor
                        old_factor = self.IMPACT_FACTORS[category]["accuracy"]
                        new_factor = (old_factor + avg_impact) / 2
                        self.IMPACT_FACTORS[category]["accuracy"] = max(0, new_factor)
                except ValueError:
                    pass

        logger.info("Recalibrated impact factors with %d records", len(self._calibration_data))

    def get_priority_ranking(self, top_n: int = 20) -> list[tuple[int, float]]:
        """Get priority ranking of samples to fix.

        Args:
            top_n: Number of top samples to return

        Returns:
            List of (sample_index, impact_score) tuples
        """
        sample_scores: dict[int, float] = {}

        for node in self.graph.get_nodes_by_type(NodeType.SAMPLE):
            idx = node.properties.get("index")
            if idx is None:
                continue

            issues = self.graph.get_issues_for_sample(idx)
            if not issues:
                continue

            # Calculate total impact score for this sample
            total_score = 0.0
            for issue in issues:
                category_str = issue.properties.get("category")
                confidence = issue.properties.get("confidence", 0.5)

                try:
                    category = IssueCategory(category_str)
                    factor = self.IMPACT_FACTORS.get(category, {}).get("accuracy", 0.001)
                    total_score += factor * confidence
                except (ValueError, TypeError):
                    total_score += 0.001

            sample_scores[idx] = total_score

        # Sort by score descending
        ranked = sorted(sample_scores.items(), key=lambda x: -x[1])
        return ranked[:top_n]


class QualityGraphBuilder:
    """Builder for constructing quality knowledge graphs."""

    def __init__(self):
        self.graph = QualityKnowledgeGraph()

    def add_dataset(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
    ) -> "QualityGraphBuilder":
        """Add dataset to the graph."""
        dataset_node = GraphNode(
            node_id="dataset_main",
            node_type=NodeType.DATASET,
            properties={
                "n_samples": len(data),
                "n_features": len(data.columns),
                "label_column": label_column,
            },
        )
        self.graph.add_node(dataset_node)

        # Add feature nodes
        for col in data.columns:
            if col != label_column:
                feature_node = GraphNode(
                    node_id=f"feature_{col}",
                    node_type=NodeType.FEATURE,
                    properties={
                        "name": col,
                        "dtype": str(data[col].dtype),
                        "missing_rate": float(data[col].isna().mean()),
                    },
                )
                self.graph.add_node(feature_node)
                self.graph.add_edge(
                    GraphEdge(
                        source_id="dataset_main",
                        target_id=f"feature_{col}",
                        edge_type=EdgeType.HAS_FEATURE,
                    )
                )

        return self

    def add_label_errors(
        self,
        errors: list[dict[str, Any]],
    ) -> "QualityGraphBuilder":
        """Add label error issues to the graph.

        Args:
            errors: List of dicts with 'index', 'given_label', 'predicted_label', 'confidence'
        """
        for error in errors:
            idx = error.get("index")
            if idx is None:
                continue

            issue_node = GraphNode(
                node_id=f"issue_label_{idx}",
                node_type=NodeType.ISSUE,
                properties={
                    "category": IssueCategory.LABEL_ERROR.value,
                    "sample_index": idx,
                    "given_label": error.get("given_label"),
                    "predicted_label": error.get("predicted_label"),
                    "confidence": error.get("confidence", 0.5),
                },
            )
            self.graph.add_node(issue_node)

            # Ensure sample node exists
            sample_id = f"sample_{idx}"
            if self.graph.get_node(sample_id) is None:
                self.graph.add_node(
                    GraphNode(
                        node_id=sample_id,
                        node_type=NodeType.SAMPLE,
                        properties={"index": idx},
                    )
                )

            self.graph.add_edge(
                GraphEdge(
                    source_id=sample_id,
                    target_id=f"issue_label_{idx}",
                    edge_type=EdgeType.HAS_ISSUE,
                    weight=error.get("confidence", 0.5),
                )
            )

        return self

    def add_duplicates(
        self,
        duplicates: list[dict[str, Any]],
    ) -> "QualityGraphBuilder":
        """Add duplicate issues to the graph.

        Args:
            duplicates: List of dicts with 'index1', 'index2', 'similarity'
        """
        for i, dup in enumerate(duplicates):
            issue_node = GraphNode(
                node_id=f"issue_dup_{i}",
                node_type=NodeType.ISSUE,
                properties={
                    "category": IssueCategory.DUPLICATE.value,
                    "index1": dup.get("index1"),
                    "index2": dup.get("index2"),
                    "similarity": dup.get("similarity", 1.0),
                },
            )
            self.graph.add_node(issue_node)

            for idx in [dup.get("index1"), dup.get("index2")]:
                if idx is not None:
                    sample_id = f"sample_{idx}"
                    if self.graph.get_node(sample_id) is None:
                        self.graph.add_node(
                            GraphNode(
                                node_id=sample_id,
                                node_type=NodeType.SAMPLE,
                                properties={"index": idx},
                            )
                        )
                    self.graph.add_edge(
                        GraphEdge(
                            source_id=sample_id,
                            target_id=f"issue_dup_{i}",
                            edge_type=EdgeType.HAS_ISSUE,
                            weight=dup.get("similarity", 1.0),
                        )
                    )

        return self

    def add_outliers(
        self,
        outliers: list[dict[str, Any]],
    ) -> "QualityGraphBuilder":
        """Add outlier issues to the graph.

        Args:
            outliers: List of dicts with 'index', 'score', 'method'
        """
        for outlier in outliers:
            idx = outlier.get("index")
            if idx is None:
                continue

            issue_node = GraphNode(
                node_id=f"issue_outlier_{idx}",
                node_type=NodeType.ISSUE,
                properties={
                    "category": IssueCategory.OUTLIER.value,
                    "sample_index": idx,
                    "score": outlier.get("score", 0.5),
                    "method": outlier.get("method", "unknown"),
                },
            )
            self.graph.add_node(issue_node)

            sample_id = f"sample_{idx}"
            if self.graph.get_node(sample_id) is None:
                self.graph.add_node(
                    GraphNode(
                        node_id=sample_id,
                        node_type=NodeType.SAMPLE,
                        properties={"index": idx},
                    )
                )

            self.graph.add_edge(
                GraphEdge(
                    source_id=sample_id,
                    target_id=f"issue_outlier_{idx}",
                    edge_type=EdgeType.HAS_ISSUE,
                    weight=outlier.get("score", 0.5),
                )
            )

        return self

    def build(self) -> QualityKnowledgeGraph:
        """Build and return the knowledge graph."""
        return self.graph


def create_knowledge_graph(
    data: pd.DataFrame,
    label_errors: list[dict[str, Any]] | None = None,
    duplicates: list[dict[str, Any]] | None = None,
    outliers: list[dict[str, Any]] | None = None,
    label_column: str | None = None,
) -> QualityKnowledgeGraph:
    """Convenience function to create a knowledge graph.

    Args:
        data: Input DataFrame
        label_errors: List of label error dicts
        duplicates: List of duplicate dicts
        outliers: List of outlier dicts
        label_column: Name of label column

    Returns:
        Populated QualityKnowledgeGraph
    """
    builder = QualityGraphBuilder()
    builder.add_dataset(data, label_column)

    if label_errors:
        builder.add_label_errors(label_errors)
    if duplicates:
        builder.add_duplicates(duplicates)
    if outliers:
        builder.add_outliers(outliers)

    return builder.build()


def predict_fix_impact(
    graph: QualityKnowledgeGraph,
    fix_indices: list[int],
) -> ImpactPrediction:
    """Convenience function to predict fix impact.

    Args:
        graph: Quality knowledge graph
        fix_indices: Sample indices to fix

    Returns:
        ImpactPrediction
    """
    analyzer = ImpactAnalyzer(graph)
    return analyzer.predict_impact(fix_indices=fix_indices)
