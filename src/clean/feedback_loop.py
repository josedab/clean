"""Continuous Learning Loop - Model Performance to Data Quality Feedback.

This module provides tools to track model performance, correlate drops with
data quality issues, and generate data prescriptions for improvement.

Example:
    >>> from clean.feedback_loop import FeedbackLoop, MLflowConnector
    >>>
    >>> # Create feedback loop
    >>> loop = FeedbackLoop(
    ...     data=training_df,
    ...     label_column="label",
    ...     connector=MLflowConnector(tracking_uri="http://localhost:5000"),
    ... )
    >>>
    >>> # Ingest model metrics
    >>> loop.ingest_metrics(run_id="abc123")
    >>>
    >>> # Analyze correlation between data issues and performance
    >>> analysis = loop.analyze_correlation()
    >>>
    >>> # Get data prescriptions
    >>> prescriptions = loop.get_prescriptions()
    >>> for p in prescriptions:
    ...     print(f"{p.action}: {p.expected_improvement}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.core.report import QualityReport
from clean.exceptions import CleanError, ConfigurationError, DependencyError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of model metrics."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    LOSS = "loss"
    CUSTOM = "custom"


class ActionType(Enum):
    """Types of data improvement actions."""

    REMOVE_LABEL_ERRORS = "remove_label_errors"
    REMOVE_DUPLICATES = "remove_duplicates"
    REMOVE_OUTLIERS = "remove_outliers"
    REBALANCE_CLASSES = "rebalance_classes"
    CLEAN_SLICE = "clean_slice"
    AUGMENT_DATA = "augment_data"
    RELABEL_SAMPLES = "relabel_samples"


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    run_id: str
    timestamp: datetime
    metrics: dict[str, float]
    dataset_hash: str | None = None
    model_type: str | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def get_metric(self, name: str) -> float | None:
        """Get a specific metric value."""
        return self.metrics.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "dataset_hash": self.dataset_hash,
            "model_type": self.model_type,
            "hyperparams": self.hyperparams,
            "tags": self.tags,
        }


@dataclass
class PerformanceDrop:
    """Detected performance regression."""

    metric_name: str
    previous_value: float
    current_value: float
    drop_percentage: float
    significance: float  # Statistical significance 0-1
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def is_significant(self) -> bool:
        """Check if drop is statistically significant."""
        return self.significance > 0.95 and self.drop_percentage > 2.0


@dataclass
class DataPrescription:
    """Recommended data improvement action."""

    action: ActionType
    description: str
    target_samples: list[int]  # Indices of samples to act on
    expected_improvement: float  # Estimated metric improvement
    confidence: float  # Confidence in the prescription 0-1
    reasoning: str
    priority: int = 1  # 1 = highest priority
    estimated_effort: str = "low"  # low, medium, high

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "description": self.description,
            "n_samples": len(self.target_samples),
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "priority": self.priority,
            "reasoning": self.reasoning,
        }


@dataclass
class CorrelationResult:
    """Result of correlation analysis between data quality and performance."""

    issue_type: str
    metric_name: str
    correlation: float  # Pearson correlation coefficient
    p_value: float
    sample_overlap: int  # Number of samples where issue correlates with errors
    effect_size: float  # Estimated effect on metric
    confidence_interval: tuple[float, float]

    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant."""
        return self.p_value < 0.05 and abs(self.correlation) > 0.3


@dataclass
class FeedbackAnalysis:
    """Complete feedback loop analysis results."""

    timestamp: datetime
    n_metrics_analyzed: int
    performance_drops: list[PerformanceDrop]
    correlations: list[CorrelationResult]
    prescriptions: list[DataPrescription]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "n_metrics_analyzed": self.n_metrics_analyzed,
            "performance_drops": [
                {
                    "metric": d.metric_name,
                    "drop": d.drop_percentage,
                    "significant": d.is_significant,
                }
                for d in self.performance_drops
            ],
            "correlations": [
                {
                    "issue": c.issue_type,
                    "metric": c.metric_name,
                    "correlation": c.correlation,
                    "significant": c.is_significant,
                }
                for c in self.correlations
            ],
            "prescriptions": [p.to_dict() for p in self.prescriptions],
            "summary": self.summary,
        }


class MetricsConnector(ABC):
    """Abstract base class for metrics ingestion connectors."""

    @abstractmethod
    def get_metrics(self, run_id: str | None = None) -> list[ModelMetrics]:
        """Get metrics from the source.

        Args:
            run_id: Optional specific run ID to fetch

        Returns:
            List of ModelMetrics objects
        """
        pass

    @abstractmethod
    def get_latest_metrics(self, n: int = 10) -> list[ModelMetrics]:
        """Get the N most recent metric recordings.

        Args:
            n: Number of recent recordings to fetch

        Returns:
            List of ModelMetrics objects
        """
        pass

    @property
    @abstractmethod
    def connector_name(self) -> str:
        """Return connector name."""
        pass


class MLflowConnector(MetricsConnector):
    """Connector for MLflow tracking server."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str | None = None,
    ):
        """Initialize MLflow connector.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Optional experiment to filter
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
        except ImportError as e:
            raise DependencyError(
                "mlflow",
                "MLflow connector requires mlflow package. Install with: pip install mlflow"
            ) from e

    @property
    def connector_name(self) -> str:
        return "mlflow"

    def get_metrics(self, run_id: str | None = None) -> list[ModelMetrics]:
        """Get metrics from MLflow."""
        if run_id:
            run = self._mlflow.get_run(run_id)
            return [self._run_to_metrics(run)]

        # Get all runs from experiment
        if self.experiment_name:
            experiment = self._mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = self._mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                )
                return [self._run_to_metrics(row) for _, row in runs.iterrows()]

        return []

    def get_latest_metrics(self, n: int = 10) -> list[ModelMetrics]:
        """Get latest N metrics from MLflow."""
        if self.experiment_name:
            experiment = self._mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = self._mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=n,
                )
                return [self._run_to_metrics(row) for _, row in runs.iterrows()]
        return []

    def _run_to_metrics(self, run: Any) -> ModelMetrics:
        """Convert MLflow run to ModelMetrics."""
        if hasattr(run, "data"):
            # MLflow Run object
            metrics = run.data.metrics
            params = run.data.params
            tags = run.data.tags
            run_id = run.info.run_id
            timestamp = datetime.fromtimestamp(run.info.start_time / 1000)
        else:
            # DataFrame row from search_runs
            metrics = {
                k.replace("metrics.", ""): v
                for k, v in run.items()
                if k.startswith("metrics.") and pd.notna(v)
            }
            params = {
                k.replace("params.", ""): v
                for k, v in run.items()
                if k.startswith("params.") and pd.notna(v)
            }
            tags = {
                k.replace("tags.", ""): v
                for k, v in run.items()
                if k.startswith("tags.") and pd.notna(v)
            }
            run_id = run.get("run_id", "unknown")
            timestamp = run.get("start_time", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

        return ModelMetrics(
            run_id=run_id,
            timestamp=timestamp,
            metrics=metrics,
            hyperparams=params,
            tags=tags,
        )


class CustomConnector(MetricsConnector):
    """Custom connector using user-provided functions."""

    def __init__(
        self,
        fetch_fn: Callable[[], list[dict[str, Any]]],
        name: str = "custom",
    ):
        """Initialize custom connector.

        Args:
            fetch_fn: Function that returns list of metric dictionaries
            name: Connector name
        """
        self._fetch_fn = fetch_fn
        self._name = name

    @property
    def connector_name(self) -> str:
        return self._name

    def get_metrics(self, run_id: str | None = None) -> list[ModelMetrics]:
        """Get metrics using custom function."""
        data = self._fetch_fn()
        metrics_list = []

        for item in data:
            if run_id and item.get("run_id") != run_id:
                continue

            metrics_list.append(ModelMetrics(
                run_id=item.get("run_id", f"run_{len(metrics_list)}"),
                timestamp=item.get("timestamp", datetime.now()),
                metrics=item.get("metrics", {}),
                dataset_hash=item.get("dataset_hash"),
                model_type=item.get("model_type"),
                hyperparams=item.get("hyperparams", {}),
                tags=item.get("tags", {}),
            ))

        return metrics_list

    def get_latest_metrics(self, n: int = 10) -> list[ModelMetrics]:
        """Get latest N metrics."""
        all_metrics = self.get_metrics()
        sorted_metrics = sorted(all_metrics, key=lambda m: m.timestamp, reverse=True)
        return sorted_metrics[:n]


class InMemoryConnector(MetricsConnector):
    """In-memory connector for testing and simple use cases."""

    def __init__(self):
        """Initialize in-memory connector."""
        self._metrics: list[ModelMetrics] = []

    @property
    def connector_name(self) -> str:
        return "in_memory"

    def add_metrics(self, metrics: ModelMetrics) -> None:
        """Add metrics to storage."""
        self._metrics.append(metrics)

    def get_metrics(self, run_id: str | None = None) -> list[ModelMetrics]:
        """Get metrics from memory."""
        if run_id:
            return [m for m in self._metrics if m.run_id == run_id]
        return self._metrics.copy()

    def get_latest_metrics(self, n: int = 10) -> list[ModelMetrics]:
        """Get latest N metrics."""
        sorted_metrics = sorted(self._metrics, key=lambda m: m.timestamp, reverse=True)
        return sorted_metrics[:n]

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._metrics.clear()


class FeedbackLoop:
    """Continuous learning loop connecting model performance to data quality.

    Tracks model performance over time, detects regressions, correlates them
    with data quality issues, and generates prescriptions for data improvement.

    Example:
        >>> loop = FeedbackLoop(data=df, label_column="label")
        >>> loop.record_metrics({"accuracy": 0.85, "f1": 0.82}, run_id="run_1")
        >>> loop.record_metrics({"accuracy": 0.78, "f1": 0.75}, run_id="run_2")
        >>>
        >>> analysis = loop.analyze()
        >>> print(analysis.summary)
        >>>
        >>> for prescription in analysis.prescriptions:
        ...     print(f"Action: {prescription.action}")
        ...     print(f"Expected improvement: {prescription.expected_improvement}")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        connector: MetricsConnector | None = None,
        predictions: np.ndarray | None = None,
        auto_analyze_quality: bool = True,
    ):
        """Initialize the feedback loop.

        Args:
            data: Training data DataFrame
            label_column: Column containing labels
            connector: Metrics connector (default: InMemoryConnector)
            predictions: Model predictions for correlation analysis
            auto_analyze_quality: Automatically analyze data quality
        """
        self.data = data.copy()
        self.label_column = label_column
        self.connector = connector or InMemoryConnector()
        self.predictions = predictions

        self._quality_report: QualityReport | None = None
        self._metrics_history: list[ModelMetrics] = []
        self._analysis_history: list[FeedbackAnalysis] = []

        if auto_analyze_quality:
            cleaner = DatasetCleaner(data=data, label_column=label_column)
            self._quality_report = cleaner.analyze()

    def record_metrics(
        self,
        metrics: dict[str, float],
        run_id: str | None = None,
        model_type: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelMetrics:
        """Record model performance metrics.

        Args:
            metrics: Dictionary of metric name to value
            run_id: Optional run identifier
            model_type: Optional model type description
            hyperparams: Optional hyperparameters
            tags: Optional tags

        Returns:
            Recorded ModelMetrics object
        """
        # Generate dataset hash for tracking
        dataset_hash = self._compute_dataset_hash()

        model_metrics = ModelMetrics(
            run_id=run_id or f"run_{len(self._metrics_history)}",
            timestamp=datetime.now(),
            metrics=metrics,
            dataset_hash=dataset_hash,
            model_type=model_type,
            hyperparams=hyperparams or {},
            tags=tags or {},
        )

        self._metrics_history.append(model_metrics)

        # Also store in connector if it supports it
        if isinstance(self.connector, InMemoryConnector):
            self.connector.add_metrics(model_metrics)

        return model_metrics

    def ingest_from_connector(self, run_id: str | None = None) -> list[ModelMetrics]:
        """Ingest metrics from the configured connector.

        Args:
            run_id: Optional specific run to fetch

        Returns:
            List of ingested metrics
        """
        metrics = self.connector.get_metrics(run_id)
        self._metrics_history.extend(metrics)
        return metrics

    def analyze(
        self,
        target_metric: str = "accuracy",
        baseline_window: int = 5,
    ) -> FeedbackAnalysis:
        """Analyze metrics and generate prescriptions.

        Args:
            target_metric: Primary metric to optimize
            baseline_window: Number of recent runs for baseline

        Returns:
            FeedbackAnalysis with findings and prescriptions
        """
        # Detect performance drops
        drops = self._detect_performance_drops(target_metric, baseline_window)

        # Analyze correlations
        correlations = self._analyze_correlations(target_metric)

        # Generate prescriptions
        prescriptions = self._generate_prescriptions(drops, correlations, target_metric)

        # Create summary
        summary = self._generate_summary(drops, correlations, prescriptions)

        analysis = FeedbackAnalysis(
            timestamp=datetime.now(),
            n_metrics_analyzed=len(self._metrics_history),
            performance_drops=drops,
            correlations=correlations,
            prescriptions=prescriptions,
            summary=summary,
        )

        self._analysis_history.append(analysis)
        return analysis

    def get_prescriptions(
        self,
        max_prescriptions: int = 5,
        min_confidence: float = 0.5,
    ) -> list[DataPrescription]:
        """Get data improvement prescriptions.

        Args:
            max_prescriptions: Maximum number of prescriptions
            min_confidence: Minimum confidence threshold

        Returns:
            List of DataPrescription objects
        """
        if not self._analysis_history:
            self.analyze()

        latest = self._analysis_history[-1]
        filtered = [p for p in latest.prescriptions if p.confidence >= min_confidence]
        sorted_prescriptions = sorted(filtered, key=lambda p: p.priority)
        return sorted_prescriptions[:max_prescriptions]

    def apply_prescription(
        self,
        prescription: DataPrescription,
        dry_run: bool = True,
    ) -> pd.DataFrame:
        """Apply a prescription to the data.

        Args:
            prescription: Prescription to apply
            dry_run: If True, return preview without modifying data

        Returns:
            Modified DataFrame (or preview if dry_run)
        """
        data = self.data.copy()

        if prescription.action == ActionType.REMOVE_LABEL_ERRORS:
            data = data.drop(index=prescription.target_samples, errors="ignore")

        elif prescription.action == ActionType.REMOVE_DUPLICATES:
            data = data.drop(index=prescription.target_samples, errors="ignore")

        elif prescription.action == ActionType.REMOVE_OUTLIERS:
            data = data.drop(index=prescription.target_samples, errors="ignore")

        elif prescription.action == ActionType.REBALANCE_CLASSES:
            # This would involve upsampling/downsampling
            pass

        data = data.reset_index(drop=True)

        if not dry_run:
            self.data = data

        return data

    def _compute_dataset_hash(self) -> str:
        """Compute hash of the dataset for tracking."""
        content = str(len(self.data)) + str(list(self.data.columns))
        if len(self.data) > 0:
            sample = self.data.head(100).to_json()
            content += sample
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _detect_performance_drops(
        self,
        target_metric: str,
        baseline_window: int,
    ) -> list[PerformanceDrop]:
        """Detect significant performance drops."""
        drops = []

        if len(self._metrics_history) < 2:
            return drops

        # Sort by timestamp
        sorted_metrics = sorted(self._metrics_history, key=lambda m: m.timestamp)

        # Get baseline (average of last N runs before current)
        if len(sorted_metrics) <= baseline_window:
            baseline_runs = sorted_metrics[:-1]
        else:
            baseline_runs = sorted_metrics[-(baseline_window + 1):-1]

        current = sorted_metrics[-1]

        for metric_name in current.metrics:
            current_value = current.metrics[metric_name]
            baseline_values = [
                m.metrics.get(metric_name)
                for m in baseline_runs
                if m.metrics.get(metric_name) is not None
            ]

            if not baseline_values:
                continue

            baseline_avg = np.mean(baseline_values)
            baseline_std = np.std(baseline_values) if len(baseline_values) > 1 else 0.01

            # Calculate drop percentage (for metrics where higher is better)
            if baseline_avg > 0:
                drop_pct = (baseline_avg - current_value) / baseline_avg * 100
            else:
                drop_pct = 0

            # Calculate significance (z-score based)
            if baseline_std > 0:
                z_score = abs(current_value - baseline_avg) / baseline_std
                from scipy import stats
                significance = 1 - stats.norm.sf(z_score) * 2
            else:
                significance = 1.0 if drop_pct > 5 else 0.0

            if drop_pct > 1:  # At least 1% drop
                drops.append(PerformanceDrop(
                    metric_name=metric_name,
                    previous_value=baseline_avg,
                    current_value=current_value,
                    drop_percentage=drop_pct,
                    significance=significance,
                ))

        return drops

    def _analyze_correlations(
        self,
        target_metric: str,
    ) -> list[CorrelationResult]:
        """Analyze correlations between data issues and performance."""
        correlations = []

        if self._quality_report is None:
            return correlations

        # Get error predictions if available
        if self.predictions is None or self.label_column is None:
            return correlations

        labels = self.data[self.label_column].values
        correct = (self.predictions == labels).astype(int)

        # Check correlation with label errors
        if hasattr(self._quality_report, "label_errors"):
            error_df = self._quality_report.label_errors()
            if len(error_df) > 0:
                is_label_error = np.zeros(len(self.data))
                error_indices = error_df["index"].values if "index" in error_df.columns else error_df.index.values
                is_label_error[error_indices] = 1

                if np.sum(is_label_error) > 0:
                    corr, p_value = self._compute_correlation(is_label_error, 1 - correct)
                    effect_size = np.mean(correct[is_label_error == 0]) - np.mean(correct[is_label_error == 1])

                    correlations.append(CorrelationResult(
                        issue_type="label_errors",
                        metric_name=target_metric,
                        correlation=corr,
                        p_value=p_value,
                        sample_overlap=int(np.sum((is_label_error == 1) & (correct == 0))),
                        effect_size=effect_size,
                        confidence_interval=(effect_size - 0.1, effect_size + 0.1),
                    ))

        # Check correlation with outliers
        if hasattr(self._quality_report, "outliers"):
            out_df = self._quality_report.outliers()
            if len(out_df) > 0:
                is_outlier = np.zeros(len(self.data))
                out_indices = out_df["index"].values if "index" in out_df.columns else out_df.index.values
                is_outlier[out_indices] = 1

                if np.sum(is_outlier) > 0:
                    corr, p_value = self._compute_correlation(is_outlier, 1 - correct)
                    effect_size = np.mean(correct[is_outlier == 0]) - np.mean(correct[is_outlier == 1])

                    correlations.append(CorrelationResult(
                        issue_type="outliers",
                        metric_name=target_metric,
                        correlation=corr,
                        p_value=p_value,
                        sample_overlap=int(np.sum((is_outlier == 1) & (correct == 0))),
                        effect_size=effect_size,
                        confidence_interval=(effect_size - 0.1, effect_size + 0.1),
                    ))

        return correlations

    def _compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        """Compute Pearson correlation with p-value."""
        try:
            from scipy import stats
            corr, p_value = stats.pearsonr(x, y)
            return float(corr), float(p_value)
        except Exception:
            # Fallback to simple correlation
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                return float(corr), 0.5
            return 0.0, 1.0

    def _generate_prescriptions(
        self,
        drops: list[PerformanceDrop],
        correlations: list[CorrelationResult],
        target_metric: str,
    ) -> list[DataPrescription]:
        """Generate data improvement prescriptions."""
        prescriptions = []
        priority = 1

        if self._quality_report is None:
            return prescriptions

        # Prescription 1: Remove label errors if correlated
        label_corr = next((c for c in correlations if c.issue_type == "label_errors"), None)
        if label_corr and label_corr.is_significant:
            if hasattr(self._quality_report, "label_errors"):
                error_df = self._quality_report.label_errors()
                # Focus on high-confidence errors
                if "confidence" in error_df.columns:
                    high_conf = error_df[error_df["confidence"] >= 0.8]
                    indices = list(high_conf["index"]) if "index" in high_conf.columns else list(high_conf.index)
                else:
                    indices = list(error_df.index)[:50]

                if indices:
                    prescriptions.append(DataPrescription(
                        action=ActionType.REMOVE_LABEL_ERRORS,
                        description=f"Remove {len(indices)} high-confidence label errors",
                        target_samples=indices,
                        expected_improvement=label_corr.effect_size * 0.5,
                        confidence=min(0.9, 1 - label_corr.p_value),
                        reasoning=f"Label errors correlate {label_corr.correlation:.2f} with prediction errors",
                        priority=priority,
                        estimated_effort="low",
                    ))
                    priority += 1

        # Prescription 2: Remove outliers if correlated
        outlier_corr = next((c for c in correlations if c.issue_type == "outliers"), None)
        if outlier_corr and outlier_corr.correlation > 0.2:
            if hasattr(self._quality_report, "outliers"):
                out_df = self._quality_report.outliers()
                indices = list(out_df["index"]) if "index" in out_df.columns else list(out_df.index)

                if indices:
                    prescriptions.append(DataPrescription(
                        action=ActionType.REMOVE_OUTLIERS,
                        description=f"Remove {len(indices)} outliers",
                        target_samples=indices,
                        expected_improvement=outlier_corr.effect_size * 0.3,
                        confidence=min(0.8, 1 - outlier_corr.p_value),
                        reasoning=f"Outliers correlate {outlier_corr.correlation:.2f} with prediction errors",
                        priority=priority,
                        estimated_effort="low",
                    ))
                    priority += 1

        # Prescription 3: Remove duplicates (always good practice)
        if hasattr(self._quality_report, "duplicates"):
            dup_df = self._quality_report.duplicates()
            if len(dup_df) > 0:
                indices = list(dup_df["index_2"]) if "index_2" in dup_df.columns else []
                if indices:
                    prescriptions.append(DataPrescription(
                        action=ActionType.REMOVE_DUPLICATES,
                        description=f"Remove {len(indices)} duplicate samples",
                        target_samples=indices,
                        expected_improvement=0.01,  # Small but reliable improvement
                        confidence=0.95,
                        reasoning="Duplicates can cause data leakage and overfitting",
                        priority=priority,
                        estimated_effort="low",
                    ))
                    priority += 1

        # Prescription 4: Rebalance if significant drops
        significant_drops = [d for d in drops if d.is_significant]
        if significant_drops and self.label_column:
            class_counts = self.data[self.label_column].value_counts()
            max_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else 1

            if max_ratio > 5:
                prescriptions.append(DataPrescription(
                    action=ActionType.REBALANCE_CLASSES,
                    description=f"Rebalance classes (current ratio: {max_ratio:.1f}:1)",
                    target_samples=[],  # Would be computed during execution
                    expected_improvement=0.03,
                    confidence=0.7,
                    reasoning=f"Class imbalance may be causing performance issues",
                    priority=priority,
                    estimated_effort="medium",
                ))

        return prescriptions

    def _generate_summary(
        self,
        drops: list[PerformanceDrop],
        correlations: list[CorrelationResult],
        prescriptions: list[DataPrescription],
    ) -> str:
        """Generate human-readable summary."""
        lines = ["Feedback Loop Analysis Summary", "=" * 40]

        # Performance drops
        if drops:
            sig_drops = [d for d in drops if d.is_significant]
            lines.append(f"\nPerformance Drops: {len(drops)} detected, {len(sig_drops)} significant")
            for d in sig_drops[:3]:
                lines.append(f"  - {d.metric_name}: {d.drop_percentage:.1f}% drop")
        else:
            lines.append("\nNo significant performance drops detected.")

        # Correlations
        if correlations:
            sig_corrs = [c for c in correlations if c.is_significant]
            lines.append(f"\nData Issue Correlations: {len(sig_corrs)} significant")
            for c in sig_corrs:
                lines.append(f"  - {c.issue_type} → {c.metric_name}: r={c.correlation:.2f}")
        else:
            lines.append("\nNo significant correlations found.")

        # Prescriptions
        if prescriptions:
            lines.append(f"\nPrescriptions: {len(prescriptions)} recommended")
            for p in prescriptions[:3]:
                lines.append(f"  {p.priority}. {p.description}")
                lines.append(f"     Expected improvement: {p.expected_improvement:.1%}")
        else:
            lines.append("\nNo immediate actions recommended.")

        return "\n".join(lines)

    @property
    def metrics_history(self) -> list[ModelMetrics]:
        """Get metrics history."""
        return self._metrics_history.copy()

    @property
    def analysis_history(self) -> list[FeedbackAnalysis]:
        """Get analysis history."""
        return self._analysis_history.copy()


def create_feedback_loop(
    data: pd.DataFrame,
    label_column: str | None = None,
    connector_type: str = "memory",
    **kwargs: Any,
) -> FeedbackLoop:
    """Create a feedback loop with the specified connector.

    Args:
        data: Training data DataFrame
        label_column: Column containing labels
        connector_type: Type of connector ("memory", "mlflow", "custom")
        **kwargs: Additional connector-specific arguments

    Returns:
        FeedbackLoop instance

    Example:
        >>> loop = create_feedback_loop(df, "label", connector_type="mlflow",
        ...                             tracking_uri="http://localhost:5000")
    """
    if connector_type == "memory":
        connector = InMemoryConnector()
    elif connector_type == "mlflow":
        connector = MLflowConnector(
            tracking_uri=kwargs.get("tracking_uri", "http://localhost:5000"),
            experiment_name=kwargs.get("experiment_name"),
        )
    elif connector_type == "custom":
        fetch_fn = kwargs.get("fetch_fn")
        if not fetch_fn:
            raise ConfigurationError("Custom connector requires 'fetch_fn' argument")
        connector = CustomConnector(fetch_fn, kwargs.get("name", "custom"))
    else:
        raise ConfigurationError(f"Unknown connector type: {connector_type}")

    return FeedbackLoop(
        data=data,
        label_column=label_column,
        connector=connector,
        predictions=kwargs.get("predictions"),
        auto_analyze_quality=kwargs.get("auto_analyze_quality", True),
    )


__all__ = [
    # Core classes
    "FeedbackLoop",
    "FeedbackAnalysis",
    "DataPrescription",
    "PerformanceDrop",
    "CorrelationResult",
    "ModelMetrics",
    # Enums
    "MetricType",
    "ActionType",
    # Connectors
    "MetricsConnector",
    "MLflowConnector",
    "CustomConnector",
    "InMemoryConnector",
    # Functions
    "create_feedback_loop",
]
