"""MLflow and Weights & Biases integration for Clean.

This module provides first-class integrations that log quality reports as
artifacts, track scores as metrics, and trigger analysis on data changes.

Example:
    >>> from clean.mlops import MLflowIntegration, WandbIntegration
    >>>
    >>> # MLflow integration
    >>> mlflow_int = MLflowIntegration(tracking_uri="http://localhost:5000")
    >>> mlflow_int.log_quality_report(report)
    >>>
    >>> # W&B integration
    >>> wandb_int = WandbIntegration(project="my-project")
    >>> wandb_int.log_quality_report(report)
    >>>
    >>> # Decorator for automatic quality tracking
    >>> @track_data_quality(backend="mlflow")
    >>> def train_model(data):
    ...     # your training code
    ...     pass
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.core.report import QualityReport
from clean.exceptions import CleanError, ConfigurationError, DependencyError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MLOpsBackend(ABC):
    """Abstract base class for MLOps integrations."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact file."""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        pass

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the run."""
        pass

    def log_quality_report(
        self,
        report: QualityReport,
        prefix: str = "data_quality",
        log_details: bool = True,
    ) -> None:
        """Log a quality report.

        Args:
            report: Quality report to log
            prefix: Metric prefix
            log_details: Whether to log detailed artifacts
        """
        # Helper to get issue count from result
        def get_issue_count(result: Any) -> int:
            if result is None:
                return 0
            if hasattr(result, 'issues'):
                return len(result.issues)
            return 0

        # Get quality score (could be object or float)
        score = report.quality_score
        if hasattr(score, 'overall'):
            score_value = score.overall
        elif hasattr(score, 'score'):
            score_value = score.score
        else:
            score_value = float(score)

        # Log main metrics
        metrics = {
            f"{prefix}/score": score_value,
            f"{prefix}/n_samples": report.dataset_info.n_samples,
            f"{prefix}/label_errors": get_issue_count(report.label_errors_result),
            f"{prefix}/duplicates": get_issue_count(report.duplicates_result),
            f"{prefix}/outliers": get_issue_count(report.outliers_result),
        }

        self.log_metrics(metrics)

        # Log detailed report as artifact
        if log_details:
            with tempfile.TemporaryDirectory() as tmpdir:
                # JSON report
                report_path = Path(tmpdir) / "quality_report.json"
                with open(report_path, "w") as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)
                self.log_artifact(report_path, f"{prefix}/reports")

                # Text summary
                summary_path = Path(tmpdir) / "quality_summary.txt"
                with open(summary_path, "w") as f:
                    f.write(report.summary())
                self.log_artifact(summary_path, f"{prefix}/reports")

        # Set tags
        self.set_tags({
            f"{prefix}_score": str(int(score_value)),
            f"{prefix}_timestamp": datetime.now().isoformat(),
        })

        logger.info(f"Logged quality report with score {score_value:.1f}")


class MLflowIntegration(MLOpsBackend):
    """MLflow integration for Clean.

    Example:
        >>> from clean.mlops import MLflowIntegration
        >>>
        >>> # Initialize
        >>> mlflow_int = MLflowIntegration(tracking_uri="http://localhost:5000")
        >>>
        >>> # Start a run
        >>> with mlflow_int.start_run(run_name="data_quality_check"):
        ...     report = cleaner.analyze()
        ...     mlflow_int.log_quality_report(report)
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
    ):
        """Initialize MLflow integration.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Experiment name
            run_id: Existing run ID to use
        """
        try:
            import mlflow
        except ImportError as e:
            raise DependencyError("mlflow", "pip install mlflow", "MLflow integration") from e

        self._mlflow = mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        self.run_id = run_id

    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> Any:
        """Start an MLflow run.

        Args:
            run_name: Name for the run
            nested: Whether this is a nested run
            tags: Tags to set on the run

        Returns:
            MLflow run context manager
        """
        return self._mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=tags,
            run_id=self.run_id,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and (value != value)  # NaN check
            ):
                self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log artifact to MLflow."""
        self._mlflow.log_artifact(str(local_path), artifact_path)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        self._mlflow.log_params(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on MLflow run."""
        self._mlflow.set_tags(tags)

    def log_dataset(
        self,
        data: pd.DataFrame,
        name: str = "training_data",
        context: str = "train",
    ) -> str:
        """Log dataset with hash for tracking.

        Args:
            data: DataFrame to log
            name: Dataset name
            context: Dataset context (train, test, validation)

        Returns:
            Dataset hash
        """
        # Calculate hash
        data_hash = hashlib.md5(  # noqa: S324
            pd.util.hash_pandas_object(data, index=True).values
        ).hexdigest()[:12]

        # Log dataset info
        self._mlflow.log_params({
            f"dataset_{name}_hash": data_hash,
            f"dataset_{name}_rows": len(data),
            f"dataset_{name}_cols": len(data.columns),
        })

        self.set_tags({
            f"dataset_{name}": data_hash,
            f"dataset_{name}_context": context,
        })

        return data_hash


class WandbIntegration(MLOpsBackend):
    """Weights & Biases integration for Clean.

    Example:
        >>> from clean.mlops import WandbIntegration
        >>>
        >>> # Initialize
        >>> wandb_int = WandbIntegration(project="my-project")
        >>>
        >>> # Start a run
        >>> with wandb_int.start_run(name="data_quality_check"):
        ...     report = cleaner.analyze()
        ...     wandb_int.log_quality_report(report)
    """

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_id: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize W&B integration.

        Args:
            project: W&B project name
            entity: W&B entity (user or team)
            run_id: Existing run ID to resume
            api_key: W&B API key (or set WANDB_API_KEY)
        """
        try:
            import wandb
        except ImportError as e:
            raise DependencyError("wandb", "pip install wandb", "W&B integration") from e

        self._wandb = wandb
        self.project = project
        self.entity = entity
        self.run_id = run_id

        if api_key:
            wandb.login(key=api_key)

        self._run: Any = None

    def start_run(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """Start a W&B run.

        Args:
            name: Run name
            config: Configuration dict
            tags: Tags for the run

        Returns:
            W&B run context
        """
        self._run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags,
            id=self.run_id,
            resume="allow" if self.run_id else None,
        )
        return self._run

    def _ensure_run(self) -> None:
        """Ensure a run is active."""
        if self._run is None:
            self._run = self._wandb.init(
                project=self.project,
                entity=self.entity,
            )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to W&B."""
        self._ensure_run()
        log_dict = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if step is not None:
            log_dict["step"] = step
        self._wandb.log(log_dict)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log artifact to W&B."""
        self._ensure_run()
        artifact = self._wandb.Artifact(
            name=artifact_path or Path(local_path).stem,
            type="data_quality",
        )
        artifact.add_file(str(local_path))
        self._wandb.log_artifact(artifact)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to W&B config."""
        self._ensure_run()
        self._wandb.config.update(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on W&B run."""
        self._ensure_run()
        if self._run:
            for key, value in tags.items():
                self._run.tags = list(self._run.tags or []) + [f"{key}:{value}"]

    def log_table(
        self,
        data: pd.DataFrame,
        name: str = "quality_issues",
    ) -> None:
        """Log a DataFrame as a W&B Table.

        Args:
            data: DataFrame to log
            name: Table name
        """
        self._ensure_run()
        table = self._wandb.Table(dataframe=data)
        self._wandb.log({name: table})

    def log_quality_report(
        self,
        report: QualityReport,
        prefix: str = "data_quality",
        log_details: bool = True,
    ) -> None:
        """Log quality report with W&B visualizations."""
        # Call parent implementation
        super().log_quality_report(report, prefix, log_details)

        # Add W&B-specific visualizations
        self._ensure_run()

        # Log issue distribution as bar chart
        if report.issue_counts:
            data = [[k.value, v] for k, v in report.issue_counts.items()]
            table = self._wandb.Table(data=data, columns=["Issue Type", "Count"])
            self._wandb.log({
                f"{prefix}/issue_distribution": self._wandb.plot.bar(
                    table, "Issue Type", "Count", title="Issue Distribution"
                )
            })

        # Log class distribution if available
        if report.class_distribution:
            data = [[k, v] for k, v in report.class_distribution.items()]
            table = self._wandb.Table(data=data, columns=["Class", "Count"])
            self._wandb.log({
                f"{prefix}/class_distribution": self._wandb.plot.bar(
                    table, "Class", "Count", title="Class Distribution"
                )
            })


@dataclass
class QualityCallback:
    """Callback for quality monitoring during training."""

    backend: MLOpsBackend
    data: pd.DataFrame
    label_column: str
    check_interval: int = 1  # Check every N epochs
    min_quality_score: float = 70.0  # Minimum acceptable score
    _check_count: int = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Called at the end of each epoch.

        Returns:
            True if training should continue, False to stop
        """
        self._check_count += 1

        if self._check_count % self.check_interval != 0:
            return True

        # Run quality analysis
        cleaner = DatasetCleaner(data=self.data, label_column=self.label_column)
        report = cleaner.analyze()

        # Log to backend
        self.backend.log_quality_report(report, prefix=f"epoch_{epoch}")

        # Get quality score value
        score = report.quality_score
        if hasattr(score, 'overall'):
            score_value = score.overall
        elif hasattr(score, 'score'):
            score_value = score.score
        else:
            score_value = float(score)

        # Check if quality is acceptable
        if score_value < self.min_quality_score:
            logger.warning(
                f"Quality score {score_value:.1f} below threshold "
                f"{self.min_quality_score}. Consider stopping training."
            )
            return False

        return True


def track_data_quality(
    backend: str = "mlflow",
    auto_analyze: bool = True,
    min_score: float = 0.0,
    **backend_kwargs: Any,
) -> Callable:
    """Decorator for automatic data quality tracking.

    Args:
        backend: Backend to use ("mlflow" or "wandb")
        auto_analyze: Automatically analyze data before function
        min_score: Minimum quality score required (raises if below)
        **backend_kwargs: Arguments passed to backend constructor

    Returns:
        Decorated function

    Example:
        >>> @track_data_quality(backend="mlflow")
        >>> def train_model(data, label_column):
        ...     # Training code
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get data from args/kwargs
            data = kwargs.get("data") or (args[0] if args else None)
            label_column = kwargs.get("label_column")

            if data is None or not isinstance(data, pd.DataFrame):
                logger.warning("No DataFrame found, skipping quality tracking")
                return func(*args, **kwargs)

            # Create backend
            if backend == "mlflow":
                mlops = MLflowIntegration(**backend_kwargs)
            elif backend == "wandb":
                mlops = WandbIntegration(**backend_kwargs)
            else:
                raise ConfigurationError(f"Unknown backend: {backend}")

            # Auto-analyze if enabled
            if auto_analyze and label_column:
                cleaner = DatasetCleaner(data=data, label_column=label_column)
                report = cleaner.analyze()
                mlops.log_quality_report(report, prefix="pre_training")

                if min_score > 0 and report.quality_score < min_score:
                    raise CleanError(
                        f"Data quality score {report.quality_score:.1f} below "
                        f"minimum threshold {min_score}"
                    )

            # Run the actual function
            return func(*args, **kwargs)

        return wrapper
    return decorator


def create_mlflow_integration(
    tracking_uri: str | None = None,
    experiment_name: str = "data_quality",
) -> MLflowIntegration:
    """Create MLflow integration.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name

    Returns:
        Configured MLflow integration
    """
    return MLflowIntegration(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


def create_wandb_integration(
    project: str = "data-quality",
    entity: str | None = None,
) -> WandbIntegration:
    """Create W&B integration.

    Args:
        project: W&B project name
        entity: W&B entity

    Returns:
        Configured W&B integration
    """
    return WandbIntegration(project=project, entity=entity)


__all__ = [
    "MLOpsBackend",
    "MLflowIntegration",
    "WandbIntegration",
    "QualityCallback",
    "track_data_quality",
    "create_mlflow_integration",
    "create_wandb_integration",
]
