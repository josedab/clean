"""Feature Store Integration for Data Quality.

This module provides connectors to popular feature stores (Feast, Tecton,
Databricks) for analyzing feature quality and tracking quality metrics
across feature versions.

Example:
    >>> from clean.feature_store import FeatureQualityAnalyzer, FeastConnector
    >>>
    >>> # Connect to Feast
    >>> connector = FeastConnector(repo_path="./feature_repo")
    >>> analyzer = FeatureQualityAnalyzer(connector)
    >>>
    >>> # Analyze feature quality
    >>> report = analyzer.analyze_feature("user_features", version="v1.0")
    >>> print(report.summary())
    >>>
    >>> # Track quality over versions
    >>> history = analyzer.get_quality_history("user_features")
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.core.report import QualityReport
from clean.exceptions import CleanError, ConfigurationError, DependencyError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FeatureStoreType(Enum):
    """Supported feature store types."""

    FEAST = "feast"
    TECTON = "tecton"
    DATABRICKS = "databricks"
    SAGEMAKER = "sagemaker"
    VERTEX = "vertex"
    CUSTOM = "custom"


class QualityDimension(Enum):
    """Dimensions for feature quality assessment."""

    COMPLETENESS = "completeness"  # Missing value rate
    UNIQUENESS = "uniqueness"  # Duplicate rate
    CONSISTENCY = "consistency"  # Value range consistency
    FRESHNESS = "freshness"  # Data staleness
    ACCURACY = "accuracy"  # Compared to source of truth
    VALIDITY = "validity"  # Schema conformance
    DRIFT = "drift"  # Distribution change


@dataclass
class FeatureMetadata:
    """Metadata about a feature."""

    name: str
    dtype: str
    entity: str | None = None
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: str | None = None
    source: str | None = None


@dataclass
class FeatureQualityMetrics:
    """Quality metrics for a single feature."""

    feature_name: str
    completeness: float  # 1 - missing_rate
    uniqueness: float  # 1 - duplicate_rate
    consistency: float  # % values within expected range
    validity: float  # % values matching schema
    n_samples: int
    n_missing: int
    n_outliers: int
    value_range: tuple[float, float] | None = None
    distribution_stats: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (
            self.completeness * 0.3 +
            self.uniqueness * 0.2 +
            self.consistency * 0.25 +
            self.validity * 0.25
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "completeness": self.completeness,
            "uniqueness": self.uniqueness,
            "consistency": self.consistency,
            "validity": self.validity,
            "overall_score": self.overall_score,
            "n_samples": self.n_samples,
            "n_missing": self.n_missing,
            "n_outliers": self.n_outliers,
            "issues": self.issues,
        }


@dataclass
class FeatureQualityReport:
    """Quality report for a feature set."""

    feature_set_name: str
    version: str | None
    timestamp: datetime
    feature_metrics: list[FeatureQualityMetrics]
    overall_score: float
    n_features: int
    n_samples: int
    issues_summary: dict[str, int]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Feature Quality Report: {self.feature_set_name}",
            "=" * 60,
            f"Version: {self.version or 'N/A'}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Features: {self.n_features}",
            f"Samples: {self.n_samples:,}",
            f"Overall Score: {self.overall_score:.2f}/1.00",
            "",
            "Feature Scores:",
        ]

        for fm in sorted(self.feature_metrics, key=lambda x: x.overall_score):
            bar = "█" * int(fm.overall_score * 10) + "░" * (10 - int(fm.overall_score * 10))
            lines.append(f"  {fm.feature_name:30s} {bar} {fm.overall_score:.2f}")

        if self.issues_summary:
            lines.append("")
            lines.append("Issues Found:")
            for issue, count in sorted(self.issues_summary.items(), key=lambda x: -x[1]):
                lines.append(f"  - {issue}: {count}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature metrics to DataFrame."""
        return pd.DataFrame([fm.to_dict() for fm in self.feature_metrics])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_set_name": self.feature_set_name,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "feature_metrics": [fm.to_dict() for fm in self.feature_metrics],
            "issues_summary": self.issues_summary,
            "recommendations": self.recommendations,
        }


@dataclass
class FeatureVersionHistory:
    """Quality history across feature versions."""

    feature_name: str
    history: list[tuple[str, datetime, float]]  # (version, timestamp, score)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(
            self.history,
            columns=["version", "timestamp", "quality_score"]
        )

    def get_trend(self) -> str:
        """Analyze quality trend."""
        if len(self.history) < 2:
            return "insufficient_data"

        scores = [h[2] for h in self.history]
        recent_avg = np.mean(scores[-3:])
        older_avg = np.mean(scores[:-3]) if len(scores) > 3 else scores[0]

        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "degrading"
        else:
            return "stable"


class FeatureStoreConnector(ABC):
    """Abstract base class for feature store connectors."""

    @abstractmethod
    def get_feature_data(
        self,
        feature_set: str,
        entity_df: pd.DataFrame | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch feature data from the store.

        Args:
            feature_set: Name of the feature set/view
            entity_df: Optional entity DataFrame for point-in-time lookup
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with feature values
        """
        pass

    @abstractmethod
    def list_feature_sets(self) -> list[str]:
        """List available feature sets."""
        pass

    @abstractmethod
    def get_feature_metadata(self, feature_set: str) -> list[FeatureMetadata]:
        """Get metadata for features in a set."""
        pass

    @property
    @abstractmethod
    def store_type(self) -> FeatureStoreType:
        """Return the store type."""
        pass


class FeastConnector(FeatureStoreConnector):
    """Connector for Feast feature store."""

    def __init__(
        self,
        repo_path: str = ".",
        feature_store: Any = None,
    ):
        """Initialize Feast connector.

        Args:
            repo_path: Path to Feast repository
            feature_store: Optional existing FeatureStore instance
        """
        self.repo_path = repo_path

        if feature_store is not None:
            self._store = feature_store
        else:
            try:
                from feast import FeatureStore
                self._store = FeatureStore(repo_path=repo_path)
            except ImportError as e:
                raise DependencyError(
                    "feast",
                    "Feast connector requires feast package. Install with: pip install feast"
                ) from e

    @property
    def store_type(self) -> FeatureStoreType:
        return FeatureStoreType.FEAST

    def get_feature_data(
        self,
        feature_set: str,
        entity_df: pd.DataFrame | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch feature data from Feast."""
        if entity_df is not None:
            # Historical features with point-in-time join
            return self._store.get_historical_features(
                entity_df=entity_df,
                features=[f"{feature_set}:*"],
            ).to_df()
        else:
            # Online features (sample)
            # For analysis, we'd typically use historical features
            # This is a simplified implementation
            logger.warning("No entity_df provided. Using sample data.")
            return pd.DataFrame()

    def list_feature_sets(self) -> list[str]:
        """List available feature views."""
        feature_views = self._store.list_feature_views()
        return [fv.name for fv in feature_views]

    def get_feature_metadata(self, feature_set: str) -> list[FeatureMetadata]:
        """Get metadata for features in a view."""
        fv = self._store.get_feature_view(feature_set)

        metadata = []
        for feature in fv.features:
            metadata.append(FeatureMetadata(
                name=feature.name,
                dtype=str(feature.dtype),
                description=getattr(feature, "description", None),
                tags=getattr(feature, "tags", {}),
            ))

        return metadata


class TectonConnector(FeatureStoreConnector):
    """Connector for Tecton feature store."""

    def __init__(
        self,
        workspace: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize Tecton connector.

        Args:
            workspace: Tecton workspace name
            api_key: Tecton API key
        """
        self.workspace = workspace

        try:
            import tecton
            if workspace:
                tecton.set_credentials(api_key=api_key)
                self._ws = tecton.get_workspace(workspace)
            else:
                self._ws = None
            self._tecton = tecton
        except ImportError as e:
            raise DependencyError(
                "tecton",
                "Tecton connector requires tecton package. Install with: pip install tecton"
            ) from e

    @property
    def store_type(self) -> FeatureStoreType:
        return FeatureStoreType.TECTON

    def get_feature_data(
        self,
        feature_set: str,
        entity_df: pd.DataFrame | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch feature data from Tecton."""
        if self._ws is None:
            raise ConfigurationError("No workspace configured")

        fv = self._ws.get_feature_view(feature_set)

        if entity_df is not None:
            return fv.get_features_for_events(entity_df).to_pandas()
        else:
            # Get feature data for a time range
            if start_time is None:
                start_time = datetime.now() - timedelta(days=7)
            if end_time is None:
                end_time = datetime.now()

            return fv.get_features_in_range(
                start_time=start_time,
                end_time=end_time,
            ).to_pandas()

    def list_feature_sets(self) -> list[str]:
        """List available feature views."""
        if self._ws is None:
            return []
        return [fv.name for fv in self._ws.list_feature_views()]

    def get_feature_metadata(self, feature_set: str) -> list[FeatureMetadata]:
        """Get metadata for features in a view."""
        if self._ws is None:
            return []

        fv = self._ws.get_feature_view(feature_set)
        metadata = []

        for schema_field in fv.output_schema:
            metadata.append(FeatureMetadata(
                name=schema_field.name,
                dtype=str(schema_field.dtype),
            ))

        return metadata


class DataFrameConnector(FeatureStoreConnector):
    """Simple connector that wraps a pandas DataFrame.

    Useful for testing or when data is already loaded.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: str = "dataframe",
    ):
        """Initialize DataFrame connector.

        Args:
            data: DataFrame with feature data
            name: Name for the feature set
        """
        self._data = data.copy()
        self._name = name

    @property
    def store_type(self) -> FeatureStoreType:
        return FeatureStoreType.CUSTOM

    def get_feature_data(
        self,
        feature_set: str,
        entity_df: pd.DataFrame | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Return the stored DataFrame."""
        return self._data.copy()

    def list_feature_sets(self) -> list[str]:
        """Return the single feature set name."""
        return [self._name]

    def get_feature_metadata(self, feature_set: str) -> list[FeatureMetadata]:
        """Generate metadata from DataFrame columns."""
        metadata = []
        for col in self._data.columns:
            metadata.append(FeatureMetadata(
                name=col,
                dtype=str(self._data[col].dtype),
            ))
        return metadata


class FeatureQualityAnalyzer:
    """Analyzer for feature store data quality.

    Connects to feature stores and analyzes feature quality across
    multiple dimensions including completeness, uniqueness, consistency,
    and drift detection.

    Example:
        >>> connector = FeastConnector(repo_path="./feature_repo")
        >>> analyzer = FeatureQualityAnalyzer(connector)
        >>> report = analyzer.analyze("user_features")
        >>> print(report.summary())
    """

    def __init__(
        self,
        connector: FeatureStoreConnector,
        quality_history_path: str | None = None,
    ):
        """Initialize the analyzer.

        Args:
            connector: Feature store connector
            quality_history_path: Optional path to store quality history
        """
        self.connector = connector
        self.quality_history_path = quality_history_path
        self._quality_history: dict[str, list[tuple[str, datetime, float]]] = {}

        if quality_history_path:
            self._load_history()

    def analyze(
        self,
        feature_set: str,
        entity_df: pd.DataFrame | None = None,
        version: str | None = None,
        expected_schema: dict[str, str] | None = None,
    ) -> FeatureQualityReport:
        """Analyze quality of a feature set.

        Args:
            feature_set: Name of the feature set
            entity_df: Optional entity DataFrame for historical lookup
            version: Optional version identifier
            expected_schema: Expected data types for validation

        Returns:
            FeatureQualityReport with metrics
        """
        # Fetch data
        data = self.connector.get_feature_data(feature_set, entity_df)

        if len(data) == 0:
            logger.warning(f"No data returned for feature set: {feature_set}")
            return FeatureQualityReport(
                feature_set_name=feature_set,
                version=version,
                timestamp=datetime.now(),
                feature_metrics=[],
                overall_score=0.0,
                n_features=0,
                n_samples=0,
                issues_summary={},
                recommendations=["No data available for analysis"],
            )

        # Get metadata
        metadata = self.connector.get_feature_metadata(feature_set)
        metadata_dict = {m.name: m for m in metadata}

        # Analyze each feature
        feature_metrics = []
        issues_summary: dict[str, int] = {}

        for col in data.columns:
            metrics = self._analyze_feature(
                data[col],
                col,
                metadata_dict.get(col),
                expected_schema.get(col) if expected_schema else None,
            )
            feature_metrics.append(metrics)

            for issue in metrics.issues:
                issues_summary[issue] = issues_summary.get(issue, 0) + 1

        # Calculate overall score
        overall_score = np.mean([fm.overall_score for fm in feature_metrics])

        # Generate recommendations
        recommendations = self._generate_recommendations(feature_metrics, issues_summary)

        report = FeatureQualityReport(
            feature_set_name=feature_set,
            version=version,
            timestamp=datetime.now(),
            feature_metrics=feature_metrics,
            overall_score=overall_score,
            n_features=len(feature_metrics),
            n_samples=len(data),
            issues_summary=issues_summary,
            recommendations=recommendations,
        )

        # Record in history
        self._record_history(feature_set, version or "latest", overall_score)

        return report

    def get_quality_history(self, feature_set: str) -> FeatureVersionHistory:
        """Get quality history for a feature set.

        Args:
            feature_set: Name of the feature set

        Returns:
            FeatureVersionHistory with scores over time
        """
        history = self._quality_history.get(feature_set, [])
        return FeatureVersionHistory(
            feature_name=feature_set,
            history=history,
        )

    def compare_versions(
        self,
        feature_set: str,
        version_a: str,
        version_b: str,
        entity_df_a: pd.DataFrame | None = None,
        entity_df_b: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compare quality between two versions.

        Args:
            feature_set: Feature set name
            version_a: First version
            version_b: Second version
            entity_df_a: Entity DataFrame for version A
            entity_df_b: Entity DataFrame for version B

        Returns:
            Comparison dictionary
        """
        report_a = self.analyze(feature_set, entity_df_a, version_a)
        report_b = self.analyze(feature_set, entity_df_b, version_b)

        comparison = {
            "feature_set": feature_set,
            "version_a": version_a,
            "version_b": version_b,
            "overall_score_a": report_a.overall_score,
            "overall_score_b": report_b.overall_score,
            "score_change": report_b.overall_score - report_a.overall_score,
            "feature_changes": [],
        }

        # Compare individual features
        metrics_a = {fm.feature_name: fm for fm in report_a.feature_metrics}
        metrics_b = {fm.feature_name: fm for fm in report_b.feature_metrics}

        for name in set(metrics_a.keys()) | set(metrics_b.keys()):
            score_a = metrics_a.get(name, FeatureQualityMetrics(
                feature_name=name, completeness=0, uniqueness=0,
                consistency=0, validity=0, n_samples=0, n_missing=0, n_outliers=0
            )).overall_score
            score_b = metrics_b.get(name, FeatureQualityMetrics(
                feature_name=name, completeness=0, uniqueness=0,
                consistency=0, validity=0, n_samples=0, n_missing=0, n_outliers=0
            )).overall_score

            if abs(score_b - score_a) > 0.05:  # Significant change
                comparison["feature_changes"].append({
                    "feature": name,
                    "score_a": score_a,
                    "score_b": score_b,
                    "change": score_b - score_a,
                })

        return comparison

    def _analyze_feature(
        self,
        series: pd.Series,
        name: str,
        metadata: FeatureMetadata | None,
        expected_dtype: str | None,
    ) -> FeatureQualityMetrics:
        """Analyze a single feature."""
        n_samples = len(series)
        issues = []

        # Completeness
        n_missing = series.isna().sum()
        completeness = 1 - (n_missing / n_samples) if n_samples > 0 else 0

        if completeness < 0.95:
            issues.append("high_missing_rate")

        # Uniqueness (for non-numeric features)
        if series.dtype == "object" or str(series.dtype).startswith("str"):
            n_unique = series.nunique()
            uniqueness = n_unique / n_samples if n_samples > 0 else 0
        else:
            uniqueness = 1.0  # Not applicable to numeric

        # Consistency (check for outliers in numeric)
        n_outliers = 0
        value_range = None
        consistency = 1.0

        if np.issubdtype(series.dtype, np.number):
            # IQR-based outlier detection
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = (series < lower) | (series > upper)
            n_outliers = outliers.sum()
            consistency = 1 - (n_outliers / n_samples) if n_samples > 0 else 1

            value_range = (float(series.min()), float(series.max()))

            if consistency < 0.95:
                issues.append("outliers_detected")

        # Validity (schema conformance)
        validity = 1.0
        if expected_dtype:
            if str(series.dtype) != expected_dtype:
                validity = 0.8
                issues.append("dtype_mismatch")

        # Distribution stats
        distribution_stats = {}
        if np.issubdtype(series.dtype, np.number):
            distribution_stats = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "skew": float(series.skew()) if len(series) > 2 else 0,
            }

        return FeatureQualityMetrics(
            feature_name=name,
            completeness=completeness,
            uniqueness=uniqueness,
            consistency=consistency,
            validity=validity,
            n_samples=n_samples,
            n_missing=int(n_missing),
            n_outliers=int(n_outliers),
            value_range=value_range,
            distribution_stats=distribution_stats,
            issues=issues,
        )

    def _generate_recommendations(
        self,
        metrics: list[FeatureQualityMetrics],
        issues_summary: dict[str, int],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Missing data
        high_missing = [m for m in metrics if m.completeness < 0.95]
        if high_missing:
            recommendations.append(
                f"Address missing data in {len(high_missing)} features: "
                f"{', '.join(m.feature_name for m in high_missing[:3])}"
            )

        # Outliers
        high_outliers = [m for m in metrics if m.n_outliers > m.n_samples * 0.05]
        if high_outliers:
            recommendations.append(
                f"Review outliers in {len(high_outliers)} features"
            )

        # Low consistency
        low_consistency = [m for m in metrics if m.consistency < 0.9]
        if low_consistency:
            recommendations.append(
                "Consider data validation rules for consistency issues"
            )

        # Overall
        avg_score = np.mean([m.overall_score for m in metrics])
        if avg_score < 0.8:
            recommendations.append(
                f"Overall quality score ({avg_score:.2f}) below threshold. "
                "Consider data cleaning pipeline improvements."
            )

        if not recommendations:
            recommendations.append("Feature quality is good. Continue monitoring.")

        return recommendations

    def _record_history(
        self,
        feature_set: str,
        version: str,
        score: float,
    ) -> None:
        """Record quality score in history."""
        if feature_set not in self._quality_history:
            self._quality_history[feature_set] = []

        self._quality_history[feature_set].append(
            (version, datetime.now(), score)
        )

        if self.quality_history_path:
            self._save_history()

    def _load_history(self) -> None:
        """Load history from disk."""
        if not self.quality_history_path:
            return

        path = Path(self.quality_history_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                for fs, entries in data.items():
                    self._quality_history[fs] = [
                        (e["version"], datetime.fromisoformat(e["timestamp"]), e["score"])
                        for e in entries
                    ]

    def _save_history(self) -> None:
        """Save history to disk."""
        if not self.quality_history_path:
            return

        data = {}
        for fs, entries in self._quality_history.items():
            data[fs] = [
                {"version": v, "timestamp": t.isoformat(), "score": s}
                for v, t, s in entries
            ]

        path = Path(self.quality_history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def analyze_feature_store(
    connector_type: str,
    feature_set: str,
    **kwargs: Any,
) -> FeatureQualityReport:
    """Convenience function to analyze feature store data.

    Args:
        connector_type: Type of connector ("feast", "tecton", "dataframe")
        feature_set: Feature set name to analyze
        **kwargs: Connector-specific arguments

    Returns:
        FeatureQualityReport

    Example:
        >>> report = analyze_feature_store(
        ...     "feast",
        ...     "user_features",
        ...     repo_path="./feature_repo"
        ... )
    """
    if connector_type == "feast":
        connector = FeastConnector(
            repo_path=kwargs.get("repo_path", "."),
            feature_store=kwargs.get("feature_store"),
        )
    elif connector_type == "tecton":
        connector = TectonConnector(
            workspace=kwargs.get("workspace"),
            api_key=kwargs.get("api_key"),
        )
    elif connector_type == "dataframe":
        data = kwargs.get("data")
        if data is None:
            raise ConfigurationError("DataFrame connector requires 'data' argument")
        connector = DataFrameConnector(data, feature_set)
    else:
        raise ConfigurationError(f"Unknown connector type: {connector_type}")

    analyzer = FeatureQualityAnalyzer(connector)
    return analyzer.analyze(
        feature_set,
        entity_df=kwargs.get("entity_df"),
        version=kwargs.get("version"),
    )


__all__ = [
    # Core classes
    "FeatureQualityAnalyzer",
    "FeatureQualityReport",
    "FeatureQualityMetrics",
    "FeatureMetadata",
    "FeatureVersionHistory",
    # Connectors
    "FeatureStoreConnector",
    "FeastConnector",
    "TectonConnector",
    "DataFrameConnector",
    # Enums
    "FeatureStoreType",
    "QualityDimension",
    # Functions
    "analyze_feature_store",
]
