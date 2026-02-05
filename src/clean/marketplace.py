"""Multi-Organization Data Marketplace.

This module provides a platform for sharing anonymized quality benchmarks
across organizations to establish industry standards.

Example:
    >>> from clean.marketplace import QualityMarketplace
    >>>
    >>> marketplace = QualityMarketplace()
    >>> marketplace.contribute_benchmark(report, domain="healthcare")
    >>> percentile = marketplace.get_percentile(my_score, domain="healthcare")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from clean.core.report import QualityReport

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Industry domains for benchmarking."""

    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    GOVERNMENT = "government"
    GENERAL = "general"


class DataType(Enum):
    """Types of datasets for benchmarking."""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    MIXED = "mixed"


class PrivacyLevel(Enum):
    """Privacy levels for shared benchmarks."""

    PUBLIC = "public"  # Fully anonymized stats
    ORGANIZATION = "organization"  # Visible to org members
    PRIVATE = "private"  # Only aggregated into overall stats


@dataclass
class AnonymizedBenchmark:
    """Anonymized benchmark contribution."""

    benchmark_id: str
    contributed_at: datetime

    # Organization (anonymized)
    org_hash: str
    privacy_level: PrivacyLevel

    # Domain and data type
    domain: Domain
    data_type: DataType

    # Anonymized metrics (no raw data)
    quality_score: float
    n_samples_bucket: str  # "small", "medium", "large", "enterprise"
    label_error_rate: float
    duplicate_rate: float
    outlier_rate: float

    # Additional anonymous stats
    task_type: str | None  # classification, regression, etc.
    n_classes_bucket: str | None  # "binary", "multiclass_small", "multiclass_large"

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "contributed_at": self.contributed_at.isoformat(),
            "org_hash": self.org_hash[:8] + "...",  # Truncated for privacy
            "privacy_level": self.privacy_level.value,
            "domain": self.domain.value,
            "data_type": self.data_type.value,
            "quality_score": self.quality_score,
            "n_samples_bucket": self.n_samples_bucket,
            "label_error_rate": self.label_error_rate,
            "duplicate_rate": self.duplicate_rate,
            "outlier_rate": self.outlier_rate,
        }


@dataclass
class IndustryBenchmark:
    """Aggregated industry benchmark."""

    domain: Domain
    data_type: DataType | None
    n_contributions: int
    last_updated: datetime

    # Quality score statistics
    quality_score_mean: float
    quality_score_median: float
    quality_score_std: float
    quality_score_p25: float
    quality_score_p75: float
    quality_score_p90: float

    # Issue rate statistics
    label_error_rate_mean: float
    label_error_rate_median: float
    duplicate_rate_mean: float
    outlier_rate_mean: float

    # Percentile thresholds
    percentile_thresholds: dict[int, float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Industry Benchmark: {self.domain.value}",
            "=" * 50,
            "",
            f"Contributors: {self.n_contributions}",
            f"Last updated: {self.last_updated.strftime('%Y-%m-%d')}",
            "",
            "Quality Score Distribution:",
            f"  Mean: {self.quality_score_mean:.1f}",
            f"  Median: {self.quality_score_median:.1f}",
            f"  25th percentile: {self.quality_score_p25:.1f}",
            f"  75th percentile: {self.quality_score_p75:.1f}",
            f"  90th percentile: {self.quality_score_p90:.1f}",
            "",
            "Issue Rates (mean):",
            f"  Label errors: {self.label_error_rate_mean:.2%}",
            f"  Duplicates: {self.duplicate_rate_mean:.2%}",
            f"  Outliers: {self.outlier_rate_mean:.2%}",
        ]

        return "\n".join(lines)

    def get_percentile(self, score: float) -> int:
        """Get percentile for a given score.

        Args:
            score: Quality score to evaluate

        Returns:
            Percentile (0-100)
        """
        for percentile in sorted(self.percentile_thresholds.keys()):
            if score <= self.percentile_thresholds[percentile]:
                return percentile

        return 99


@dataclass
class OrganizationProfile:
    """Organization profile for marketplace."""

    org_id: str
    org_hash: str  # Anonymized identifier
    created_at: datetime

    n_contributions: int
    domains: list[Domain]
    reputation_score: float  # Based on contribution quality

    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class PercentileResult:
    """Result of percentile lookup."""

    score: float
    percentile: int
    domain: Domain
    better_than_percent: float
    benchmark_sample_size: int
    recommendation: str

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Your quality score of {self.score:.1f} is at the "
            f"{self.percentile}th percentile for {self.domain.value}.\n"
            f"Better than {self.better_than_percent:.0f}% of benchmarks "
            f"(based on {self.benchmark_sample_size} contributions).\n\n"
            f"Recommendation: {self.recommendation}"
        )


class BenchmarkStore(ABC):
    """Abstract store for benchmarks."""

    @abstractmethod
    def save_benchmark(self, benchmark: AnonymizedBenchmark) -> None:
        """Save a benchmark."""
        pass

    @abstractmethod
    def get_benchmarks(
        self,
        domain: Domain | None = None,
        data_type: DataType | None = None,
        limit: int = 1000,
    ) -> list[AnonymizedBenchmark]:
        """Get benchmarks matching criteria."""
        pass

    @abstractmethod
    def get_industry_benchmark(
        self,
        domain: Domain,
        data_type: DataType | None = None,
    ) -> IndustryBenchmark | None:
        """Get aggregated industry benchmark."""
        pass


class SQLiteBenchmarkStore(BenchmarkStore):
    """SQLite-based benchmark store."""

    def __init__(self, path: str | Path | None = None):
        """Initialize store.

        Args:
            path: Path to SQLite database
        """
        self.path = Path(path) if path else None

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
        else:
            self._conn = sqlite3.connect(":memory:")

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                benchmark_id TEXT PRIMARY KEY,
                contributed_at TEXT NOT NULL,
                org_hash TEXT NOT NULL,
                privacy_level TEXT NOT NULL,
                domain TEXT NOT NULL,
                data_type TEXT NOT NULL,
                quality_score REAL NOT NULL,
                n_samples_bucket TEXT NOT NULL,
                label_error_rate REAL NOT NULL,
                duplicate_rate REAL NOT NULL,
                outlier_rate REAL NOT NULL,
                task_type TEXT,
                n_classes_bucket TEXT,
                metadata TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                org_id TEXT PRIMARY KEY,
                org_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                n_contributions INTEGER DEFAULT 0,
                domains TEXT,
                reputation_score REAL DEFAULT 50.0,
                settings TEXT
            )
        """)

        self._conn.commit()

    def save_benchmark(self, benchmark: AnonymizedBenchmark) -> None:
        """Save a benchmark."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO benchmarks
            (benchmark_id, contributed_at, org_hash, privacy_level, domain,
             data_type, quality_score, n_samples_bucket, label_error_rate,
             duplicate_rate, outlier_rate, task_type, n_classes_bucket, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                benchmark.benchmark_id,
                benchmark.contributed_at.isoformat(),
                benchmark.org_hash,
                benchmark.privacy_level.value,
                benchmark.domain.value,
                benchmark.data_type.value,
                benchmark.quality_score,
                benchmark.n_samples_bucket,
                benchmark.label_error_rate,
                benchmark.duplicate_rate,
                benchmark.outlier_rate,
                benchmark.task_type,
                benchmark.n_classes_bucket,
                json.dumps(benchmark.metadata),
            ),
        )
        self._conn.commit()

    def get_benchmarks(
        self,
        domain: Domain | None = None,
        data_type: DataType | None = None,
        limit: int = 1000,
    ) -> list[AnonymizedBenchmark]:
        """Get benchmarks matching criteria."""
        query = "SELECT * FROM benchmarks WHERE privacy_level != 'private'"
        params: list[Any] = []

        if domain:
            query += " AND domain = ?"
            params.append(domain.value)

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type.value)

        query += " ORDER BY contributed_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        benchmarks = []

        for row in cursor.fetchall():
            benchmarks.append(AnonymizedBenchmark(
                benchmark_id=row[0],
                contributed_at=datetime.fromisoformat(row[1]),
                org_hash=row[2],
                privacy_level=PrivacyLevel(row[3]),
                domain=Domain(row[4]),
                data_type=DataType(row[5]),
                quality_score=row[6],
                n_samples_bucket=row[7],
                label_error_rate=row[8],
                duplicate_rate=row[9],
                outlier_rate=row[10],
                task_type=row[11],
                n_classes_bucket=row[12],
                metadata=json.loads(row[13]) if row[13] else {},
            ))

        return benchmarks

    def get_industry_benchmark(
        self,
        domain: Domain,
        data_type: DataType | None = None,
    ) -> IndustryBenchmark | None:
        """Get aggregated industry benchmark."""
        benchmarks = self.get_benchmarks(domain, data_type, limit=10000)

        if len(benchmarks) < 3:
            return None

        scores = [b.quality_score for b in benchmarks]
        label_errors = [b.label_error_rate for b in benchmarks]
        duplicates = [b.duplicate_rate for b in benchmarks]
        outliers = [b.outlier_rate for b in benchmarks]

        # Calculate percentile thresholds
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        thresholds = {
            p: float(np.percentile(scores, p))
            for p in percentiles
        }

        return IndustryBenchmark(
            domain=domain,
            data_type=data_type,
            n_contributions=len(benchmarks),
            last_updated=max(b.contributed_at for b in benchmarks),
            quality_score_mean=float(np.mean(scores)),
            quality_score_median=float(np.median(scores)),
            quality_score_std=float(np.std(scores)),
            quality_score_p25=float(np.percentile(scores, 25)),
            quality_score_p75=float(np.percentile(scores, 75)),
            quality_score_p90=float(np.percentile(scores, 90)),
            label_error_rate_mean=float(np.mean(label_errors)),
            label_error_rate_median=float(np.median(label_errors)),
            duplicate_rate_mean=float(np.mean(duplicates)),
            outlier_rate_mean=float(np.mean(outliers)),
            percentile_thresholds=thresholds,
        )

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


class QualityMarketplace:
    """Marketplace for sharing and comparing quality benchmarks.

    Enables organizations to:
    - Contribute anonymized quality benchmarks
    - Compare scores against industry percentiles
    - Discover best practices from top performers
    """

    def __init__(
        self,
        store: BenchmarkStore | None = None,
        org_id: str | None = None,
    ):
        """Initialize marketplace.

        Args:
            store: Benchmark store (defaults to in-memory)
            org_id: Organization identifier
        """
        self.store = store or SQLiteBenchmarkStore()
        self.org_id = org_id or "anonymous"
        self.org_hash = hashlib.sha256(self.org_id.encode()).hexdigest()

    def contribute_benchmark(
        self,
        report: QualityReport | None = None,
        quality_score: float | None = None,
        n_samples: int | None = None,
        label_error_rate: float = 0.0,
        duplicate_rate: float = 0.0,
        outlier_rate: float = 0.0,
        domain: Domain = Domain.GENERAL,
        data_type: DataType = DataType.TABULAR,
        privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC,
        task_type: str | None = None,
    ) -> AnonymizedBenchmark:
        """Contribute an anonymized benchmark.

        Args:
            report: Quality report to contribute (extracts metrics)
            quality_score: Manual quality score (if no report)
            n_samples: Number of samples
            label_error_rate: Label error rate
            duplicate_rate: Duplicate rate
            outlier_rate: Outlier rate
            domain: Industry domain
            data_type: Type of data
            privacy_level: How public to make the benchmark
            task_type: ML task type

        Returns:
            AnonymizedBenchmark
        """
        # Extract from report if provided
        if report is not None:
            quality_score = report.quality_score
            n_samples = report.n_samples

            if hasattr(report, "label_errors"):
                errors = report.label_errors()
                if errors is not None:
                    label_error_rate = len(errors) / n_samples

            if hasattr(report, "duplicates"):
                dupes = report.duplicates()
                if dupes is not None:
                    duplicate_rate = len(dupes) / n_samples

            if hasattr(report, "outliers"):
                outliers = report.outliers()
                if outliers is not None:
                    outlier_rate = len(outliers) / n_samples

        if quality_score is None:
            raise ValueError("Either report or quality_score must be provided")

        # Anonymize sample count into buckets
        n_samples = n_samples or 0
        if n_samples < 1000:
            n_samples_bucket = "small"
        elif n_samples < 10000:
            n_samples_bucket = "medium"
        elif n_samples < 100000:
            n_samples_bucket = "large"
        else:
            n_samples_bucket = "enterprise"

        # Generate benchmark ID
        benchmark_id = hashlib.sha256(
            f"{self.org_hash}_{datetime.now().isoformat()}_{quality_score}".encode()
        ).hexdigest()[:16]

        benchmark = AnonymizedBenchmark(
            benchmark_id=benchmark_id,
            contributed_at=datetime.now(),
            org_hash=self.org_hash,
            privacy_level=privacy_level,
            domain=domain,
            data_type=data_type,
            quality_score=quality_score,
            n_samples_bucket=n_samples_bucket,
            label_error_rate=label_error_rate,
            duplicate_rate=duplicate_rate,
            outlier_rate=outlier_rate,
            task_type=task_type,
            n_classes_bucket=None,
        )

        self.store.save_benchmark(benchmark)
        logger.info(f"Contributed benchmark {benchmark_id} to {domain.value}")

        return benchmark

    def get_percentile(
        self,
        score: float,
        domain: Domain = Domain.GENERAL,
        data_type: DataType | None = None,
    ) -> PercentileResult:
        """Get percentile rank for a quality score.

        Args:
            score: Quality score to evaluate
            domain: Industry domain for comparison
            data_type: Optional data type filter

        Returns:
            PercentileResult
        """
        benchmark = self.store.get_industry_benchmark(domain, data_type)

        if benchmark is None:
            # Not enough data - provide estimate
            return PercentileResult(
                score=score,
                percentile=50,  # Default to median
                domain=domain,
                better_than_percent=50.0,
                benchmark_sample_size=0,
                recommendation="Not enough industry data for comparison. Contribute your benchmarks!",
            )

        percentile = benchmark.get_percentile(score)

        # Generate recommendation
        if percentile >= 90:
            recommendation = (
                "Excellent! Your data quality is in the top 10% for your industry. "
                "Consider sharing best practices."
            )
        elif percentile >= 75:
            recommendation = (
                "Good data quality. Focus on addressing remaining issues to reach "
                "the top tier."
            )
        elif percentile >= 50:
            recommendation = (
                "Average data quality for your industry. Focus on label error "
                "reduction for quick wins."
            )
        elif percentile >= 25:
            recommendation = (
                "Below average. Prioritize data cleaning to improve model performance. "
                "Start with duplicate removal."
            )
        else:
            recommendation = (
                "Significant data quality issues. Consider comprehensive data audit "
                "before training models."
            )

        return PercentileResult(
            score=score,
            percentile=percentile,
            domain=domain,
            better_than_percent=float(percentile),
            benchmark_sample_size=benchmark.n_contributions,
            recommendation=recommendation,
        )

    def get_industry_benchmark(
        self,
        domain: Domain,
        data_type: DataType | None = None,
    ) -> IndustryBenchmark | None:
        """Get industry benchmark for a domain.

        Args:
            domain: Industry domain
            data_type: Optional data type filter

        Returns:
            IndustryBenchmark or None
        """
        return self.store.get_industry_benchmark(domain, data_type)

    def compare_to_industry(
        self,
        report: QualityReport,
        domain: Domain = Domain.GENERAL,
    ) -> dict[str, Any]:
        """Compare a report to industry benchmarks.

        Args:
            report: Quality report to compare
            domain: Industry domain

        Returns:
            Comparison results
        """
        benchmark = self.get_industry_benchmark(domain)

        if benchmark is None:
            return {
                "status": "no_benchmark",
                "message": "Not enough industry data available",
            }

        # Get percentiles for all metrics
        percentile_result = self.get_percentile(report.quality_score, domain)

        comparison = {
            "status": "success",
            "domain": domain.value,
            "n_industry_benchmarks": benchmark.n_contributions,
            "your_score": report.quality_score,
            "industry_median": benchmark.quality_score_median,
            "industry_mean": benchmark.quality_score_mean,
            "percentile": percentile_result.percentile,
            "better_than_percent": percentile_result.better_than_percent,
            "recommendation": percentile_result.recommendation,
        }

        # Add issue rate comparisons
        if hasattr(report, "label_errors"):
            errors = report.label_errors()
            if errors is not None:
                your_rate = len(errors) / report.n_samples
                comparison["label_error_comparison"] = {
                    "your_rate": your_rate,
                    "industry_mean": benchmark.label_error_rate_mean,
                    "better_than_industry": your_rate < benchmark.label_error_rate_mean,
                }

        return comparison

    def get_leaderboard(
        self,
        domain: Domain | None = None,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Get anonymized leaderboard of top performers.

        Args:
            domain: Filter by domain
            top_n: Number of entries to return

        Returns:
            List of leaderboard entries
        """
        benchmarks = self.store.get_benchmarks(domain, limit=1000)

        # Filter to public benchmarks
        public_benchmarks = [
            b for b in benchmarks
            if b.privacy_level == PrivacyLevel.PUBLIC
        ]

        # Sort by quality score
        sorted_benchmarks = sorted(
            public_benchmarks,
            key=lambda b: b.quality_score,
            reverse=True,
        )

        # Return top N as anonymized entries
        leaderboard = []
        for i, b in enumerate(sorted_benchmarks[:top_n]):
            leaderboard.append({
                "rank": i + 1,
                "quality_score": b.quality_score,
                "domain": b.domain.value,
                "data_type": b.data_type.value,
                "size_bucket": b.n_samples_bucket,
                "org_hash": b.org_hash[:8] + "...",
                "contributed_at": b.contributed_at.isoformat(),
            })

        return leaderboard


def create_marketplace(
    store_path: str | Path | None = None,
    org_id: str | None = None,
) -> QualityMarketplace:
    """Create a quality marketplace.

    Args:
        store_path: Path to SQLite database
        org_id: Organization identifier

    Returns:
        QualityMarketplace
    """
    store = SQLiteBenchmarkStore(store_path) if store_path else SQLiteBenchmarkStore()
    return QualityMarketplace(store=store, org_id=org_id)


def get_industry_percentile(
    score: float,
    domain: str = "general",
    marketplace: QualityMarketplace | None = None,
) -> PercentileResult:
    """Convenience function to get industry percentile.

    Args:
        score: Quality score
        domain: Industry domain
        marketplace: Optional marketplace instance

    Returns:
        PercentileResult
    """
    if marketplace is None:
        marketplace = QualityMarketplace()

    return marketplace.get_percentile(score, Domain(domain))
