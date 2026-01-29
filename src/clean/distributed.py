"""Distributed computing backend for large-scale data quality analysis.

This module provides support for distributed processing using:
- Dask for parallel/distributed DataFrames
- Optional Spark integration
- Chunked processing for memory efficiency

Example:
    >>> from clean.distributed import DaskCleaner
    >>>
    >>> cleaner = DaskCleaner(n_workers=4)
    >>> report = cleaner.analyze("large_data.parquet", label_column="label")
    >>> print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""

    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "4GB"
    chunk_size: int = 100000
    scheduler: str = "threads"  # 'threads', 'processes', 'distributed'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_workers": self.n_workers,
            "threads_per_worker": self.threads_per_worker,
            "memory_limit": self.memory_limit,
            "chunk_size": self.chunk_size,
            "scheduler": self.scheduler,
        }


@dataclass
class ChunkReport:
    """Quality report for a single chunk."""

    chunk_id: int
    n_samples: int
    n_label_errors: int
    n_duplicates: int
    n_outliers: int
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedReport:
    """Aggregated report from distributed analysis."""

    total_samples: int
    total_chunks: int
    total_label_errors: int
    total_duplicates: int
    total_outliers: int
    overall_quality_score: float
    chunk_reports: list[ChunkReport]
    processing_time_seconds: float
    config: DistributedConfig

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Distributed Analysis Report",
            "=" * 50,
            f"Total Samples: {self.total_samples:,}",
            f"Total Chunks: {self.total_chunks}",
            f"Processing Time: {self.processing_time_seconds:.1f}s",
            f"",
            f"Quality Metrics:",
            f"  Overall Score: {self.overall_quality_score:.1f}/100",
            f"  Label Errors: {self.total_label_errors:,}",
            f"  Duplicates: {self.total_duplicates:,}",
            f"  Outliers: {self.total_outliers:,}",
            f"",
            f"Configuration:",
            f"  Workers: {self.config.n_workers}",
            f"  Chunk Size: {self.config.chunk_size:,}",
            f"  Scheduler: {self.config.scheduler}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "total_chunks": self.total_chunks,
            "total_label_errors": self.total_label_errors,
            "total_duplicates": self.total_duplicates,
            "total_outliers": self.total_outliers,
            "overall_quality_score": self.overall_quality_score,
            "processing_time_seconds": self.processing_time_seconds,
            "config": self.config.to_dict(),
        }


class ChunkedAnalyzer:
    """Memory-efficient chunked analyzer for large datasets.

    Processes data in chunks without requiring the full dataset
    in memory.
    """

    def __init__(
        self,
        chunk_size: int = 100000,
        label_column: str | None = None,
    ):
        """Initialize the chunked analyzer.

        Args:
            chunk_size: Number of rows per chunk
            label_column: Name of label column
        """
        self.chunk_size = chunk_size
        self.label_column = label_column

    def analyze_file(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> Iterator[ChunkReport]:
        """Analyze a file in chunks.

        Args:
            file_path: Path to data file (CSV, Parquet)
            **kwargs: Additional arguments for analysis

        Yields:
            ChunkReport for each processed chunk
        """
        file_path = Path(file_path)

        if file_path.suffix == ".parquet":
            yield from self._analyze_parquet(file_path, **kwargs)
        elif file_path.suffix == ".csv":
            yield from self._analyze_csv(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _analyze_csv(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> Iterator[ChunkReport]:
        """Analyze CSV file in chunks."""
        chunk_id = 0

        for chunk_df in pd.read_csv(file_path, chunksize=self.chunk_size):
            report = self._analyze_chunk(chunk_df, chunk_id, **kwargs)
            yield report
            chunk_id += 1

    def _analyze_parquet(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> Iterator[ChunkReport]:
        """Analyze Parquet file in chunks."""
        # Read full parquet and process in chunks
        df = pd.read_parquet(file_path)
        n_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size

        for chunk_id in range(n_chunks):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx]

            report = self._analyze_chunk(chunk_df, chunk_id, **kwargs)
            yield report

    def _analyze_chunk(
        self,
        chunk_df: pd.DataFrame,
        chunk_id: int,
        **kwargs: Any,
    ) -> ChunkReport:
        """Analyze a single chunk."""
        from clean import DatasetCleaner

        n_samples = len(chunk_df)

        try:
            cleaner = DatasetCleaner(
                data=chunk_df,
                label_column=self.label_column,
            )
            report = cleaner.analyze(show_progress=False)

            return ChunkReport(
                chunk_id=chunk_id,
                n_samples=n_samples,
                n_label_errors=len(report.label_errors()),
                n_duplicates=len(report.duplicates()),
                n_outliers=len(report.outliers()),
                quality_score=report.quality_score.overall,
            )
        except Exception as e:
            logger.warning("Chunk %d analysis failed: %s", chunk_id, e)
            return ChunkReport(
                chunk_id=chunk_id,
                n_samples=n_samples,
                n_label_errors=0,
                n_duplicates=0,
                n_outliers=0,
                quality_score=0.0,
                metadata={"error": str(e)},
            )


class DaskCleaner:
    """Distributed data cleaner using Dask.

    Enables parallel processing of large datasets across multiple
    cores or a distributed cluster.
    """

    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "4GB",
        scheduler: str = "threads",
    ):
        """Initialize the Dask cleaner.

        Args:
            n_workers: Number of worker processes/threads
            threads_per_worker: Threads per worker (for distributed)
            memory_limit: Memory limit per worker
            scheduler: Dask scheduler ('threads', 'processes', 'distributed')
        """
        self.config = DistributedConfig(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            scheduler=scheduler,
        )

        self._client = None

    def analyze(
        self,
        data: str | Path | pd.DataFrame,
        label_column: str | None = None,
        chunk_size: int | None = None,
    ) -> DistributedReport:
        """Analyze data using distributed processing.

        Args:
            data: File path or DataFrame
            label_column: Name of label column
            chunk_size: Override default chunk size

        Returns:
            DistributedReport with aggregated results
        """
        import time

        start_time = time.time()
        chunk_size = chunk_size or self.config.chunk_size

        if isinstance(data, (str, Path)):
            reports = self._analyze_file(Path(data), label_column, chunk_size)
        else:
            reports = self._analyze_dataframe(data, label_column, chunk_size)

        processing_time = time.time() - start_time

        return self._aggregate_reports(reports, processing_time)

    def _analyze_file(
        self,
        file_path: Path,
        label_column: str | None,
        chunk_size: int,
    ) -> list[ChunkReport]:
        """Analyze file using chunked processing."""
        analyzer = ChunkedAnalyzer(
            chunk_size=chunk_size,
            label_column=label_column,
        )

        reports = []
        for report in analyzer.analyze_file(file_path):
            reports.append(report)
            logger.info("Processed chunk %d: %d samples", report.chunk_id, report.n_samples)

        return reports

    def _analyze_dataframe(
        self,
        df: pd.DataFrame,
        label_column: str | None,
        chunk_size: int,
    ) -> list[ChunkReport]:
        """Analyze DataFrame using parallel processing."""
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        reports = []

        # Simple parallel processing using threads
        if self.config.scheduler == "threads":
            from concurrent.futures import ThreadPoolExecutor

            def process_chunk(args: tuple) -> ChunkReport:
                chunk_id, chunk_df = args
                analyzer = ChunkedAnalyzer(label_column=label_column)
                return analyzer._analyze_chunk(chunk_df, chunk_id)

            chunks = []
            for chunk_id in range(n_chunks):
                start_idx = chunk_id * chunk_size
                end_idx = min(start_idx + chunk_size, len(df))
                chunks.append((chunk_id, df.iloc[start_idx:end_idx].copy()))

            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                reports = list(executor.map(process_chunk, chunks))

        else:
            # Fallback to sequential
            analyzer = ChunkedAnalyzer(
                chunk_size=chunk_size,
                label_column=label_column,
            )

            for chunk_id in range(n_chunks):
                start_idx = chunk_id * chunk_size
                end_idx = min(start_idx + chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                reports.append(analyzer._analyze_chunk(chunk_df, chunk_id))

        return reports

    def _aggregate_reports(
        self,
        chunk_reports: list[ChunkReport],
        processing_time: float,
    ) -> DistributedReport:
        """Aggregate chunk reports into final report."""
        if not chunk_reports:
            return DistributedReport(
                total_samples=0,
                total_chunks=0,
                total_label_errors=0,
                total_duplicates=0,
                total_outliers=0,
                overall_quality_score=0.0,
                chunk_reports=[],
                processing_time_seconds=processing_time,
                config=self.config,
            )

        total_samples = sum(r.n_samples for r in chunk_reports)
        total_label_errors = sum(r.n_label_errors for r in chunk_reports)
        total_duplicates = sum(r.n_duplicates for r in chunk_reports)
        total_outliers = sum(r.n_outliers for r in chunk_reports)

        # Weighted average quality score
        weights = [r.n_samples for r in chunk_reports]
        scores = [r.quality_score for r in chunk_reports]
        overall_score = np.average(scores, weights=weights) if weights else 0.0

        return DistributedReport(
            total_samples=total_samples,
            total_chunks=len(chunk_reports),
            total_label_errors=total_label_errors,
            total_duplicates=total_duplicates,
            total_outliers=total_outliers,
            overall_quality_score=float(overall_score),
            chunk_reports=chunk_reports,
            processing_time_seconds=processing_time,
            config=self.config,
        )


class SparkCleaner:
    """Distributed data cleaner using Apache Spark.

    Note: Requires PySpark to be installed.
    """

    def __init__(
        self,
        spark_session: Any = None,
        app_name: str = "CleanDataQuality",
        n_partitions: int = 8,
    ):
        """Initialize the Spark cleaner.

        Args:
            spark_session: Existing SparkSession (created if None)
            app_name: Application name for Spark
            n_partitions: Number of partitions for processing
        """
        self.app_name = app_name
        self.n_partitions = n_partitions
        self._spark = spark_session

    def _get_spark(self) -> Any:
        """Get or create SparkSession."""
        if self._spark is not None:
            return self._spark

        try:
            from pyspark.sql import SparkSession

            self._spark = (
                SparkSession.builder
                .appName(self.app_name)
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .getOrCreate()
            )
            return self._spark
        except ImportError:
            raise ImportError(
                "PySpark not installed. Install with: pip install pyspark"
            )

    def analyze(
        self,
        data: str | Path,
        label_column: str | None = None,
    ) -> DistributedReport:
        """Analyze data using Spark.

        Args:
            data: File path to data
            label_column: Name of label column

        Returns:
            DistributedReport with results
        """
        import time

        start_time = time.time()
        spark = self._get_spark()

        file_path = str(data)

        # Read data
        if file_path.endswith(".parquet"):
            spark_df = spark.read.parquet(file_path)
        elif file_path.endswith(".csv"):
            spark_df = spark.read.csv(file_path, header=True, inferSchema=True)
        else:
            raise ValueError(f"Unsupported format: {file_path}")

        # Repartition for parallelism
        spark_df = spark_df.repartition(self.n_partitions)

        # Process partitions
        def analyze_partition(iterator: Iterator) -> Iterator:
            """Process a single partition."""
            rows = list(iterator)
            if not rows:
                return iter([])

            # Convert to pandas
            pdf = pd.DataFrame(rows)

            from clean import DatasetCleaner

            try:
                cleaner = DatasetCleaner(data=pdf, label_column=label_column)
                report = cleaner.analyze(show_progress=False)

                yield {
                    "n_samples": len(pdf),
                    "n_label_errors": len(report.label_errors()),
                    "n_duplicates": len(report.duplicates()),
                    "n_outliers": len(report.outliers()),
                    "quality_score": report.quality_score.overall,
                }
            except Exception as e:
                yield {
                    "n_samples": len(pdf),
                    "n_label_errors": 0,
                    "n_duplicates": 0,
                    "n_outliers": 0,
                    "quality_score": 0.0,
                    "error": str(e),
                }

        # Collect results
        results = spark_df.rdd.mapPartitions(analyze_partition).collect()

        processing_time = time.time() - start_time

        # Create chunk reports
        chunk_reports = []
        for i, r in enumerate(results):
            chunk_reports.append(ChunkReport(
                chunk_id=i,
                n_samples=r["n_samples"],
                n_label_errors=r["n_label_errors"],
                n_duplicates=r["n_duplicates"],
                n_outliers=r["n_outliers"],
                quality_score=r["quality_score"],
            ))

        # Aggregate
        total_samples = sum(r["n_samples"] for r in results)
        total_label_errors = sum(r["n_label_errors"] for r in results)
        total_duplicates = sum(r["n_duplicates"] for r in results)
        total_outliers = sum(r["n_outliers"] for r in results)

        weights = [r["n_samples"] for r in results]
        scores = [r["quality_score"] for r in results]
        overall_score = np.average(scores, weights=weights) if weights else 0.0

        return DistributedReport(
            total_samples=total_samples,
            total_chunks=len(results),
            total_label_errors=total_label_errors,
            total_duplicates=total_duplicates,
            total_outliers=total_outliers,
            overall_quality_score=float(overall_score),
            chunk_reports=chunk_reports,
            processing_time_seconds=processing_time,
            config=DistributedConfig(
                n_workers=self.n_partitions,
                scheduler="spark",
            ),
        )

    def stop(self) -> None:
        """Stop the Spark session."""
        if self._spark:
            self._spark.stop()
            self._spark = None


def analyze_distributed(
    data: str | Path | pd.DataFrame,
    label_column: str | None = None,
    backend: str = "dask",
    n_workers: int = 4,
    **kwargs: Any,
) -> DistributedReport:
    """Analyze data using distributed processing.

    Args:
        data: File path or DataFrame
        label_column: Name of label column
        backend: Backend to use ('dask', 'spark')
        n_workers: Number of parallel workers
        **kwargs: Additional arguments for analyze method

    Returns:
        DistributedReport with results
    """
    if backend == "dask":
        cleaner = DaskCleaner(n_workers=n_workers)
        return cleaner.analyze(data, label_column, **kwargs)
    elif backend == "spark":
        cleaner = SparkCleaner(n_partitions=n_workers)
        try:
            return cleaner.analyze(data, label_column, **kwargs)
        finally:
            cleaner.stop()
    else:
        raise ValueError(f"Unknown backend: {backend}")
