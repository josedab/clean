"""Async streaming support for processing large datasets.

This module provides streaming capabilities for Clean, allowing analysis
of datasets that don't fit in memory through chunked processing.

Example:
    >>> from clean.streaming import StreamingCleaner
    >>> async for result in StreamingCleaner.analyze_file("large.csv", chunk_size=10000):
    ...     print(f"Processed chunk: {result.chunk_id}, issues: {len(result.issues)}")
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.constants import DEFAULT_CHUNK_SIZE, DEFAULT_IQR_MULTIPLIER
from clean.core.types import IssueType

if TYPE_CHECKING:
    pass


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_id: int
    start_row: int
    end_row: int
    issues: dict[str, list[int]] = field(default_factory=dict)
    quality_score: float = 100.0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_issues(self) -> int:
        """Total number of issues found in this chunk."""
        return sum(len(v) for v in self.issues.values())


@dataclass
class StreamingSummary:
    """Summary of streaming analysis across all chunks."""

    total_rows: int
    total_chunks: int
    total_issues: int
    issue_breakdown: dict[str, int] = field(default_factory=dict)
    average_quality_score: float = 100.0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "total_rows": self.total_rows,
            "total_chunks": self.total_chunks,
            "total_issues": self.total_issues,
            "issue_breakdown": self.issue_breakdown,
            "average_quality_score": self.average_quality_score,
            "processing_time_seconds": self.processing_time_seconds,
        }


class StreamingCleaner:
    """Streaming data quality analyzer for large datasets.

    Processes data in chunks to handle datasets that don't fit in memory.

    Args:
        label_column: Name of the label column
        chunk_size: Number of rows per chunk (default: 10000)
        detectors: List of detectors to run (default: all)

    Example:
        >>> cleaner = StreamingCleaner(label_column="label", chunk_size=5000)
        >>> async for result in cleaner.analyze_file("large_dataset.csv"):
        ...     print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
    """

    def __init__(
        self,
        label_column: str = "label",
        chunk_size: int = 10000,
        detectors: list[str] | None = None,
    ) -> None:
        self.label_column = label_column
        self.chunk_size = chunk_size
        self.detectors = detectors or ["duplicates", "outliers", "imbalance"]
        self._chunk_results: list[ChunkResult] = []

    async def analyze_file(
        self, file_path: str | Path, **read_kwargs: Any
    ) -> AsyncIterator[ChunkResult]:
        """Analyze a CSV file in streaming fashion.

        Args:
            file_path: Path to CSV file
            **read_kwargs: Additional arguments passed to pd.read_csv

        Yields:
            ChunkResult for each processed chunk
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        chunk_id = 0
        start_row = 0

        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, **read_kwargs):
            result = await self._analyze_chunk(chunk, chunk_id, start_row)
            self._chunk_results.append(result)
            yield result

            chunk_id += 1
            start_row += len(chunk)
            await asyncio.sleep(0)  # Yield control to event loop

    async def analyze_dataframe(
        self, df: pd.DataFrame
    ) -> AsyncIterator[ChunkResult]:
        """Analyze a DataFrame in streaming fashion.

        Args:
            df: DataFrame to analyze

        Yields:
            ChunkResult for each processed chunk
        """
        chunk_id = 0
        start_row = 0

        for chunk_start in range(0, len(df), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end]

            result = await self._analyze_chunk(chunk, chunk_id, start_row)
            self._chunk_results.append(result)
            yield result

            chunk_id += 1
            start_row += len(chunk)
            await asyncio.sleep(0)

    async def _analyze_chunk(
        self, chunk: pd.DataFrame, chunk_id: int, start_row: int
    ) -> ChunkResult:
        """Analyze a single chunk of data."""
        issues: dict[str, list[int]] = {}
        stats: dict[str, Any] = {"rows": len(chunk)}

        if self.label_column in chunk.columns:
            labels = chunk[self.label_column]
            stats["unique_labels"] = labels.nunique()
            stats["label_distribution"] = labels.value_counts().to_dict()

        # Run detectors
        if "outliers" in self.detectors:
            outlier_indices = self._detect_outliers_chunk(chunk)
            if outlier_indices:
                issues[IssueType.OUTLIER.value] = [i + start_row for i in outlier_indices]

        if "duplicates" in self.detectors:
            dup_indices = self._detect_duplicates_chunk(chunk)
            if dup_indices:
                issues[IssueType.DUPLICATE.value] = [i + start_row for i in dup_indices]

        # Calculate chunk quality score
        total_issues = sum(len(v) for v in issues.values())
        quality_score = max(0.0, 100.0 - (total_issues / max(1, len(chunk)) * 100))

        return ChunkResult(
            chunk_id=chunk_id,
            start_row=start_row,
            end_row=start_row + len(chunk),
            issues=issues,
            quality_score=quality_score,
            stats=stats,
        )

    def _detect_outliers_chunk(self, chunk: pd.DataFrame) -> list[int]:
        """Simple outlier detection for a chunk using IQR method."""
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.label_column]

        if not numeric_cols:
            return []

        outlier_mask = pd.Series([False] * len(chunk), index=chunk.index)

        for col in numeric_cols:
            q1 = chunk[col].quantile(0.25)
            q3 = chunk[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - DEFAULT_IQR_MULTIPLIER * iqr
            upper = q3 + DEFAULT_IQR_MULTIPLIER * iqr
            col_outliers = (chunk[col] < lower) | (chunk[col] > upper)
            outlier_mask = outlier_mask | col_outliers

        return list(chunk.index[outlier_mask].tolist())

    def _detect_duplicates_chunk(self, chunk: pd.DataFrame) -> list[int]:
        """Detect duplicates within a chunk."""
        duplicated = chunk.duplicated(keep="first")
        return list(chunk.index[duplicated].tolist())

    def get_summary(self) -> StreamingSummary:
        """Get summary of all processed chunks."""
        if not self._chunk_results:
            return StreamingSummary(
                total_rows=0,
                total_chunks=0,
                total_issues=0,
            )

        total_rows = sum(r.end_row - r.start_row for r in self._chunk_results)
        total_issues = sum(r.total_issues for r in self._chunk_results)

        issue_breakdown: dict[str, int] = {}
        for result in self._chunk_results:
            for issue_type, indices in result.issues.items():
                issue_breakdown[issue_type] = issue_breakdown.get(issue_type, 0) + len(indices)

        avg_score = sum(r.quality_score for r in self._chunk_results) / len(self._chunk_results)

        return StreamingSummary(
            total_rows=total_rows,
            total_chunks=len(self._chunk_results),
            total_issues=total_issues,
            issue_breakdown=issue_breakdown,
            average_quality_score=avg_score,
        )

    def reset(self) -> None:
        """Reset the cleaner state for a new analysis."""
        self._chunk_results = []


def stream_analyze(
    df: pd.DataFrame,
    label_column: str = "label",
    chunk_size: int = 10000,
) -> Iterator[ChunkResult]:
    """Synchronous streaming analysis helper.

    Args:
        df: DataFrame to analyze
        label_column: Name of the label column
        chunk_size: Number of rows per chunk

    Yields:
        ChunkResult for each processed chunk

    Example:
        >>> for result in stream_analyze(large_df, chunk_size=5000):
        ...     print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
    """
    cleaner = StreamingCleaner(label_column=label_column, chunk_size=chunk_size)

    async def _run() -> list[ChunkResult]:
        results = []
        async for result in cleaner.analyze_dataframe(df):
            results.append(result)
        return results

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(_run())
        yield from results
    finally:
        loop.close()
