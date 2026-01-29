"""Chunk processing utilities for large dataset handling.

This module provides base classes and utilities for processing data in chunks,
shared between the streaming and distributed processing modules.

Example:
    >>> processor = SyncChunkProcessor(chunk_size=10000)
    >>> for result in processor.process_dataframe(df, analyze_chunk):
    ...     print(f"Processed chunk {result.chunk_id}")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

from clean.constants import DEFAULT_CHUNK_SIZE, DEFAULT_IQR_MULTIPLIER
from clean.core.types import IssueType

# Type variable for chunk results
T = TypeVar("T")


@dataclass
class ChunkInfo:
    """Information about a data chunk."""

    chunk_id: int
    start_row: int
    end_row: int
    n_rows: int

    @classmethod
    def from_bounds(cls, chunk_id: int, start: int, end: int) -> ChunkInfo:
        """Create ChunkInfo from start/end bounds."""
        return cls(
            chunk_id=chunk_id,
            start_row=start,
            end_row=end,
            n_rows=end - start,
        )


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

    @property
    def n_rows(self) -> int:
        """Number of rows in this chunk."""
        return self.end_row - self.start_row


@dataclass
class ProcessingSummary:
    """Summary of processing across all chunks."""

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

    @classmethod
    def from_results(
        cls,
        results: list[ChunkResult],
        processing_time: float = 0.0,
    ) -> ProcessingSummary:
        """Create summary from list of chunk results."""
        if not results:
            return cls(
                total_rows=0,
                total_chunks=0,
                total_issues=0,
                processing_time_seconds=processing_time,
            )

        total_rows = sum(r.n_rows for r in results)
        total_issues = sum(r.total_issues for r in results)

        issue_breakdown: dict[str, int] = {}
        for result in results:
            for issue_type, indices in result.issues.items():
                issue_breakdown[issue_type] = (
                    issue_breakdown.get(issue_type, 0) + len(indices)
                )

        avg_score = sum(r.quality_score for r in results) / len(results)

        return cls(
            total_rows=total_rows,
            total_chunks=len(results),
            total_issues=total_issues,
            issue_breakdown=issue_breakdown,
            average_quality_score=avg_score,
            processing_time_seconds=processing_time,
        )


class ChunkAnalyzer:
    """Shared analysis logic for chunk processing.

    Provides common detection methods used by both streaming and
    distributed processing modules.
    """

    def __init__(
        self,
        label_column: str | None = None,
        detectors: list[str] | None = None,
    ):
        """Initialize the analyzer.

        Args:
            label_column: Name of the label column
            detectors: List of detectors to run
        """
        self.label_column = label_column
        self.detectors = detectors or ["duplicates", "outliers"]

    def analyze_chunk(
        self,
        chunk: pd.DataFrame,
        chunk_info: ChunkInfo,
    ) -> ChunkResult:
        """Analyze a single chunk of data.

        Args:
            chunk: DataFrame chunk to analyze
            chunk_info: Information about this chunk

        Returns:
            ChunkResult with detected issues
        """
        issues: dict[str, list[int]] = {}
        stats: dict[str, Any] = {"rows": len(chunk)}

        # Extract label statistics if available
        if self.label_column and self.label_column in chunk.columns:
            labels = chunk[self.label_column]
            stats["unique_labels"] = int(labels.nunique())
            stats["label_distribution"] = labels.value_counts().to_dict()

        # Run detectors
        if "outliers" in self.detectors:
            outlier_indices = self._detect_outliers(chunk)
            if outlier_indices:
                issues[IssueType.OUTLIER.value] = [
                    i + chunk_info.start_row for i in outlier_indices
                ]

        if "duplicates" in self.detectors:
            dup_indices = self._detect_duplicates(chunk)
            if dup_indices:
                issues[IssueType.DUPLICATE.value] = [
                    i + chunk_info.start_row for i in dup_indices
                ]

        # Calculate quality score
        total_issues = sum(len(v) for v in issues.values())
        quality_score = max(0.0, 100.0 - (total_issues / max(1, len(chunk)) * 100))

        return ChunkResult(
            chunk_id=chunk_info.chunk_id,
            start_row=chunk_info.start_row,
            end_row=chunk_info.end_row,
            issues=issues,
            quality_score=quality_score,
            stats=stats,
        )

    def _detect_outliers(self, chunk: pd.DataFrame) -> list[int]:
        """Detect outliers using IQR method."""
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

        # Convert to positional indices
        return [i for i, is_outlier in enumerate(outlier_mask) if is_outlier]

    def _detect_duplicates(self, chunk: pd.DataFrame) -> list[int]:
        """Detect duplicates within a chunk."""
        duplicated = chunk.duplicated(keep="first")
        return [i for i, is_dup in enumerate(duplicated) if is_dup]


class BaseChunkProcessor(ABC, Generic[T]):
    """Abstract base class for chunk processors.

    Provides common functionality for iterating over data in chunks.
    Subclasses implement sync or async processing.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        label_column: str | None = None,
        detectors: list[str] | None = None,
    ):
        """Initialize the processor.

        Args:
            chunk_size: Number of rows per chunk
            label_column: Name of the label column
            detectors: List of detectors to run
        """
        self.chunk_size = chunk_size
        self.analyzer = ChunkAnalyzer(
            label_column=label_column,
            detectors=detectors,
        )
        self._results: list[T] = []

    def reset(self) -> None:
        """Reset processor state for a new analysis."""
        self._results = []

    @abstractmethod
    def process_dataframe(self, df: pd.DataFrame) -> Iterator[T] | AsyncIterator[T]:
        """Process a DataFrame in chunks."""
        ...

    @abstractmethod
    def process_file(
        self, file_path: Path, **read_kwargs: Any
    ) -> Iterator[T] | AsyncIterator[T]:
        """Process a file in chunks."""
        ...

    def _iter_dataframe_chunks(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, ChunkInfo]]:
        """Iterate over DataFrame chunks with info."""
        for chunk_id, start in enumerate(range(0, len(df), self.chunk_size)):
            end = min(start + self.chunk_size, len(df))
            chunk = df.iloc[start:end]
            info = ChunkInfo.from_bounds(chunk_id, start, end)
            yield chunk, info

    def _iter_csv_chunks(
        self, file_path: Path, **read_kwargs: Any
    ) -> Iterator[tuple[pd.DataFrame, ChunkInfo]]:
        """Iterate over CSV file chunks with info."""
        start_row = 0

        for chunk_id, chunk in enumerate(
            pd.read_csv(file_path, chunksize=self.chunk_size, **read_kwargs)
        ):
            end_row = start_row + len(chunk)
            info = ChunkInfo.from_bounds(chunk_id, start_row, end_row)
            yield chunk, info
            start_row = end_row


class SyncChunkProcessor(BaseChunkProcessor[ChunkResult]):
    """Synchronous chunk processor.

    Processes data chunks synchronously, yielding results as they complete.

    Example:
        >>> processor = SyncChunkProcessor(chunk_size=10000)
        >>> for result in processor.process_dataframe(df):
        ...     print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
    """

    def process_dataframe(self, df: pd.DataFrame) -> Iterator[ChunkResult]:
        """Process a DataFrame in chunks.

        Args:
            df: DataFrame to process

        Yields:
            ChunkResult for each processed chunk
        """
        for chunk, info in self._iter_dataframe_chunks(df):
            result = self.analyzer.analyze_chunk(chunk, info)
            self._results.append(result)
            yield result

    def process_file(
        self, file_path: Path, **read_kwargs: Any
    ) -> Iterator[ChunkResult]:
        """Process a CSV file in chunks.

        Args:
            file_path: Path to CSV file
            **read_kwargs: Additional arguments for pd.read_csv

        Yields:
            ChunkResult for each processed chunk
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        for chunk, info in self._iter_csv_chunks(file_path, **read_kwargs):
            result = self.analyzer.analyze_chunk(chunk, info)
            self._results.append(result)
            yield result

    def get_summary(self) -> ProcessingSummary:
        """Get summary of all processed chunks."""
        return ProcessingSummary.from_results(self._results)


class AsyncChunkProcessor(BaseChunkProcessor[ChunkResult]):
    """Asynchronous chunk processor.

    Processes data chunks asynchronously, allowing for non-blocking I/O.

    Example:
        >>> processor = AsyncChunkProcessor(chunk_size=10000)
        >>> async for result in processor.process_dataframe(df):
        ...     print(f"Chunk {result.chunk_id}: {result.total_issues} issues")
    """

    async def process_dataframe(  # type: ignore[override]
        self, df: pd.DataFrame
    ) -> AsyncIterator[ChunkResult]:
        """Process a DataFrame in chunks asynchronously.

        Args:
            df: DataFrame to process

        Yields:
            ChunkResult for each processed chunk
        """
        for chunk, info in self._iter_dataframe_chunks(df):
            result = self.analyzer.analyze_chunk(chunk, info)
            self._results.append(result)
            yield result
            await asyncio.sleep(0)  # Yield control to event loop

    async def process_file(  # type: ignore[override]
        self, file_path: Path, **read_kwargs: Any
    ) -> AsyncIterator[ChunkResult]:
        """Process a CSV file in chunks asynchronously.

        Args:
            file_path: Path to CSV file
            **read_kwargs: Additional arguments for pd.read_csv

        Yields:
            ChunkResult for each processed chunk
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        for chunk, info in self._iter_csv_chunks(file_path, **read_kwargs):
            result = self.analyzer.analyze_chunk(chunk, info)
            self._results.append(result)
            yield result
            await asyncio.sleep(0)

    def get_summary(self) -> ProcessingSummary:
        """Get summary of all processed chunks."""
        return ProcessingSummary.from_results(self._results)


__all__ = [
    "ChunkInfo",
    "ChunkResult",
    "ProcessingSummary",
    "ChunkAnalyzer",
    "BaseChunkProcessor",
    "SyncChunkProcessor",
    "AsyncChunkProcessor",
]
