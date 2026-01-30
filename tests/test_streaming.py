"""Tests for streaming module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.streaming import ChunkResult, StreamingCleaner, StreamingSummary, stream_analyze


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_create_chunk_result(self) -> None:
        result = ChunkResult(
            chunk_id=0,
            start_row=0,
            end_row=100,
            issues={"outlier": [1, 2, 3]},
            quality_score=97.0,
        )
        assert result.chunk_id == 0
        assert result.total_issues == 3

    def test_total_issues_empty(self) -> None:
        result = ChunkResult(chunk_id=0, start_row=0, end_row=100)
        assert result.total_issues == 0

    def test_total_issues_multiple_types(self) -> None:
        result = ChunkResult(
            chunk_id=0,
            start_row=0,
            end_row=100,
            issues={"outlier": [1, 2], "duplicate": [3, 4, 5]},
        )
        assert result.total_issues == 5


class TestStreamingSummary:
    """Tests for StreamingSummary dataclass."""

    def test_to_dict(self) -> None:
        summary = StreamingSummary(
            total_rows=1000,
            total_chunks=10,
            total_issues=50,
            issue_breakdown={"outlier": 30, "duplicate": 20},
            average_quality_score=95.0,
        )
        result = summary.to_dict()
        assert result["total_rows"] == 1000
        assert result["total_chunks"] == 10
        assert result["total_issues"] == 50


class TestStreamingCleaner:
    """Tests for StreamingCleaner class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1, 2], 100),
        })

    @pytest.fixture
    def sample_csv(self, sample_df: pd.DataFrame, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        return csv_path

    def test_init_defaults(self) -> None:
        cleaner = StreamingCleaner()
        assert cleaner.label_column == "label"
        assert cleaner.chunk_size == 10000
        assert "duplicates" in cleaner.detectors

    def test_init_custom(self) -> None:
        cleaner = StreamingCleaner(
            label_column="target",
            chunk_size=5000,
            detectors=["outliers"],
        )
        assert cleaner.label_column == "target"
        assert cleaner.chunk_size == 5000
        assert cleaner.detectors == ["outliers"]

    @pytest.mark.asyncio
    async def test_analyze_dataframe(self, sample_df: pd.DataFrame) -> None:
        cleaner = StreamingCleaner(label_column="label", chunk_size=25)
        results = []
        async for result in cleaner.analyze_dataframe(sample_df):
            results.append(result)

        assert len(results) == 4  # 100 rows / 25 chunk_size
        assert results[0].chunk_id == 0
        assert results[-1].chunk_id == 3

    @pytest.mark.asyncio
    async def test_analyze_file(self, sample_csv: Path) -> None:
        cleaner = StreamingCleaner(label_column="label", chunk_size=25)
        results = []
        async for result in cleaner.analyze_file(sample_csv):
            results.append(result)

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_analyze_file_not_found(self) -> None:
        cleaner = StreamingCleaner()
        with pytest.raises(FileNotFoundError):
            async for _ in cleaner.analyze_file("/nonexistent/file.csv"):
                pass

    @pytest.mark.asyncio
    async def test_get_summary(self, sample_df: pd.DataFrame) -> None:
        cleaner = StreamingCleaner(label_column="label", chunk_size=25)
        async for _ in cleaner.analyze_dataframe(sample_df):
            pass

        summary = cleaner.get_summary()
        assert summary.total_rows == 100
        assert summary.total_chunks == 4
        assert summary.average_quality_score <= 100.0

    def test_get_summary_empty(self) -> None:
        cleaner = StreamingCleaner()
        summary = cleaner.get_summary()
        assert summary.total_rows == 0
        assert summary.total_chunks == 0

    def test_reset(self, sample_df: pd.DataFrame) -> None:
        cleaner = StreamingCleaner(label_column="label", chunk_size=25)

        # Run analysis using sync helper
        list(stream_analyze(sample_df, label_column="label", chunk_size=25))

        # Reset
        cleaner.reset()
        summary = cleaner.get_summary()
        assert summary.total_chunks == 0


class TestStreamAnalyze:
    """Tests for stream_analyze helper function."""

    def test_stream_analyze(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(50),
            "label": np.random.choice([0, 1], 50),
        })

        results = list(stream_analyze(df, label_column="label", chunk_size=10))
        assert len(results) == 5
        assert all(isinstance(r, ChunkResult) for r in results)


class TestOutlierDetection:
    """Tests for chunk-level outlier detection."""

    @pytest.mark.asyncio
    async def test_detects_outliers(self) -> None:
        # Create data with clear outliers
        np.random.seed(42)
        data = np.random.randn(100)
        data[0] = 100  # Clear outlier
        data[1] = -100  # Clear outlier

        df = pd.DataFrame({
            "feature": data,
            "label": np.zeros(100),
        })

        cleaner = StreamingCleaner(label_column="label", chunk_size=100)
        async for result in cleaner.analyze_dataframe(df):
            assert "outlier" in result.issues
            assert len(result.issues["outlier"]) >= 2


class TestDuplicateDetection:
    """Tests for chunk-level duplicate detection."""

    @pytest.mark.asyncio
    async def test_detects_duplicates(self) -> None:
        df = pd.DataFrame({
            "feature": [1, 1, 2, 2, 3],
            "label": [0, 0, 1, 1, 2],
        })

        cleaner = StreamingCleaner(label_column="label", chunk_size=10)
        async for result in cleaner.analyze_dataframe(df):
            assert "duplicate" in result.issues
            assert len(result.issues["duplicate"]) == 2
