"""Tests for processing module - chunk processing utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.processing import (
    AsyncChunkProcessor,
    ChunkAnalyzer,
    ChunkInfo,
    ChunkResult,
    ProcessingSummary,
    SyncChunkProcessor,
)


class TestChunkInfo:
    """Tests for ChunkInfo dataclass."""

    def test_chunk_info_creation(self):
        """Test basic ChunkInfo creation."""
        info = ChunkInfo(
            chunk_id=0,
            start_row=0,
            end_row=100,
            n_rows=100,
        )
        assert info.chunk_id == 0
        assert info.start_row == 0
        assert info.end_row == 100
        assert info.n_rows == 100

    def test_from_bounds(self):
        """Test ChunkInfo.from_bounds factory method."""
        info = ChunkInfo.from_bounds(chunk_id=2, start=200, end=350)

        assert info.chunk_id == 2
        assert info.start_row == 200
        assert info.end_row == 350
        assert info.n_rows == 150


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_chunk_result_creation(self):
        """Test basic ChunkResult creation."""
        result = ChunkResult(
            chunk_id=0,
            start_row=0,
            end_row=100,
        )
        assert result.chunk_id == 0
        assert result.total_issues == 0
        assert result.quality_score == 100.0

    def test_chunk_result_with_issues(self):
        """Test ChunkResult with issues."""
        result = ChunkResult(
            chunk_id=1,
            start_row=100,
            end_row=200,
            issues={
                "outlier": [105, 110, 115],
                "duplicate": [150, 160],
            },
            quality_score=85.0,
        )

        assert result.total_issues == 5
        assert result.n_rows == 100
        assert result.quality_score == 85.0

    def test_n_rows_property(self):
        """Test n_rows computed property."""
        result = ChunkResult(chunk_id=0, start_row=50, end_row=150)
        assert result.n_rows == 100


class TestProcessingSummary:
    """Tests for ProcessingSummary dataclass."""

    def test_empty_summary(self):
        """Test ProcessingSummary with no results."""
        summary = ProcessingSummary.from_results([])

        assert summary.total_rows == 0
        assert summary.total_chunks == 0
        assert summary.total_issues == 0
        assert summary.average_quality_score == 100.0

    def test_summary_from_results(self):
        """Test ProcessingSummary from multiple chunk results."""
        results = [
            ChunkResult(
                chunk_id=0,
                start_row=0,
                end_row=100,
                issues={"outlier": [5, 10]},
                quality_score=90.0,
            ),
            ChunkResult(
                chunk_id=1,
                start_row=100,
                end_row=200,
                issues={"outlier": [120], "duplicate": [150, 160]},
                quality_score=80.0,
            ),
        ]

        summary = ProcessingSummary.from_results(results, processing_time=1.5)

        assert summary.total_rows == 200
        assert summary.total_chunks == 2
        assert summary.total_issues == 5
        assert summary.issue_breakdown == {"outlier": 3, "duplicate": 2}
        assert summary.average_quality_score == 85.0
        assert summary.processing_time_seconds == 1.5

    def test_to_dict(self):
        """Test converting summary to dictionary."""
        summary = ProcessingSummary(
            total_rows=500,
            total_chunks=5,
            total_issues=20,
            issue_breakdown={"outlier": 15, "duplicate": 5},
            average_quality_score=92.5,
            processing_time_seconds=2.3,
        )

        d = summary.to_dict()

        assert d["total_rows"] == 500
        assert d["total_chunks"] == 5
        assert d["total_issues"] == 20
        assert d["average_quality_score"] == 92.5


class TestChunkAnalyzer:
    """Tests for ChunkAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ChunkAnalyzer(label_column="label")

    @pytest.fixture
    def sample_chunk(self):
        """Create sample DataFrame chunk."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": ["a"] * 50 + ["b"] * 50,
        })

    def test_analyze_chunk_basic(self, analyzer, sample_chunk):
        """Test basic chunk analysis."""
        info = ChunkInfo.from_bounds(0, 0, len(sample_chunk))
        result = analyzer.analyze_chunk(sample_chunk, info)

        assert isinstance(result, ChunkResult)
        assert result.chunk_id == 0
        assert result.n_rows == 100
        assert "rows" in result.stats

    def test_analyze_chunk_with_labels(self, analyzer, sample_chunk):
        """Test chunk analysis extracts label stats."""
        info = ChunkInfo.from_bounds(0, 0, len(sample_chunk))
        result = analyzer.analyze_chunk(sample_chunk, info)

        assert "unique_labels" in result.stats
        assert result.stats["unique_labels"] == 2
        assert "label_distribution" in result.stats

    def test_detect_outliers(self, analyzer):
        """Test outlier detection within chunk."""
        # Create data with obvious outlier
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 100],  # 100 is outlier
        })

        outliers = analyzer._detect_outliers(df)

        assert len(outliers) > 0
        assert 5 in outliers  # Index of 100

    def test_detect_outliers_no_numeric(self, analyzer):
        """Test outlier detection with no numeric columns."""
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
        })

        outliers = analyzer._detect_outliers(df)

        assert outliers == []

    def test_detect_duplicates(self, analyzer):
        """Test duplicate detection within chunk."""
        df = pd.DataFrame({
            "a": [1, 2, 1, 3],
            "b": ["x", "y", "x", "z"],
        })

        dups = analyzer._detect_duplicates(df)

        assert len(dups) == 1
        assert 2 in dups  # Second occurrence of (1, "x")

    def test_detect_duplicates_none(self, analyzer):
        """Test no duplicates detected."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": ["x", "y", "z", "w"],
        })

        dups = analyzer._detect_duplicates(df)

        assert dups == []

    def test_quality_score_calculation(self, analyzer):
        """Test quality score reflects issue count."""
        # Create chunk with known duplicates
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All duplicates
        })

        info = ChunkInfo.from_bounds(0, 0, len(df))
        result = analyzer.analyze_chunk(df, info)

        # 9 duplicates out of 10 = 90% issues
        assert result.quality_score < 20  # Low quality


class TestSyncChunkProcessor:
    """Tests for SyncChunkProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return SyncChunkProcessor(chunk_size=50)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "label": ["a"] * 100 + ["b"] * 100,
        })

    def test_process_dataframe(self, processor, sample_df):
        """Test processing DataFrame in chunks."""
        results = list(processor.process_dataframe(sample_df))

        assert len(results) == 4  # 200 rows / 50 chunk_size
        assert all(isinstance(r, ChunkResult) for r in results)

    def test_chunk_ordering(self, processor, sample_df):
        """Test chunks are processed in order."""
        results = list(processor.process_dataframe(sample_df))

        for i, result in enumerate(results):
            assert result.chunk_id == i

    def test_reset(self, processor, sample_df):
        """Test reset clears results."""
        list(processor.process_dataframe(sample_df))

        processor.reset()
        summary = processor.get_summary()

        assert summary.total_chunks == 0

    def test_get_summary(self, processor, sample_df):
        """Test getting summary after processing."""
        list(processor.process_dataframe(sample_df))

        summary = processor.get_summary()

        assert summary.total_rows == 200
        assert summary.total_chunks == 4

    def test_process_file(self, processor):
        """Test processing CSV file."""
        # Create temp CSV file
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
        })

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)

            try:
                results = list(processor.process_file(Path(f.name)))
                assert len(results) == 2  # 100 rows / 50 chunk_size
            finally:
                Path(f.name).unlink()

    def test_process_file_not_found(self, processor):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            list(processor.process_file(Path("/nonexistent/file.csv")))


class TestAsyncChunkProcessor:
    """Tests for AsyncChunkProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create async processor instance."""
        return AsyncChunkProcessor(chunk_size=50)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(150),
            "feature2": np.random.randn(150),
        })

    @pytest.mark.asyncio
    async def test_process_dataframe_async(self, processor, sample_df):
        """Test async processing of DataFrame."""
        results = []
        async for result in processor.process_dataframe(sample_df):
            results.append(result)

        assert len(results) == 3  # 150 rows / 50 chunk_size
        assert all(isinstance(r, ChunkResult) for r in results)

    @pytest.mark.asyncio
    async def test_async_chunk_ordering(self, processor, sample_df):
        """Test async chunks maintain order."""
        results = []
        async for result in processor.process_dataframe(sample_df):
            results.append(result)

        for i, result in enumerate(results):
            assert result.chunk_id == i

    @pytest.mark.asyncio
    async def test_async_process_file(self, processor):
        """Test async processing of CSV file."""
        df = pd.DataFrame({
            "x": range(75),
            "y": range(75, 150),
        })

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)

            try:
                results = []
                async for result in processor.process_file(Path(f.name)):
                    results.append(result)
                assert len(results) == 2  # 75 rows / 50 chunk_size
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_async_file_not_found(self, processor):
        """Test async error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            async for _ in processor.process_file(Path("/nonexistent/file.csv")):
                pass

    def test_get_summary_async(self, processor):
        """Test summary works with async processor."""
        # Even without processing, should return empty summary
        summary = processor.get_summary()
        assert summary.total_chunks == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test processing empty DataFrame."""
        processor = SyncChunkProcessor(chunk_size=10)
        df = pd.DataFrame()

        results = list(processor.process_dataframe(df))

        assert results == []

    def test_single_row_dataframe(self):
        """Test processing single-row DataFrame."""
        processor = SyncChunkProcessor(chunk_size=10)
        df = pd.DataFrame({"a": [1], "b": [2]})

        results = list(processor.process_dataframe(df))

        assert len(results) == 1
        assert results[0].n_rows == 1

    def test_chunk_size_larger_than_data(self):
        """Test when chunk size exceeds data size."""
        processor = SyncChunkProcessor(chunk_size=1000)
        df = pd.DataFrame({"a": range(50)})

        results = list(processor.process_dataframe(df))

        assert len(results) == 1
        assert results[0].n_rows == 50

    def test_exact_chunk_boundary(self):
        """Test when data size is exact multiple of chunk size."""
        processor = SyncChunkProcessor(chunk_size=25)
        df = pd.DataFrame({"a": range(100)})

        results = list(processor.process_dataframe(df))

        assert len(results) == 4
        assert all(r.n_rows == 25 for r in results)

    def test_custom_detectors(self):
        """Test with custom detector list."""
        processor = SyncChunkProcessor(
            chunk_size=10,
            detectors=["outliers"],  # Only outliers, no duplicates
        )
        df = pd.DataFrame({
            "a": [1, 1, 1, 2, 3],  # Has duplicates
            "b": [1, 2, 3, 4, 100],  # Has outlier
        })

        results = list(processor.process_dataframe(df))

        # Should detect outliers but not duplicates
        for result in results:
            assert "duplicate" not in result.issues or result.issues.get("duplicate") == []
