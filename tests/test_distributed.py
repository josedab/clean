"""Tests for distributed processing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.distributed import (
    ChunkedAnalyzer,
    DaskCleaner,
    DistributedConfig,
    DistributedReport,
    analyze_distributed,
)


class TestDistributedConfig:
    """Tests for DistributedConfig class."""

    def test_config_defaults(self) -> None:
        config = DistributedConfig()
        assert config.n_workers == 4
        assert config.chunk_size == 100000

    def test_config_to_dict(self) -> None:
        config = DistributedConfig(n_workers=8, chunk_size=50000)
        d = config.to_dict()

        assert d["n_workers"] == 8
        assert d["chunk_size"] == 50000


class TestChunkedAnalyzer:
    """Tests for ChunkedAnalyzer class."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(500),
            "feature2": np.random.randn(500),
            "label": np.random.choice([0, 1, 2], 500),
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_analyzer_init(self) -> None:
        analyzer = ChunkedAnalyzer(chunk_size=1000)
        assert analyzer.chunk_size == 1000

    def test_analyze_csv_file(self, sample_csv: Path) -> None:
        analyzer = ChunkedAnalyzer(chunk_size=100, label_column="label")

        reports = list(analyzer.analyze_file(sample_csv))

        assert len(reports) == 5  # 500 samples / 100 chunk_size
        for report in reports:
            assert report.n_samples <= 100

    def test_analyze_chunk_quality(self, sample_csv: Path) -> None:
        analyzer = ChunkedAnalyzer(chunk_size=250, label_column="label")

        reports = list(analyzer.analyze_file(sample_csv))

        for report in reports:
            assert 0 <= report.quality_score <= 100


class TestDaskCleaner:
    """Tests for DaskCleaner class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(300),
            "feature2": np.random.randn(300),
            "label": np.random.choice([0, 1, 2], 300),
        })

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(300),
            "feature2": np.random.randn(300),
            "label": np.random.choice([0, 1, 2], 300),
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_cleaner_init(self) -> None:
        cleaner = DaskCleaner(n_workers=2)
        assert cleaner.config.n_workers == 2

    def test_analyze_dataframe(self, sample_df: pd.DataFrame) -> None:
        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(
            sample_df,
            label_column="label",
            chunk_size=100,
        )

        assert isinstance(report, DistributedReport)
        assert report.total_samples == 300
        assert report.total_chunks == 3

    def test_analyze_file(self, sample_csv: Path) -> None:
        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(
            sample_csv,
            label_column="label",
            chunk_size=100,
        )

        assert isinstance(report, DistributedReport)
        assert report.total_samples == 300

    def test_report_quality_score(self, sample_df: pd.DataFrame) -> None:
        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(
            sample_df,
            label_column="label",
            chunk_size=100,
        )

        assert 0 <= report.overall_quality_score <= 100

    def test_report_summary(self, sample_df: pd.DataFrame) -> None:
        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(
            sample_df,
            label_column="label",
            chunk_size=100,
        )

        summary = report.summary()
        assert "Distributed Analysis Report" in summary
        assert "Total Samples" in summary

    def test_report_to_dict(self, sample_df: pd.DataFrame) -> None:
        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(
            sample_df,
            label_column="label",
            chunk_size=100,
        )

        d = report.to_dict()
        assert "total_samples" in d
        assert "overall_quality_score" in d
        assert "processing_time_seconds" in d


class TestAnalyzeDistributed:
    """Tests for analyze_distributed convenience function."""

    def test_function_with_dataframe(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(200),
            "y": np.random.randn(200),
            "label": np.random.choice([0, 1], 200),
        })

        report = analyze_distributed(
            df,
            label_column="label",
            backend="dask",
            n_workers=2,
            chunk_size=100,
        )

        assert isinstance(report, DistributedReport)

    def test_invalid_backend(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(ValueError, match="Unknown backend"):
            analyze_distributed(df, backend="invalid")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"x": [], "label": []})

        cleaner = DaskCleaner(n_workers=1)
        report = cleaner.analyze(df, label_column="label", chunk_size=10)

        assert report.total_samples == 0
        assert report.total_chunks == 0

    def test_single_chunk(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "label": np.random.choice([0, 1], 50),
        })

        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(df, label_column="label", chunk_size=100)

        assert report.total_chunks == 1

    def test_many_small_chunks(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })

        cleaner = DaskCleaner(n_workers=4)
        report = cleaner.analyze(df, label_column="label", chunk_size=10)

        assert report.total_chunks == 10

    def test_processing_time_recorded(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })

        cleaner = DaskCleaner(n_workers=2)
        report = cleaner.analyze(df, label_column="label", chunk_size=50)

        assert report.processing_time_seconds > 0
