"""Tests for marketplace module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestMarketplaceModule:
    """Tests for multi-organization data marketplace module."""

    def test_imports(self) -> None:
        from clean.marketplace import (
            QualityMarketplace,
            IndustryBenchmark,
            Domain,
            DataType,
            PrivacyLevel,
            create_marketplace,
            get_industry_percentile,
        )
        assert QualityMarketplace is not None
        assert IndustryBenchmark is not None
        assert Domain is not None
        assert DataType is not None

    def test_domain_enum(self) -> None:
        from clean.marketplace import Domain
        
        assert Domain.HEALTHCARE is not None
        assert Domain.FINANCE is not None
        assert Domain.RETAIL is not None
        assert Domain.MANUFACTURING is not None
        assert Domain.TECHNOLOGY is not None
        assert Domain.EDUCATION is not None
        assert Domain.GOVERNMENT is not None
        assert Domain.GENERAL is not None

    def test_data_type_enum(self) -> None:
        from clean.marketplace import DataType
        
        assert DataType.TABULAR is not None
        assert DataType.TEXT is not None
        assert DataType.IMAGE is not None
        assert DataType.TIME_SERIES is not None
        assert DataType.MIXED is not None

    def test_privacy_level_enum(self) -> None:
        from clean.marketplace import PrivacyLevel
        
        assert PrivacyLevel.PUBLIC is not None
        assert PrivacyLevel.PRIVATE is not None
        assert PrivacyLevel.ORGANIZATION is not None

    def test_marketplace_init(self) -> None:
        from clean.marketplace import QualityMarketplace
        
        marketplace = QualityMarketplace()
        assert marketplace is not None
        
        marketplace_with_org = QualityMarketplace(org_id="test-org")
        assert marketplace_with_org is not None

    def test_contribute_benchmark_basic(self) -> None:
        from clean.marketplace import QualityMarketplace, Domain, DataType
        
        marketplace = QualityMarketplace(org_id="test-org")
        
        benchmark = marketplace.contribute_benchmark(
            quality_score=0.85,
            n_samples=10000,
            label_error_rate=0.02,
            duplicate_rate=0.05,
            outlier_rate=0.03,
            domain=Domain.TECHNOLOGY,
            data_type=DataType.TABULAR,
        )
        
        assert benchmark is not None
        assert benchmark.quality_score == 0.85

    def test_get_percentile(self) -> None:
        from clean.marketplace import QualityMarketplace, Domain, DataType
        
        marketplace = QualityMarketplace()
        
        # Add some benchmarks first
        for score in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            marketplace.contribute_benchmark(
                quality_score=score,
                n_samples=1000,
                domain=Domain.GENERAL,
                data_type=DataType.TABULAR,
            )
        
        result = marketplace.get_percentile(0.85, domain=Domain.GENERAL)
        
        assert result is not None
        assert 0 <= result.percentile <= 100

    def test_get_industry_benchmark(self) -> None:
        from clean.marketplace import QualityMarketplace, Domain, DataType
        
        marketplace = QualityMarketplace()
        
        # Add benchmarks
        for score in [0.7, 0.75, 0.8, 0.85]:
            marketplace.contribute_benchmark(
                quality_score=score,
                n_samples=1000,
                domain=Domain.FINANCE,
                data_type=DataType.TABULAR,
            )
        
        benchmark = marketplace.get_industry_benchmark(Domain.FINANCE, DataType.TABULAR)
        
        # May be None if not enough data, but should not error
        if benchmark is not None:
            assert benchmark.domain == Domain.FINANCE
            assert benchmark.n_contributions >= 0

    def test_get_leaderboard(self) -> None:
        from clean.marketplace import QualityMarketplace, Domain
        
        marketplace = QualityMarketplace()
        
        # Add benchmarks from different domains
        for domain in [Domain.FINANCE, Domain.HEALTHCARE, Domain.GENERAL]:
            marketplace.contribute_benchmark(
                quality_score=0.85,
                n_samples=1000,
                domain=domain,
            )
        
        leaderboard = marketplace.get_leaderboard(domain=Domain.GENERAL, top_n=10)
        
        assert isinstance(leaderboard, list)

    def test_convenience_function_create_marketplace(self) -> None:
        from clean.marketplace import create_marketplace
        
        marketplace = create_marketplace(org_id="test-org")
        assert marketplace is not None

    def test_industry_benchmark_fields(self) -> None:
        from clean.marketplace import IndustryBenchmark, Domain, DataType
        from datetime import datetime
        
        benchmark = IndustryBenchmark(
            domain=Domain.FINANCE,
            data_type=DataType.TABULAR,
            n_contributions=100,
            last_updated=datetime.now(),
            quality_score_mean=0.85,
            quality_score_median=0.86,
            quality_score_std=0.05,
            quality_score_p25=0.80,
            quality_score_p75=0.90,
            quality_score_p90=0.95,
            label_error_rate_mean=0.02,
            label_error_rate_median=0.015,
            duplicate_rate_mean=0.03,
            outlier_rate_mean=0.05,
            percentile_thresholds={},
        )
        
        assert benchmark.domain == Domain.FINANCE
        assert benchmark.quality_score_mean == 0.85
        assert benchmark.n_contributions == 100
