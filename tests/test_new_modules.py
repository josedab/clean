"""Smoke tests for new modules - verify imports and basic functionality."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestBenchmarkModule:
    """Smoke tests for benchmark module."""

    def test_imports(self) -> None:
        from clean.benchmark import (
            BenchmarkSuite,
            BenchmarkRunner,
            BenchmarkResult,
            SyntheticDataGenerator,
            run_benchmark,
            compare_detectors,
        )
        assert BenchmarkSuite is not None
        assert BenchmarkRunner is not None

    def test_synthetic_generator_basic(self) -> None:
        from clean.benchmark import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        assert gen is not None

    def test_benchmark_suite_init(self) -> None:
        from clean.benchmark import BenchmarkSuite
        suite = BenchmarkSuite()
        assert suite is not None


class TestFeatureStoreModule:
    """Smoke tests for feature store module."""

    def test_imports(self) -> None:
        from clean.feature_store import (
            FeatureQualityAnalyzer,
            FeatureQualityReport,
            FeatureStoreConnector,
            DataFrameConnector,
            analyze_feature_store,
        )
        assert FeatureQualityAnalyzer is not None

    def test_dataframe_connector_init(self) -> None:
        from clean.feature_store import DataFrameConnector
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        connector = DataFrameConnector(df)
        assert connector is not None

    def test_analyzer_init(self) -> None:
        from clean.feature_store import FeatureQualityAnalyzer, DataFrameConnector
        df = pd.DataFrame({"x": [1, 2, 3]})
        connector = DataFrameConnector(df)
        analyzer = FeatureQualityAnalyzer(connector=connector)
        assert analyzer is not None


class TestAugmentationModule:
    """Smoke tests for augmentation module."""

    def test_imports(self) -> None:
        from clean.augmentation import (
            DataAugmenter,
            AugmentationConfig,
            AugmentationResult,
            AugmentationStrategy,
            augment_for_quality,
        )
        assert DataAugmenter is not None
        assert AugmentationStrategy is not None

    def test_config_default(self) -> None:
        from clean.augmentation import AugmentationConfig
        config = AugmentationConfig()
        assert config is not None

    def test_augmenter_init(self) -> None:
        from clean.augmentation import DataAugmenter
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "label": [0] * 40 + [1] * 10,
        })
        augmenter = DataAugmenter(data=df, label_column="label")
        assert augmenter is not None


class TestMultilingualModule:
    """Smoke tests for multilingual module."""

    def test_imports(self) -> None:
        from clean.multilingual import (
            MultilingualDetector,
            MultilingualReport,
            LanguageDetector,
            detect_multilingual_errors,
        )
        assert MultilingualDetector is not None
        assert LanguageDetector is not None

    def test_language_detector_init(self) -> None:
        from clean.multilingual import LanguageDetector
        detector = LanguageDetector()
        assert detector is not None

    def test_detector_init(self) -> None:
        from clean.multilingual import MultilingualDetector
        detector = MultilingualDetector()
        assert detector is not None


class TestScoreAPIModule:
    """Smoke tests for score API module."""

    def test_imports(self) -> None:
        from clean.score_api import (
            QualityScoreAPI,
            QuickScore,
            TierLevel,
            TierLimits,
            RateLimiter,
        )
        assert QualityScoreAPI is not None
        assert TierLevel is not None

    def test_tier_levels(self) -> None:
        from clean.score_api import TierLevel
        assert TierLevel.FREE is not None
        assert TierLevel.BASIC is not None
        assert TierLevel.PRO is not None
        assert TierLevel.ENTERPRISE is not None

    def test_api_init(self) -> None:
        from clean.score_api import QualityScoreAPI, TierLevel
        api = QualityScoreAPI(default_tier=TierLevel.FREE)
        assert api is not None

    def test_rate_limiter(self) -> None:
        from clean.score_api import RateLimiter, TierLevel
        limiter = RateLimiter()
        status = limiter.check_limit("test_key", TierLevel.FREE)
        assert status is not None
        assert not status.is_limited


class TestDistillationModule:
    """Smoke tests for distillation module."""

    def test_imports(self) -> None:
        from clean.distillation import (
            ModelDistiller,
            DistillationPipeline,
            DistillationConfig,
            LightweightDetector,
            ModelFormat,
            CompressionLevel,
        )
        assert ModelDistiller is not None
        assert CompressionLevel is not None

    def test_config_default(self) -> None:
        from clean.distillation import DistillationConfig
        config = DistillationConfig()
        assert config is not None
        assert config.temperature > 0

    def test_compression_levels(self) -> None:
        from clean.distillation import CompressionLevel
        assert CompressionLevel.NONE is not None
        assert CompressionLevel.LOW is not None
        assert CompressionLevel.MEDIUM is not None
        assert CompressionLevel.HIGH is not None
        assert CompressionLevel.EXTREME is not None

    def test_distiller_init(self) -> None:
        from clean.distillation import ModelDistiller
        distiller = ModelDistiller()
        assert distiller is not None
