"""Tests for quality_augmentation module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestQualityAugmentationModule:
    """Tests for quality-aware augmentation module."""

    def test_imports(self) -> None:
        from clean.quality_augmentation import (
            QualityAwareAugmenter,
            AugmentationConfig,
            AugmentationResult,
            AugmentationMethod,
            GapType,
        )
        assert QualityAwareAugmenter is not None
        assert AugmentationConfig is not None
        assert AugmentationResult is not None

    def test_augmentation_method_enum(self) -> None:
        from clean.quality_augmentation import AugmentationMethod
        
        assert AugmentationMethod.SMOTE is not None
        assert AugmentationMethod.MIXUP is not None
        assert AugmentationMethod.NOISE_INJECTION is not None

    def test_gap_type_enum(self) -> None:
        from clean.quality_augmentation import GapType
        
        # These are the actual enum values
        assert GapType.CLASS_IMBALANCE is not None
        assert GapType.UNDERREPRESENTED_SLICE is not None

    def test_config_defaults(self) -> None:
        from clean.quality_augmentation import AugmentationConfig
        
        config = AugmentationConfig()
        assert config.target_balance_ratio > 0
        assert config.quality_threshold >= 0
        assert config.diversity_weight >= 0

    def test_augmenter_init(self) -> None:
        from clean.quality_augmentation import QualityAwareAugmenter, AugmentationConfig
        
        augmenter = QualityAwareAugmenter()
        assert augmenter is not None
        
        config = AugmentationConfig(target_balance_ratio=0.5)
        augmenter_with_config = QualityAwareAugmenter(config=config)
        assert augmenter_with_config is not None

    def test_augment_basic(self) -> None:
        from clean.quality_augmentation import QualityAwareAugmenter, AugmentationResult
        
        np.random.seed(42)
        # Create imbalanced data
        X = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
        })
        y = np.array([0] * 90 + [1] * 10)  # 90% class 0, 10% class 1
        
        augmenter = QualityAwareAugmenter()
        result = augmenter.augment(X, y)
        
        assert isinstance(result, AugmentationResult)
        assert result.n_samples_original == 100
        assert result.n_samples_generated >= 0

    def test_augmentation_result_fields(self) -> None:
        from clean.quality_augmentation import AugmentationResult
        
        # AugmentationResult doesn't have to_dict, check fields directly
        result = AugmentationResult(
            n_samples_original=100,
            n_samples_generated=50,
            n_samples_accepted=45,
            n_samples_rejected=5,
            gaps_addressed=[],
            samples=None,
            quality_improvement=0.1,
            class_balance_improvement=0.2,
            diversity_improvement=0.05,
            rejection_reasons={},
            method_breakdown={},
        )
        
        assert result.n_samples_original == 100
        assert result.n_samples_generated == 50
        assert result.quality_improvement == 0.1

    def test_augment_with_report(self) -> None:
        from clean import DatasetCleaner
        from clean.quality_augmentation import QualityAwareAugmenter
        
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "label": [0] * 90 + [1] * 10,
        })
        
        cleaner = DatasetCleaner(data=df, label_column="label")
        report = cleaner.analyze()
        
        X = df[["feature_0", "feature_1"]]
        y = df["label"].values
        
        augmenter = QualityAwareAugmenter()
        result = augmenter.augment(X, y, report=report)
        
        assert result is not None
        assert result.n_samples_original == 100
