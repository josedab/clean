"""Tests for curriculum module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestCurriculumModule:
    """Tests for curriculum learning optimizer module."""

    def test_imports(self) -> None:
        from clean.curriculum import (
            CurriculumOptimizer,
            CurriculumConfig,
            CurriculumSchedule,
            DifficultyMetric,
            CurriculumStrategy,
        )
        assert CurriculumOptimizer is not None
        assert CurriculumConfig is not None
        assert CurriculumSchedule is not None
        assert DifficultyMetric is not None

    def test_difficulty_metrics(self) -> None:
        from clean.curriculum import DifficultyMetric
        
        assert DifficultyMetric.QUALITY_SCORE is not None
        assert DifficultyMetric.MODEL_CONFIDENCE is not None
        assert DifficultyMetric.LABEL_CERTAINTY is not None
        assert DifficultyMetric.OUTLIER_SCORE is not None
        assert DifficultyMetric.NEIGHBOR_AGREEMENT is not None
        assert DifficultyMetric.LOSS_VALUE is not None

    def test_curriculum_strategies(self) -> None:
        from clean.curriculum import CurriculumStrategy
        
        assert CurriculumStrategy.EASY_TO_HARD is not None
        assert CurriculumStrategy.SELF_PACED is not None
        assert CurriculumStrategy.DIVERSITY is not None

    def test_config_defaults(self) -> None:
        from clean.curriculum import CurriculumConfig
        
        config = CurriculumConfig()
        assert config.n_epochs > 0
        assert config.warmup_epochs >= 0
        assert config.initial_fraction > 0

    def test_optimizer_init(self) -> None:
        from clean.curriculum import CurriculumOptimizer, CurriculumConfig, CurriculumStrategy
        
        optimizer = CurriculumOptimizer()
        assert optimizer is not None
        
        config = CurriculumConfig(n_epochs=5)
        optimizer_with_config = CurriculumOptimizer(config=config)
        assert optimizer_with_config is not None
        
        optimizer_with_strategy = CurriculumOptimizer(strategy=CurriculumStrategy.EASY_TO_HARD)
        assert optimizer_with_strategy is not None

    def test_optimize_basic(self) -> None:
        from clean.curriculum import CurriculumOptimizer, CurriculumSchedule
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        optimizer = CurriculumOptimizer()
        schedule = optimizer.optimize(X, y)
        
        assert isinstance(schedule, CurriculumSchedule)
        assert schedule.n_samples == 100
        assert schedule.n_epochs > 0
        assert len(schedule.sample_order) == 100

    def test_optimize_with_quality_scores(self) -> None:
        from clean.curriculum import CurriculumOptimizer, CurriculumSchedule
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        quality_scores = np.random.rand(100)
        
        optimizer = CurriculumOptimizer()
        schedule = optimizer.optimize(X, y, quality_scores=quality_scores)
        
        assert isinstance(schedule, CurriculumSchedule)
        assert schedule.n_samples == 100
        assert schedule.difficulty_scores is not None

    def test_optimize_with_dataframe(self) -> None:
        from clean.curriculum import CurriculumOptimizer
        
        np.random.seed(42)
        X = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        })
        y = np.random.choice([0, 1], 100)
        
        optimizer = CurriculumOptimizer()
        schedule = optimizer.optimize(X, y)
        
        assert schedule is not None
        assert schedule.n_samples == 100

    def test_schedule_fields(self) -> None:
        from clean.curriculum import CurriculumSchedule, CurriculumStrategy
        
        # CurriculumSchedule doesn't have to_dict, check fields directly
        schedule = CurriculumSchedule(
            strategy=CurriculumStrategy.EASY_TO_HARD,
            n_samples=100,
            n_epochs=10,
            sample_order=list(range(100)),
            epoch_schedules={},
            samples_per_epoch=[10] * 10,
            difficulty_scores=list(np.random.rand(100)),
            metadata={},
        )
        
        assert schedule.n_samples == 100
        assert schedule.n_epochs == 10
        assert len(schedule.sample_order) == 100

    def test_curriculum_data_loader(self) -> None:
        from clean.curriculum import CurriculumOptimizer, create_curriculum_loader
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        optimizer = CurriculumOptimizer()
        schedule = optimizer.optimize(X, y)
        
        loader = create_curriculum_loader(X, y, schedule, batch_size=16)
        assert loader is not None
