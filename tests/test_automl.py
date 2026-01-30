"""Tests for AutoML quality tuning."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from clean.automl import (
    AdaptiveThresholdManager,
    OptimizationMethod,
    QualityTuner,
    ThresholdParams,
    TuningConfig,
    TuningMetric,
    TuningResult,
    tune_quality_thresholds,
)


class TestTuningConfig:
    """Tests for TuningConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TuningConfig()
        assert config.metric == TuningMetric.ACCURACY
        assert config.method == OptimizationMethod.RANDOM_SEARCH
        assert config.n_trials == 50

    def test_string_enums(self):
        """Test string enum conversion."""
        config = TuningConfig(metric="f1", method="grid")
        assert config.metric == TuningMetric.F1
        assert config.method == OptimizationMethod.GRID_SEARCH

    def test_custom_ranges(self):
        """Test custom threshold ranges."""
        config = TuningConfig(
            label_error_threshold_range=(0.5, 0.9),
            outlier_contamination_range=(0.05, 0.15),
        )
        assert config.label_error_threshold_range == (0.5, 0.9)
        assert config.outlier_contamination_range == (0.05, 0.15)


class TestThresholdParams:
    """Tests for ThresholdParams."""

    def test_default_params(self):
        """Test default parameters."""
        params = ThresholdParams()
        assert params.label_error_threshold == 0.5
        assert params.outlier_contamination == 0.1
        assert params.duplicate_threshold == 0.9

    def test_to_dict(self):
        """Test dictionary conversion."""
        params = ThresholdParams(
            label_error_threshold=0.7,
            remove_outliers=False,
        )
        d = params.to_dict()
        assert d["label_error_threshold"] == 0.7
        assert d["remove_outliers"] is False


class TestQualityTuner:
    """Tests for QualityTuner."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        return X, y

    def test_tuner_initialization(self):
        """Test tuner initialization."""
        tuner = QualityTuner()
        assert tuner.config.metric == TuningMetric.ACCURACY

    def test_random_search(self, sample_data):
        """Test random search optimization."""
        X, y = sample_data
        config = TuningConfig(
            method=OptimizationMethod.RANDOM_SEARCH,
            n_trials=10,
            verbose=False,
        )
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        assert isinstance(result, TuningResult)
        assert result.best_params is not None
        assert result.n_trials_completed <= 10
        assert result.baseline_score > 0

    def test_grid_search(self, sample_data):
        """Test grid search optimization."""
        X, y = sample_data
        config = TuningConfig(
            method=OptimizationMethod.GRID_SEARCH,
            n_trials=125,  # 5^3
            verbose=False,
        )
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        assert isinstance(result, TuningResult)
        assert result.best_params is not None

    def test_bayesian_search(self, sample_data):
        """Test Bayesian optimization."""
        X, y = sample_data
        config = TuningConfig(
            method=OptimizationMethod.BAYESIAN,
            n_trials=20,
            verbose=False,
        )
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        assert isinstance(result, TuningResult)
        assert result.best_params is not None

    def test_evolutionary_search(self, sample_data):
        """Test evolutionary optimization."""
        X, y = sample_data
        config = TuningConfig(
            method=OptimizationMethod.EVOLUTIONARY,
            n_trials=40,  # 2 generations of 20
            verbose=False,
        )
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        assert isinstance(result, TuningResult)
        assert result.best_params is not None

    def test_early_stopping(self, sample_data):
        """Test early stopping."""
        X, y = sample_data
        config = TuningConfig(
            n_trials=100,
            early_stopping_rounds=5,
            verbose=False,
        )
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        # Should stop early if no improvement
        assert result.n_trials_completed <= 100

    def test_timeout(self, sample_data):
        """Test timeout."""
        X, y = sample_data
        config = TuningConfig(
            n_trials=1000,
            timeout_seconds=1.0,  # 1 second timeout
            verbose=False,
        )
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        assert result.optimization_time_seconds < 5.0  # Should stop quickly

    def test_custom_model(self, sample_data):
        """Test with custom model."""
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier

        config = TuningConfig(n_trials=5, verbose=False)
        tuner = QualityTuner(config=config)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = tuner.tune(X, y, model=model)

        assert result.best_score > 0

    def test_dataframe_input(self, sample_data):
        """Test with DataFrame input."""
        X, y = sample_data
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        config = TuningConfig(n_trials=5, verbose=False)
        tuner = QualityTuner(config=config)
        result = tuner.tune(df, y)

        assert result.best_params is not None

    def test_result_summary(self, sample_data):
        """Test result summary generation."""
        X, y = sample_data
        config = TuningConfig(n_trials=5, verbose=False)
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        summary = result.summary()
        assert "AutoML Quality Tuning Results" in summary
        assert "Baseline score" in summary
        assert "Best score" in summary

    def test_result_to_dict(self, sample_data):
        """Test result dictionary conversion."""
        X, y = sample_data
        config = TuningConfig(n_trials=5, verbose=False)
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        d = result.to_dict()
        assert "best_params" in d
        assert "best_score" in d
        assert "improvement" in d


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_tune_quality_thresholds(self):
        """Test tune_quality_thresholds function."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        result = tune_quality_thresholds(X, y, n_trials=5, verbose=False)

        assert isinstance(result, TuningResult)
        assert result.best_params is not None


class TestAdaptiveThresholdManager:
    """Tests for AdaptiveThresholdManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = AdaptiveThresholdManager()
        assert manager.params is not None
        assert manager.performance_window == 100

    def test_record_performance(self):
        """Test recording performance."""
        manager = AdaptiveThresholdManager(performance_window=10)

        for i in range(15):
            manager.record_performance(0.8 + i * 0.01)

        assert len(manager._performance_history) == 10  # Window size

    def test_baseline_calculation(self):
        """Test baseline performance calculation."""
        manager = AdaptiveThresholdManager()

        for _ in range(10):
            manager.record_performance(0.85)

        assert manager._baseline_performance == pytest.approx(0.85)

    def test_degradation_detection(self):
        """Test degradation detection."""
        manager = AdaptiveThresholdManager(degradation_threshold=0.1)

        # Establish baseline
        for _ in range(10):
            manager.record_performance(0.90)

        # Degrade performance
        for _ in range(10):
            manager.record_performance(0.75)

        assert manager.check_degradation()

    def test_no_degradation(self):
        """Test when no degradation."""
        manager = AdaptiveThresholdManager(degradation_threshold=0.1)

        # Establish baseline
        for _ in range(10):
            manager.record_performance(0.85)

        # Maintain performance
        for _ in range(10):
            manager.record_performance(0.84)

        assert not manager.check_degradation()

    def test_get_current_params(self):
        """Test getting current parameters."""
        initial = ThresholdParams(label_error_threshold=0.7)
        manager = AdaptiveThresholdManager(initial_params=initial)

        params = manager.get_current_params()
        assert params.label_error_threshold == 0.7


class TestOutlierDetection:
    """Tests for outlier detection in tuning."""

    def test_detect_outliers(self):
        """Test outlier detection."""
        config = TuningConfig(n_trials=1, verbose=False)
        tuner = QualityTuner(config=config)

        # Create data with clear outliers
        X = np.vstack([
            np.random.randn(100, 5),  # Normal data
            np.random.randn(10, 5) * 10,  # Outliers
        ])

        outliers = tuner._detect_outliers(X, contamination=0.1)
        assert outliers.sum() > 0
        assert outliers.sum() < len(X)


class TestDuplicateDetection:
    """Tests for duplicate detection in tuning."""

    def test_detect_duplicates(self):
        """Test duplicate detection."""
        config = TuningConfig(n_trials=1, verbose=False)
        tuner = QualityTuner(config=config)

        # Create data with duplicates
        base = np.random.randn(50, 5)
        X = np.vstack([base, base[:10]])  # Add 10 duplicates

        duplicates = tuner._detect_duplicates(X, threshold=0.99)
        assert duplicates.sum() > 0


class TestIntegration:
    """Integration tests."""

    def test_full_tuning_workflow(self):
        """Test complete tuning workflow."""
        # Generate data with noise
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            flip_y=0.1,  # 10% label noise
            random_state=42,
        )

        # Add some outliers
        outlier_indices = np.random.choice(len(X), 20, replace=False)
        X[outlier_indices] *= 10

        # Tune
        config = TuningConfig(n_trials=20, verbose=False)
        tuner = QualityTuner(config=config)
        result = tuner.tune(X, y)

        # Verify result
        assert result.best_score >= result.baseline_score * 0.9  # At least close to baseline
        assert result.best_params.label_error_threshold > 0
        assert result.best_params.outlier_contamination > 0
