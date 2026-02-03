"""Tests for benchmark module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    DetectionMetrics,
    DetectorType,
    SyntheticDataGenerator,
    compare_detectors,
    run_benchmark,
)


class TestDetectionMetrics:
    """Tests for DetectionMetrics class."""

    def test_perfect_precision(self):
        """Test perfect precision."""
        metrics = DetectionMetrics(
            true_positives=10, false_positives=0, false_negatives=5, true_negatives=85
        )
        assert metrics.precision == 1.0

    def test_perfect_recall(self):
        """Test perfect recall."""
        metrics = DetectionMetrics(
            true_positives=10, false_positives=5, false_negatives=0, true_negatives=85
        )
        assert metrics.recall == 1.0

    def test_f1_score(self):
        """Test F1 score calculation."""
        metrics = DetectionMetrics(
            true_positives=8, false_positives=2, false_negatives=2, true_negatives=88
        )
        precision = 8 / 10  # 0.8
        recall = 8 / 10  # 0.8
        expected_f1 = 2 * precision * recall / (precision + recall)
        assert abs(metrics.f1 - expected_f1) < 0.001

    def test_accuracy(self):
        """Test accuracy calculation."""
        metrics = DetectionMetrics(
            true_positives=8, false_positives=2, false_negatives=2, true_negatives=88
        )
        assert metrics.accuracy == (8 + 88) / 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DetectionMetrics(
            true_positives=10, false_positives=5, false_negatives=3, true_negatives=82
        )
        d = metrics.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert "accuracy" in d

    def test_zero_predictions(self):
        """Test metrics with zero predictions."""
        metrics = DetectionMetrics(
            true_positives=0, false_positives=0, false_negatives=10, true_negatives=90
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.n_samples == 10000
        assert config.corruption_rate == 0.05
        assert config.random_seed == 42

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(n_samples=5000, corruption_rate=0.1, random_seed=123)
        assert config.n_samples == 5000
        assert config.corruption_rate == 0.1
        assert config.random_seed == 123


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""

    def test_generate_label_errors(self):
        """Test label error generation."""
        gen = SyntheticDataGenerator(random_seed=42)
        data, labels, error_mask = gen.generate_label_errors(
            n_samples=100, n_classes=3, error_rate=0.1, n_features=5
        )

        assert len(data) == 100
        assert len(labels) == 100
        assert len(error_mask) == 100
        assert error_mask.sum() > 0  # Some errors injected
        assert error_mask.sum() < 100  # Not all errors

    def test_generate_duplicates(self):
        """Test duplicate generation."""
        gen = SyntheticDataGenerator(random_seed=42)
        data, duplicate_groups = gen.generate_duplicates(
            n_samples=100, n_features=5, duplicate_rate=0.1
        )

        assert len(data) > 0
        assert len(duplicate_groups) >= 0

    def test_generate_outliers(self):
        """Test outlier generation."""
        gen = SyntheticDataGenerator(random_seed=42)
        data, outlier_mask = gen.generate_outliers(
            n_samples=100, n_features=5, outlier_rate=0.05
        )

        assert len(data) == 100
        assert len(outlier_mask) == 100
        assert outlier_mask.sum() > 0  # Some outliers

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        gen1 = SyntheticDataGenerator(random_seed=42)
        gen2 = SyntheticDataGenerator(random_seed=42)

        data1, _, _ = gen1.generate_label_errors(n_samples=50, n_classes=2)
        data2, _, _ = gen2.generate_label_errors(n_samples=50, n_classes=2)

        np.testing.assert_array_equal(data1, data2)


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_init(self):
        """Test runner initialization."""
        runner = BenchmarkRunner()
        assert runner is not None

    def test_init_with_config(self):
        """Test runner with custom config."""
        config = BenchmarkConfig(n_samples=500)
        runner = BenchmarkRunner(config=config)
        assert runner.config.n_samples == 500

    def test_run_label_error_benchmark(self):
        """Test running label error benchmark."""
        config = BenchmarkConfig(n_samples=200, corruption_rate=0.1)
        runner = BenchmarkRunner(config=config)

        result = runner.run_label_error_benchmark()

        assert isinstance(result, BenchmarkResult)
        assert result.metrics is not None

    def test_run_duplicate_benchmark(self):
        """Test running duplicate benchmark."""
        config = BenchmarkConfig(n_samples=200, corruption_rate=0.1)
        runner = BenchmarkRunner(config=config)

        result = runner.run_duplicate_benchmark()

        assert isinstance(result, BenchmarkResult)

    def test_run_outlier_benchmark(self):
        """Test running outlier benchmark."""
        config = BenchmarkConfig(n_samples=200, corruption_rate=0.1)
        runner = BenchmarkRunner(config=config)

        result = runner.run_outlier_benchmark()

        assert isinstance(result, BenchmarkResult)


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result."""
        metrics = DetectionMetrics(
            true_positives=8, false_positives=2, false_negatives=2, true_negatives=88
        )
        return BenchmarkResult(
            dataset_name="test_dataset",
            detector_type="label_errors",
            metrics=metrics,
            execution_time=1.5,
            n_samples=100,
            n_ground_truth_issues=10,
            n_detected_issues=10,
        )

    def test_summary(self, sample_result):
        """Test summary generation."""
        summary = sample_result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict(self, sample_result):
        """Test conversion to dict."""
        d = sample_result.to_dict()
        assert "dataset" in d
        assert "metrics" in d
        assert "execution_time" in d


class TestBenchmarkSuiteResult:
    """Tests for BenchmarkSuiteResult class."""

    @pytest.fixture
    def sample_suite_result(self):
        """Create sample suite result."""
        metrics1 = DetectionMetrics(10, 2, 3, 85)
        metrics2 = DetectionMetrics(8, 3, 4, 85)

        results = [
            BenchmarkResult(
                dataset_name="test1",
                detector_type="label_errors",
                metrics=metrics1,
                execution_time=1.0,
                n_samples=100,
                n_ground_truth_issues=13,
                n_detected_issues=12,
            ),
            BenchmarkResult(
                dataset_name="test2",
                detector_type="outliers",
                metrics=metrics2,
                execution_time=0.5,
                n_samples=100,
                n_ground_truth_issues=12,
                n_detected_issues=11,
            ),
        ]

        return BenchmarkSuiteResult(
            results=results,
            total_time=1.5,
            config=BenchmarkConfig(),
        )

    def test_summary(self, sample_suite_result):
        """Test summary generation."""
        summary = sample_suite_result.summary()
        assert isinstance(summary, str)

    def test_to_dataframe(self, sample_suite_result):
        """Test DataFrame conversion."""
        df = sample_suite_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_to_json(self, sample_suite_result):
        """Test JSON export."""
        json_str = sample_suite_result.to_json()
        data = json.loads(json_str)
        assert "results" in data

    def test_get_leaderboard(self, sample_suite_result):
        """Test leaderboard generation."""
        leaderboard = sample_suite_result.get_leaderboard()
        assert isinstance(leaderboard, pd.DataFrame)


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_init(self):
        """Test suite initialization."""
        suite = BenchmarkSuite()
        assert suite is not None

    def test_run_single(self):
        """Test running single benchmark."""
        config = BenchmarkConfig(n_samples=200)
        suite = BenchmarkSuite(config=config)

        result = suite.run_single(
            dataset=BenchmarkDataset.SYNTHETIC_LABEL_ERRORS,
        )

        assert isinstance(result, BenchmarkResult)

    def test_run_all_label_errors(self):
        """Test running all benchmarks."""
        config = BenchmarkConfig(n_samples=200)
        suite = BenchmarkSuite(config=config)

        results = suite.run_all(detector_types=[DetectorType.LABEL_ERRORS])

        assert isinstance(results, BenchmarkSuiteResult)
        assert len(results.results) > 0

    def test_results_property(self):
        """Test results property."""
        config = BenchmarkConfig(n_samples=200)
        suite = BenchmarkSuite(config=config)

        suite.run_single(
            dataset=BenchmarkDataset.SYNTHETIC_LABEL_ERRORS,
        )

        results = suite.results
        assert isinstance(results, list)


class TestRunBenchmark:
    """Tests for run_benchmark convenience function."""

    def test_run_benchmark_basic(self):
        """Test basic benchmark run."""
        result = run_benchmark(
            dataset=BenchmarkDataset.SYNTHETIC_LABEL_ERRORS,
            detector=DetectorType.LABEL_ERRORS,
            n_samples=200,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.metrics is not None

    def test_run_benchmark_outliers(self):
        """Test outlier benchmark."""
        result = run_benchmark(
            dataset=BenchmarkDataset.SYNTHETIC_OUTLIERS,
            detector=DetectorType.OUTLIERS,
            n_samples=200,
        )

        assert isinstance(result, BenchmarkResult)


class TestCompareDetectors:
    """Tests for compare_detectors function."""

    def test_compare_basic(self):
        """Test basic comparison."""

        def dummy_detector(data, labels):
            return np.zeros(len(data), dtype=bool)

        result = compare_detectors(
            detectors=[dummy_detector],
            n_samples=200,
            n_runs=1,
        )

        assert isinstance(result, pd.DataFrame)


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset enum."""

    def test_synthetic_datasets(self):
        """Test synthetic dataset values."""
        assert BenchmarkDataset.SYNTHETIC_LABEL_ERRORS.value == "synthetic_label_errors"
        assert BenchmarkDataset.SYNTHETIC_DUPLICATES.value == "synthetic_duplicates"
        assert BenchmarkDataset.SYNTHETIC_OUTLIERS.value == "synthetic_outliers"


class TestDetectorType:
    """Tests for DetectorType enum."""

    def test_detector_types(self):
        """Test detector type values."""
        assert DetectorType.LABEL_ERRORS.value == "label_errors"
        assert DetectorType.DUPLICATES.value == "duplicates"
        assert DetectorType.OUTLIERS.value == "outliers"
        assert DetectorType.ALL.value == "all"
