"""Data Quality Benchmark Suite.

This module provides standardized benchmarks for evaluating data quality
detection algorithms on well-known datasets with ground truth labels.

Example:
    >>> from clean.benchmark import BenchmarkSuite, run_benchmark
    >>>
    >>> # Run benchmark on standard datasets
    >>> suite = BenchmarkSuite()
    >>> results = suite.run_all()
    >>> print(results.summary())
    >>>
    >>> # Run on specific dataset
    >>> result = run_benchmark("cifar10_corrupted", detector="label_errors")
    >>> print(f"Recall: {result.recall:.3f}, Precision: {result.precision:.3f}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.detection import LabelErrorDetector, DuplicateDetector, OutlierDetector
from clean.exceptions import CleanError, ConfigurationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BenchmarkDataset(Enum):
    """Available benchmark datasets."""

    # Label error benchmarks
    CIFAR10_CORRUPTED = "cifar10_corrupted"
    MNIST_CORRUPTED = "mnist_corrupted"
    IMAGENET_CORRUPTED = "imagenet_corrupted"

    # Duplicate benchmarks
    TEXT_DUPLICATES = "text_duplicates"
    IMAGE_DUPLICATES = "image_duplicates"

    # Outlier benchmarks
    TABULAR_OUTLIERS = "tabular_outliers"
    CREDIT_FRAUD = "credit_fraud"

    # Synthetic benchmarks (generated on demand)
    SYNTHETIC_LABEL_ERRORS = "synthetic_label_errors"
    SYNTHETIC_DUPLICATES = "synthetic_duplicates"
    SYNTHETIC_OUTLIERS = "synthetic_outliers"


class DetectorType(Enum):
    """Types of detectors to benchmark."""

    LABEL_ERRORS = "label_errors"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    ALL = "all"


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks."""

    n_samples: int = 10000
    corruption_rate: float = 0.05  # Rate of injected issues
    random_seed: int = 42
    cv_folds: int = 5
    timeout_seconds: float = 300.0
    save_results: bool = True
    results_dir: str = ".benchmark_results"


@dataclass
class DetectionMetrics:
    """Metrics for detection performance."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        """Calculate precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity)."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
        }


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    dataset_name: str
    detector_type: str
    metrics: DetectionMetrics
    execution_time: float
    n_samples: int
    n_ground_truth_issues: int
    n_detected_issues: int
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Benchmark: {self.dataset_name} - {self.detector_type}",
            "=" * 50,
            f"Samples: {self.n_samples:,}",
            f"Ground Truth Issues: {self.n_ground_truth_issues}",
            f"Detected Issues: {self.n_detected_issues}",
            "",
            "Metrics:",
            f"  Precision: {self.metrics.precision:.3f}",
            f"  Recall:    {self.metrics.recall:.3f}",
            f"  F1 Score:  {self.metrics.f1:.3f}",
            "",
            f"Execution Time: {self.execution_time:.2f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset_name,
            "detector": self.detector_type,
            "metrics": self.metrics.to_dict(),
            "execution_time": self.execution_time,
            "n_samples": self.n_samples,
            "n_ground_truth": self.n_ground_truth_issues,
            "n_detected": self.n_detected_issues,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkSuiteResult:
    """Results from running multiple benchmarks."""

    results: list[BenchmarkResult]
    total_time: float
    config: BenchmarkConfig

    def summary(self) -> str:
        """Generate summary of all results."""
        lines = [
            "Benchmark Suite Results",
            "=" * 60,
            f"Total Benchmarks: {len(self.results)}",
            f"Total Time: {self.total_time:.1f}s",
            "",
            "Results by Dataset:",
        ]

        for result in self.results:
            lines.append(
                f"  {result.dataset_name:25s} | "
                f"P: {result.metrics.precision:.3f} | "
                f"R: {result.metrics.recall:.3f} | "
                f"F1: {result.metrics.f1:.3f}"
            )

        # Overall averages
        if self.results:
            avg_precision = np.mean([r.metrics.precision for r in self.results])
            avg_recall = np.mean([r.metrics.recall for r in self.results])
            avg_f1 = np.mean([r.metrics.f1 for r in self.results])
            lines.extend([
                "",
                "Overall Averages:",
                f"  Precision: {avg_precision:.3f}",
                f"  Recall:    {avg_recall:.3f}",
                f"  F1 Score:  {avg_f1:.3f}",
            ])

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for result in self.results:
            row = {
                "dataset": result.dataset_name,
                "detector": result.detector_type,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1": result.metrics.f1,
                "execution_time": result.execution_time,
                "n_samples": result.n_samples,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "results": [r.to_dict() for r in self.results],
            "total_time": self.total_time,
        }, indent=2)

    def get_leaderboard(self) -> pd.DataFrame:
        """Generate leaderboard sorted by F1 score."""
        df = self.to_dataframe()
        return df.sort_values("f1", ascending=False).reset_index(drop=True)


class SyntheticDataGenerator:
    """Generator for synthetic benchmark data with known issues."""

    def __init__(self, random_seed: int = 42):
        """Initialize generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)

    def generate_label_errors(
        self,
        n_samples: int = 10000,
        n_classes: int = 10,
        n_features: int = 100,
        error_rate: float = 0.05,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Generate synthetic dataset with label errors.

        Args:
            n_samples: Number of samples
            n_classes: Number of classes
            n_features: Number of features
            error_rate: Rate of label corruption

        Returns:
            Tuple of (data DataFrame, ground truth labels, error mask)
        """
        # Generate features with class-dependent patterns
        X = np.zeros((n_samples, n_features))
        true_labels = self.rng.randint(0, n_classes, n_samples)

        for i in range(n_samples):
            class_idx = true_labels[i]
            # Each class has a different mean pattern
            mean = np.zeros(n_features)
            mean[class_idx * (n_features // n_classes):(class_idx + 1) * (n_features // n_classes)] = 1.0
            X[i] = mean + self.rng.randn(n_features) * 0.5

        # Corrupt labels
        n_errors = int(n_samples * error_rate)
        error_indices = self.rng.choice(n_samples, n_errors, replace=False)
        error_mask = np.zeros(n_samples, dtype=bool)
        error_mask[error_indices] = True

        corrupted_labels = true_labels.copy()
        for idx in error_indices:
            # Change to a different random class
            new_label = self.rng.randint(0, n_classes)
            while new_label == true_labels[idx]:
                new_label = self.rng.randint(0, n_classes)
            corrupted_labels[idx] = new_label

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)
        df["label"] = corrupted_labels
        df["true_label"] = true_labels

        return df, corrupted_labels, error_mask

    def generate_duplicates(
        self,
        n_samples: int = 10000,
        n_features: int = 50,
        duplicate_rate: float = 0.05,
        near_duplicate_noise: float = 0.01,
    ) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
        """Generate synthetic dataset with duplicates.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            duplicate_rate: Rate of duplicate pairs
            near_duplicate_noise: Noise level for near-duplicates

        Returns:
            Tuple of (data DataFrame, list of duplicate pairs)
        """
        # Generate base samples
        n_unique = int(n_samples * (1 - duplicate_rate))
        X_unique = self.rng.randn(n_unique, n_features)

        # Generate duplicates
        n_duplicates = n_samples - n_unique
        duplicate_sources = self.rng.choice(n_unique, n_duplicates)
        X_duplicates = X_unique[duplicate_sources] + self.rng.randn(n_duplicates, n_features) * near_duplicate_noise

        # Combine
        X = np.vstack([X_unique, X_duplicates])

        # Record duplicate pairs
        duplicate_pairs = []
        for i, source in enumerate(duplicate_sources):
            duplicate_pairs.append((int(source), n_unique + i))

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)

        return df, duplicate_pairs

    def generate_outliers(
        self,
        n_samples: int = 10000,
        n_features: int = 20,
        outlier_rate: float = 0.05,
        outlier_magnitude: float = 5.0,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic dataset with outliers.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            outlier_rate: Rate of outliers
            outlier_magnitude: How far outliers deviate

        Returns:
            Tuple of (data DataFrame, outlier mask)
        """
        # Generate normal data
        X = self.rng.randn(n_samples, n_features)

        # Add outliers
        n_outliers = int(n_samples * outlier_rate)
        outlier_indices = self.rng.choice(n_samples, n_outliers, replace=False)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        outlier_mask[outlier_indices] = True

        for idx in outlier_indices:
            # Randomly select features to make outliers
            n_outlier_features = self.rng.randint(1, n_features // 2 + 1)
            outlier_features = self.rng.choice(n_features, n_outlier_features, replace=False)
            for f in outlier_features:
                X[idx, f] += self.rng.choice([-1, 1]) * outlier_magnitude

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)

        return df, outlier_mask


class BenchmarkRunner:
    """Runner for individual benchmarks."""

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.generator = SyntheticDataGenerator(self.config.random_seed)

    def run_label_error_benchmark(
        self,
        data: pd.DataFrame | None = None,
        ground_truth: np.ndarray | None = None,
        label_column: str = "label",
    ) -> BenchmarkResult:
        """Run label error detection benchmark.

        Args:
            data: Optional data (generates synthetic if None)
            ground_truth: Boolean mask of true label errors
            label_column: Column containing labels

        Returns:
            BenchmarkResult with metrics
        """
        start_time = time.time()

        # Generate synthetic data if not provided
        if data is None:
            data, _, ground_truth = self.generator.generate_label_errors(
                n_samples=self.config.n_samples,
                error_rate=self.config.corruption_rate,
            )

        # Get features (exclude label columns)
        feature_cols = [c for c in data.columns if c not in [label_column, "true_label"]]
        X = data[feature_cols].values
        y = data[label_column].values

        # Run detector
        detector = LabelErrorDetector(cv_folds=self.config.cv_folds)
        detected_indices = set()

        try:
            errors_df = detector.find_errors(X, y)
            if len(errors_df) > 0:
                if "index" in errors_df.columns:
                    detected_indices = set(errors_df["index"].tolist())
                else:
                    detected_indices = set(errors_df.index.tolist())
        except Exception as e:
            logger.warning(f"Label error detection failed: {e}")

        execution_time = time.time() - start_time

        # Calculate metrics
        if ground_truth is not None:
            gt_indices = set(np.where(ground_truth)[0])
        else:
            gt_indices = set()

        metrics = self._calculate_metrics(gt_indices, detected_indices, len(data))

        return BenchmarkResult(
            dataset_name="label_errors",
            detector_type="label_error_detector",
            metrics=metrics,
            execution_time=execution_time,
            n_samples=len(data),
            n_ground_truth_issues=len(gt_indices),
            n_detected_issues=len(detected_indices),
            config={"cv_folds": self.config.cv_folds},
        )

    def run_duplicate_benchmark(
        self,
        data: pd.DataFrame | None = None,
        ground_truth_pairs: list[tuple[int, int]] | None = None,
    ) -> BenchmarkResult:
        """Run duplicate detection benchmark.

        Args:
            data: Optional data (generates synthetic if None)
            ground_truth_pairs: List of (idx1, idx2) duplicate pairs

        Returns:
            BenchmarkResult with metrics
        """
        start_time = time.time()

        # Generate synthetic data if not provided
        if data is None:
            data, ground_truth_pairs = self.generator.generate_duplicates(
                n_samples=self.config.n_samples,
                duplicate_rate=self.config.corruption_rate,
            )

        # Run detector
        detector = DuplicateDetector(methods=["hash", "embedding"])
        detected_pairs: set[tuple[int, int]] = set()

        try:
            dup_df = detector.find_duplicates(data)
            if len(dup_df) > 0:
                for _, row in dup_df.iterrows():
                    idx1 = row.get("index_1", row.get("index", 0))
                    idx2 = row.get("index_2", idx1)
                    pair = tuple(sorted([int(idx1), int(idx2)]))
                    detected_pairs.add(pair)
        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e}")

        execution_time = time.time() - start_time

        # Calculate pair-level metrics
        if ground_truth_pairs is not None:
            gt_pairs = {tuple(sorted(p)) for p in ground_truth_pairs}
        else:
            gt_pairs = set()

        tp = len(gt_pairs & detected_pairs)
        fp = len(detected_pairs - gt_pairs)
        fn = len(gt_pairs - detected_pairs)
        tn = 0  # Not meaningful for pair detection

        metrics = DetectionMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
        )

        return BenchmarkResult(
            dataset_name="duplicates",
            detector_type="duplicate_detector",
            metrics=metrics,
            execution_time=execution_time,
            n_samples=len(data),
            n_ground_truth_issues=len(gt_pairs),
            n_detected_issues=len(detected_pairs),
        )

    def run_outlier_benchmark(
        self,
        data: pd.DataFrame | None = None,
        ground_truth: np.ndarray | None = None,
    ) -> BenchmarkResult:
        """Run outlier detection benchmark.

        Args:
            data: Optional data (generates synthetic if None)
            ground_truth: Boolean mask of true outliers

        Returns:
            BenchmarkResult with metrics
        """
        start_time = time.time()

        # Generate synthetic data if not provided
        if data is None:
            data, ground_truth = self.generator.generate_outliers(
                n_samples=self.config.n_samples,
                outlier_rate=self.config.corruption_rate,
            )

        # Run detector
        detector = OutlierDetector(method="isolation_forest")
        detected_indices: set[int] = set()

        try:
            outliers_df = detector.find_outliers(data)
            if len(outliers_df) > 0:
                if "index" in outliers_df.columns:
                    detected_indices = set(outliers_df["index"].tolist())
                else:
                    detected_indices = set(outliers_df.index.tolist())
        except Exception as e:
            logger.warning(f"Outlier detection failed: {e}")

        execution_time = time.time() - start_time

        # Calculate metrics
        if ground_truth is not None:
            gt_indices = set(np.where(ground_truth)[0])
        else:
            gt_indices = set()

        metrics = self._calculate_metrics(gt_indices, detected_indices, len(data))

        return BenchmarkResult(
            dataset_name="outliers",
            detector_type="outlier_detector",
            metrics=metrics,
            execution_time=execution_time,
            n_samples=len(data),
            n_ground_truth_issues=len(gt_indices),
            n_detected_issues=len(detected_indices),
        )

    def _calculate_metrics(
        self,
        ground_truth: set[int],
        detected: set[int],
        total_samples: int,
    ) -> DetectionMetrics:
        """Calculate detection metrics."""
        tp = len(ground_truth & detected)
        fp = len(detected - ground_truth)
        fn = len(ground_truth - detected)
        tn = total_samples - tp - fp - fn

        return DetectionMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
        )


class BenchmarkSuite:
    """Suite for running multiple benchmarks.

    Example:
        >>> suite = BenchmarkSuite()
        >>> results = suite.run_all()
        >>> print(results.summary())
        >>> results.to_dataframe().to_csv("benchmark_results.csv")
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.runner = BenchmarkRunner(self.config)
        self._results: list[BenchmarkResult] = []

    def run_all(
        self,
        detector_types: list[DetectorType] | None = None,
    ) -> BenchmarkSuiteResult:
        """Run all benchmarks.

        Args:
            detector_types: Types of detectors to benchmark (default: all)

        Returns:
            BenchmarkSuiteResult with all results
        """
        start_time = time.time()
        results = []

        detector_types = detector_types or [
            DetectorType.LABEL_ERRORS,
            DetectorType.DUPLICATES,
            DetectorType.OUTLIERS,
        ]

        # Run synthetic benchmarks for each detector type
        for detector_type in detector_types:
            if detector_type == DetectorType.LABEL_ERRORS or detector_type == DetectorType.ALL:
                logger.info("Running label error benchmark...")
                results.append(self.runner.run_label_error_benchmark())

            if detector_type == DetectorType.DUPLICATES or detector_type == DetectorType.ALL:
                logger.info("Running duplicate benchmark...")
                results.append(self.runner.run_duplicate_benchmark())

            if detector_type == DetectorType.OUTLIERS or detector_type == DetectorType.ALL:
                logger.info("Running outlier benchmark...")
                results.append(self.runner.run_outlier_benchmark())

        total_time = time.time() - start_time
        self._results = results

        suite_result = BenchmarkSuiteResult(
            results=results,
            total_time=total_time,
            config=self.config,
        )

        # Save results if configured
        if self.config.save_results:
            self._save_results(suite_result)

        return suite_result

    def run_single(
        self,
        dataset: BenchmarkDataset,
        data: pd.DataFrame | None = None,
        ground_truth: Any = None,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            dataset: Which benchmark dataset to use
            data: Optional custom data
            ground_truth: Optional ground truth

        Returns:
            BenchmarkResult
        """
        if dataset in [BenchmarkDataset.CIFAR10_CORRUPTED, BenchmarkDataset.MNIST_CORRUPTED,
                       BenchmarkDataset.IMAGENET_CORRUPTED, BenchmarkDataset.SYNTHETIC_LABEL_ERRORS]:
            return self.runner.run_label_error_benchmark(data, ground_truth)

        elif dataset in [BenchmarkDataset.TEXT_DUPLICATES, BenchmarkDataset.IMAGE_DUPLICATES,
                         BenchmarkDataset.SYNTHETIC_DUPLICATES]:
            return self.runner.run_duplicate_benchmark(data, ground_truth)

        elif dataset in [BenchmarkDataset.TABULAR_OUTLIERS, BenchmarkDataset.CREDIT_FRAUD,
                         BenchmarkDataset.SYNTHETIC_OUTLIERS]:
            return self.runner.run_outlier_benchmark(data, ground_truth)

        else:
            raise ConfigurationError(f"Unknown dataset: {dataset}")

    def add_custom_benchmark(
        self,
        name: str,
        data: pd.DataFrame,
        ground_truth: Any,
        detector_type: DetectorType,
    ) -> BenchmarkResult:
        """Add and run a custom benchmark.

        Args:
            name: Name for the benchmark
            data: Data to benchmark on
            ground_truth: Ground truth labels/indices
            detector_type: Which detector to use

        Returns:
            BenchmarkResult
        """
        if detector_type == DetectorType.LABEL_ERRORS:
            result = self.runner.run_label_error_benchmark(data, ground_truth)
        elif detector_type == DetectorType.DUPLICATES:
            result = self.runner.run_duplicate_benchmark(data, ground_truth)
        elif detector_type == DetectorType.OUTLIERS:
            result = self.runner.run_outlier_benchmark(data, ground_truth)
        else:
            raise ConfigurationError(f"Invalid detector type: {detector_type}")

        # Override name
        result.dataset_name = name
        self._results.append(result)
        return result

    def _save_results(self, results: BenchmarkSuiteResult) -> None:
        """Save results to disk."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"benchmark_{timestamp}.json"

        with open(filepath, "w") as f:
            f.write(results.to_json())

        logger.info(f"Results saved to {filepath}")

    @property
    def results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results.copy()


def run_benchmark(
    dataset: str | BenchmarkDataset,
    detector: str = "all",
    n_samples: int = 10000,
    corruption_rate: float = 0.05,
    **kwargs: Any,
) -> BenchmarkResult | BenchmarkSuiteResult:
    """Run a benchmark with the specified configuration.

    Args:
        dataset: Dataset name or BenchmarkDataset enum
        detector: Detector type ("label_errors", "duplicates", "outliers", "all")
        n_samples: Number of samples
        corruption_rate: Rate of injected issues
        **kwargs: Additional configuration

    Returns:
        BenchmarkResult or BenchmarkSuiteResult

    Example:
        >>> result = run_benchmark("synthetic_label_errors", n_samples=5000)
        >>> print(f"F1: {result.metrics.f1:.3f}")
    """
    config = BenchmarkConfig(
        n_samples=n_samples,
        corruption_rate=corruption_rate,
        **{k: v for k, v in kwargs.items() if k in BenchmarkConfig.__dataclass_fields__},
    )

    suite = BenchmarkSuite(config)

    if detector == "all":
        return suite.run_all()

    if isinstance(dataset, str):
        try:
            dataset = BenchmarkDataset(dataset.lower())
        except ValueError:
            # Treat as synthetic
            dataset = BenchmarkDataset.SYNTHETIC_LABEL_ERRORS

    return suite.run_single(dataset)


def compare_detectors(
    detectors: list[Callable],
    n_samples: int = 10000,
    n_runs: int = 5,
) -> pd.DataFrame:
    """Compare multiple detector implementations.

    Args:
        detectors: List of detector callables
        n_samples: Samples per run
        n_runs: Number of runs for averaging

    Returns:
        DataFrame with comparison results
    """
    results = []
    generator = SyntheticDataGenerator()

    for run in range(n_runs):
        # Generate fresh data each run
        data, _, ground_truth = generator.generate_label_errors(
            n_samples=n_samples,
            error_rate=0.05,
        )

        gt_indices = set(np.where(ground_truth)[0])

        for i, detector in enumerate(detectors):
            name = getattr(detector, "__name__", f"detector_{i}")

            start = time.time()
            try:
                detected = detector(data)
                detected_indices = set(detected) if detected is not None else set()
            except Exception as e:
                logger.warning(f"Detector {name} failed: {e}")
                detected_indices = set()
            elapsed = time.time() - start

            tp = len(gt_indices & detected_indices)
            fp = len(detected_indices - gt_indices)
            fn = len(gt_indices - detected_indices)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                "detector": name,
                "run": run,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "time": elapsed,
            })

    df = pd.DataFrame(results)

    # Aggregate by detector
    summary = df.groupby("detector").agg({
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"],
        "time": ["mean", "std"],
    }).round(3)

    return summary


__all__ = [
    # Core classes
    "BenchmarkSuite",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "DetectionMetrics",
    "SyntheticDataGenerator",
    # Config
    "BenchmarkConfig",
    # Enums
    "BenchmarkDataset",
    "DetectorType",
    # Functions
    "run_benchmark",
    "compare_detectors",
]
