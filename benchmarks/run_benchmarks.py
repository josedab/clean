"""Benchmark suite for Clean data quality detection.

This module provides utilities for benchmarking Clean's detection
capabilities against known ground truth datasets.

Usage:
    python -m benchmarks.run_benchmarks
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from clean import DatasetCleaner


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    dataset_size: int
    n_injected_issues: int
    n_detected_issues: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Detection accuracy."""
        total = self.true_positives + self.false_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def summary(self) -> str:
        """Generate summary of all benchmarks."""
        lines = [
            f"Benchmark Suite: {self.name}",
            f"Timestamp: {self.timestamp}",
            "=" * 60,
            "",
            f"{'Benchmark':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>10}",
            "-" * 60,
        ]

        for r in self.results:
            lines.append(
                f"{r.name:<25} {r.precision:>10.3f} {r.recall:>10.3f} "
                f"{r.f1_score:>10.3f} {r.runtime_seconds:>9.2f}s"
            )

        lines.append("-" * 60)

        # Averages
        avg_precision = np.mean([r.precision for r in self.results])
        avg_recall = np.mean([r.recall for r in self.results])
        avg_f1 = np.mean([r.f1_score for r in self.results])
        total_time = sum(r.runtime_seconds for r in self.results)

        lines.append(
            f"{'AVERAGE':<25} {avg_precision:>10.3f} {avg_recall:>10.3f} "
            f"{avg_f1:>10.3f} {total_time:>9.2f}s"
        )

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export to JSON."""
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        """Save results to file."""
        Path(path).write_text(self.to_json())


def create_label_error_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 5,
    error_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, set[int]]:
    """Create a synthetic dataset with known label errors.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        error_rate: Fraction of labels to corrupt
        random_state: Random seed

    Returns:
        Tuple of (features DataFrame, corrupted labels, set of error indices)
    """
    rng = np.random.RandomState(random_state)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])

    # Inject label errors
    n_errors = int(n_samples * error_rate)
    error_indices = set(rng.choice(n_samples, size=n_errors, replace=False))

    corrupted_labels = y.copy()
    for idx in error_indices:
        original = y[idx]
        # Change to a different random class
        new_label = rng.choice([c for c in range(n_classes) if c != original])
        corrupted_labels[idx] = new_label

    return df, corrupted_labels, error_indices


def create_duplicate_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    duplicate_rate: float = 0.05,
    near_duplicate_noise: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, set[tuple[int, int]]]:
    """Create a synthetic dataset with known duplicates.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        duplicate_rate: Fraction of samples to duplicate
        near_duplicate_noise: Noise level for near-duplicates
        random_state: Random seed

    Returns:
        Tuple of (features DataFrame, labels, set of duplicate pair indices)
    """
    rng = np.random.RandomState(random_state)

    # Create base data
    n_base = int(n_samples * (1 - duplicate_rate))
    X, y = make_classification(
        n_samples=n_base,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=3,
        random_state=random_state,
    )

    # Create duplicates
    n_duplicates = n_samples - n_base
    dup_indices = rng.choice(n_base, size=n_duplicates, replace=True)

    X_dups = X[dup_indices] + rng.normal(0, near_duplicate_noise, (n_duplicates, n_features))
    y_dups = y[dup_indices]

    # Combine
    X_full = np.vstack([X, X_dups])
    y_full = np.concatenate([y, y_dups])

    df = pd.DataFrame(X_full, columns=[f"feature_{i}" for i in range(n_features)])

    # Track duplicate pairs
    duplicate_pairs = set()
    for i, orig_idx in enumerate(dup_indices):
        duplicate_pairs.add((orig_idx, n_base + i))

    return df, y_full, duplicate_pairs


def create_outlier_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    outlier_rate: float = 0.05,
    outlier_magnitude: float = 5.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, set[int]]:
    """Create a synthetic dataset with known outliers.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        outlier_rate: Fraction of samples to make outliers
        outlier_magnitude: How many std devs away outliers are
        random_state: Random seed

    Returns:
        Tuple of (features DataFrame, labels, set of outlier indices)
    """
    rng = np.random.RandomState(random_state)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=3,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])

    # Inject outliers
    n_outliers = int(n_samples * outlier_rate)
    outlier_indices = set(rng.choice(n_samples, size=n_outliers, replace=False))

    for idx in outlier_indices:
        # Add outlier magnitude to random features
        n_outlier_features = rng.randint(1, n_features // 2)
        outlier_features = rng.choice(n_features, size=n_outlier_features, replace=False)
        for feat_idx in outlier_features:
            col = f"feature_{feat_idx}"
            direction = rng.choice([-1, 1])
            df.loc[idx, col] += direction * outlier_magnitude * df[col].std()

    return df, y, outlier_indices


def benchmark_label_errors(
    n_samples: int = 1000,
    error_rate: float = 0.05,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark label error detection.

    Args:
        n_samples: Dataset size
        error_rate: Fraction of label errors
        random_state: Random seed

    Returns:
        BenchmarkResult with detection metrics
    """
    # Create dataset
    df, labels, true_errors = create_label_error_dataset(
        n_samples=n_samples,
        error_rate=error_rate,
        random_state=random_state,
    )

    df["label"] = labels

    # Run detection
    start_time = time.time()
    cleaner = DatasetCleaner(data=df, label_column="label")
    report = cleaner.analyze(
        detect_label_errors=True,
        detect_duplicates=False,
        detect_outliers=False,
        detect_imbalance=False,
        detect_bias=False,
        show_progress=False,
    )
    runtime = time.time() - start_time

    # Compute metrics
    if report.label_errors_result:
        detected = {e.index for e in report.label_errors_result.issues}
    else:
        detected = set()

    tp = len(detected & true_errors)
    fp = len(detected - true_errors)
    fn = len(true_errors - detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return BenchmarkResult(
        name="label_errors",
        dataset_size=n_samples,
        n_injected_issues=len(true_errors),
        n_detected_issues=len(detected),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        runtime_seconds=runtime,
        metadata={"error_rate": error_rate},
    )


def benchmark_duplicates(
    n_samples: int = 1000,
    duplicate_rate: float = 0.05,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark duplicate detection.

    Args:
        n_samples: Dataset size
        duplicate_rate: Fraction of duplicates
        random_state: Random seed

    Returns:
        BenchmarkResult with detection metrics
    """
    df, labels, true_pairs = create_duplicate_dataset(
        n_samples=n_samples,
        duplicate_rate=duplicate_rate,
        random_state=random_state,
    )

    df["label"] = labels

    # Run detection
    start_time = time.time()
    cleaner = DatasetCleaner(data=df, label_column="label")
    report = cleaner.analyze(
        detect_label_errors=False,
        detect_duplicates=True,
        detect_outliers=False,
        detect_imbalance=False,
        detect_bias=False,
        show_progress=False,
    )
    runtime = time.time() - start_time

    # Compute metrics (for duplicates, we check if pairs are detected)
    if report.duplicates_result:
        detected_pairs = {(d.index1, d.index2) for d in report.duplicates_result.issues}
        # Also check reverse pairs
        detected_pairs_both = detected_pairs | {(d.index2, d.index1) for d in report.duplicates_result.issues}
    else:
        detected_pairs_both = set()

    # True pairs that were detected (in either direction)
    true_pairs_both = true_pairs | {(b, a) for a, b in true_pairs}
    tp = len(detected_pairs_both & true_pairs_both) // 2  # Divide by 2 since we doubled
    fp = len(detected_pairs_both - true_pairs_both) // 2
    fn = len(true_pairs) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return BenchmarkResult(
        name="duplicates",
        dataset_size=n_samples,
        n_injected_issues=len(true_pairs),
        n_detected_issues=len(detected_pairs_both) // 2,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        runtime_seconds=runtime,
        metadata={"duplicate_rate": duplicate_rate},
    )


def benchmark_outliers(
    n_samples: int = 1000,
    outlier_rate: float = 0.05,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark outlier detection.

    Args:
        n_samples: Dataset size
        outlier_rate: Fraction of outliers
        random_state: Random seed

    Returns:
        BenchmarkResult with detection metrics
    """
    df, labels, true_outliers = create_outlier_dataset(
        n_samples=n_samples,
        outlier_rate=outlier_rate,
        random_state=random_state,
    )

    df["label"] = labels

    # Run detection
    start_time = time.time()
    cleaner = DatasetCleaner(data=df, label_column="label")
    report = cleaner.analyze(
        detect_label_errors=False,
        detect_duplicates=False,
        detect_outliers=True,
        detect_imbalance=False,
        detect_bias=False,
        show_progress=False,
    )
    runtime = time.time() - start_time

    # Compute metrics
    if report.outliers_result:
        detected = {o.index for o in report.outliers_result.issues}
    else:
        detected = set()

    tp = len(detected & true_outliers)
    fp = len(detected - true_outliers)
    fn = len(true_outliers - detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return BenchmarkResult(
        name="outliers",
        dataset_size=n_samples,
        n_injected_issues=len(true_outliers),
        n_detected_issues=len(detected),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        runtime_seconds=runtime,
        metadata={"outlier_rate": outlier_rate},
    )


def run_all_benchmarks(
    sizes: list[int] | None = None,
    rates: list[float] | None = None,
    random_state: int = 42,
) -> BenchmarkSuite:
    """Run all benchmarks with various configurations.

    Args:
        sizes: Dataset sizes to test
        rates: Issue rates to test
        random_state: Base random seed

    Returns:
        BenchmarkSuite with all results
    """
    from datetime import datetime

    sizes = sizes or [500, 1000, 2000]
    rates = rates or [0.05, 0.10]

    suite = BenchmarkSuite(
        name="Clean Detection Benchmarks",
        timestamp=datetime.now().isoformat(),
        metadata={"sizes": sizes, "rates": rates, "random_state": random_state},
    )

    for size in sizes:
        for rate in rates:
            # Label errors
            result = benchmark_label_errors(
                n_samples=size,
                error_rate=rate,
                random_state=random_state,
            )
            result.name = f"label_errors_{size}_{int(rate*100)}pct"
            suite.add_result(result)

            # Duplicates
            result = benchmark_duplicates(
                n_samples=size,
                duplicate_rate=rate,
                random_state=random_state,
            )
            result.name = f"duplicates_{size}_{int(rate*100)}pct"
            suite.add_result(result)

            # Outliers
            result = benchmark_outliers(
                n_samples=size,
                outlier_rate=rate,
                random_state=random_state,
            )
            result.name = f"outliers_{size}_{int(rate*100)}pct"
            suite.add_result(result)

    return suite


if __name__ == "__main__":
    print("Running Clean Detection Benchmarks...")
    print()

    suite = run_all_benchmarks(
        sizes=[500, 1000],
        rates=[0.05, 0.10],
    )

    print(suite.summary())
    print()

    # Save results
    output_path = Path(__file__).parent / "results" / "latest.json"
    output_path.parent.mkdir(exist_ok=True)
    suite.save(output_path)
    print(f"Results saved to: {output_path}")
