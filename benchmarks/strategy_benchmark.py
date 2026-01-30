#!/usr/bin/env python
"""Performance benchmarks for Clean library strategy pattern implementations.

Run with: python benchmarks/strategy_benchmark.py

This benchmark compares the performance of the new strategy-based implementations
against baseline measurements to ensure the refactoring doesn't introduce overhead.
"""

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    std_time: float
    samples_per_second: float

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Avg time: {self.avg_time * 1000:.2f} ms\n"
            f"  Std time: {self.std_time * 1000:.2f} ms\n"
            f"  Throughput: {self.samples_per_second:.0f} samples/sec"
        )


def benchmark(
    func: Callable,
    iterations: int = 10,
    warmup: int = 2,
    n_samples: int = 1000,
) -> BenchmarkResult:
    """Run a benchmark function multiple times."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_arr = np.array(times)
    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=float(np.sum(times_arr)),
        avg_time=float(np.mean(times_arr)),
        std_time=float(np.std(times_arr)),
        samples_per_second=n_samples / float(np.mean(times_arr)),
    )


def create_test_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Create test data for benchmarks."""
    np.random.seed(42)
    data = {}

    # Numeric features
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_samples)

    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data["feature_0"][outlier_indices] *= 10

    # Add labels
    data["label"] = np.random.choice(["a", "b", "c"], size=n_samples)

    return pd.DataFrame(data)


def benchmark_outlier_detection():
    """Benchmark outlier detection strategies."""
    print("\n" + "=" * 60)
    print("Outlier Detection Strategy Benchmarks")
    print("=" * 60)

    from clean.detection.strategies import (
        IsolationForestStrategy,
        LOFStrategy,
        ZScoreStrategy,
        create_strategy,
    )

    n_samples = 1000
    df = create_test_data(n_samples=n_samples)
    features = df.drop(columns=["label"])
    labels = df["label"].values

    # Z-Score (fastest)
    zscore = ZScoreStrategy(threshold=3.0)

    def run_zscore():
        zscore.detect(features, labels)

    result = benchmark(run_zscore, n_samples=n_samples)
    print(f"\n{result}")

    # Isolation Forest
    try:
        iso_forest = IsolationForestStrategy(contamination=0.05)

        def run_iso_forest():
            iso_forest.detect(features, labels)

        result = benchmark(run_iso_forest, iterations=5, n_samples=n_samples)
        print(f"\n{result}")
    except Exception as e:
        print(f"\nIsolation Forest: Skipped ({e})")

    # LOF
    try:
        lof = LOFStrategy(n_neighbors=20, contamination=0.05)

        def run_lof():
            lof.detect(features, labels)

        result = benchmark(run_lof, iterations=5, n_samples=n_samples)
        print(f"\n{result}")
    except Exception as e:
        print(f"\nLOF: Skipped ({e})")


def benchmark_duplicate_detection():
    """Benchmark duplicate detection strategies."""
    print("\n" + "=" * 60)
    print("Duplicate Detection Strategy Benchmarks")
    print("=" * 60)

    from clean.detection.duplicate_strategies import (
        HashStrategy,
        FuzzyStrategy,
    )

    n_samples = 1000
    df = create_test_data(n_samples=n_samples)

    # Add some exact duplicates
    dup_indices = np.random.choice(n_samples, size=50, replace=False)
    for idx in dup_indices[:25]:
        df.iloc[dup_indices[25 + (idx % 25)]] = df.iloc[idx]

    # Hash Strategy - fit then detect
    hash_strat = HashStrategy()

    def run_hash():
        hash_strat.fit(df)
        hash_strat.detect(df, similarity_threshold=1.0, seen_pairs=set())

    result = benchmark(run_hash, n_samples=n_samples)
    print(f"\nHash Strategy (exact):\n{result}")

    # Fuzzy Strategy - fit then detect
    fuzzy = FuzzyStrategy(max_samples=1000)

    def run_fuzzy():
        fuzzy.fit(df)
        fuzzy.detect(df, similarity_threshold=0.9, seen_pairs=set())

    result = benchmark(run_fuzzy, iterations=5, n_samples=n_samples)
    print(f"\nFuzzy Strategy:\n{result}")


def benchmark_chunk_processing():
    """Benchmark chunk processing."""
    print("\n" + "=" * 60)
    print("Chunk Processing Benchmarks")
    print("=" * 60)

    from clean.processing import SyncChunkProcessor

    n_samples = 10000
    df = create_test_data(n_samples=n_samples, n_features=10)

    # Different chunk sizes
    for chunk_size in [100, 500, 1000, 2000]:
        processor = SyncChunkProcessor(chunk_size=chunk_size)

        def run_processor():
            list(processor.process_dataframe(df))
            processor.reset()

        result = benchmark(run_processor, iterations=5, n_samples=n_samples)
        print(f"\nChunk size {chunk_size}:")
        print(f"  Avg time: {result.avg_time * 1000:.2f} ms")
        print(f"  Throughput: {result.samples_per_second:.0f} samples/sec")


def benchmark_strategy_factory():
    """Benchmark strategy factory overhead."""
    print("\n" + "=" * 60)
    print("Strategy Factory Overhead Benchmarks")
    print("=" * 60)

    from clean.detection.strategies import create_strategy

    iterations = 10000

    # Measure factory creation time
    start = time.perf_counter()
    for _ in range(iterations):
        create_strategy("zscore", zscore_threshold=3.0)
    factory_time = time.perf_counter() - start

    print(f"\nStrategy factory creation:")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {factory_time * 1000:.2f} ms")
    print(f"  Avg time: {factory_time / iterations * 1000000:.2f} µs")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("Clean Library Strategy Pattern Benchmarks")
    print("=" * 60)

    benchmark_strategy_factory()
    benchmark_outlier_detection()
    benchmark_duplicate_detection()
    benchmark_chunk_processing()

    print("\n" + "=" * 60)
    print("Benchmarks Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
