"""Automated Data Augmentation based on Quality Issues.

This module provides intelligent data augmentation that targets specific
quality issues identified in the dataset, such as class imbalance,
underrepresented slices, or low-quality regions.

Example:
    >>> from clean.augmentation import DataAugmenter, AugmentationConfig
    >>>
    >>> # Create augmenter with quality report
    >>> augmenter = DataAugmenter(data=df, label_column="label", report=quality_report)
    >>>
    >>> # Get augmentation recommendations
    >>> recommendations = augmenter.recommend()
    >>> for rec in recommendations:
    ...     print(f"Augment {rec.target_class} by {rec.multiplier}x")
    >>>
    >>> # Apply augmentation
    >>> augmented_df = augmenter.augment(strategy="smart")
    >>> print(f"Original: {len(df)}, Augmented: {len(augmented_df)}")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.core.report import QualityReport
from clean.exceptions import CleanError, ConfigurationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AugmentationStrategy(Enum):
    """Augmentation strategies."""

    RANDOM = "random"  # Random oversampling
    SMOTE = "smote"  # Synthetic Minority Over-sampling
    SMART = "smart"  # Quality-aware augmentation
    CLASS_BALANCE = "class_balance"  # Balance classes only
    SLICE_BALANCE = "slice_balance"  # Balance underperforming slices
    CUSTOM = "custom"  # User-defined strategy


class AugmentationType(Enum):
    """Types of augmentation operations."""

    OVERSAMPLE = "oversample"  # Duplicate existing samples
    SYNTHETIC = "synthetic"  # Generate synthetic samples
    NOISE = "noise"  # Add noise to existing samples
    INTERPOLATE = "interpolate"  # Interpolate between samples
    REMOVE = "remove"  # Remove samples (downsampling)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    strategy: AugmentationStrategy = AugmentationStrategy.SMART
    target_balance_ratio: float = 1.0  # Target ratio between classes
    max_augmentation_factor: float = 5.0  # Max multiplier for any class
    min_class_samples: int = 100  # Minimum samples per class
    noise_level: float = 0.1  # Noise level for noisy augmentation
    random_seed: int = 42
    preserve_outliers: bool = False  # Whether to augment outliers


@dataclass
class AugmentationRecommendation:
    """Recommendation for augmentation."""

    target: str  # Class name or slice description
    target_type: str  # "class", "slice", "feature"
    current_count: int
    recommended_count: int
    multiplier: float
    augmentation_type: AugmentationType
    reason: str
    priority: int = 1  # 1 = highest

    @property
    def samples_to_add(self) -> int:
        """Calculate samples to add."""
        return max(0, self.recommended_count - self.current_count)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "target_type": self.target_type,
            "current_count": self.current_count,
            "recommended_count": self.recommended_count,
            "multiplier": self.multiplier,
            "samples_to_add": self.samples_to_add,
            "augmentation_type": self.augmentation_type.value,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class AugmentationResult:
    """Result of augmentation operation."""

    original_size: int
    augmented_size: int
    samples_added: int
    samples_removed: int
    augmentation_breakdown: dict[str, int]  # target -> count added
    quality_score_before: float | None
    quality_score_after: float | None
    recommendations_applied: list[str]

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Augmentation Summary",
            "=" * 40,
            f"Original Size: {self.original_size:,}",
            f"Augmented Size: {self.augmented_size:,}",
            f"Samples Added: {self.samples_added:,}",
            f"Samples Removed: {self.samples_removed:,}",
        ]

        if self.quality_score_before and self.quality_score_after:
            change = self.quality_score_after - self.quality_score_before
            lines.append(f"Quality Score: {self.quality_score_before:.2f} → {self.quality_score_after:.2f} ({change:+.2f})")

        if self.augmentation_breakdown:
            lines.append("")
            lines.append("Breakdown by Target:")
            for target, count in sorted(self.augmentation_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"  {target}: +{count}")

        return "\n".join(lines)


class AugmentationOperation(ABC):
    """Abstract base class for augmentation operations."""

    @abstractmethod
    def augment(
        self,
        data: pd.DataFrame,
        indices: list[int],
        n_samples: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Apply augmentation to selected samples.

        Args:
            data: Original data
            indices: Indices of samples to augment from
            n_samples: Number of new samples to generate
            **kwargs: Operation-specific parameters

        Returns:
            DataFrame with new augmented samples
        """
        pass


class OversamplingOperation(AugmentationOperation):
    """Simple oversampling (random duplication)."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def augment(
        self,
        data: pd.DataFrame,
        indices: list[int],
        n_samples: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Duplicate random samples."""
        if len(indices) == 0:
            return pd.DataFrame(columns=data.columns)

        # Sample with replacement
        selected = self.rng.choice(indices, size=n_samples, replace=True)
        return data.iloc[selected].copy()


class NoiseAugmentationOperation(AugmentationOperation):
    """Add Gaussian noise to numeric features."""

    def __init__(self, noise_level: float = 0.1, random_seed: int = 42):
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_seed)

    def augment(
        self,
        data: pd.DataFrame,
        indices: list[int],
        n_samples: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Add noise to samples."""
        if len(indices) == 0:
            return pd.DataFrame(columns=data.columns)

        # Sample base samples
        selected = self.rng.choice(indices, size=n_samples, replace=True)
        augmented = data.iloc[selected].copy()

        # Add noise to numeric columns
        numeric_cols = augmented.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = data[col].std()
            noise = self.rng.randn(n_samples) * std * self.noise_level
            augmented[col] = augmented[col].values + noise

        return augmented


class InterpolationOperation(AugmentationOperation):
    """Interpolate between pairs of samples (SMOTE-like)."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def augment(
        self,
        data: pd.DataFrame,
        indices: list[int],
        n_samples: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Interpolate between sample pairs."""
        if len(indices) < 2:
            return pd.DataFrame(columns=data.columns)

        augmented_rows = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in data.columns if c not in numeric_cols]

        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = self.rng.choice(indices, size=2, replace=False)
            sample1 = data.iloc[idx1]
            sample2 = data.iloc[idx2]

            # Interpolation factor
            alpha = self.rng.uniform(0.2, 0.8)

            # Create new sample
            new_sample = {}

            # Interpolate numeric columns
            for col in numeric_cols:
                new_sample[col] = sample1[col] * alpha + sample2[col] * (1 - alpha)

            # For non-numeric, randomly pick one
            for col in non_numeric_cols:
                new_sample[col] = sample1[col] if self.rng.random() < 0.5 else sample2[col]

            augmented_rows.append(new_sample)

        return pd.DataFrame(augmented_rows)


class SMOTEOperation(AugmentationOperation):
    """SMOTE (Synthetic Minority Over-sampling Technique)."""

    def __init__(self, k_neighbors: int = 5, random_seed: int = 42):
        self.k_neighbors = k_neighbors
        self.rng = np.random.RandomState(random_seed)

    def augment(
        self,
        data: pd.DataFrame,
        indices: list[int],
        n_samples: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate synthetic samples using SMOTE."""
        if len(indices) < 2:
            return pd.DataFrame(columns=data.columns)

        subset = data.iloc[indices]
        numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in subset.columns if c not in numeric_cols]

        if not numeric_cols:
            # Fall back to simple oversampling
            return OversamplingOperation(self.rng.randint(0, 10000)).augment(
                data, indices, n_samples
            )

        # Get numeric data
        X = subset[numeric_cols].values

        # Compute nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        k = min(self.k_neighbors, len(indices) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        neighbors = nn.kneighbors(X, return_distance=False)

        augmented_rows = []

        for _ in range(n_samples):
            # Pick random sample
            idx = self.rng.randint(len(indices))
            sample_idx = indices[idx]

            # Pick random neighbor (excluding self)
            neighbor_idx = neighbors[idx, self.rng.randint(1, k + 1)]
            neighbor_sample_idx = indices[neighbor_idx]

            # Interpolation
            alpha = self.rng.uniform(0, 1)
            new_numeric = X[idx] * alpha + X[neighbor_idx] * (1 - alpha)

            # Create new sample
            new_sample = dict(zip(numeric_cols, new_numeric))

            # Copy non-numeric from original
            for col in non_numeric_cols:
                new_sample[col] = subset.iloc[idx][col]

            augmented_rows.append(new_sample)

        return pd.DataFrame(augmented_rows)


class DataAugmenter:
    """Intelligent data augmenter based on quality analysis.

    Analyzes data quality issues and applies targeted augmentation
    to improve dataset balance and quality.

    Example:
        >>> augmenter = DataAugmenter(data=df, label_column="label")
        >>> recommendations = augmenter.recommend()
        >>> augmented_df = augmenter.augment(strategy="smart")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        report: QualityReport | None = None,
        config: AugmentationConfig | None = None,
    ):
        """Initialize the augmenter.

        Args:
            data: Input DataFrame
            label_column: Column containing labels
            report: Pre-computed quality report
            config: Augmentation configuration
        """
        self.data = data.copy()
        self.label_column = label_column
        self.config = config or AugmentationConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

        # Initialize operations
        self._operations = {
            AugmentationType.OVERSAMPLE: OversamplingOperation(self.config.random_seed),
            AugmentationType.NOISE: NoiseAugmentationOperation(
                self.config.noise_level, self.config.random_seed
            ),
            AugmentationType.INTERPOLATE: InterpolationOperation(self.config.random_seed),
            AugmentationType.SYNTHETIC: SMOTEOperation(random_seed=self.config.random_seed),
        }

        # Get or compute quality report
        if report is not None:
            self._report = report
        else:
            cleaner = DatasetCleaner(data=data, label_column=label_column)
            self._report = cleaner.analyze()

    def recommend(self) -> list[AugmentationRecommendation]:
        """Generate augmentation recommendations based on quality issues.

        Returns:
            List of AugmentationRecommendation objects
        """
        recommendations = []
        priority = 1

        # Class imbalance recommendations
        if self.label_column:
            recommendations.extend(self._recommend_class_balance())
            priority = len(recommendations) + 1

        # Slice-based recommendations (if quality report has slice info)
        # This would integrate with SliceDiscovery

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations

    def augment(
        self,
        strategy: AugmentationStrategy | str | None = None,
        recommendations: list[AugmentationRecommendation] | None = None,
        reanalyze_quality: bool = True,
    ) -> AugmentationResult:
        """Apply augmentation to the data.

        Args:
            strategy: Augmentation strategy to use
            recommendations: Optional specific recommendations to apply
            reanalyze_quality: Re-run quality analysis after augmentation

        Returns:
            AugmentationResult with augmented data accessible via .data
        """
        if isinstance(strategy, str):
            strategy = AugmentationStrategy(strategy)
        strategy = strategy or self.config.strategy

        original_size = len(self.data)
        quality_before = self._report.score.overall if self._report else None

        # Get recommendations if not provided
        if recommendations is None:
            if strategy == AugmentationStrategy.CLASS_BALANCE:
                recommendations = self._recommend_class_balance()
            elif strategy == AugmentationStrategy.SMART:
                recommendations = self.recommend()
            else:
                recommendations = self._recommend_class_balance()

        # Apply recommendations
        augmented_data = self.data.copy()
        augmentation_breakdown: dict[str, int] = {}
        samples_added = 0
        samples_removed = 0
        recommendations_applied = []

        for rec in recommendations:
            if rec.samples_to_add > 0:
                # Get indices for this target
                if rec.target_type == "class" and self.label_column:
                    indices = list(
                        augmented_data[augmented_data[self.label_column] == rec.target].index
                    )
                else:
                    indices = list(augmented_data.index)

                if not indices:
                    continue

                # Apply augmentation
                operation = self._operations.get(rec.augmentation_type)
                if operation is None:
                    operation = self._operations[AugmentationType.OVERSAMPLE]

                new_samples = operation.augment(
                    augmented_data,
                    indices,
                    rec.samples_to_add,
                )

                if len(new_samples) > 0:
                    augmented_data = pd.concat(
                        [augmented_data, new_samples],
                        ignore_index=True,
                    )
                    augmentation_breakdown[rec.target] = len(new_samples)
                    samples_added += len(new_samples)
                    recommendations_applied.append(f"{rec.target}: +{len(new_samples)}")

            elif rec.augmentation_type == AugmentationType.REMOVE:
                # Downsampling
                if rec.target_type == "class" and self.label_column:
                    class_indices = augmented_data[
                        augmented_data[self.label_column] == rec.target
                    ].index.tolist()

                    n_to_remove = rec.current_count - rec.recommended_count
                    if n_to_remove > 0 and len(class_indices) > n_to_remove:
                        remove_indices = self.rng.choice(
                            class_indices, size=n_to_remove, replace=False
                        )
                        augmented_data = augmented_data.drop(index=remove_indices)
                        samples_removed += n_to_remove
                        recommendations_applied.append(f"{rec.target}: -{n_to_remove}")

        # Store augmented data
        self.augmented_data = augmented_data.reset_index(drop=True)

        # Reanalyze quality
        quality_after = None
        if reanalyze_quality and self.label_column:
            cleaner = DatasetCleaner(
                data=self.augmented_data,
                label_column=self.label_column,
            )
            new_report = cleaner.analyze()
            quality_after = new_report.score.overall

        return AugmentationResult(
            original_size=original_size,
            augmented_size=len(self.augmented_data),
            samples_added=samples_added,
            samples_removed=samples_removed,
            augmentation_breakdown=augmentation_breakdown,
            quality_score_before=quality_before,
            quality_score_after=quality_after,
            recommendations_applied=recommendations_applied,
        )

    def get_augmented_data(self) -> pd.DataFrame:
        """Get the augmented data.

        Returns:
            Augmented DataFrame
        """
        if hasattr(self, "augmented_data"):
            return self.augmented_data.copy()
        return self.data.copy()

    def _recommend_class_balance(self) -> list[AugmentationRecommendation]:
        """Generate class balance recommendations."""
        recommendations = []

        if not self.label_column:
            return recommendations

        # Get class distribution
        class_counts = self.data[self.label_column].value_counts()
        majority_count = class_counts.max()
        minority_count = class_counts.min()

        if majority_count == 0:
            return recommendations

        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float("inf")

        # If imbalance is significant, recommend augmentation
        if imbalance_ratio > 2:
            target_count = int(majority_count * self.config.target_balance_ratio)

            priority = 1
            for class_name, count in class_counts.items():
                if count < target_count:
                    # Limit augmentation factor
                    max_augmented = int(count * self.config.max_augmentation_factor)
                    recommended_count = min(target_count, max_augmented)

                    # Ensure minimum samples
                    recommended_count = max(recommended_count, self.config.min_class_samples)

                    multiplier = recommended_count / count if count > 0 else 1.0

                    # Choose augmentation type based on multiplier
                    if multiplier > 3:
                        aug_type = AugmentationType.SYNTHETIC
                    elif multiplier > 2:
                        aug_type = AugmentationType.INTERPOLATE
                    else:
                        aug_type = AugmentationType.NOISE

                    recommendations.append(AugmentationRecommendation(
                        target=str(class_name),
                        target_type="class",
                        current_count=count,
                        recommended_count=recommended_count,
                        multiplier=multiplier,
                        augmentation_type=aug_type,
                        reason=f"Class imbalance ({imbalance_ratio:.1f}:1 ratio)",
                        priority=priority,
                    ))
                    priority += 1

        return recommendations

    def add_custom_operation(
        self,
        name: str,
        operation: AugmentationOperation,
    ) -> None:
        """Add a custom augmentation operation.

        Args:
            name: Name for the operation
            operation: AugmentationOperation instance
        """
        # Create a new enum value dynamically isn't ideal, so we use a dict
        self._operations[AugmentationType.CUSTOM] = operation


def augment_for_quality(
    data: pd.DataFrame,
    label_column: str | None = None,
    strategy: str = "smart",
    target_balance: float = 1.0,
    **kwargs: Any,
) -> tuple[pd.DataFrame, AugmentationResult]:
    """Convenience function for quality-aware augmentation.

    Args:
        data: Input DataFrame
        label_column: Column containing labels
        strategy: Augmentation strategy
        target_balance: Target class balance ratio
        **kwargs: Additional configuration options

    Returns:
        Tuple of (augmented DataFrame, AugmentationResult)

    Example:
        >>> augmented_df, result = augment_for_quality(df, "label", strategy="smart")
        >>> print(result.summary())
    """
    config = AugmentationConfig(
        strategy=AugmentationStrategy(strategy),
        target_balance_ratio=target_balance,
        **{k: v for k, v in kwargs.items() if k in AugmentationConfig.__dataclass_fields__},
    )

    augmenter = DataAugmenter(
        data=data,
        label_column=label_column,
        config=config,
    )

    result = augmenter.augment()
    return augmenter.get_augmented_data(), result


__all__ = [
    # Core classes
    "DataAugmenter",
    "AugmentationConfig",
    "AugmentationResult",
    "AugmentationRecommendation",
    # Operations
    "AugmentationOperation",
    "OversamplingOperation",
    "NoiseAugmentationOperation",
    "InterpolationOperation",
    "SMOTEOperation",
    # Enums
    "AugmentationStrategy",
    "AugmentationType",
    # Functions
    "augment_for_quality",
]
