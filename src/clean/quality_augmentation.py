"""Quality-Aware Data Augmentation.

This module provides intelligent data augmentation that specifically
addresses detected quality gaps like class imbalance, underrepresented
slices, and edge cases.

Example:
    >>> from clean.quality_augmentation import QualityAwareAugmenter
    >>>
    >>> augmenter = QualityAwareAugmenter()
    >>> result = augmenter.augment(data, quality_report)
    >>> print(f"Generated {result.n_samples} new samples")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from clean.core.report import QualityReport

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of quality gaps that can be addressed."""

    CLASS_IMBALANCE = "class_imbalance"
    UNDERREPRESENTED_SLICE = "underrepresented_slice"
    EDGE_CASE = "edge_case"
    LOW_DIVERSITY = "low_diversity"
    BOUNDARY_REGION = "boundary_region"


class AugmentationMethod(Enum):
    """Available augmentation methods."""

    SMOTE = "smote"
    ADASYN = "adasyn"
    INTERPOLATION = "interpolation"
    NOISE_INJECTION = "noise_injection"
    MIXUP = "mixup"
    CUTMIX = "cutmix"
    FEATURE_VARIATION = "feature_variation"


@dataclass
class QualityGap:
    """Identified quality gap in the dataset."""

    gap_type: GapType
    description: str
    severity: float  # 0-1
    affected_samples: int
    target_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_type": self.gap_type.value,
            "description": self.description,
            "severity": self.severity,
            "affected_samples": self.affected_samples,
            "target_samples": self.target_samples,
            "metadata": self.metadata,
        }


@dataclass
class AugmentedSample:
    """A single augmented sample."""

    features: np.ndarray
    label: Any
    source_indices: list[int]
    method: AugmentationMethod
    quality_score: float  # Estimated quality of augmented sample
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationResult:
    """Result of quality-aware augmentation."""

    n_samples_original: int
    n_samples_generated: int
    n_samples_accepted: int
    n_samples_rejected: int

    gaps_addressed: list[QualityGap]
    samples: list[AugmentedSample]

    quality_improvement: float  # Estimated quality score improvement
    class_balance_improvement: float
    diversity_improvement: float

    rejection_reasons: dict[str, int] = field(default_factory=dict)
    method_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert augmented samples to DataFrame."""
        if not self.samples:
            return pd.DataFrame()

        data = []
        for sample in self.samples:
            row = {"label": sample.label, "quality_score": sample.quality_score}
            for i, val in enumerate(sample.features):
                row[f"feature_{i}"] = val
            data.append(row)

        return pd.DataFrame(data)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Quality-Aware Augmentation Result",
            "=" * 50,
            "",
            f"Original samples: {self.n_samples_original:,}",
            f"Generated samples: {self.n_samples_generated:,}",
            f"Accepted samples: {self.n_samples_accepted:,}",
            f"Rejected samples: {self.n_samples_rejected:,}",
            "",
            f"Quality improvement: +{self.quality_improvement:.1f}%",
            f"Class balance improvement: +{self.class_balance_improvement:.1f}%",
            f"Diversity improvement: +{self.diversity_improvement:.1f}%",
            "",
            "Gaps addressed:",
        ]

        for gap in self.gaps_addressed:
            lines.append(f"  • {gap.description} ({gap.affected_samples} → {gap.target_samples})")

        if self.method_breakdown:
            lines.append("")
            lines.append("Methods used:")
            for method, count in self.method_breakdown.items():
                lines.append(f"  • {method}: {count} samples")

        return "\n".join(lines)


@dataclass
class AugmentationConfig:
    """Configuration for quality-aware augmentation."""

    target_balance_ratio: float = 2.0  # Max imbalance ratio after augmentation
    min_samples_per_class: int = 100
    max_augmentation_factor: float = 3.0  # Max multiplier for any class
    quality_threshold: float = 0.7  # Min quality score for accepted samples
    diversity_weight: float = 0.3  # Weight for diversity in sample selection

    # Method preferences
    preferred_methods: list[AugmentationMethod] = field(
        default_factory=lambda: [
            AugmentationMethod.SMOTE,
            AugmentationMethod.MIXUP,
            AugmentationMethod.NOISE_INJECTION,
        ]
    )

    # Quality validation
    validate_samples: bool = True
    use_quality_filter: bool = True
    max_rejection_rate: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_balance_ratio": self.target_balance_ratio,
            "min_samples_per_class": self.min_samples_per_class,
            "max_augmentation_factor": self.max_augmentation_factor,
            "quality_threshold": self.quality_threshold,
            "diversity_weight": self.diversity_weight,
            "validate_samples": self.validate_samples,
        }


class GapAnalyzer:
    """Analyze quality gaps in dataset."""

    def __init__(self, config: AugmentationConfig):
        """Initialize gap analyzer.

        Args:
            config: Augmentation configuration
        """
        self.config = config

    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray,
        report: QualityReport | None = None,
    ) -> list[QualityGap]:
        """Analyze quality gaps in dataset.

        Args:
            X: Feature matrix
            y: Labels
            report: Optional quality report for additional insights

        Returns:
            List of identified quality gaps
        """
        gaps = []

        # Analyze class imbalance
        gaps.extend(self._analyze_class_imbalance(y))

        # Analyze boundary regions
        gaps.extend(self._analyze_boundary_regions(X, y))

        # Analyze diversity
        gaps.extend(self._analyze_diversity(X, y))

        # Sort by severity
        gaps.sort(key=lambda g: g.severity, reverse=True)

        return gaps

    def _analyze_class_imbalance(self, y: np.ndarray) -> list[QualityGap]:
        """Analyze class imbalance issues."""
        gaps = []
        unique, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        for cls, count in zip(unique, counts):
            ratio = max_count / count

            if ratio > self.config.target_balance_ratio:
                target = int(max_count / self.config.target_balance_ratio)
                target = min(target, int(count * self.config.max_augmentation_factor))

                severity = min(ratio / 10, 1.0)  # Normalize severity

                gaps.append(QualityGap(
                    gap_type=GapType.CLASS_IMBALANCE,
                    description=f"Class '{cls}' is underrepresented ({count} vs {max_count})",
                    severity=severity,
                    affected_samples=count,
                    target_samples=target,
                    metadata={"class": cls, "ratio": ratio},
                ))

        return gaps

    def _analyze_boundary_regions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[QualityGap]:
        """Analyze undersampled boundary regions."""
        gaps = []

        if len(X) < 50:
            return gaps

        # Find samples near class boundaries using k-NN
        k = min(5, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        boundary_samples = []
        for i, (_, neighbor_idx) in enumerate(zip(distances, indices)):
            neighbor_labels = y[neighbor_idx[1:]]  # Exclude self
            if len(np.unique(neighbor_labels)) > 1:
                boundary_samples.append(i)

        if len(boundary_samples) > 0:
            # Check if boundary regions are undersampled
            boundary_fraction = len(boundary_samples) / len(X)

            if boundary_fraction < 0.1:
                target = int(len(X) * 0.15)  # Aim for ~15% boundary samples

                gaps.append(QualityGap(
                    gap_type=GapType.BOUNDARY_REGION,
                    description="Decision boundary regions are undersampled",
                    severity=0.6,
                    affected_samples=len(boundary_samples),
                    target_samples=target,
                    metadata={"boundary_indices": boundary_samples[:100]},
                ))

        return gaps

    def _analyze_diversity(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[QualityGap]:
        """Analyze feature diversity issues."""
        gaps = []

        for cls in np.unique(y):
            X_class = X[y == cls]

            if len(X_class) < 10:
                continue

            # Calculate variance across features
            variances = np.var(X_class, axis=0)
            low_variance_features = np.sum(variances < 0.01)

            if low_variance_features > X.shape[1] * 0.3:
                gaps.append(QualityGap(
                    gap_type=GapType.LOW_DIVERSITY,
                    description=f"Class '{cls}' has low feature diversity",
                    severity=0.4,
                    affected_samples=len(X_class),
                    target_samples=int(len(X_class) * 1.5),
                    metadata={
                        "class": cls,
                        "low_variance_features": int(low_variance_features),
                    },
                ))

        return gaps


class AugmentationStrategy(ABC):
    """Base class for augmentation strategies."""

    @abstractmethod
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        target_class: Any | None = None,
    ) -> list[AugmentedSample]:
        """Generate augmented samples.

        Args:
            X: Feature matrix
            y: Labels
            n_samples: Number of samples to generate
            target_class: Optional class to focus on

        Returns:
            List of augmented samples
        """
        pass


class SMOTEStrategy(AugmentationStrategy):
    """SMOTE-based augmentation strategy."""

    def __init__(self, k_neighbors: int = 5):
        """Initialize SMOTE strategy.

        Args:
            k_neighbors: Number of neighbors for interpolation
        """
        self.k_neighbors = k_neighbors

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        target_class: Any | None = None,
    ) -> list[AugmentedSample]:
        """Generate samples using SMOTE."""
        samples = []

        if target_class is not None:
            mask = y == target_class
            X_class = X[mask]
            indices = np.where(mask)[0]
        else:
            X_class = X
            indices = np.arange(len(X))

        if len(X_class) < 2:
            return samples

        # Fit nearest neighbors
        k = min(self.k_neighbors, len(X_class) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_class)

        for _ in range(n_samples):
            # Select random sample
            idx = np.random.randint(len(X_class))
            sample = X_class[idx]

            # Get neighbors
            _, neighbor_indices = nn.kneighbors([sample])
            neighbor_idx = np.random.choice(neighbor_indices[0][1:])
            neighbor = X_class[neighbor_idx]

            # Interpolate
            alpha = np.random.random()
            new_sample = sample + alpha * (neighbor - sample)

            # Estimate quality based on distance
            distance = np.linalg.norm(sample - neighbor)
            quality_score = max(0.5, 1.0 - distance / 10)

            samples.append(AugmentedSample(
                features=new_sample,
                label=target_class if target_class is not None else y[idx],
                source_indices=[int(indices[idx]), int(indices[neighbor_idx])],
                method=AugmentationMethod.SMOTE,
                quality_score=quality_score,
            ))

        return samples


class MixupStrategy(AugmentationStrategy):
    """Mixup-based augmentation strategy."""

    def __init__(self, alpha: float = 0.4):
        """Initialize Mixup strategy.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        target_class: Any | None = None,
    ) -> list[AugmentedSample]:
        """Generate samples using Mixup."""
        samples = []

        if target_class is not None:
            mask = y == target_class
            X_class = X[mask]
            indices = np.where(mask)[0]
        else:
            X_class = X
            indices = np.arange(len(X))

        if len(X_class) < 2:
            return samples

        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(X_class), 2, replace=False)

            # Generate mixing weight from beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Mix samples
            new_sample = lam * X_class[idx1] + (1 - lam) * X_class[idx2]

            # Quality based on mixing ratio (closer to 0.5 = more novel but riskier)
            quality_score = 0.7 + 0.3 * abs(lam - 0.5) * 2

            samples.append(AugmentedSample(
                features=new_sample,
                label=target_class if target_class is not None else y[indices[idx1]],
                source_indices=[int(indices[idx1]), int(indices[idx2])],
                method=AugmentationMethod.MIXUP,
                quality_score=quality_score,
                metadata={"lambda": lam},
            ))

        return samples


class NoiseInjectionStrategy(AugmentationStrategy):
    """Noise injection augmentation strategy."""

    def __init__(self, noise_scale: float = 0.1):
        """Initialize noise injection strategy.

        Args:
            noise_scale: Scale of Gaussian noise relative to feature std
        """
        self.noise_scale = noise_scale

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        target_class: Any | None = None,
    ) -> list[AugmentedSample]:
        """Generate samples by injecting noise."""
        samples = []

        if target_class is not None:
            mask = y == target_class
            X_class = X[mask]
            indices = np.where(mask)[0]
        else:
            X_class = X
            indices = np.arange(len(X))

        if len(X_class) < 1:
            return samples

        # Calculate feature-wise std
        feature_std = np.std(X_class, axis=0)
        feature_std[feature_std == 0] = 1.0  # Avoid zero std

        for _ in range(n_samples):
            # Select random sample
            idx = np.random.randint(len(X_class))
            sample = X_class[idx]

            # Add Gaussian noise
            noise = np.random.randn(*sample.shape) * feature_std * self.noise_scale
            new_sample = sample + noise

            # Quality based on noise magnitude
            noise_magnitude = np.linalg.norm(noise) / np.linalg.norm(sample + 1e-10)
            quality_score = max(0.6, 1.0 - noise_magnitude)

            samples.append(AugmentedSample(
                features=new_sample,
                label=target_class if target_class is not None else y[indices[idx]],
                source_indices=[int(indices[idx])],
                method=AugmentationMethod.NOISE_INJECTION,
                quality_score=quality_score,
            ))

        return samples


class QualityFilter:
    """Filter augmented samples by quality."""

    def __init__(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        threshold: float = 0.7,
    ):
        """Initialize quality filter.

        Args:
            X_original: Original feature matrix
            y_original: Original labels
            threshold: Minimum quality score
        """
        self.X_original = X_original
        self.y_original = y_original
        self.threshold = threshold

        # Fit nearest neighbors for novelty check
        self.nn = NearestNeighbors(n_neighbors=1)
        self.nn.fit(X_original)

    def filter(
        self,
        samples: list[AugmentedSample],
    ) -> tuple[list[AugmentedSample], dict[str, int]]:
        """Filter samples by quality.

        Args:
            samples: Augmented samples to filter

        Returns:
            Tuple of (accepted_samples, rejection_reasons)
        """
        accepted = []
        rejections: dict[str, int] = {}

        for sample in samples:
            passed, reason = self._check_sample(sample)
            if passed:
                accepted.append(sample)
            else:
                rejections[reason] = rejections.get(reason, 0) + 1

        return accepted, rejections

    def _check_sample(self, sample: AugmentedSample) -> tuple[bool, str]:
        """Check if sample passes quality filters."""
        # Check quality score
        if sample.quality_score < self.threshold:
            return False, "low_quality_score"

        # Check for NaN/inf
        if not np.isfinite(sample.features).all():
            return False, "invalid_values"

        # Check novelty (not too close to existing samples)
        distance, _ = self.nn.kneighbors([sample.features])
        if distance[0][0] < 0.01:  # Too similar to existing
            return False, "too_similar"

        # Check bounds (not too far from data distribution)
        feature_mins = self.X_original.min(axis=0)
        feature_maxs = self.X_original.max(axis=0)
        feature_range = feature_maxs - feature_mins + 1e-10

        normalized = (sample.features - feature_mins) / feature_range
        if (normalized < -0.5).any() or (normalized > 1.5).any():
            return False, "out_of_distribution"

        return True, ""


class QualityAwareAugmenter:
    """Intelligent data augmentation that addresses quality gaps.

    Analyzes detected quality issues and generates samples to
    specifically address class imbalance, underrepresented slices,
    and boundary regions.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
    ):
        """Initialize augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self.gap_analyzer = GapAnalyzer(self.config)

        # Initialize strategies
        self.strategies: dict[AugmentationMethod, AugmentationStrategy] = {
            AugmentationMethod.SMOTE: SMOTEStrategy(),
            AugmentationMethod.MIXUP: MixupStrategy(),
            AugmentationMethod.NOISE_INJECTION: NoiseInjectionStrategy(),
        }

    def augment(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        report: QualityReport | None = None,
        label_column: str | None = None,
    ) -> AugmentationResult:
        """Perform quality-aware augmentation.

        Args:
            X: Feature matrix or DataFrame
            y: Labels
            report: Optional quality report for additional insights
            label_column: Label column name if X is DataFrame

        Returns:
            AugmentationResult with generated samples
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        # Analyze quality gaps
        gaps = self.gap_analyzer.analyze(X_arr, y_arr, report)

        if not gaps:
            logger.info("No quality gaps found requiring augmentation")
            return AugmentationResult(
                n_samples_original=len(X_arr),
                n_samples_generated=0,
                n_samples_accepted=0,
                n_samples_rejected=0,
                gaps_addressed=[],
                samples=[],
                quality_improvement=0.0,
                class_balance_improvement=0.0,
                diversity_improvement=0.0,
            )

        # Create quality filter
        quality_filter = QualityFilter(
            X_arr, y_arr,
            threshold=self.config.quality_threshold,
        ) if self.config.use_quality_filter else None

        # Generate samples for each gap
        all_samples: list[AugmentedSample] = []
        method_breakdown: dict[str, int] = {}

        for gap in gaps:
            samples = self._augment_for_gap(X_arr, y_arr, gap)
            all_samples.extend(samples)

            for sample in samples:
                method = sample.method.value
                method_breakdown[method] = method_breakdown.get(method, 0) + 1

        # Filter samples
        n_generated = len(all_samples)
        rejection_reasons: dict[str, int] = {}

        if quality_filter:
            all_samples, rejection_reasons = quality_filter.filter(all_samples)

        n_accepted = len(all_samples)
        n_rejected = n_generated - n_accepted

        # Calculate improvements
        quality_improvement = self._estimate_quality_improvement(
            X_arr, y_arr, all_samples
        )
        balance_improvement = self._calculate_balance_improvement(
            y_arr, all_samples
        )
        diversity_improvement = self._calculate_diversity_improvement(
            X_arr, all_samples
        )

        return AugmentationResult(
            n_samples_original=len(X_arr),
            n_samples_generated=n_generated,
            n_samples_accepted=n_accepted,
            n_samples_rejected=n_rejected,
            gaps_addressed=gaps,
            samples=all_samples,
            quality_improvement=quality_improvement,
            class_balance_improvement=balance_improvement,
            diversity_improvement=diversity_improvement,
            rejection_reasons=rejection_reasons,
            method_breakdown=method_breakdown,
        )

    def _augment_for_gap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gap: QualityGap,
    ) -> list[AugmentedSample]:
        """Generate samples to address a specific gap."""
        n_samples = gap.target_samples - gap.affected_samples
        n_samples = max(0, n_samples)

        if n_samples == 0:
            return []

        # Select strategy based on gap type
        if gap.gap_type == GapType.CLASS_IMBALANCE:
            strategy = self.strategies[AugmentationMethod.SMOTE]
            target_class = gap.metadata.get("class")
        elif gap.gap_type == GapType.BOUNDARY_REGION:
            strategy = self.strategies[AugmentationMethod.MIXUP]
            target_class = None
        elif gap.gap_type == GapType.LOW_DIVERSITY:
            strategy = self.strategies[AugmentationMethod.NOISE_INJECTION]
            target_class = gap.metadata.get("class")
        else:
            strategy = self.strategies[AugmentationMethod.SMOTE]
            target_class = None

        return strategy.augment(X, y, n_samples, target_class)

    def _estimate_quality_improvement(
        self,
        X: np.ndarray,
        y: np.ndarray,
        samples: list[AugmentedSample],
    ) -> float:
        """Estimate quality score improvement from augmentation."""
        if not samples:
            return 0.0

        # Base improvement on balance and diversity gains
        avg_quality = np.mean([s.quality_score for s in samples])
        n_new = len(samples)
        n_original = len(X)

        # Weighted improvement
        improvement = (avg_quality - 0.5) * (n_new / n_original) * 10

        return float(max(0, improvement))

    def _calculate_balance_improvement(
        self,
        y: np.ndarray,
        samples: list[AugmentedSample],
    ) -> float:
        """Calculate class balance improvement."""
        if not samples:
            return 0.0

        unique, counts = np.unique(y, return_counts=True)
        original_ratio = counts.max() / counts.min()

        # Count new samples per class
        new_counts = counts.copy()
        class_to_idx = {c: i for i, c in enumerate(unique)}

        for sample in samples:
            if sample.label in class_to_idx:
                new_counts[class_to_idx[sample.label]] += 1

        new_ratio = new_counts.max() / new_counts.min()

        # Improvement percentage
        improvement = (original_ratio - new_ratio) / original_ratio * 100

        return float(max(0, improvement))

    def _calculate_diversity_improvement(
        self,
        X: np.ndarray,
        samples: list[AugmentedSample],
    ) -> float:
        """Calculate feature diversity improvement."""
        if not samples or len(samples) < 2:
            return 0.0

        # Calculate original variance
        original_var = np.var(X, axis=0).mean()

        # Calculate combined variance
        new_features = np.array([s.features for s in samples])
        combined = np.vstack([X, new_features])
        combined_var = np.var(combined, axis=0).mean()

        # Improvement percentage
        if original_var > 0:
            improvement = (combined_var - original_var) / original_var * 100
        else:
            improvement = 0.0

        return float(max(0, improvement))

    def add_strategy(
        self,
        method: AugmentationMethod,
        strategy: AugmentationStrategy,
    ) -> None:
        """Add a custom augmentation strategy.

        Args:
            method: Method identifier
            strategy: Strategy implementation
        """
        self.strategies[method] = strategy


def augment_for_quality(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    report: QualityReport | None = None,
    config: AugmentationConfig | None = None,
) -> AugmentationResult:
    """Convenience function for quality-aware augmentation.

    Args:
        X: Feature matrix
        y: Labels
        report: Optional quality report
        config: Optional configuration

    Returns:
        AugmentationResult
    """
    augmenter = QualityAwareAugmenter(config=config)
    return augmenter.augment(X, y, report)


def apply_augmentation(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    result: AugmentationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation result to dataset.

    Args:
        X: Original features
        y: Original labels
        result: Augmentation result

    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.select_dtypes(include=[np.number]).values
    else:
        X_arr = np.asarray(X)

    y_arr = np.asarray(y)

    if not result.samples:
        return X_arr, y_arr

    # Combine original and augmented
    new_X = np.array([s.features for s in result.samples])
    new_y = np.array([s.label for s in result.samples])

    combined_X = np.vstack([X_arr, new_X])
    combined_y = np.concatenate([y_arr, new_y])

    return combined_X, combined_y


def create_augmenter(
    config: AugmentationConfig | None = None,
    **kwargs: Any,
) -> QualityAwareAugmenter:
    """Create a quality-aware augmenter.

    Args:
        config: Configuration
        **kwargs: Additional config parameters

    Returns:
        QualityAwareAugmenter instance
    """
    if config is None:
        config = AugmentationConfig(**kwargs)
    return QualityAwareAugmenter(config=config)
