"""Curriculum Learning Optimizer.

This module optimizes training sample ordering based on quality scores
to improve model learning through curriculum learning strategies.

Example:
    >>> from clean.curriculum import CurriculumOptimizer
    >>>
    >>> optimizer = CurriculumOptimizer(strategy="easy_to_hard")
    >>> ordered_indices = optimizer.optimize(X, y, quality_scores)
    >>> # Train model with ordered data
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Curriculum learning strategies."""

    EASY_TO_HARD = "easy_to_hard"
    HARD_TO_EASY = "hard_to_easy"
    SELF_PACED = "self_paced"
    COMPETENCE_BASED = "competence_based"
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    ANTI_CURRICULUM = "anti_curriculum"
    MIXED = "mixed"


class DifficultyMetric(Enum):
    """Metrics for sample difficulty."""

    QUALITY_SCORE = "quality_score"
    MODEL_CONFIDENCE = "model_confidence"
    LABEL_CERTAINTY = "label_certainty"
    OUTLIER_SCORE = "outlier_score"
    NEIGHBOR_AGREEMENT = "neighbor_agreement"
    LOSS_VALUE = "loss_value"


@dataclass
class SampleDifficulty:
    """Difficulty assessment for a single sample."""

    index: int
    difficulty_score: float  # 0 = easy, 1 = hard
    quality_score: float
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "difficulty_score": self.difficulty_score,
            "quality_score": self.quality_score,
            "metrics": self.metrics,
        }


@dataclass
class CurriculumSchedule:
    """Schedule for curriculum learning."""

    strategy: CurriculumStrategy
    n_samples: int
    n_epochs: int
    sample_order: list[int]
    epoch_schedules: list[list[int]]

    # Per-epoch sample counts (for pacing)
    samples_per_epoch: list[int]

    # Difficulty distribution
    difficulty_scores: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)

    def get_epoch_samples(self, epoch: int) -> list[int]:
        """Get sample indices for a specific epoch."""
        if epoch >= len(self.epoch_schedules):
            return self.epoch_schedules[-1]
        return self.epoch_schedules[epoch]

    def get_batch_iterator(
        self,
        batch_size: int,
        epoch: int = 0,
    ) -> Iterator[list[int]]:
        """Get batch iterator for an epoch.

        Args:
            batch_size: Batch size
            epoch: Epoch number

        Yields:
            Lists of sample indices per batch
        """
        samples = self.get_epoch_samples(epoch)

        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Curriculum Learning Schedule",
            "=" * 50,
            "",
            f"Strategy: {self.strategy.value}",
            f"Total samples: {self.n_samples:,}",
            f"Epochs: {self.n_epochs}",
            "",
            "Difficulty distribution:",
            f"  Mean: {self.difficulty_scores.mean():.3f}",
            f"  Std: {self.difficulty_scores.std():.3f}",
            f"  Min: {self.difficulty_scores.min():.3f}",
            f"  Max: {self.difficulty_scores.max():.3f}",
            "",
            "Samples per epoch:",
        ]

        for i, count in enumerate(self.samples_per_epoch[:5]):
            lines.append(f"  Epoch {i + 1}: {count:,} samples")

        if len(self.samples_per_epoch) > 5:
            lines.append(f"  ... ({len(self.samples_per_epoch) - 5} more epochs)")

        return "\n".join(lines)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    strategy: CurriculumStrategy = CurriculumStrategy.EASY_TO_HARD
    n_epochs: int = 10
    warmup_epochs: int = 2

    # Pacing parameters
    initial_fraction: float = 0.3  # Start with easiest 30%
    growth_rate: float = 1.5  # How fast to add more samples

    # Self-paced learning parameters
    pace_function: str = "linear"  # linear, logarithmic, exponential
    pace_parameter: float = 0.1

    # Difficulty weights
    quality_weight: float = 0.4
    confidence_weight: float = 0.3
    neighbor_weight: float = 0.2
    outlier_weight: float = 0.1

    # Diversity settings
    diversity_factor: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "n_epochs": self.n_epochs,
            "warmup_epochs": self.warmup_epochs,
            "initial_fraction": self.initial_fraction,
            "growth_rate": self.growth_rate,
        }


class DifficultyScorer:
    """Score sample difficulty based on multiple metrics."""

    def __init__(self, config: CurriculumConfig):
        """Initialize difficulty scorer.

        Args:
            config: Curriculum configuration
        """
        self.config = config

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quality_scores: np.ndarray | None = None,
        model: BaseEstimator | None = None,
    ) -> list[SampleDifficulty]:
        """Score difficulty of all samples.

        Args:
            X: Feature matrix
            y: Labels
            quality_scores: Optional pre-computed quality scores
            model: Optional model for confidence estimation

        Returns:
            List of SampleDifficulty objects
        """
        n_samples = len(X)

        # Initialize metrics
        metrics = {
            "quality": np.ones(n_samples),
            "confidence": np.ones(n_samples),
            "neighbor": np.ones(n_samples),
            "outlier": np.zeros(n_samples),
        }

        # Quality scores (inverted: high quality = easy)
        if quality_scores is not None:
            # Normalize to 0-1
            min_q, max_q = quality_scores.min(), quality_scores.max()
            if max_q > min_q:
                metrics["quality"] = 1 - (quality_scores - min_q) / (max_q - min_q)
            else:
                metrics["quality"] = np.zeros(n_samples)

        # Model confidence (low confidence = hard)
        if model is not None:
            try:
                proba = cross_val_predict(
                    model, X, y,
                    cv=min(5, len(np.unique(y))),
                    method="predict_proba",
                )
                # Get confidence for true class
                confidence = np.array([
                    proba[i, np.where(model.classes_ == y[i])[0][0]]
                    if y[i] in model.classes_ else 0.5
                    for i in range(n_samples)
                ])
                metrics["confidence"] = 1 - confidence
            except Exception:
                pass

        # Neighbor agreement (disagreement = hard)
        metrics["neighbor"] = self._compute_neighbor_difficulty(X, y)

        # Outlier score (outliers = hard)
        metrics["outlier"] = self._compute_outlier_scores(X)

        # Combine metrics
        difficulties = []
        for i in range(n_samples):
            weighted_score = (
                self.config.quality_weight * metrics["quality"][i] +
                self.config.confidence_weight * metrics["confidence"][i] +
                self.config.neighbor_weight * metrics["neighbor"][i] +
                self.config.outlier_weight * metrics["outlier"][i]
            )

            total_weight = (
                self.config.quality_weight +
                self.config.confidence_weight +
                self.config.neighbor_weight +
                self.config.outlier_weight
            )

            difficulty_score = weighted_score / total_weight

            difficulties.append(SampleDifficulty(
                index=i,
                difficulty_score=float(difficulty_score),
                quality_score=float(quality_scores[i]) if quality_scores is not None else 1.0,
                metrics={
                    "quality": float(metrics["quality"][i]),
                    "confidence": float(metrics["confidence"][i]),
                    "neighbor": float(metrics["neighbor"][i]),
                    "outlier": float(metrics["outlier"][i]),
                },
            ))

        return difficulties

    def _compute_neighbor_difficulty(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute difficulty based on neighbor label agreement."""
        if len(X) < 10:
            return np.zeros(len(X))

        k = min(5, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)

        _, indices = nn.kneighbors(X)

        difficulties = np.zeros(len(X))
        for i, neighbor_idx in enumerate(indices):
            neighbor_labels = y[neighbor_idx[1:]]  # Exclude self
            agreement = (neighbor_labels == y[i]).mean()
            difficulties[i] = 1 - agreement

        return difficulties

    def _compute_outlier_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute outlier scores using Local Outlier Factor."""
        if len(X) < 20:
            return np.zeros(len(X))

        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1))
        lof.fit_predict(X)

        # LOF scores are negative, more negative = more outlier
        scores = -lof.negative_outlier_factor_

        # Normalize to 0-1
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            return (scores - min_s) / (max_s - min_s)

        return np.zeros(len(X))


class CurriculumScheduler(ABC):
    """Base class for curriculum schedulers."""

    @abstractmethod
    def schedule(
        self,
        difficulties: list[SampleDifficulty],
        n_epochs: int,
    ) -> CurriculumSchedule:
        """Create curriculum schedule.

        Args:
            difficulties: Sample difficulties
            n_epochs: Number of training epochs

        Returns:
            CurriculumSchedule
        """
        pass


class EasyToHardScheduler(CurriculumScheduler):
    """Schedule samples from easy to hard."""

    def __init__(self, config: CurriculumConfig):
        """Initialize scheduler.

        Args:
            config: Curriculum configuration
        """
        self.config = config

    def schedule(
        self,
        difficulties: list[SampleDifficulty],
        n_epochs: int,
    ) -> CurriculumSchedule:
        """Create easy-to-hard schedule."""
        n_samples = len(difficulties)

        # Sort by difficulty (easy first)
        sorted_samples = sorted(difficulties, key=lambda d: d.difficulty_score)
        sorted_indices = [d.index for d in sorted_samples]

        # Calculate samples per epoch
        samples_per_epoch = []
        current_fraction = self.config.initial_fraction

        for epoch in range(n_epochs):
            n_include = min(int(n_samples * current_fraction), n_samples)
            samples_per_epoch.append(n_include)
            current_fraction *= self.config.growth_rate
            current_fraction = min(current_fraction, 1.0)

        # Create epoch schedules
        epoch_schedules = []
        for n_include in samples_per_epoch:
            epoch_samples = sorted_indices[:n_include]
            # Shuffle within epoch for SGD
            np.random.shuffle(epoch_samples)
            epoch_schedules.append(list(epoch_samples))

        difficulty_scores = np.array([d.difficulty_score for d in difficulties])

        return CurriculumSchedule(
            strategy=CurriculumStrategy.EASY_TO_HARD,
            n_samples=n_samples,
            n_epochs=n_epochs,
            sample_order=sorted_indices,
            epoch_schedules=epoch_schedules,
            samples_per_epoch=samples_per_epoch,
            difficulty_scores=difficulty_scores,
        )


class SelfPacedScheduler(CurriculumScheduler):
    """Self-paced learning scheduler."""

    def __init__(self, config: CurriculumConfig):
        """Initialize scheduler.

        Args:
            config: Curriculum configuration
        """
        self.config = config

    def schedule(
        self,
        difficulties: list[SampleDifficulty],
        n_epochs: int,
    ) -> CurriculumSchedule:
        """Create self-paced schedule."""
        n_samples = len(difficulties)
        difficulty_scores = np.array([d.difficulty_score for d in difficulties])

        # Self-paced: start with threshold, gradually increase
        epoch_schedules = []
        samples_per_epoch = []

        for epoch in range(n_epochs):
            # Calculate threshold for this epoch
            if self.config.pace_function == "linear":
                threshold = self.config.pace_parameter + (1 - self.config.pace_parameter) * (epoch / n_epochs)
            elif self.config.pace_function == "logarithmic":
                threshold = self.config.pace_parameter + (1 - self.config.pace_parameter) * np.log1p(epoch) / np.log1p(n_epochs)
            else:  # exponential
                threshold = 1 - (1 - self.config.pace_parameter) * np.exp(-epoch / (n_epochs / 3))

            # Select samples below threshold
            selected = [
                d.index for d in difficulties
                if d.difficulty_score <= threshold
            ]

            if not selected:
                selected = [difficulties[0].index]  # Always include at least one

            np.random.shuffle(selected)
            epoch_schedules.append(selected)
            samples_per_epoch.append(len(selected))

        return CurriculumSchedule(
            strategy=CurriculumStrategy.SELF_PACED,
            n_samples=n_samples,
            n_epochs=n_epochs,
            sample_order=list(range(n_samples)),
            epoch_schedules=epoch_schedules,
            samples_per_epoch=samples_per_epoch,
            difficulty_scores=difficulty_scores,
        )


class DiversityScheduler(CurriculumScheduler):
    """Schedule with diversity consideration."""

    def __init__(self, config: CurriculumConfig, X: np.ndarray):
        """Initialize scheduler.

        Args:
            config: Curriculum configuration
            X: Feature matrix for diversity calculation
        """
        self.config = config
        self.X = X

    def schedule(
        self,
        difficulties: list[SampleDifficulty],
        n_epochs: int,
    ) -> CurriculumSchedule:
        """Create diversity-aware schedule."""
        n_samples = len(difficulties)
        difficulty_scores = np.array([d.difficulty_score for d in difficulties])

        # Sort by difficulty
        sorted_by_difficulty = sorted(difficulties, key=lambda d: d.difficulty_score)

        # For each epoch, select diverse samples within difficulty range
        epoch_schedules = []
        samples_per_epoch = []

        for epoch in range(n_epochs):
            # Determine difficulty range for this epoch
            fraction = min(
                self.config.initial_fraction * (self.config.growth_rate ** epoch),
                1.0,
            )
            n_include = max(1, int(n_samples * fraction))

            # Get candidates (within difficulty range)
            candidates = [d.index for d in sorted_by_difficulty[:n_include]]

            # Select diverse subset
            selected = self._select_diverse(candidates, n_include)

            np.random.shuffle(selected)
            epoch_schedules.append(selected)
            samples_per_epoch.append(len(selected))

        return CurriculumSchedule(
            strategy=CurriculumStrategy.DIVERSITY,
            n_samples=n_samples,
            n_epochs=n_epochs,
            sample_order=list(range(n_samples)),
            epoch_schedules=epoch_schedules,
            samples_per_epoch=samples_per_epoch,
            difficulty_scores=difficulty_scores,
        )

    def _select_diverse(
        self,
        candidates: list[int],
        n_select: int,
    ) -> list[int]:
        """Select diverse samples from candidates."""
        if len(candidates) <= n_select:
            return candidates

        # Use farthest point sampling
        X_candidates = self.X[candidates]

        selected = [0]  # Start with first candidate
        selected_features = [X_candidates[0]]

        while len(selected) < n_select:
            # Find farthest point from selected set
            max_min_dist = -1
            best_idx = -1

            for i, x in enumerate(X_candidates):
                if i in selected:
                    continue

                # Distance to nearest selected point
                min_dist = min(
                    np.linalg.norm(x - s)
                    for s in selected_features
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
                selected_features.append(X_candidates[best_idx])
            else:
                break

        return [candidates[i] for i in selected]


class CurriculumOptimizer:
    """Optimize training sample ordering for curriculum learning.

    Analyzes sample difficulty and creates optimized training schedules
    to improve model learning.
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        strategy: str | CurriculumStrategy | None = None,
    ):
        """Initialize curriculum optimizer.

        Args:
            config: Curriculum configuration
            strategy: Strategy name or enum (overrides config)
        """
        self.config = config or CurriculumConfig()

        if strategy:
            if isinstance(strategy, str):
                self.config.strategy = CurriculumStrategy(strategy)
            else:
                self.config.strategy = strategy

        self.scorer = DifficultyScorer(self.config)

    def optimize(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        quality_scores: np.ndarray | None = None,
        model: BaseEstimator | None = None,
    ) -> CurriculumSchedule:
        """Create optimized curriculum schedule.

        Args:
            X: Feature matrix
            y: Labels
            quality_scores: Optional quality scores (0-100)
            model: Optional model for confidence estimation

        Returns:
            CurriculumSchedule
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        # Score difficulties
        difficulties = self.scorer.score(X_arr, y_arr, quality_scores, model)

        # Create scheduler based on strategy
        scheduler = self._create_scheduler(X_arr)

        # Generate schedule
        schedule = scheduler.schedule(difficulties, self.config.n_epochs)

        return schedule

    def get_ordered_indices(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        quality_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get simply ordered indices (for single-pass training).

        Args:
            X: Feature matrix
            y: Labels
            quality_scores: Optional quality scores

        Returns:
            Ordered sample indices
        """
        schedule = self.optimize(X, y, quality_scores)
        return np.array(schedule.sample_order)

    def _create_scheduler(self, X: np.ndarray) -> CurriculumScheduler:
        """Create appropriate scheduler for strategy."""
        strategy = self.config.strategy

        if strategy == CurriculumStrategy.EASY_TO_HARD:
            return EasyToHardScheduler(self.config)
        elif strategy == CurriculumStrategy.SELF_PACED:
            return SelfPacedScheduler(self.config)
        elif strategy == CurriculumStrategy.DIVERSITY:
            return DiversityScheduler(self.config, X)
        elif strategy == CurriculumStrategy.HARD_TO_EASY:
            # Invert the easy-to-hard schedule
            scheduler = EasyToHardScheduler(self.config)
            return scheduler  # Will be inverted in post-processing
        else:
            return EasyToHardScheduler(self.config)


class CurriculumDataLoader:
    """Data loader that implements curriculum learning.

    Compatible with PyTorch-style iteration.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        schedule: CurriculumSchedule,
        batch_size: int = 32,
    ):
        """Initialize curriculum data loader.

        Args:
            X: Feature matrix
            y: Labels
            schedule: Curriculum schedule
            batch_size: Batch size
        """
        self.X = X
        self.y = y
        self.schedule = schedule
        self.batch_size = batch_size
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for pacing.

        Args:
            epoch: Epoch number
        """
        self._current_epoch = epoch

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches for current epoch."""
        for batch_indices in self.schedule.get_batch_iterator(
            self.batch_size,
            self._current_epoch,
        ):
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            yield X_batch, y_batch

    def __len__(self) -> int:
        """Number of batches in current epoch."""
        n_samples = len(self.schedule.get_epoch_samples(self._current_epoch))
        return (n_samples + self.batch_size - 1) // self.batch_size


def create_curriculum(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    strategy: str = "easy_to_hard",
    quality_scores: np.ndarray | None = None,
    **kwargs: Any,
) -> CurriculumSchedule:
    """Convenience function to create curriculum schedule.

    Args:
        X: Feature matrix
        y: Labels
        strategy: Curriculum strategy
        quality_scores: Optional quality scores
        **kwargs: Additional config parameters

    Returns:
        CurriculumSchedule
    """
    config = CurriculumConfig(
        strategy=CurriculumStrategy(strategy),
        **kwargs,
    )
    optimizer = CurriculumOptimizer(config=config)
    return optimizer.optimize(X, y, quality_scores)


def create_curriculum_loader(
    X: np.ndarray,
    y: np.ndarray,
    schedule: CurriculumSchedule,
    batch_size: int = 32,
) -> CurriculumDataLoader:
    """Create curriculum-aware data loader.

    Args:
        X: Feature matrix
        y: Labels
        schedule: Curriculum schedule
        batch_size: Batch size

    Returns:
        CurriculumDataLoader
    """
    return CurriculumDataLoader(X, y, schedule, batch_size)
