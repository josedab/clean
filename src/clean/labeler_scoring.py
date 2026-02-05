"""Automated Labeler Performance Scoring.

This module provides tools to evaluate individual labeler performance
and enable smart routing to appropriate labelers.

Example:
    >>> from clean.labeler_scoring import LabelerEvaluator
    >>>
    >>> evaluator = LabelerEvaluator()
    >>> evaluator.fit(labels, labeler_ids, ground_truth)
    >>> report = evaluator.get_labeler_report("labeler_001")
    >>> recommended = evaluator.recommend_labelers(task_type="sentiment")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExpertiseLevel(Enum):
    """Labeler expertise level."""

    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    MASTER = "master"


class PerformanceStatus(Enum):
    """Performance status indicator."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    INCONSISTENT = "inconsistent"


@dataclass
class LabelerMetrics:
    """Comprehensive metrics for a labeler."""

    labeler_id: str
    n_labels: int
    n_tasks: int

    # Quality metrics
    accuracy: float
    error_rate: float
    agreement_rate: float  # Inter-annotator agreement

    # Consistency metrics
    consistency_score: float
    self_agreement: float  # Agreement with own repeated labels

    # Efficiency metrics
    avg_labels_per_day: float
    completion_rate: float

    # Expertise
    expertise_level: ExpertiseLevel
    strong_categories: list[str]
    weak_categories: list[str]

    # Trend
    performance_trend: PerformanceStatus
    recent_accuracy: float
    accuracy_change: float

    # Time analysis
    first_label_date: datetime | None
    last_label_date: datetime | None
    active_days: int

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "labeler_id": self.labeler_id,
            "n_labels": self.n_labels,
            "n_tasks": self.n_tasks,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "agreement_rate": self.agreement_rate,
            "consistency_score": self.consistency_score,
            "expertise_level": self.expertise_level.value,
            "strong_categories": self.strong_categories,
            "weak_categories": self.weak_categories,
            "performance_trend": self.performance_trend.value,
            "recent_accuracy": self.recent_accuracy,
        }


@dataclass
class LabelerReport:
    """Detailed report for a labeler."""

    labeler_id: str
    metrics: LabelerMetrics
    generated_at: datetime

    # Performance breakdown by category
    category_performance: dict[str, float]

    # Error analysis
    common_error_patterns: list[dict[str, Any]]
    confused_label_pairs: list[tuple[Any, Any, int]]  # (true, pred, count)

    # Recommendations
    training_recommendations: list[str]
    suitable_task_types: list[str]
    unsuitable_task_types: list[str]

    # Comparison
    percentile_rank: int
    comparison_to_avg: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Labeler Performance Report: {self.labeler_id}",
            "=" * 50,
            "",
            f"Experience: {self.metrics.expertise_level.value.title()}",
            f"Total labels: {self.metrics.n_labels:,}",
            f"Accuracy: {self.metrics.accuracy:.1%}",
            f"Consistency: {self.metrics.consistency_score:.1%}",
            "",
            f"Performance trend: {self.metrics.performance_trend.value}",
            f"Percentile rank: {self.percentile_rank}th",
            "",
            "Strong categories:",
        ]

        for cat in self.metrics.strong_categories[:3]:
            perf = self.category_performance.get(cat, 0)
            lines.append(f"  - {cat}: {perf:.1%}")

        if self.metrics.weak_categories:
            lines.append("")
            lines.append("Areas for improvement:")
            for cat in self.metrics.weak_categories[:3]:
                perf = self.category_performance.get(cat, 0)
                lines.append(f"  - {cat}: {perf:.1%}")

        if self.training_recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.training_recommendations[:3]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "labeler_id": self.labeler_id,
            "metrics": self.metrics.to_dict(),
            "generated_at": self.generated_at.isoformat(),
            "category_performance": self.category_performance,
            "percentile_rank": self.percentile_rank,
            "training_recommendations": self.training_recommendations,
            "suitable_task_types": self.suitable_task_types,
        }


@dataclass
class LabelerRecommendation:
    """Recommendation for labeler assignment."""

    labeler_id: str
    score: float
    reason: str
    expected_accuracy: float
    confidence: float


@dataclass
class TaskAssignment:
    """Optimized task assignment."""

    task_id: str | int
    task_category: str | None
    recommended_labelers: list[LabelerRecommendation]
    n_labelers_needed: int
    difficulty_estimate: float


class LabelerEvaluator:
    """Evaluator for individual labeler performance.

    Analyzes labeling patterns to identify:
    - Per-labeler accuracy and consistency
    - Expertise areas and weaknesses
    - Performance trends over time
    """

    def __init__(
        self,
        window_size: int = 100,
        min_labels_for_evaluation: int = 10,
    ):
        """Initialize evaluator.

        Args:
            window_size: Window size for trend analysis
            min_labels_for_evaluation: Minimum labels required
        """
        self.window_size = window_size
        self.min_labels = min_labels_for_evaluation

        # Data storage
        self._labels: list[Any] = []
        self._labeler_ids: list[str] = []
        self._ground_truth: list[Any] = []
        self._categories: list[str | None] = []
        self._timestamps: list[datetime | None] = []
        self._task_ids: list[str | int] = []

        # Computed metrics
        self._labeler_metrics: dict[str, LabelerMetrics] = {}
        self._is_fitted = False

    def fit(
        self,
        labels: list[Any] | np.ndarray,
        labeler_ids: list[str] | np.ndarray,
        ground_truth: list[Any] | np.ndarray | None = None,
        categories: list[str | None] | np.ndarray | None = None,
        timestamps: list[datetime | None] | None = None,
        task_ids: list[str | int] | None = None,
    ) -> LabelerEvaluator:
        """Fit evaluator to labeling data.

        Args:
            labels: Labels provided by labelers
            labeler_ids: IDs of labelers for each label
            ground_truth: True labels (if known)
            categories: Category/task type for each item
            timestamps: Timestamps for each label
            task_ids: Task identifiers for each item

        Returns:
            self
        """
        self._labels = list(labels)
        self._labeler_ids = list(labeler_ids)
        self._ground_truth = list(ground_truth) if ground_truth is not None else []
        self._categories = list(categories) if categories is not None else [None] * len(labels)
        self._timestamps = timestamps or [None] * len(labels)
        self._task_ids = list(task_ids) if task_ids is not None else list(range(len(labels)))

        if len(self._labels) != len(self._labeler_ids):
            raise ValueError("labels and labeler_ids must have same length")

        # Compute metrics for each labeler
        self._compute_all_metrics()
        self._is_fitted = True

        return self

    def _compute_all_metrics(self) -> None:
        """Compute metrics for all labelers."""
        # Group data by labeler
        labeler_data: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: {
                "labels": [],
                "ground_truth": [],
                "categories": [],
                "timestamps": [],
                "task_ids": [],
                "indices": [],
            }
        )

        for i, (label, labeler_id, gt, cat, ts, task_id) in enumerate(zip(
            self._labels,
            self._labeler_ids,
            self._ground_truth if self._ground_truth else [None] * len(self._labels),
            self._categories,
            self._timestamps,
            self._task_ids,
        )):
            labeler_data[labeler_id]["labels"].append(label)
            labeler_data[labeler_id]["ground_truth"].append(gt)
            labeler_data[labeler_id]["categories"].append(cat)
            labeler_data[labeler_id]["timestamps"].append(ts)
            labeler_data[labeler_id]["task_ids"].append(task_id)
            labeler_data[labeler_id]["indices"].append(i)

        # Compute metrics for each labeler
        for labeler_id, data in labeler_data.items():
            if len(data["labels"]) >= self.min_labels:
                metrics = self._compute_labeler_metrics(labeler_id, data)
                self._labeler_metrics[labeler_id] = metrics

    def _compute_labeler_metrics(
        self,
        labeler_id: str,
        data: dict[str, list[Any]],
    ) -> LabelerMetrics:
        """Compute metrics for a single labeler."""
        labels = data["labels"]
        ground_truth = data["ground_truth"]
        categories = data["categories"]
        timestamps = data["timestamps"]
        task_ids = data["task_ids"]

        n_labels = len(labels)
        n_tasks = len(set(task_ids))

        # Accuracy (if ground truth available)
        has_gt = ground_truth and ground_truth[0] is not None
        if has_gt:
            correct = sum(1 for l, gt in zip(labels, ground_truth) if l == gt)
            accuracy = correct / n_labels
            error_rate = 1 - accuracy
        else:
            accuracy = 0.5  # Unknown
            error_rate = 0.5

        # Agreement rate (with other labelers on same tasks)
        agreement_rate = self._compute_agreement_rate(labeler_id, task_ids, labels)

        # Consistency score
        consistency_score = self._compute_consistency(labels, categories)

        # Self-agreement (repeated labels on same items)
        self_agreement = self._compute_self_agreement(task_ids, labels)

        # Efficiency metrics
        valid_timestamps = [ts for ts in timestamps if ts is not None]
        if valid_timestamps:
            days = (max(valid_timestamps) - min(valid_timestamps)).days + 1
            avg_labels_per_day = n_labels / max(days, 1)
            first_date = min(valid_timestamps)
            last_date = max(valid_timestamps)
            active_days = days
        else:
            avg_labels_per_day = 0.0
            first_date = None
            last_date = None
            active_days = 0

        # Completion rate (assume all are complete)
        completion_rate = 1.0

        # Category-wise performance
        category_performance = self._compute_category_performance(
            labels, ground_truth, categories
        )

        # Identify strong and weak categories
        strong_categories = [
            cat for cat, perf in category_performance.items()
            if perf >= accuracy + 0.1
        ]
        weak_categories = [
            cat for cat, perf in category_performance.items()
            if perf <= accuracy - 0.1
        ]

        # Expertise level
        if accuracy >= 0.95 and n_labels >= 1000:
            expertise_level = ExpertiseLevel.MASTER
        elif accuracy >= 0.90 and n_labels >= 500:
            expertise_level = ExpertiseLevel.EXPERT
        elif accuracy >= 0.80 and n_labels >= 100:
            expertise_level = ExpertiseLevel.INTERMEDIATE
        else:
            expertise_level = ExpertiseLevel.NOVICE

        # Performance trend
        performance_trend, recent_accuracy, accuracy_change = self._compute_trend(
            labels, ground_truth, timestamps
        )

        return LabelerMetrics(
            labeler_id=labeler_id,
            n_labels=n_labels,
            n_tasks=n_tasks,
            accuracy=accuracy,
            error_rate=error_rate,
            agreement_rate=agreement_rate,
            consistency_score=consistency_score,
            self_agreement=self_agreement,
            avg_labels_per_day=avg_labels_per_day,
            completion_rate=completion_rate,
            expertise_level=expertise_level,
            strong_categories=strong_categories,
            weak_categories=weak_categories,
            performance_trend=performance_trend,
            recent_accuracy=recent_accuracy,
            accuracy_change=accuracy_change,
            first_label_date=first_date,
            last_label_date=last_date,
            active_days=active_days,
        )

    def _compute_agreement_rate(
        self,
        labeler_id: str,
        task_ids: list[str | int],
        labels: list[Any],
    ) -> float:
        """Compute agreement with other labelers."""
        # Build task -> labels mapping for all labelers
        task_labels: dict[str | int, list[tuple[str, Any]]] = defaultdict(list)

        for label, lid, task_id in zip(self._labels, self._labeler_ids, self._task_ids):
            task_labels[task_id].append((lid, label))

        # Compute agreement for this labeler's tasks
        agreements = []
        for task_id, label in zip(task_ids, labels):
            other_labels = [
                l for lid, l in task_labels[task_id]
                if lid != labeler_id
            ]
            if other_labels:
                agreement = sum(1 for l in other_labels if l == label) / len(other_labels)
                agreements.append(agreement)

        return float(np.mean(agreements)) if agreements else 0.5

    def _compute_consistency(
        self,
        labels: list[Any],
        categories: list[str | None],
    ) -> float:
        """Compute labeling consistency within categories."""
        # Group labels by category
        category_labels: dict[str, list[Any]] = defaultdict(list)
        for label, cat in zip(labels, categories):
            if cat:
                category_labels[cat].append(label)

        if not category_labels:
            return 1.0

        # Compute variance within each category
        consistencies = []
        for cat, cat_labels in category_labels.items():
            if len(cat_labels) >= 2:
                # For categorical labels, use mode frequency
                from collections import Counter
                counter = Counter(cat_labels)
                mode_freq = counter.most_common(1)[0][1] / len(cat_labels)
                consistencies.append(mode_freq)

        return float(np.mean(consistencies)) if consistencies else 1.0

    def _compute_self_agreement(
        self,
        task_ids: list[str | int],
        labels: list[Any],
    ) -> float:
        """Compute self-agreement on repeated tasks."""
        # Find tasks with multiple labels from same labeler
        task_label_counts: dict[str | int, list[Any]] = defaultdict(list)

        for task_id, label in zip(task_ids, labels):
            task_label_counts[task_id].append(label)

        # Compute agreement on repeated tasks
        agreements = []
        for labels_list in task_label_counts.values():
            if len(labels_list) >= 2:
                # Agreement = proportion that matches first label
                first_label = labels_list[0]
                agreement = sum(1 for l in labels_list if l == first_label) / len(labels_list)
                agreements.append(agreement)

        return float(np.mean(agreements)) if agreements else 1.0

    def _compute_category_performance(
        self,
        labels: list[Any],
        ground_truth: list[Any],
        categories: list[str | None],
    ) -> dict[str, float]:
        """Compute performance by category."""
        category_results: dict[str, list[bool]] = defaultdict(list)

        has_gt = ground_truth and ground_truth[0] is not None
        if not has_gt:
            return {}

        for label, gt, cat in zip(labels, ground_truth, categories):
            if cat:
                category_results[cat].append(label == gt)

        return {
            cat: float(np.mean(results))
            for cat, results in category_results.items()
            if len(results) >= 5  # Minimum sample
        }

    def _compute_trend(
        self,
        labels: list[Any],
        ground_truth: list[Any],
        timestamps: list[datetime | None],
    ) -> tuple[PerformanceStatus, float, float]:
        """Compute performance trend."""
        has_gt = ground_truth and ground_truth[0] is not None
        if not has_gt:
            return PerformanceStatus.STABLE, 0.5, 0.0

        # Sort by timestamp if available
        valid_data = [
            (ts, l == gt)
            for ts, l, gt in zip(timestamps, labels, ground_truth)
            if ts is not None
        ]

        if len(valid_data) < self.window_size:
            # Use index order
            correct = [l == gt for l, gt in zip(labels, ground_truth)]
        else:
            valid_data.sort(key=lambda x: x[0])
            correct = [c for _, c in valid_data]

        if len(correct) < 20:
            return PerformanceStatus.STABLE, float(np.mean(correct)), 0.0

        # Compare first half to second half
        mid = len(correct) // 2
        early_accuracy = np.mean(correct[:mid])
        recent_accuracy = np.mean(correct[mid:])
        accuracy_change = recent_accuracy - early_accuracy

        # Compute variance
        window = min(self.window_size, len(correct) // 4)
        windowed_acc = [
            np.mean(correct[i:i + window])
            for i in range(0, len(correct) - window, window)
        ]

        if len(windowed_acc) >= 2:
            variance = np.std(windowed_acc)
        else:
            variance = 0.0

        # Determine trend
        if variance > 0.1:
            status = PerformanceStatus.INCONSISTENT
        elif accuracy_change > 0.05:
            status = PerformanceStatus.IMPROVING
        elif accuracy_change < -0.05:
            status = PerformanceStatus.DECLINING
        else:
            status = PerformanceStatus.STABLE

        return status, float(recent_accuracy), float(accuracy_change)

    def get_labeler_metrics(self, labeler_id: str) -> LabelerMetrics | None:
        """Get metrics for a specific labeler.

        Args:
            labeler_id: Labeler identifier

        Returns:
            LabelerMetrics or None
        """
        return self._labeler_metrics.get(labeler_id)

    def get_labeler_report(self, labeler_id: str) -> LabelerReport | None:
        """Get detailed report for a labeler.

        Args:
            labeler_id: Labeler identifier

        Returns:
            LabelerReport or None
        """
        metrics = self._labeler_metrics.get(labeler_id)
        if metrics is None:
            return None

        # Compute category performance
        labeler_mask = [lid == labeler_id for lid in self._labeler_ids]
        labeler_labels = [l for l, m in zip(self._labels, labeler_mask) if m]
        labeler_gt = [gt for gt, m in zip(self._ground_truth, labeler_mask) if m] if self._ground_truth else []
        labeler_cats = [c for c, m in zip(self._categories, labeler_mask) if m]

        category_performance = self._compute_category_performance(
            labeler_labels, labeler_gt, labeler_cats
        )

        # Compute error patterns
        common_errors, confused_pairs = self._compute_error_patterns(
            labeler_labels, labeler_gt
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, confused_pairs)

        # Determine suitable task types
        suitable_tasks = metrics.strong_categories.copy()
        unsuitable_tasks = metrics.weak_categories.copy()

        # Compute percentile rank
        all_accuracies = [m.accuracy for m in self._labeler_metrics.values()]
        percentile = int(np.percentile(
            all_accuracies,
            (sum(a <= metrics.accuracy for a in all_accuracies) / len(all_accuracies)) * 100
        ))

        comparison_to_avg = metrics.accuracy - np.mean(all_accuracies)

        return LabelerReport(
            labeler_id=labeler_id,
            metrics=metrics,
            generated_at=datetime.now(),
            category_performance=category_performance,
            common_error_patterns=common_errors,
            confused_label_pairs=confused_pairs,
            training_recommendations=recommendations,
            suitable_task_types=suitable_tasks,
            unsuitable_task_types=unsuitable_tasks,
            percentile_rank=percentile,
            comparison_to_avg=float(comparison_to_avg),
        )

    def _compute_error_patterns(
        self,
        labels: list[Any],
        ground_truth: list[Any],
    ) -> tuple[list[dict[str, Any]], list[tuple[Any, Any, int]]]:
        """Compute common error patterns."""
        if not ground_truth or ground_truth[0] is None:
            return [], []

        # Track confusion pairs
        from collections import Counter
        confusion_counts: Counter[tuple[Any, Any]] = Counter()

        for label, gt in zip(labels, ground_truth):
            if label != gt:
                confusion_counts[(gt, label)] += 1

        # Get top confused pairs
        top_confused = confusion_counts.most_common(5)
        confused_pairs = [(gt, pred, count) for (gt, pred), count in top_confused]

        # Identify patterns
        patterns = []
        for (gt, pred), count in top_confused[:3]:
            patterns.append({
                "type": "confusion",
                "true_label": gt,
                "predicted_label": pred,
                "count": count,
                "description": f"Often labels '{gt}' as '{pred}'",
            })

        return patterns, confused_pairs

    def _generate_recommendations(
        self,
        metrics: LabelerMetrics,
        confused_pairs: list[tuple[Any, Any, int]],
    ) -> list[str]:
        """Generate training recommendations."""
        recommendations = []

        # Based on accuracy
        if metrics.accuracy < 0.8:
            recommendations.append(
                "Consider additional training on labeling guidelines"
            )

        # Based on trend
        if metrics.performance_trend == PerformanceStatus.DECLINING:
            recommendations.append(
                "Performance declining - schedule refresher training"
            )

        # Based on consistency
        if metrics.consistency_score < 0.8:
            recommendations.append(
                "Improve consistency by reviewing similar examples together"
            )

        # Based on confused pairs
        for gt, pred, count in confused_pairs[:2]:
            if count >= 5:
                recommendations.append(
                    f"Review distinction between '{gt}' and '{pred}'"
                )

        # Based on weak categories
        for cat in metrics.weak_categories[:2]:
            recommendations.append(
                f"Additional training needed for '{cat}' category"
            )

        return recommendations

    def get_all_labeler_metrics(self) -> dict[str, LabelerMetrics]:
        """Get metrics for all labelers.

        Returns:
            Dictionary mapping labeler ID to metrics
        """
        return self._labeler_metrics.copy()

    def get_labeler_ranking(
        self,
        metric: str = "accuracy",
        ascending: bool = False,
    ) -> list[tuple[str, float]]:
        """Get ranking of labelers by metric.

        Args:
            metric: Metric to rank by
            ascending: Sort ascending if True

        Returns:
            List of (labeler_id, metric_value) tuples
        """
        rankings = []
        for labeler_id, metrics in self._labeler_metrics.items():
            value = getattr(metrics, metric, 0.0)
            rankings.append((labeler_id, value))

        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings


class SmartRouter:
    """Smart routing of tasks to appropriate labelers."""

    def __init__(
        self,
        evaluator: LabelerEvaluator,
        min_accuracy: float = 0.8,
        diversity_weight: float = 0.3,
    ):
        """Initialize router.

        Args:
            evaluator: Fitted LabelerEvaluator
            min_accuracy: Minimum accuracy threshold
            diversity_weight: Weight for labeler diversity
        """
        self.evaluator = evaluator
        self.min_accuracy = min_accuracy
        self.diversity_weight = diversity_weight

    def recommend_labelers(
        self,
        task_category: str | None = None,
        n_labelers: int = 3,
        exclude_labelers: list[str] | None = None,
    ) -> list[LabelerRecommendation]:
        """Recommend labelers for a task.

        Args:
            task_category: Task category/type
            n_labelers: Number of labelers to recommend
            exclude_labelers: Labelers to exclude

        Returns:
            List of LabelerRecommendation
        """
        exclude_set = set(exclude_labelers or [])
        recommendations = []

        for labeler_id, metrics in self.evaluator.get_all_labeler_metrics().items():
            if labeler_id in exclude_set:
                continue

            if metrics.accuracy < self.min_accuracy:
                continue

            # Calculate score
            score = self._calculate_suitability_score(metrics, task_category)

            # Determine reason
            if task_category and task_category in metrics.strong_categories:
                reason = f"Expert in {task_category}"
            elif metrics.expertise_level == ExpertiseLevel.MASTER:
                reason = "Master-level labeler"
            elif metrics.expertise_level == ExpertiseLevel.EXPERT:
                reason = "Expert labeler with high accuracy"
            else:
                reason = f"Suitable labeler with {metrics.accuracy:.0%} accuracy"

            recommendations.append(LabelerRecommendation(
                labeler_id=labeler_id,
                score=score,
                reason=reason,
                expected_accuracy=metrics.accuracy,
                confidence=min(1.0, metrics.n_labels / 100),
            ))

        # Sort by score
        recommendations.sort(key=lambda x: x.score, reverse=True)

        return recommendations[:n_labelers]

    def _calculate_suitability_score(
        self,
        metrics: LabelerMetrics,
        task_category: str | None,
    ) -> float:
        """Calculate suitability score for a labeler."""
        score = 0.0

        # Accuracy component (40%)
        score += metrics.accuracy * 0.4

        # Consistency component (20%)
        score += metrics.consistency_score * 0.2

        # Availability/efficiency component (10%)
        availability = min(1.0, metrics.avg_labels_per_day / 100)
        score += availability * 0.1

        # Category expertise (30% if category specified)
        if task_category:
            if task_category in metrics.strong_categories:
                score += 0.3
            elif task_category in metrics.weak_categories:
                score += 0.0
            else:
                score += 0.15

        # Trend adjustment
        if metrics.performance_trend == PerformanceStatus.IMPROVING:
            score *= 1.1
        elif metrics.performance_trend == PerformanceStatus.DECLINING:
            score *= 0.9

        return min(1.0, score)

    def create_task_assignments(
        self,
        task_ids: list[str | int],
        task_categories: list[str | None] | None = None,
        n_labelers_per_task: int = 3,
    ) -> list[TaskAssignment]:
        """Create optimized task assignments.

        Args:
            task_ids: List of task identifiers
            task_categories: Optional categories for each task
            n_labelers_per_task: Labelers to assign per task

        Returns:
            List of TaskAssignment
        """
        if task_categories is None:
            task_categories = [None] * len(task_ids)

        assignments = []

        for task_id, category in zip(task_ids, task_categories):
            # Get recommendations for this task
            recommendations = self.recommend_labelers(
                task_category=category,
                n_labelers=n_labelers_per_task,
            )

            # Estimate difficulty (based on category performance)
            if category:
                all_metrics = self.evaluator.get_all_labeler_metrics()
                cat_accuracies = []
                for m in all_metrics.values():
                    # Estimate from overall accuracy if no category data
                    cat_accuracies.append(m.accuracy)

                avg_accuracy = np.mean(cat_accuracies) if cat_accuracies else 0.8
                difficulty = 1.0 - avg_accuracy
            else:
                difficulty = 0.3  # Default medium difficulty

            assignments.append(TaskAssignment(
                task_id=task_id,
                task_category=category,
                recommended_labelers=recommendations,
                n_labelers_needed=n_labelers_per_task,
                difficulty_estimate=difficulty,
            ))

        return assignments


def evaluate_labelers(
    labels: list[Any] | np.ndarray,
    labeler_ids: list[str] | np.ndarray,
    ground_truth: list[Any] | np.ndarray | None = None,
    categories: list[str | None] | np.ndarray | None = None,
) -> LabelerEvaluator:
    """Convenience function to evaluate labelers.

    Args:
        labels: Labels provided
        labeler_ids: Labeler IDs
        ground_truth: True labels
        categories: Categories

    Returns:
        Fitted LabelerEvaluator
    """
    evaluator = LabelerEvaluator()
    evaluator.fit(labels, labeler_ids, ground_truth, categories)
    return evaluator


def get_labeler_report(
    labels: list[Any],
    labeler_ids: list[str],
    ground_truth: list[Any] | None,
    labeler_id: str,
) -> LabelerReport | None:
    """Convenience function to get a labeler report.

    Args:
        labels: Labels provided
        labeler_ids: Labeler IDs
        ground_truth: True labels
        labeler_id: Specific labeler to report on

    Returns:
        LabelerReport or None
    """
    evaluator = evaluate_labelers(labels, labeler_ids, ground_truth)
    return evaluator.get_labeler_report(labeler_id)
