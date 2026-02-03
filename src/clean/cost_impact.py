"""Cost-Impact Estimator for data quality improvements.

This module calculates the estimated labeling cost to fix issues and
predicted model accuracy improvement from data cleaning.

Example:
    >>> from clean.cost_impact import CostImpactEstimator, estimate_impact
    >>>
    >>> # Estimate impact of cleaning
    >>> estimator = CostImpactEstimator(
    ...     labeling_cost_per_sample=0.10,  # $0.10 per label
    ...     compute_cost_per_hour=0.50,     # $0.50 per GPU hour
    ... )
    >>> impact = estimator.estimate(quality_report, model_accuracy=0.85)
    >>> print(impact.summary())
    >>>
    >>> # Quick estimation
    >>> impact = estimate_impact(quality_report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from clean.core.report import QualityReport
from clean.core.types import IssueType

logger = logging.getLogger(__name__)


class CleaningAction(Enum):
    """Types of data cleaning actions."""

    RELABEL = "relabel"
    REMOVE = "remove"
    DEDUPLICATE = "deduplicate"
    REVIEW_OUTLIERS = "review_outliers"
    REBALANCE = "rebalance"
    AUGMENT = "augment"


@dataclass
class ActionCost:
    """Cost breakdown for a single action."""

    action: CleaningAction
    issue_type: IssueType | None
    n_samples: int
    human_cost_usd: float
    compute_cost_usd: float
    time_hours: float
    expected_accuracy_gain: float
    confidence: float  # Confidence in the estimate

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD."""
        return self.human_cost_usd + self.compute_cost_usd

    @property
    def roi(self) -> float:
        """Return on investment (accuracy gain per dollar)."""
        if self.total_cost_usd == 0:
            return float("inf")
        return self.expected_accuracy_gain / self.total_cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "issue_type": self.issue_type.value if self.issue_type else None,
            "n_samples": self.n_samples,
            "human_cost_usd": self.human_cost_usd,
            "compute_cost_usd": self.compute_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "time_hours": self.time_hours,
            "expected_accuracy_gain": self.expected_accuracy_gain,
            "roi": self.roi,
            "confidence": self.confidence,
        }


@dataclass
class ImpactReport:
    """Complete cost-impact assessment."""

    total_issues: int
    total_cost_usd: float
    total_time_hours: float
    expected_accuracy_gain: float
    current_accuracy: float | None
    projected_accuracy: float | None
    actions: list[ActionCost]
    recommendations: list[str]
    roi_ranking: list[str]  # Actions ranked by ROI

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "COST-IMPACT ASSESSMENT",
            "=" * 60,
            "",
            f"Total Issues Found: {self.total_issues:,}",
            "",
            "💰 COST ESTIMATE",
            "-" * 40,
            f"  Total Human Cost: ${self.total_cost_usd - sum(a.compute_cost_usd for a in self.actions):,.2f}",
            f"  Total Compute Cost: ${sum(a.compute_cost_usd for a in self.actions):,.2f}",
            f"  Total Cost: ${self.total_cost_usd:,.2f}",
            f"  Total Time: {self.total_time_hours:.1f} hours",
            "",
            "📈 EXPECTED IMPACT",
            "-" * 40,
        ]

        if self.current_accuracy is not None:
            lines.extend([
                f"  Current Accuracy: {self.current_accuracy:.1%}",
                f"  Projected Accuracy: {self.projected_accuracy:.1%}",
                f"  Expected Gain: +{self.expected_accuracy_gain:.1%}",
            ])
        else:
            lines.append(f"  Expected Accuracy Gain: +{self.expected_accuracy_gain:.1%}")

        lines.extend([
            "",
            "🎯 ACTION BREAKDOWN (by ROI)",
            "-" * 40,
        ])

        # Sort by ROI
        sorted_actions = sorted(self.actions, key=lambda a: a.roi, reverse=True)
        for action in sorted_actions[:5]:
            roi_str = f"{action.roi:.2f}" if action.roi < float("inf") else "∞"
            lines.append(
                f"  {action.action.value:15s} | "
                f"${action.total_cost_usd:8.2f} | "
                f"+{action.expected_accuracy_gain:.2%} | "
                f"ROI: {roi_str}"
            )

        if self.recommendations:
            lines.extend([
                "",
                "💡 RECOMMENDATIONS",
                "-" * 40,
            ])
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_issues": self.total_issues,
            "total_cost_usd": self.total_cost_usd,
            "total_time_hours": self.total_time_hours,
            "expected_accuracy_gain": self.expected_accuracy_gain,
            "current_accuracy": self.current_accuracy,
            "projected_accuracy": self.projected_accuracy,
            "actions": [a.to_dict() for a in self.actions],
            "recommendations": self.recommendations,
            "roi_ranking": self.roi_ranking,
        }


@dataclass
class CostConfig:
    """Configuration for cost estimation."""

    # Human labeling costs
    labeling_cost_per_sample: float = 0.10  # USD per sample for labeling
    review_cost_per_sample: float = 0.05  # USD per sample for review
    expert_review_cost_per_sample: float = 0.25  # USD for expert review

    # Time estimates (hours per sample)
    labeling_time_per_sample: float = 0.01  # 36 seconds per sample
    review_time_per_sample: float = 0.005  # 18 seconds per sample

    # Compute costs
    compute_cost_per_hour: float = 1.00  # USD per GPU hour
    embedding_cost_per_1k: float = 0.001  # USD per 1000 embeddings

    # Model training assumptions
    training_hours_per_epoch: float = 0.5
    expected_epochs_after_cleaning: int = 5


class CostImpactEstimator:
    """Estimate costs and impact of data cleaning actions."""

    # Empirical accuracy gain factors (based on data cleaning research)
    ACCURACY_GAIN_FACTORS = {
        IssueType.LABEL_ERROR: 0.03,  # 3% gain per 10% label errors fixed
        IssueType.DUPLICATE: 0.01,  # 1% gain per 10% duplicates removed
        IssueType.OUTLIER: 0.005,  # 0.5% gain per 10% outliers addressed
        IssueType.CLASS_IMBALANCE: 0.02,  # 2% gain from addressing imbalance
        IssueType.BIAS: 0.01,  # 1% gain from bias mitigation
    }

    def __init__(
        self,
        config: CostConfig | None = None,
        labeling_cost_per_sample: float | None = None,
        compute_cost_per_hour: float | None = None,
    ):
        """Initialize cost estimator.

        Args:
            config: Full cost configuration
            labeling_cost_per_sample: Override labeling cost
            compute_cost_per_hour: Override compute cost
        """
        self.config = config or CostConfig()

        if labeling_cost_per_sample is not None:
            self.config.labeling_cost_per_sample = labeling_cost_per_sample
        if compute_cost_per_hour is not None:
            self.config.compute_cost_per_hour = compute_cost_per_hour

    def estimate(
        self,
        report: QualityReport,
        current_accuracy: float | None = None,
        dataset_size: int | None = None,
    ) -> ImpactReport:
        """Estimate cost and impact of cleaning based on quality report.

        Args:
            report: Quality report from analysis
            current_accuracy: Current model accuracy (optional)
            dataset_size: Override dataset size

        Returns:
            Impact report with cost breakdown
        """
        actions = []
        total_accuracy_gain = 0.0

        n_samples = dataset_size or report.n_samples

        # Estimate for label errors
        if report.label_error_count > 0:
            action = self._estimate_label_error_action(
                report.label_error_count, n_samples
            )
            actions.append(action)
            total_accuracy_gain += action.expected_accuracy_gain

        # Estimate for duplicates
        if report.duplicate_count > 0:
            action = self._estimate_duplicate_action(
                report.duplicate_count, n_samples
            )
            actions.append(action)
            total_accuracy_gain += action.expected_accuracy_gain

        # Estimate for outliers
        if report.outlier_count > 0:
            action = self._estimate_outlier_action(report.outlier_count, n_samples)
            actions.append(action)
            total_accuracy_gain += action.expected_accuracy_gain

        # Estimate for imbalance
        if report.class_distribution:
            imbalance_ratio = self._calculate_imbalance_ratio(
                report.class_distribution
            )
            if imbalance_ratio > 3:  # Significant imbalance
                action = self._estimate_imbalance_action(
                    imbalance_ratio, n_samples, report.class_distribution
                )
                actions.append(action)
                total_accuracy_gain += action.expected_accuracy_gain

        # Calculate totals
        total_cost = sum(a.total_cost_usd for a in actions)
        total_time = sum(a.time_hours for a in actions)
        total_issues = sum(a.n_samples for a in actions)

        # Calculate projected accuracy
        projected_accuracy = None
        if current_accuracy is not None:
            projected_accuracy = min(current_accuracy + total_accuracy_gain, 1.0)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            actions, total_accuracy_gain, current_accuracy
        )

        # ROI ranking
        roi_ranking = [
            a.action.value
            for a in sorted(actions, key=lambda x: x.roi, reverse=True)
        ]

        return ImpactReport(
            total_issues=total_issues,
            total_cost_usd=total_cost,
            total_time_hours=total_time,
            expected_accuracy_gain=total_accuracy_gain,
            current_accuracy=current_accuracy,
            projected_accuracy=projected_accuracy,
            actions=actions,
            recommendations=recommendations,
            roi_ranking=roi_ranking,
        )

    def _estimate_label_error_action(
        self, n_errors: int, total_samples: int
    ) -> ActionCost:
        """Estimate cost for fixing label errors."""
        # Cost: relabeling
        human_cost = n_errors * self.config.labeling_cost_per_sample
        time_hours = n_errors * self.config.labeling_time_per_sample

        # Compute cost for retraining
        compute_cost = (
            self.config.training_hours_per_epoch
            * self.config.expected_epochs_after_cleaning
            * self.config.compute_cost_per_hour
        )

        # Expected accuracy gain
        error_rate = n_errors / total_samples
        accuracy_gain = min(
            error_rate * self.ACCURACY_GAIN_FACTORS[IssueType.LABEL_ERROR] * 10,
            0.15,  # Cap at 15%
        )

        return ActionCost(
            action=CleaningAction.RELABEL,
            issue_type=IssueType.LABEL_ERROR,
            n_samples=n_errors,
            human_cost_usd=human_cost,
            compute_cost_usd=compute_cost,
            time_hours=time_hours,
            expected_accuracy_gain=accuracy_gain,
            confidence=0.7,  # Medium-high confidence
        )

    def _estimate_duplicate_action(
        self, n_duplicates: int, total_samples: int
    ) -> ActionCost:
        """Estimate cost for removing duplicates."""
        # Cost: mostly automated, some review
        human_cost = n_duplicates * self.config.review_cost_per_sample * 0.1  # Only review 10%
        time_hours = n_duplicates * self.config.review_time_per_sample * 0.1

        # Compute cost for reprocessing
        compute_cost = (
            self.config.embedding_cost_per_1k * (total_samples / 1000)
            + self.config.training_hours_per_epoch
            * self.config.expected_epochs_after_cleaning
            * self.config.compute_cost_per_hour
        )

        # Expected accuracy gain
        dup_rate = n_duplicates / total_samples
        accuracy_gain = min(
            dup_rate * self.ACCURACY_GAIN_FACTORS[IssueType.DUPLICATE] * 10,
            0.05,  # Cap at 5%
        )

        return ActionCost(
            action=CleaningAction.DEDUPLICATE,
            issue_type=IssueType.DUPLICATE,
            n_samples=n_duplicates,
            human_cost_usd=human_cost,
            compute_cost_usd=compute_cost,
            time_hours=time_hours,
            expected_accuracy_gain=accuracy_gain,
            confidence=0.8,  # High confidence - duplicates are clear
        )

    def _estimate_outlier_action(
        self, n_outliers: int, total_samples: int
    ) -> ActionCost:
        """Estimate cost for addressing outliers."""
        # Cost: expert review needed
        human_cost = n_outliers * self.config.expert_review_cost_per_sample * 0.5
        time_hours = n_outliers * self.config.review_time_per_sample

        # Compute cost
        compute_cost = (
            self.config.training_hours_per_epoch
            * self.config.expected_epochs_after_cleaning
            * self.config.compute_cost_per_hour
        )

        # Expected accuracy gain (conservative - outliers are tricky)
        outlier_rate = n_outliers / total_samples
        accuracy_gain = min(
            outlier_rate * self.ACCURACY_GAIN_FACTORS[IssueType.OUTLIER] * 10,
            0.03,  # Cap at 3%
        )

        return ActionCost(
            action=CleaningAction.REVIEW_OUTLIERS,
            issue_type=IssueType.OUTLIER,
            n_samples=n_outliers,
            human_cost_usd=human_cost,
            compute_cost_usd=compute_cost,
            time_hours=time_hours,
            expected_accuracy_gain=accuracy_gain,
            confidence=0.5,  # Lower confidence - outliers depend on context
        )

    def _estimate_imbalance_action(
        self,
        imbalance_ratio: float,
        total_samples: int,
        class_distribution: dict[str, int],
    ) -> ActionCost:
        """Estimate cost for addressing class imbalance."""
        # Calculate samples needed for balance
        max_class_size = max(class_distribution.values())
        min_class_size = min(class_distribution.values())
        samples_needed = max_class_size - min_class_size

        # Cost: augmentation and potential labeling
        # Assume 20% need manual labeling, 80% can be augmented
        human_cost = samples_needed * 0.2 * self.config.labeling_cost_per_sample
        time_hours = samples_needed * 0.2 * self.config.labeling_time_per_sample

        # Compute cost for augmentation and training
        compute_cost = (
            self.config.compute_cost_per_hour * 2  # Augmentation
            + self.config.training_hours_per_epoch
            * self.config.expected_epochs_after_cleaning
            * self.config.compute_cost_per_hour
        )

        # Expected accuracy gain
        # Higher imbalance = more gain from fixing
        accuracy_gain = min(
            np.log10(imbalance_ratio) * self.ACCURACY_GAIN_FACTORS[IssueType.CLASS_IMBALANCE],
            0.08,  # Cap at 8%
        )

        return ActionCost(
            action=CleaningAction.REBALANCE,
            issue_type=IssueType.CLASS_IMBALANCE,
            n_samples=samples_needed,
            human_cost_usd=human_cost,
            compute_cost_usd=compute_cost,
            time_hours=time_hours,
            expected_accuracy_gain=accuracy_gain,
            confidence=0.6,  # Medium confidence
        )

    def _calculate_imbalance_ratio(
        self, class_distribution: dict[str, int]
    ) -> float:
        """Calculate the imbalance ratio."""
        if not class_distribution:
            return 1.0
        counts = list(class_distribution.values())
        return max(counts) / max(min(counts), 1)

    def _generate_recommendations(
        self,
        actions: list[ActionCost],
        total_gain: float,
        current_accuracy: float | None,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Sort by ROI
        sorted_actions = sorted(actions, key=lambda a: a.roi, reverse=True)

        if sorted_actions:
            best_action = sorted_actions[0]
            recommendations.append(
                f"Start with {best_action.action.value} - highest ROI "
                f"(+{best_action.expected_accuracy_gain:.1%} for ${best_action.total_cost_usd:.2f})"
            )

        # Check if label errors are significant
        label_actions = [a for a in actions if a.issue_type == IssueType.LABEL_ERROR]
        if label_actions and label_actions[0].n_samples > 100:
            recommendations.append(
                f"High priority: {label_actions[0].n_samples} label errors detected. "
                "Consider using confident learning for prioritized review."
            )

        # Budget recommendations
        total_cost = sum(a.total_cost_usd for a in actions)
        if total_cost > 1000:
            recommendations.append(
                f"Total cleaning cost is ${total_cost:,.2f}. "
                "Consider phased approach starting with highest-ROI actions."
            )

        # Accuracy-based recommendations
        if current_accuracy is not None:
            if current_accuracy < 0.8:
                recommendations.append(
                    "Current accuracy is below 80%. Data quality improvements "
                    "likely to have significant impact."
                )
            if current_accuracy + total_gain > 0.95:
                recommendations.append(
                    "Projected accuracy exceeds 95%. Verify with validation set "
                    "to avoid overfitting to training data issues."
                )

        return recommendations


def estimate_impact(
    report: QualityReport,
    current_accuracy: float | None = None,
    labeling_cost: float = 0.10,
    compute_cost: float = 1.00,
) -> ImpactReport:
    """Convenience function to estimate impact of data cleaning.

    Args:
        report: Quality report from analysis
        current_accuracy: Current model accuracy
        labeling_cost: Cost per sample for labeling (USD)
        compute_cost: Cost per GPU hour (USD)

    Returns:
        Impact report with cost breakdown

    Example:
        >>> report = cleaner.analyze()
        >>> impact = estimate_impact(report, current_accuracy=0.85)
        >>> print(impact.summary())
    """
    estimator = CostImpactEstimator(
        labeling_cost_per_sample=labeling_cost,
        compute_cost_per_hour=compute_cost,
    )
    return estimator.estimate(report, current_accuracy=current_accuracy)


__all__ = [
    "CleaningAction",
    "ActionCost",
    "ImpactReport",
    "CostConfig",
    "CostImpactEstimator",
    "estimate_impact",
]
