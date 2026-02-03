"""Proactive Quality Advisor.

This module provides AI-powered recommendations for improving data quality,
analyzing patterns in issues, and suggesting prioritized remediation actions.

Example:
    >>> from clean.advisor import QualityAdvisor, get_recommendations
    >>>
    >>> # Get recommendations from a report
    >>> advisor = QualityAdvisor()
    >>> recommendations = advisor.analyze(report)
    >>> print(recommendations.summary())
    >>>
    >>> # Quick function
    >>> recs = get_recommendations(report)
    >>> for rec in recs.top(5):
    ...     print(f"- {rec.title}: {rec.description}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from clean.core.report import QualityReport
from clean.core.types import IssueType
from clean.exceptions import CleanError, ConfigurationError, DependencyError

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Recommendation priority level."""

    CRITICAL = "critical"  # Must fix - blocks production use
    HIGH = "high"  # Should fix soon - significant impact
    MEDIUM = "medium"  # Nice to fix - moderate impact
    LOW = "low"  # Optional - minor improvement


class ActionCategory(Enum):
    """Category of recommended action."""

    RELABEL = "relabel"  # Fix label errors
    DEDUPLICATE = "deduplicate"  # Remove duplicates
    CLEAN = "clean"  # Clean outliers/anomalies
    AUGMENT = "augment"  # Add more data
    BALANCE = "balance"  # Address imbalance
    TRANSFORM = "transform"  # Transform features
    VALIDATE = "validate"  # Add validation rules
    MONITOR = "monitor"  # Set up monitoring


@dataclass
class Recommendation:
    """A specific recommendation for improving data quality."""

    id: str
    title: str
    description: str
    priority: Priority
    category: ActionCategory
    impact_score: float  # 0-100, estimated improvement
    effort_score: float  # 0-100, estimated effort (lower = easier)
    affected_samples: int = 0
    affected_columns: list[str] = field(default_factory=list)
    code_snippet: str | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def roi_score(self) -> float:
        """Return on investment score (impact / effort)."""
        if self.effort_score == 0:
            return self.impact_score
        return self.impact_score / self.effort_score * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "category": self.category.value,
            "impact_score": self.impact_score,
            "effort_score": self.effort_score,
            "roi_score": self.roi_score,
            "affected_samples": self.affected_samples,
            "affected_columns": self.affected_columns,
            "code_snippet": self.code_snippet,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass
class AdvisorReport:
    """Complete set of recommendations from the advisor."""

    recommendations: list[Recommendation]
    quality_score: float
    projected_score: float  # Score after implementing recommendations
    analysis_timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def top(self, n: int = 5) -> list[Recommendation]:
        """Get top N recommendations by ROI score."""
        return sorted(
            self.recommendations, key=lambda r: r.roi_score, reverse=True
        )[:n]

    def by_priority(self, priority: Priority) -> list[Recommendation]:
        """Filter recommendations by priority."""
        return [r for r in self.recommendations if r.priority == priority]

    def by_category(self, category: ActionCategory) -> list[Recommendation]:
        """Filter recommendations by category."""
        return [r for r in self.recommendations if r.category == category]

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "Quality Advisor Report",
            "=" * 50,
            "",
            f"Current Quality Score: {self.quality_score:.1f}/100",
            f"Projected Score (after fixes): {self.projected_score:.1f}/100",
            f"Potential Improvement: +{self.projected_score - self.quality_score:.1f}",
            "",
            f"Total Recommendations: {len(self.recommendations)}",
        ]

        # Count by priority
        by_priority = {}
        for rec in self.recommendations:
            by_priority[rec.priority.value] = by_priority.get(rec.priority.value, 0) + 1

        if by_priority:
            lines.append("")
            lines.append("By Priority:")
            for priority in ["critical", "high", "medium", "low"]:
                if priority in by_priority:
                    lines.append(f"  - {priority.title()}: {by_priority[priority]}")

        # Top recommendations
        lines.append("")
        lines.append("Top Recommendations:")
        for i, rec in enumerate(self.top(5), 1):
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            lines.append(
                f"  {i}. {emoji.get(rec.priority.value, '•')} [{rec.priority.value.upper()}] {rec.title}"
            )
            lines.append(f"     Impact: {rec.impact_score:.0f} | Effort: {rec.effort_score:.0f} | ROI: {rec.roi_score:.0f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality_score": self.quality_score,
            "projected_score": self.projected_score,
            "analysis_timestamp": self.analysis_timestamp,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            "# Quality Advisor Report",
            "",
            f"**Current Score**: {self.quality_score:.1f}/100",
            f"**Projected Score**: {self.projected_score:.1f}/100",
            f"**Generated**: {self.analysis_timestamp}",
            "",
            "## Recommendations",
            "",
        ]

        priority_order = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]

        for priority in priority_order:
            recs = self.by_priority(priority)
            if not recs:
                continue

            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            lines.append(f"### {emoji[priority.value]} {priority.value.title()} Priority")
            lines.append("")

            for rec in sorted(recs, key=lambda r: r.roi_score, reverse=True):
                lines.append(f"#### {rec.title}")
                lines.append("")
                lines.append(f"**Category**: {rec.category.value} | **Impact**: {rec.impact_score:.0f} | **Effort**: {rec.effort_score:.0f}")
                lines.append("")
                lines.append(rec.description)
                lines.append("")

                if rec.rationale:
                    lines.append(f"_Rationale: {rec.rationale}_")
                    lines.append("")

                if rec.code_snippet:
                    lines.append("```python")
                    lines.append(rec.code_snippet)
                    lines.append("```")
                    lines.append("")

        return "\n".join(lines)


class QualityAdvisor:
    """AI-powered quality advisor that generates recommendations.

    Example:
        >>> advisor = QualityAdvisor()
        >>> report = advisor.analyze(quality_report)
        >>> print(report.summary())
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        use_llm: bool = False,
        custom_rules: list[Callable] | None = None,
    ):
        """Initialize advisor.

        Args:
            api_key: OpenAI API key for LLM-powered recommendations
            model: LLM model to use
            use_llm: Whether to use LLM for enhanced recommendations
            custom_rules: Custom rule functions for recommendations
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.use_llm = use_llm
        self.custom_rules = custom_rules or []
        self._openai = None
        self._rec_counter = 0

    def analyze(
        self,
        report: QualityReport,
        data: pd.DataFrame | None = None,
    ) -> AdvisorReport:
        """Analyze a quality report and generate recommendations.

        Args:
            report: Quality report to analyze
            data: Optional DataFrame for deeper analysis

        Returns:
            AdvisorReport with recommendations
        """
        from datetime import datetime

        logger.info("Analyzing quality report for recommendations")

        recommendations = []

        # Run built-in analyzers
        recommendations.extend(self._analyze_label_errors(report))
        recommendations.extend(self._analyze_duplicates(report))
        recommendations.extend(self._analyze_outliers(report))
        recommendations.extend(self._analyze_imbalance(report))
        recommendations.extend(self._analyze_completeness(report))

        # Run custom rules
        for rule in self.custom_rules:
            try:
                custom_recs = rule(report, data)
                if custom_recs:
                    recommendations.extend(custom_recs)
            except Exception as e:
                logger.warning(f"Custom rule failed: {e}")

        # Enhance with LLM if enabled
        if self.use_llm and self.api_key:
            recommendations = self._enhance_with_llm(recommendations, report)

        # Calculate projected score
        current_score = self._get_score(report)
        projected_score = self._calculate_projected_score(current_score, recommendations)

        return AdvisorReport(
            recommendations=recommendations,
            quality_score=current_score,
            projected_score=projected_score,
            analysis_timestamp=datetime.now().isoformat(),
        )

    def _get_score(self, report: QualityReport) -> float:
        """Extract quality score from report."""
        score = report.quality_score
        if hasattr(score, "overall"):
            return score.overall
        return float(score)

    def _next_id(self) -> str:
        """Generate next recommendation ID."""
        self._rec_counter += 1
        return f"REC-{self._rec_counter:03d}"

    def _analyze_label_errors(self, report: QualityReport) -> list[Recommendation]:
        """Analyze label errors and generate recommendations."""
        recs = []

        if report.label_errors_result is None:
            return recs

        issues = report.label_errors_result.issues
        n_errors = len(issues)
        n_samples = report.dataset_info.n_samples

        if n_errors == 0:
            return recs

        error_rate = n_errors / n_samples * 100

        # Determine priority based on error rate
        if error_rate > 10:
            priority = Priority.CRITICAL
        elif error_rate > 5:
            priority = Priority.HIGH
        elif error_rate > 2:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        # Main recommendation
        recs.append(Recommendation(
            id=self._next_id(),
            title=f"Review and fix {n_errors:,} potential label errors",
            description=(
                f"Found {n_errors:,} samples ({error_rate:.1f}%) with potential label errors. "
                f"These may be mislabeled examples that could hurt model performance."
            ),
            priority=priority,
            category=ActionCategory.RELABEL,
            impact_score=min(error_rate * 5, 50),  # Up to 50 points
            effort_score=min(n_errors / 100, 80),  # Scale with count
            affected_samples=n_errors,
            rationale=(
                "Label errors are one of the most impactful data quality issues. "
                "Studies show that fixing just 5% of label errors can improve model accuracy by 1-3%."
            ),
            code_snippet="""# Export label errors for review
label_errors = report.label_errors_result.issues
error_indices = [issue.index for issue in label_errors]
df_errors = df.iloc[error_indices]
df_errors.to_csv('label_errors_to_review.csv')""",
        ))

        # Additional recommendation for high error rates
        if error_rate > 20:
            recs.append(Recommendation(
                id=self._next_id(),
                title="Consider re-labeling a sample of data",
                description=(
                    f"With {error_rate:.1f}% potential label errors, consider having a subset "
                    "professionally re-labeled to establish ground truth."
                ),
                priority=Priority.HIGH,
                category=ActionCategory.RELABEL,
                impact_score=30,
                effort_score=60,
                affected_samples=min(1000, n_samples // 10),
                rationale="High error rates may indicate systematic labeling issues.",
            ))

        return recs

    def _analyze_duplicates(self, report: QualityReport) -> list[Recommendation]:
        """Analyze duplicates and generate recommendations."""
        recs = []

        if report.duplicates_result is None:
            return recs

        issues = report.duplicates_result.issues
        n_duplicates = len(issues)
        n_samples = report.dataset_info.n_samples

        if n_duplicates == 0:
            return recs

        dup_rate = n_duplicates / n_samples * 100

        # Determine priority
        if dup_rate > 20:
            priority = Priority.HIGH
        elif dup_rate > 10:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        recs.append(Recommendation(
            id=self._next_id(),
            title=f"Remove {n_duplicates:,} duplicate samples",
            description=(
                f"Found {n_duplicates:,} duplicate or near-duplicate samples ({dup_rate:.1f}%). "
                "Duplicates can cause data leakage if they appear in both train and test sets."
            ),
            priority=priority,
            category=ActionCategory.DEDUPLICATE,
            impact_score=min(dup_rate * 3, 40),
            effort_score=20,  # Usually easy to fix
            affected_samples=n_duplicates,
            rationale="Duplicates inflate metrics and can cause train-test leakage.",
            code_snippet="""# Remove duplicates
duplicate_indices = [issue.index for issue in report.duplicates_result.issues]
df_clean = df.drop(index=duplicate_indices)
print(f"Removed {len(duplicate_indices)} duplicates")""",
        ))

        return recs

    def _analyze_outliers(self, report: QualityReport) -> list[Recommendation]:
        """Analyze outliers and generate recommendations."""
        recs = []

        if report.outliers_result is None:
            return recs

        issues = report.outliers_result.issues
        n_outliers = len(issues)
        n_samples = report.dataset_info.n_samples

        if n_outliers == 0:
            return recs

        outlier_rate = n_outliers / n_samples * 100

        # Determine priority
        if outlier_rate > 10:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        recs.append(Recommendation(
            id=self._next_id(),
            title=f"Review {n_outliers:,} potential outliers",
            description=(
                f"Found {n_outliers:,} outlier samples ({outlier_rate:.1f}%). "
                "Some may be data errors, others may be valuable edge cases."
            ),
            priority=priority,
            category=ActionCategory.CLEAN,
            impact_score=min(outlier_rate * 2, 20),
            effort_score=40,  # Requires manual review
            affected_samples=n_outliers,
            rationale="Outliers can be data errors or valuable edge cases - manual review recommended.",
            code_snippet="""# Review outliers
outlier_indices = [issue.index for issue in report.outliers_result.issues]
df_outliers = df.iloc[outlier_indices]
# Inspect before deciding to remove
df_outliers.describe()""",
        ))

        return recs

    def _analyze_imbalance(self, report: QualityReport) -> list[Recommendation]:
        """Analyze class imbalance and generate recommendations."""
        recs = []

        if report.imbalance_result is None and report.class_distribution is None:
            return recs

        # Get class distribution info
        class_dist = report.class_distribution
        if class_dist is None:
            return recs

        # Check for imbalance
        if hasattr(class_dist, "class_counts"):
            counts = class_dist.class_counts
            if counts:
                max_count = max(counts.values())
                min_count = min(counts.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

                if imbalance_ratio > 10:
                    priority = Priority.HIGH
                elif imbalance_ratio > 5:
                    priority = Priority.MEDIUM
                else:
                    return recs  # No significant imbalance

                recs.append(Recommendation(
                    id=self._next_id(),
                    title="Address class imbalance",
                    description=(
                        f"Class imbalance ratio is {imbalance_ratio:.1f}:1. "
                        "This can bias models toward the majority class."
                    ),
                    priority=priority,
                    category=ActionCategory.BALANCE,
                    impact_score=min(imbalance_ratio * 2, 35),
                    effort_score=30,
                    rationale="Class imbalance causes models to ignore minority classes.",
                    code_snippet="""# Options for handling imbalance:
# 1. Oversample minority class
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)

# 2. Undersample majority class
# 3. Use class_weight='balanced' in model

# 4. Use focal loss for training""",
                ))

        return recs

    def _analyze_completeness(self, report: QualityReport) -> list[Recommendation]:
        """Analyze data completeness and generate recommendations."""
        recs = []

        # Check completeness score
        score = report.quality_score
        completeness = getattr(score, "completeness", 100) if hasattr(score, "completeness") else 100

        if completeness < 90:
            if completeness < 70:
                priority = Priority.HIGH
            else:
                priority = Priority.MEDIUM

            recs.append(Recommendation(
                id=self._next_id(),
                title="Improve data completeness",
                description=(
                    f"Data completeness is {completeness:.1f}%. "
                    "Missing values can reduce model performance and require imputation."
                ),
                priority=priority,
                category=ActionCategory.TRANSFORM,
                impact_score=max(0, (100 - completeness) * 0.5),
                effort_score=35,
                rationale="Missing data reduces effective sample size and may introduce bias.",
                code_snippet="""# Analyze missing patterns
missing_pct = df.isnull().sum() / len(df) * 100
high_missing = missing_pct[missing_pct > 50]
print("Columns with >50% missing:", high_missing)

# Options:
# 1. Drop columns with too many missing values
# 2. Impute with mean/median/mode
# 3. Use advanced imputation (KNN, iterative)""",
            ))

        return recs

    def _calculate_projected_score(
        self, current_score: float, recommendations: list[Recommendation]
    ) -> float:
        """Calculate projected score after implementing recommendations."""
        # Sum up impact scores (with diminishing returns)
        total_impact = 0
        for rec in sorted(recommendations, key=lambda r: r.impact_score, reverse=True):
            # Diminishing returns - each subsequent fix has less impact
            diminish_factor = 0.9 ** (total_impact / 20)
            total_impact += rec.impact_score * diminish_factor * 0.5

        projected = min(100, current_score + total_impact)
        return projected

    def _enhance_with_llm(
        self,
        recommendations: list[Recommendation],
        report: QualityReport,
    ) -> list[Recommendation]:
        """Enhance recommendations with LLM insights."""
        try:
            import openai
        except ImportError:
            logger.warning("OpenAI not available for LLM enhancement")
            return recommendations

        if not self.api_key:
            return recommendations

        client = openai.OpenAI(api_key=self.api_key)

        # Prepare summary for LLM
        rec_summary = "\n".join([
            f"- {r.title} (Priority: {r.priority.value})"
            for r in recommendations
        ])

        prompt = f"""Analyze these data quality recommendations and suggest any additional considerations:

Quality Score: {self._get_score(report):.1f}/100
Sample Count: {report.dataset_info.n_samples}

Current Recommendations:
{rec_summary}

Provide 1-2 additional strategic recommendations that may have been missed.
Focus on practical, high-impact improvements.
Return as JSON: [{{"title": "...", "description": "...", "priority": "high/medium/low"}}]"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)

            if isinstance(result, list):
                for item in result[:2]:  # Limit to 2 LLM suggestions
                    priority_map = {
                        "critical": Priority.CRITICAL,
                        "high": Priority.HIGH,
                        "medium": Priority.MEDIUM,
                        "low": Priority.LOW,
                    }
                    recommendations.append(Recommendation(
                        id=self._next_id(),
                        title=item.get("title", "LLM Suggestion"),
                        description=item.get("description", ""),
                        priority=priority_map.get(item.get("priority", "medium").lower(), Priority.MEDIUM),
                        category=ActionCategory.VALIDATE,
                        impact_score=25,
                        effort_score=30,
                        metadata={"source": "llm"},
                    ))

            logger.info("Enhanced recommendations with LLM insights")

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")

        return recommendations


def get_recommendations(
    report: QualityReport,
    data: pd.DataFrame | None = None,
    use_llm: bool = False,
    api_key: str | None = None,
) -> AdvisorReport:
    """Get quality improvement recommendations.

    Args:
        report: Quality report to analyze
        data: Optional DataFrame for deeper analysis
        use_llm: Whether to use LLM for enhanced recommendations
        api_key: OpenAI API key (if using LLM)

    Returns:
        AdvisorReport with recommendations

    Example:
        >>> report = analyze(df, labels=labels)
        >>> recs = get_recommendations(report)
        >>> print(recs.summary())
    """
    advisor = QualityAdvisor(api_key=api_key, use_llm=use_llm)
    return advisor.analyze(report, data)


__all__ = [
    "QualityAdvisor",
    "AdvisorReport",
    "Recommendation",
    "Priority",
    "ActionCategory",
    "get_recommendations",
]
