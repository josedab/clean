"""Auto-fix engine for applying suggested fixes to datasets.

This module provides the FixEngine for generating and applying
fixes based on detected data quality issues.

Example:
    >>> from clean import DatasetCleaner
    >>> from clean.fixes import FixEngine
    >>>
    >>> cleaner = DatasetCleaner(data=df, label_column='label')
    >>> report = cleaner.analyze()
    >>>
    >>> engine = FixEngine(report, features=df, labels=labels)
    >>> fixes = engine.suggest_fixes()
    >>> print(f"Found {len(fixes)} suggested fixes")
    >>>
    >>> # Preview fixes
    >>> for fix in fixes[:5]:
    ...     print(fix)
    >>>
    >>> # Apply fixes
    >>> clean_df, clean_labels = engine.apply_fixes(fixes, dry_run=False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.plugins import SuggestedFix

if TYPE_CHECKING:
    from clean.core.report import QualityReport

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Strategy presets for fix aggressiveness."""

    CONSERVATIVE = "conservative"  # Only high-confidence fixes
    MODERATE = "moderate"  # Balanced approach
    AGGRESSIVE = "aggressive"  # Apply more fixes


@dataclass
class FixConfig:
    """Configuration for fix suggestions and application."""

    # Label error fixes
    label_error_threshold: float = 0.9  # Min confidence to suggest relabeling
    auto_relabel: bool = False  # Auto-apply relabeling

    # Duplicate fixes
    duplicate_similarity_threshold: float = 0.98  # Min similarity to suggest removal
    keep_strategy: str = "first"  # 'first', 'last', 'random'

    # Outlier fixes
    outlier_score_threshold: float = 0.9  # Min score to suggest removal
    outlier_action: str = "flag"  # 'remove', 'flag', 'impute'

    # General
    max_fixes: int | None = None  # Limit number of fixes
    require_confirmation: bool = True  # Require user confirmation

    @classmethod
    def from_strategy(cls, strategy: FixStrategy) -> FixConfig:
        """Create config from a preset strategy."""
        if strategy == FixStrategy.CONSERVATIVE:
            return cls(
                label_error_threshold=0.95,
                duplicate_similarity_threshold=0.99,
                outlier_score_threshold=0.95,
                auto_relabel=False,
            )
        elif strategy == FixStrategy.AGGRESSIVE:
            return cls(
                label_error_threshold=0.7,
                duplicate_similarity_threshold=0.9,
                outlier_score_threshold=0.7,
                auto_relabel=True,
            )
        else:  # MODERATE
            return cls()


@dataclass
class FixResult:
    """Result of applying fixes."""

    features: pd.DataFrame
    labels: np.ndarray | None
    applied_fixes: list[SuggestedFix]
    skipped_fixes: list[SuggestedFix]
    errors: list[tuple[SuggestedFix, Exception]]

    @property
    def n_applied(self) -> int:
        return len(self.applied_fixes)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_fixes)

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    def summary(self) -> str:
        """Generate summary of fix results."""
        lines = [
            "Fix Application Summary",
            "=" * 40,
            f"Applied: {self.n_applied} fixes",
            f"Skipped: {self.n_skipped} fixes",
            f"Errors:  {self.n_errors} fixes",
        ]
        if self.n_applied > 0:
            lines.append("\nApplied fixes by type:")
            type_counts: dict[str, int] = {}
            for fix in self.applied_fixes:
                type_counts[fix.fix_type] = type_counts.get(fix.fix_type, 0) + 1
            for fix_type, count in sorted(type_counts.items()):
                lines.append(f"  - {fix_type}: {count}")
        return "\n".join(lines)


class FixEngine:
    """Engine for generating and applying data fixes.

    The FixEngine analyzes a QualityReport and generates suggested
    fixes for detected issues. Fixes can be previewed and applied
    with various safety options.

    Example:
        >>> engine = FixEngine(report, features=df, labels=labels)
        >>>
        >>> # Get all suggested fixes
        >>> fixes = engine.suggest_fixes()
        >>>
        >>> # Filter to high-confidence fixes
        >>> high_conf = [f for f in fixes if f.confidence > 0.9]
        >>>
        >>> # Preview what would change
        >>> preview = engine.apply_fixes(high_conf, dry_run=True)
        >>> print(preview.summary())
        >>>
        >>> # Actually apply
        >>> result = engine.apply_fixes(high_conf, dry_run=False)
        >>> clean_df, clean_labels = result.features, result.labels
    """

    def __init__(
        self,
        report: QualityReport,
        features: pd.DataFrame,
        labels: np.ndarray | None = None,
        config: FixConfig | None = None,
    ):
        """Initialize the fix engine.

        Args:
            report: QualityReport from DatasetCleaner.analyze()
            features: Original feature DataFrame
            labels: Original labels array
            config: Fix configuration (defaults to moderate strategy)
        """
        self.report = report
        self.features = features.copy()
        self.labels = labels.copy() if labels is not None else None
        self.config = config or FixConfig()
        self._audit_log: list[dict[str, Any]] = []

    def suggest_fixes(
        self,
        include_label_errors: bool = True,
        include_duplicates: bool = True,
        include_outliers: bool = True,
    ) -> list[SuggestedFix]:
        """Generate fix suggestions based on the report.

        Args:
            include_label_errors: Include label error fixes
            include_duplicates: Include duplicate removal fixes
            include_outliers: Include outlier fixes

        Returns:
            List of SuggestedFix objects sorted by confidence
        """
        fixes: list[SuggestedFix] = []

        if include_label_errors:
            fixes.extend(self._suggest_label_fixes())

        if include_duplicates:
            fixes.extend(self._suggest_duplicate_fixes())

        if include_outliers:
            fixes.extend(self._suggest_outlier_fixes())

        # Sort by confidence descending
        fixes.sort(key=lambda f: f.confidence, reverse=True)

        # Apply max limit
        if self.config.max_fixes is not None:
            fixes = fixes[: self.config.max_fixes]

        return fixes

    def _suggest_label_fixes(self) -> list[SuggestedFix]:
        """Generate label error fix suggestions."""
        fixes = []

        if self.report.label_errors_result is None:
            return fixes

        for error in self.report.label_errors_result.issues:
            if error.confidence >= self.config.label_error_threshold:
                fixes.append(
                    SuggestedFix(
                        issue_type="label_error",
                        issue_index=error.index,
                        fix_type="relabel",
                        confidence=error.confidence,
                        description=f"Change label from '{error.given_label}' to '{error.predicted_label}'",
                        old_value=error.given_label,
                        new_value=error.predicted_label,
                        metadata={"self_confidence": error.self_confidence},
                    )
                )

        return fixes

    def _suggest_duplicate_fixes(self) -> list[SuggestedFix]:
        """Generate duplicate fix suggestions."""
        fixes = []

        if self.report.duplicates_result is None:
            return fixes

        # Track which indices we've already suggested removing
        to_remove: set[int] = set()

        for dup in self.report.duplicates_result.issues:
            if dup.similarity >= self.config.duplicate_similarity_threshold:
                # Keep first, suggest removing second
                if self.config.keep_strategy == "first":
                    remove_idx = dup.index2
                elif self.config.keep_strategy == "last":
                    remove_idx = dup.index1
                else:
                    remove_idx = dup.index2

                if remove_idx not in to_remove:
                    to_remove.add(remove_idx)
                    fixes.append(
                        SuggestedFix(
                            issue_type="duplicate",
                            issue_index=(dup.index1, dup.index2),
                            fix_type="remove",
                            confidence=dup.similarity,
                            description=f"Remove duplicate sample at index {remove_idx} (similar to {dup.index1 if remove_idx == dup.index2 else dup.index2})",
                            old_value=remove_idx,
                            new_value=None,
                            metadata={"is_exact": dup.is_exact, "keep_idx": dup.index1 if remove_idx == dup.index2 else dup.index2},
                        )
                    )

        return fixes

    def _suggest_outlier_fixes(self) -> list[SuggestedFix]:
        """Generate outlier fix suggestions."""
        fixes = []

        if self.report.outliers_result is None:
            return fixes

        for outlier in self.report.outliers_result.issues:
            if outlier.score >= self.config.outlier_score_threshold:
                if self.config.outlier_action == "remove":
                    fix_type = "remove"
                    description = f"Remove outlier at index {outlier.index}"
                elif self.config.outlier_action == "impute":
                    fix_type = "impute"
                    description = f"Impute outlier values at index {outlier.index}"
                else:
                    fix_type = "flag"
                    description = f"Flag outlier at index {outlier.index} for review"

                fixes.append(
                    SuggestedFix(
                        issue_type="outlier",
                        issue_index=outlier.index,
                        fix_type=fix_type,
                        confidence=outlier.score,
                        description=description,
                        old_value=outlier.index,
                        new_value=None,
                        metadata={"method": outlier.method, "score": outlier.score},
                    )
                )

        return fixes

    def apply_fixes(
        self,
        fixes: list[SuggestedFix],
        dry_run: bool = True,
    ) -> FixResult:
        """Apply a list of fixes to the data.

        Args:
            fixes: List of fixes to apply
            dry_run: If True, simulate but don't actually modify data

        Returns:
            FixResult with modified data and application status
        """
        # Work on copies
        features = self.features.copy()
        labels = self.labels.copy() if self.labels is not None else None

        applied: list[SuggestedFix] = []
        skipped: list[SuggestedFix] = []
        errors: list[tuple[SuggestedFix, Exception]] = []

        # Track indices to remove (applied after all other fixes)
        indices_to_remove: set[int] = set()

        for fix in fixes:
            try:
                if fix.fix_type == "relabel" and labels is not None:
                    if not dry_run:
                        idx = fix.issue_index
                        if isinstance(idx, int) and 0 <= idx < len(labels):
                            labels[idx] = fix.new_value
                            self._log_fix(fix, "applied")
                    applied.append(fix)

                elif fix.fix_type == "remove":
                    # Collect indices to remove
                    if fix.issue_type == "duplicate":
                        indices_to_remove.add(fix.old_value)
                    else:
                        idx = fix.issue_index
                        if isinstance(idx, int):
                            indices_to_remove.add(idx)
                    applied.append(fix)

                elif fix.fix_type == "flag":
                    # Flags don't modify data, just mark
                    if not dry_run and "is_flagged" not in features.columns:
                        features["is_flagged"] = False
                    if not dry_run:
                        idx = fix.issue_index
                        if isinstance(idx, int) and idx in features.index:
                            features.loc[idx, "is_flagged"] = True
                    applied.append(fix)

                elif fix.fix_type == "impute":
                    # Simple mean imputation for numeric columns
                    if not dry_run:
                        idx = fix.issue_index
                        if isinstance(idx, int) and idx in features.index:
                            for col in features.select_dtypes(include=[np.number]).columns:
                                if pd.notna(features.loc[idx, col]):
                                    col_mean = features[col].mean()
                                    features.loc[idx, col] = col_mean
                    applied.append(fix)

                else:
                    skipped.append(fix)
                    logger.warning("Unknown fix type: %s", fix.fix_type)

            except Exception as e:
                errors.append((fix, e))
                logger.exception("Error applying fix %s: %s", fix, e)

        # Remove collected indices
        if indices_to_remove and not dry_run:
            keep_mask = ~features.index.isin(indices_to_remove)
            features = features.loc[keep_mask].reset_index(drop=True)
            if labels is not None:
                mask_array = keep_mask.to_numpy() if hasattr(keep_mask, 'to_numpy') else np.array(keep_mask)
                labels = labels[mask_array]

        return FixResult(
            features=features,
            labels=labels,
            applied_fixes=applied,
            skipped_fixes=skipped,
            errors=errors,
        )

    def _log_fix(self, fix: SuggestedFix, status: str) -> None:
        """Log a fix to the audit trail."""
        self._audit_log.append({
            "fix_type": fix.fix_type,
            "issue_type": fix.issue_type,
            "index": fix.issue_index,
            "status": status,
            "confidence": fix.confidence,
            "description": fix.description,
        })

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log of applied fixes."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()


def suggest_fixes(
    report: QualityReport,
    features: pd.DataFrame,
    labels: np.ndarray | None = None,
    strategy: FixStrategy = FixStrategy.MODERATE,
) -> list[SuggestedFix]:
    """Convenience function to suggest fixes.

    Args:
        report: QualityReport from analysis
        features: Feature DataFrame
        labels: Label array
        strategy: Fix strategy preset

    Returns:
        List of suggested fixes
    """
    config = FixConfig.from_strategy(strategy)
    engine = FixEngine(report, features, labels, config)
    return engine.suggest_fixes()


def apply_fixes(
    report: QualityReport,
    features: pd.DataFrame,
    labels: np.ndarray | None = None,
    fixes: list[SuggestedFix] | None = None,
    strategy: FixStrategy = FixStrategy.MODERATE,
    dry_run: bool = True,
) -> FixResult:
    """Convenience function to apply fixes.

    Args:
        report: QualityReport from analysis
        features: Feature DataFrame
        labels: Label array
        fixes: Specific fixes to apply (if None, generates suggestions)
        strategy: Fix strategy preset
        dry_run: If True, simulate but don't modify

    Returns:
        FixResult with modified data
    """
    config = FixConfig.from_strategy(strategy)
    engine = FixEngine(report, features, labels, config)

    if fixes is None:
        fixes = engine.suggest_fixes()

    return engine.apply_fixes(fixes, dry_run=dry_run)


__all__ = [
    "FixEngine",
    "FixConfig",
    "FixResult",
    "FixStrategy",
    "suggest_fixes",
    "apply_fixes",
]
