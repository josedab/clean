"""Synthetic data quality validation.

This module provides tools for validating synthetic/generated data quality:
- Mode collapse detection
- Memorization/overfitting detection
- Distribution gap analysis
- Diversity metrics

Example:
    >>> from clean.synthetic import SyntheticDataValidator
    >>>
    >>> validator = SyntheticDataValidator()
    >>> validator.set_reference(real_df)
    >>> report = validator.validate(synthetic_df)
    >>> print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class SyntheticIssueType(Enum):
    """Types of synthetic data issues."""

    MODE_COLLAPSE = "mode_collapse"
    MEMORIZATION = "memorization"
    DISTRIBUTION_GAP = "distribution_gap"
    LOW_DIVERSITY = "low_diversity"
    FEATURE_CORRELATION = "feature_correlation"
    OUTLIER_GENERATION = "outlier_generation"


class IssueSeverity(Enum):
    """Severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SyntheticIssue:
    """A detected issue in synthetic data."""

    issue_type: SyntheticIssueType
    severity: IssueSeverity
    description: str
    affected_features: list[str] = field(default_factory=list)
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_features": self.affected_features,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SyntheticValidationReport:
    """Validation report for synthetic data."""

    n_real_samples: int
    n_synthetic_samples: int
    quality_score: float  # 0-100
    fidelity_score: float  # How well it matches real data distribution
    diversity_score: float  # Internal diversity of synthetic data
    privacy_score: float  # Risk of memorization/leakage
    issues: list[SyntheticIssue]
    feature_scores: dict[str, float]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✅ GOOD" if self.quality_score >= 70 else "⚠️ NEEDS ATTENTION"
        lines = [
            "Synthetic Data Validation Report",
            "=" * 50,
            f"Status: {status}",
            f"Real Samples: {self.n_real_samples:,}",
            f"Synthetic Samples: {self.n_synthetic_samples:,}",
            "",
            "Quality Scores (0-100):",
            f"  Overall:   {self.quality_score:.1f}",
            f"  Fidelity:  {self.fidelity_score:.1f}",
            f"  Diversity: {self.diversity_score:.1f}",
            f"  Privacy:   {self.privacy_score:.1f}",
            "",
        ]

        if self.issues:
            lines.append(f"Issues Found ({len(self.issues)}):")
            for issue in self.issues[:5]:
                lines.append(f"  - [{issue.severity.value}] {issue.issue_type.value}: {issue.description}")
            if len(self.issues) > 5:
                lines.append(f"  ... and {len(self.issues) - 5} more")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_real_samples": self.n_real_samples,
            "n_synthetic_samples": self.n_synthetic_samples,
            "quality_score": self.quality_score,
            "fidelity_score": self.fidelity_score,
            "diversity_score": self.diversity_score,
            "privacy_score": self.privacy_score,
            "issues": [i.to_dict() for i in self.issues],
            "feature_scores": self.feature_scores,
            "recommendations": self.recommendations,
        }


class SyntheticDataValidator:
    """Validator for synthetic/generated data quality.

    Assesses how well synthetic data matches real data distributions
    while maintaining diversity and privacy.
    """

    def __init__(
        self,
        memorization_threshold: float = 0.95,
        mode_collapse_threshold: float = 0.1,
        distribution_threshold: float = 0.1,
    ):
        """Initialize the validator.

        Args:
            memorization_threshold: Similarity threshold for memorization detection
            mode_collapse_threshold: Threshold for mode collapse detection
            distribution_threshold: Threshold for distribution gap detection
        """
        self.memorization_threshold = memorization_threshold
        self.mode_collapse_threshold = mode_collapse_threshold
        self.distribution_threshold = distribution_threshold

        self._reference: pd.DataFrame | None = None
        self._reference_stats: dict[str, dict[str, float]] = {}

    def set_reference(
        self,
        real_data: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> SyntheticDataValidator:
        """Set the real/reference data for comparison.

        Args:
            real_data: Real data DataFrame
            feature_columns: Columns to use for validation

        Returns:
            Self for chaining
        """
        self._reference = real_data.copy()

        if feature_columns:
            self._reference = self._reference[feature_columns]

        # Compute reference statistics
        self._compute_reference_stats()

        return self

    def _compute_reference_stats(self) -> None:
        """Compute statistics of reference data."""
        if self._reference is None:
            return

        self._reference_stats = {}

        for col in self._reference.columns:
            if pd.api.types.is_numeric_dtype(self._reference[col]):
                self._reference_stats[col] = {
                    "mean": float(self._reference[col].mean()),
                    "std": float(self._reference[col].std()),
                    "min": float(self._reference[col].min()),
                    "max": float(self._reference[col].max()),
                    "median": float(self._reference[col].median()),
                    "type": "numerical",
                }
            else:
                value_counts = self._reference[col].value_counts(normalize=True)
                self._reference_stats[col] = {
                    "n_unique": int(self._reference[col].nunique()),
                    "value_probs": value_counts.to_dict(),
                    "type": "categorical",
                }

    def validate(
        self,
        synthetic_data: pd.DataFrame,
        check_memorization: bool = True,
        check_mode_collapse: bool = True,
        check_distribution: bool = True,
        check_diversity: bool = True,
        check_correlations: bool = True,
    ) -> SyntheticValidationReport:
        """Validate synthetic data quality.

        Args:
            synthetic_data: Synthetic data to validate
            check_memorization: Check for memorized samples
            check_mode_collapse: Check for mode collapse
            check_distribution: Check distribution match
            check_diversity: Check internal diversity
            check_correlations: Check feature correlations

        Returns:
            SyntheticValidationReport with validation results
        """
        if self._reference is None:
            raise RuntimeError("Reference data not set. Call set_reference() first.")

        issues: list[SyntheticIssue] = []
        feature_scores: dict[str, float] = {}

        # Align columns
        common_cols = list(set(self._reference.columns) & set(synthetic_data.columns))
        if not common_cols:
            raise ValueError("No common columns between real and synthetic data")

        real = self._reference[common_cols]
        synth = synthetic_data[common_cols]

        # Check memorization
        privacy_score = 100.0
        if check_memorization:
            mem_issues, mem_score = self._check_memorization(real, synth)
            issues.extend(mem_issues)
            privacy_score = mem_score

        # Check mode collapse
        if check_mode_collapse:
            mc_issues = self._check_mode_collapse(synth)
            issues.extend(mc_issues)

        # Check distribution match
        fidelity_score = 100.0
        if check_distribution:
            dist_issues, dist_scores, fid_score = self._check_distribution(real, synth)
            issues.extend(dist_issues)
            feature_scores.update(dist_scores)
            fidelity_score = fid_score

        # Check diversity
        diversity_score = 100.0
        if check_diversity:
            div_issues, div_score = self._check_diversity(synth)
            issues.extend(div_issues)
            diversity_score = div_score

        # Check correlations
        if check_correlations:
            corr_issues = self._check_correlations(real, synth)
            issues.extend(corr_issues)

        # Calculate overall quality score
        quality_score = (
            0.4 * fidelity_score +
            0.3 * diversity_score +
            0.3 * privacy_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        return SyntheticValidationReport(
            n_real_samples=len(real),
            n_synthetic_samples=len(synth),
            quality_score=quality_score,
            fidelity_score=fidelity_score,
            diversity_score=diversity_score,
            privacy_score=privacy_score,
            issues=issues,
            feature_scores=feature_scores,
            recommendations=recommendations,
        )

    def _check_memorization(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> tuple[list[SyntheticIssue], float]:
        """Check for memorized samples (privacy risk)."""
        issues = []

        # Get numerical columns only
        num_cols = real.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return issues, 100.0

        real_num = real[num_cols].dropna()
        synth_num = synth[num_cols].dropna()

        if len(real_num) == 0 or len(synth_num) == 0:
            return issues, 100.0

        # Normalize data
        real_mean = real_num.mean()
        real_std = real_num.std().replace(0, 1)  # Avoid division by zero

        real_norm = (real_num - real_mean) / real_std
        synth_norm = (synth_num - real_mean) / real_std

        # Sample for efficiency
        sample_size = min(1000, len(synth_norm))
        synth_sample = synth_norm.sample(n=sample_size, random_state=42)

        # Find nearest neighbors
        try:
            distances = cdist(
                synth_sample.values, real_norm.values[:min(5000, len(real_norm))], metric='euclidean'
            )
            min_distances = distances.min(axis=1)

            # Count very similar samples (potential memorization)
            n_memorized = np.sum(min_distances < 0.1)  # Very close samples
            memorization_rate = n_memorized / len(synth_sample)

            if memorization_rate > 0.05:
                issues.append(SyntheticIssue(
                    issue_type=SyntheticIssueType.MEMORIZATION,
                    severity=IssueSeverity.HIGH if memorization_rate > 0.2 else IssueSeverity.MEDIUM,
                    description=f"{memorization_rate:.1%} of synthetic samples are very similar to real samples",
                    score=memorization_rate,
                    metadata={
                        "n_memorized": int(n_memorized),
                        "sample_size": sample_size,
                    },
                ))

            privacy_score = max(0, 100 - memorization_rate * 200)

        except Exception:
            privacy_score = 100.0

        return issues, privacy_score

    def _check_mode_collapse(
        self,
        synth: pd.DataFrame,
    ) -> list[SyntheticIssue]:
        """Check for mode collapse (lack of variety)."""
        issues = []

        for col in synth.columns:
            if pd.api.types.is_numeric_dtype(synth[col]):
                # Check for very low variance
                std = synth[col].std()
                if col in self._reference_stats:
                    ref_std = self._reference_stats[col].get("std", 1)
                    if ref_std > 0 and std / ref_std < self.mode_collapse_threshold:
                        issues.append(SyntheticIssue(
                            issue_type=SyntheticIssueType.MODE_COLLAPSE,
                            severity=IssueSeverity.HIGH,
                            description=f"Feature '{col}' has collapsed variance ({std:.4f} vs {ref_std:.4f})",
                            affected_features=[col],
                            score=std / ref_std if ref_std > 0 else 0,
                        ))
            else:
                # Check for collapsed categories
                n_unique = synth[col].nunique()
                if col in self._reference_stats:
                    ref_unique = self._reference_stats[col].get("n_unique", 1)
                    if ref_unique > 1 and n_unique / ref_unique < 0.5:
                        issues.append(SyntheticIssue(
                            issue_type=SyntheticIssueType.MODE_COLLAPSE,
                            severity=IssueSeverity.MEDIUM,
                            description=f"Feature '{col}' has fewer unique values ({n_unique} vs {ref_unique})",
                            affected_features=[col],
                        ))

        return issues

    def _check_distribution(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> tuple[list[SyntheticIssue], dict[str, float], float]:
        """Check distribution match between real and synthetic."""
        issues = []
        feature_scores = {}
        scores = []

        for col in real.columns:
            if pd.api.types.is_numeric_dtype(real[col]):
                # KS test for numerical
                try:
                    stat, p_value = stats.ks_2samp(
                        real[col].dropna(), synth[col].dropna()
                    )
                    feature_scores[col] = max(0, 100 - stat * 100)
                    scores.append(feature_scores[col])

                    if stat > self.distribution_threshold:
                        issues.append(SyntheticIssue(
                            issue_type=SyntheticIssueType.DISTRIBUTION_GAP,
                            severity=IssueSeverity.MEDIUM if stat < 0.2 else IssueSeverity.HIGH,
                            description=f"Distribution mismatch for '{col}' (KS stat={stat:.3f})",
                            affected_features=[col],
                            score=stat,
                            metadata={"p_value": p_value},
                        ))
                except Exception:
                    feature_scores[col] = 50.0
                    scores.append(50.0)
            else:
                # Chi-squared for categorical
                try:
                    real_counts = real[col].value_counts()
                    synth_counts = synth[col].value_counts()

                    # Align categories
                    all_cats = set(real_counts.index) | set(synth_counts.index)
                    real_freq = [real_counts.get(c, 0) for c in all_cats]
                    synth_freq = [synth_counts.get(c, 0) for c in all_cats]

                    # Normalize
                    real_pct = np.array(real_freq) / max(sum(real_freq), 1)
                    synth_pct = np.array(synth_freq) / max(sum(synth_freq), 1)

                    # Total variation distance
                    tvd = 0.5 * np.sum(np.abs(real_pct - synth_pct))
                    feature_scores[col] = max(0, 100 - tvd * 100)
                    scores.append(feature_scores[col])

                    if tvd > self.distribution_threshold:
                        issues.append(SyntheticIssue(
                            issue_type=SyntheticIssueType.DISTRIBUTION_GAP,
                            severity=IssueSeverity.MEDIUM,
                            description=f"Category distribution mismatch for '{col}' (TVD={tvd:.3f})",
                            affected_features=[col],
                            score=tvd,
                        ))
                except Exception:
                    feature_scores[col] = 50.0
                    scores.append(50.0)

        fidelity_score = np.mean(scores) if scores else 50.0
        return issues, feature_scores, float(fidelity_score)

    def _check_diversity(
        self,
        synth: pd.DataFrame,
    ) -> tuple[list[SyntheticIssue], float]:
        """Check internal diversity of synthetic data."""
        issues = []

        # Get numerical columns
        num_cols = synth.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return issues, 100.0

        synth_num = synth[num_cols].dropna()
        if len(synth_num) < 10:
            return issues, 50.0

        # Normalize
        synth_norm = (synth_num - synth_num.mean()) / synth_num.std().replace(0, 1)

        # Sample pairwise distances
        sample_size = min(500, len(synth_norm))
        sample = synth_norm.sample(n=sample_size, random_state=42)

        try:
            distances = cdist(sample.values, sample.values, metric='euclidean')
            np.fill_diagonal(distances, np.inf)  # Exclude self-distance

            mean_nn_dist = distances.min(axis=1).mean()
            diversity_score = min(100, mean_nn_dist * 50)  # Scale to 0-100

            if diversity_score < 30:
                issues.append(SyntheticIssue(
                    issue_type=SyntheticIssueType.LOW_DIVERSITY,
                    severity=IssueSeverity.HIGH,
                    description="Synthetic samples lack diversity (too similar to each other)",
                    score=diversity_score / 100,
                    metadata={"mean_nn_distance": mean_nn_dist},
                ))

        except Exception:
            diversity_score = 50.0

        return issues, float(diversity_score)

    def _check_correlations(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> list[SyntheticIssue]:
        """Check if correlations between features are preserved."""
        issues = []

        # Get numerical columns
        num_cols = real.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return issues

        try:
            real_corr = real[num_cols].corr()
            synth_corr = synth[num_cols].corr()

            # Compare correlation matrices
            corr_diff = np.abs(real_corr.values - synth_corr.values)
            max_diff = np.nanmax(corr_diff)
            mean_diff = np.nanmean(corr_diff)

            if mean_diff > 0.2:
                issues.append(SyntheticIssue(
                    issue_type=SyntheticIssueType.FEATURE_CORRELATION,
                    severity=IssueSeverity.MEDIUM,
                    description=f"Feature correlations not well preserved (mean diff={mean_diff:.3f})",
                    score=mean_diff,
                    metadata={
                        "max_diff": float(max_diff),
                        "mean_diff": float(mean_diff),
                    },
                ))

        except Exception:
            logger.debug("Distribution gap detection failed", exc_info=True)

        return issues

    def _generate_recommendations(
        self,
        issues: list[SyntheticIssue],
    ) -> list[str]:
        """Generate recommendations based on detected issues."""
        recommendations = []

        issue_types = {i.issue_type for i in issues}

        if SyntheticIssueType.MEMORIZATION in issue_types:
            recommendations.append(
                "Add differential privacy or reduce training epochs to prevent memorization"
            )

        if SyntheticIssueType.MODE_COLLAPSE in issue_types:
            recommendations.append(
                "Increase model diversity (higher temperature, different seeds, or ensemble)"
            )

        if SyntheticIssueType.DISTRIBUTION_GAP in issue_types:
            recommendations.append(
                "Review training data coverage and consider longer training"
            )

        if SyntheticIssueType.LOW_DIVERSITY in issue_types:
            recommendations.append(
                "Increase sampling temperature or add noise during generation"
            )

        if SyntheticIssueType.FEATURE_CORRELATION in issue_types:
            recommendations.append(
                "Consider models that better capture feature dependencies (e.g., copulas)"
            )

        if not recommendations:
            recommendations.append(
                "Synthetic data quality is good. Continue monitoring for drift."
            )

        return recommendations


def validate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    **kwargs: Any,
) -> SyntheticValidationReport:
    """Validate synthetic data against real reference data.

    Args:
        real_data: Real/reference data
        synthetic_data: Synthetic data to validate
        **kwargs: Additional arguments for SyntheticDataValidator

    Returns:
        SyntheticValidationReport with validation results
    """
    validator = SyntheticDataValidator(**kwargs)
    validator.set_reference(real_data)
    return validator.validate(synthetic_data)
