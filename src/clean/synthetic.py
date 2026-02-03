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


# =============================================================================
# Certified Synthetic Data Generation
# =============================================================================


class CertificationStatus(Enum):
    """Status of quality certification."""

    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL = "conditional"


class SynthesizerType(Enum):
    """Types of synthetic data generators."""

    GAUSSIAN_COPULA = "gaussian_copula"
    BOOTSTRAP = "bootstrap"
    SMOTE = "smote"


@dataclass
class CertificationConfig:
    """Configuration for synthetic data certification."""

    min_quality_score: float = 80.0
    max_duplicates: float = 0.01
    max_outliers: float = 0.05
    max_missing: float = 0.0
    min_diversity: float = 0.9
    max_ks_statistic: float = 0.1
    max_correlation_diff: float = 0.15
    min_distance_to_original: float = 0.1
    max_retries: int = 5
    random_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_quality_score": self.min_quality_score,
            "max_duplicates": self.max_duplicates,
            "max_outliers": self.max_outliers,
            "max_missing": self.max_missing,
            "min_diversity": self.min_diversity,
            "max_ks_statistic": self.max_ks_statistic,
            "max_correlation_diff": self.max_correlation_diff,
            "min_distance_to_original": self.min_distance_to_original,
            "max_retries": self.max_retries,
        }


@dataclass
class QualityMetric:
    """A single quality metric with pass/fail status."""

    name: str
    value: float
    threshold: float
    passed: bool
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "description": self.description,
        }


@dataclass
class QualityCertificate:
    """Certificate proving synthetic data quality."""

    certificate_id: str
    timestamp: str
    status: CertificationStatus
    quality_score: float
    n_samples: int
    metrics: list[QualityMetric]
    config: CertificationConfig
    synthesizer: str
    reference_hash: str
    synthetic_hash: str
    generation_time_seconds: float
    retries_needed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_certified(self) -> bool:
        """Check if data is certified."""
        return self.status == CertificationStatus.PASSED

    def summary(self) -> str:
        """Get certificate summary."""
        lines = [
            "=" * 60,
            "SYNTHETIC DATA QUALITY CERTIFICATE",
            "=" * 60,
            f"Certificate ID: {self.certificate_id}",
            f"Timestamp: {self.timestamp}",
            f"Status: {self.status.value.upper()}",
            f"Quality Score: {self.quality_score:.1f}/100",
            f"Samples Generated: {self.n_samples:,}",
            f"Generation Time: {self.generation_time_seconds:.2f}s",
            f"Synthesizer: {self.synthesizer}",
            "-" * 60,
            "METRICS:",
        ]
        for metric in self.metrics:
            status = "✓" if metric.passed else "✗"
            lines.append(
                f"  {status} {metric.name}: {metric.value:.4f} "
                f"(threshold: {metric.threshold:.4f})"
            )
        if self.warnings:
            lines.append("-" * 60)
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "quality_score": self.quality_score,
            "n_samples": self.n_samples,
            "metrics": [m.to_dict() for m in self.metrics],
            "config": self.config.to_dict(),
            "synthesizer": self.synthesizer,
            "reference_hash": self.reference_hash,
            "synthetic_hash": self.synthetic_hash,
            "generation_time_seconds": self.generation_time_seconds,
            "retries_needed": self.retries_needed,
            "warnings": self.warnings,
        }


@dataclass
class GenerationResult:
    """Result of certified synthetic data generation."""

    data: pd.DataFrame
    certificate: QualityCertificate
    original_n_samples: int


class GaussianCopulaSynthesizer:
    """Gaussian Copula-based synthetic data generator."""

    def __init__(self, random_seed: int | None = None):
        self.random_seed = random_seed
        self._fitted = False
        self._columns: list[str] = []
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._corr_matrix: np.ndarray | None = None
        self._category_maps: dict[str, dict[int, Any]] = {}
        self._category_probs: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "gaussian_copula"

    def fit(self, data: pd.DataFrame) -> None:
        """Fit to reference data."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self._columns = list(data.columns)
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = data.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if self._numeric_cols:
            numeric_data = data[self._numeric_cols].fillna(
                data[self._numeric_cols].mean()
            )
            self._means = numeric_data.mean().values
            self._stds = numeric_data.std().values + 1e-8
            standardized = (numeric_data - self._means) / self._stds
            self._corr_matrix = standardized.corr().values
            min_eig = np.min(np.linalg.eigvalsh(self._corr_matrix))
            if min_eig < 0:
                self._corr_matrix -= 1.1 * min_eig * np.eye(len(self._numeric_cols))

        for col in self._categorical_cols:
            value_counts = data[col].value_counts(normalize=True)
            self._category_maps[col] = {i: v for i, v in enumerate(value_counts.index)}
            self._category_probs[col] = value_counts.values

        self._fitted = True

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples."""
        if not self._fitted:
            raise RuntimeError("Synthesizer must be fitted first")

        result = pd.DataFrame()

        if self._numeric_cols and self._corr_matrix is not None:
            try:
                L = np.linalg.cholesky(self._corr_matrix)
                z = np.random.randn(n_samples, len(self._numeric_cols))
                correlated = z @ L.T
                synthetic_numeric = correlated * self._stds + self._means
                for i, col in enumerate(self._numeric_cols):
                    result[col] = synthetic_numeric[:, i]
            except np.linalg.LinAlgError:
                for i, col in enumerate(self._numeric_cols):
                    result[col] = (
                        np.random.randn(n_samples) * self._stds[i] + self._means[i]
                    )

        for col in self._categorical_cols:
            probs = self._category_probs[col]
            indices = np.random.choice(len(probs), size=n_samples, p=probs)
            result[col] = [self._category_maps[col][i] for i in indices]

        return result[self._columns]


class CertifiedDataGenerator:
    """Generate certified synthetic data with quality guarantees."""

    def __init__(
        self,
        config: CertificationConfig | None = None,
        synthesizer_type: SynthesizerType = SynthesizerType.GAUSSIAN_COPULA,
    ):
        self.config = config or CertificationConfig()
        self.synthesizer_type = synthesizer_type

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of dataframe."""
        import hashlib

        data_str = data.to_csv(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _validate(
        self, synthetic: pd.DataFrame, reference: pd.DataFrame
    ) -> tuple[list[QualityMetric], list[str]]:
        """Validate synthetic data quality."""
        metrics = []
        warnings = []

        # Duplicate rate
        n_dups = synthetic.duplicated().sum()
        dup_rate = n_dups / len(synthetic) if len(synthetic) > 0 else 0
        metrics.append(
            QualityMetric(
                name="duplicate_rate",
                value=dup_rate,
                threshold=self.config.max_duplicates,
                passed=dup_rate <= self.config.max_duplicates,
                description="Rate of duplicate rows",
            )
        )

        # Missing rate
        missing_rate = synthetic.isnull().sum().sum() / synthetic.size
        metrics.append(
            QualityMetric(
                name="missing_rate",
                value=missing_rate,
                threshold=self.config.max_missing,
                passed=missing_rate <= self.config.max_missing,
                description="Rate of missing values",
            )
        )

        # KS statistic
        numeric_cols = reference.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ks_stats = []
            for col in numeric_cols:
                ref_vals = reference[col].dropna()
                syn_vals = synthetic[col].dropna()
                if len(ref_vals) > 0 and len(syn_vals) > 0:
                    ks_stat, _ = stats.ks_2samp(ref_vals, syn_vals)
                    ks_stats.append(ks_stat)
            avg_ks = np.mean(ks_stats) if ks_stats else 0
            metrics.append(
                QualityMetric(
                    name="ks_statistic",
                    value=avg_ks,
                    threshold=self.config.max_ks_statistic,
                    passed=avg_ks <= self.config.max_ks_statistic,
                    description="KS distance from reference",
                )
            )

        # Correlation preservation
        if len(numeric_cols) >= 2:
            ref_corr = reference[numeric_cols].corr()
            syn_corr = synthetic[numeric_cols].corr()
            corr_diff = np.abs(ref_corr - syn_corr).mean().mean()
            metrics.append(
                QualityMetric(
                    name="correlation_diff",
                    value=corr_diff,
                    threshold=self.config.max_correlation_diff,
                    passed=corr_diff <= self.config.max_correlation_diff,
                    description="Correlation matrix difference",
                )
            )

        # Diversity
        if len(numeric_cols) > 0:
            ref_std = reference[numeric_cols].std().mean()
            syn_std = synthetic[numeric_cols].std().mean()
            diversity = syn_std / ref_std if ref_std > 0 else 1.0
            metrics.append(
                QualityMetric(
                    name="diversity",
                    value=diversity,
                    threshold=self.config.min_diversity,
                    passed=diversity >= self.config.min_diversity,
                    description="Diversity relative to reference",
                )
            )

        return metrics, warnings

    def _calculate_score(self, metrics: list[QualityMetric]) -> float:
        """Calculate overall quality score."""
        if not metrics:
            return 0.0

        weights = {
            "duplicate_rate": 20,
            "missing_rate": 15,
            "ks_statistic": 25,
            "correlation_diff": 20,
            "diversity": 20,
        }

        total_weight = sum(weights.get(m.name, 10) for m in metrics)
        weighted_score = 0

        for metric in metrics:
            weight = weights.get(metric.name, 10)
            if metric.passed:
                weighted_score += weight * 100
            else:
                ratio = max(0, 1 - abs(metric.value - metric.threshold) / max(metric.threshold, 0.01))
                weighted_score += weight * ratio * 100

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def generate(
        self,
        reference_data: pd.DataFrame,
        n_samples: int | None = None,
    ) -> GenerationResult:
        """Generate certified synthetic data.

        Args:
            reference_data: Reference dataset to learn from
            n_samples: Number of samples to generate

        Returns:
            GenerationResult with certified synthetic data
        """
        import secrets
        import time
        from datetime import datetime

        start_time = time.time()
        n_samples = n_samples or len(reference_data)

        logger.info("Generating %d certified synthetic samples", n_samples)

        synthesizer = GaussianCopulaSynthesizer(random_seed=self.config.random_seed)
        synthesizer.fit(reference_data)

        best_result = None
        best_score = -1
        retries = 0

        for attempt in range(self.config.max_retries):
            synthetic_data = synthesizer.sample(n_samples)
            metrics, warnings = self._validate(synthetic_data, reference_data)
            score = self._calculate_score(metrics)

            logger.info("Attempt %d: Quality score %.1f", attempt + 1, score)

            if score > best_score:
                best_score = score
                best_result = (synthetic_data, metrics, warnings)
                retries = attempt

            if score >= self.config.min_quality_score:
                if all(m.passed for m in metrics):
                    break

        synthetic_data, metrics, warnings = best_result  # type: ignore

        all_passed = all(m.passed for m in metrics)
        if best_score >= self.config.min_quality_score and all_passed:
            status = CertificationStatus.PASSED
        elif best_score >= self.config.min_quality_score * 0.9:
            status = CertificationStatus.CONDITIONAL
        else:
            status = CertificationStatus.FAILED

        certificate = QualityCertificate(
            certificate_id=secrets.token_hex(8),
            timestamp=datetime.now().isoformat(),
            status=status,
            quality_score=best_score,
            n_samples=n_samples,
            metrics=metrics,
            config=self.config,
            synthesizer=synthesizer.name,
            reference_hash=self._compute_hash(reference_data),
            synthetic_hash=self._compute_hash(synthetic_data),
            generation_time_seconds=time.time() - start_time,
            retries_needed=retries,
            warnings=warnings,
        )

        return GenerationResult(
            data=synthetic_data,
            certificate=certificate,
            original_n_samples=len(reference_data),
        )


def generate_certified_data(
    reference_data: pd.DataFrame,
    n_samples: int | None = None,
    min_quality_score: float = 80.0,
) -> GenerationResult:
    """Generate certified synthetic data with quality guarantees.

    Args:
        reference_data: Reference dataset to learn from
        n_samples: Number of samples to generate
        min_quality_score: Minimum required quality score (0-100)

    Returns:
        GenerationResult with certified data and quality certificate

    Example:
        >>> result = generate_certified_data(df, n_samples=10000)
        >>> print(result.certificate.summary())
        >>> print(f"Certified: {result.certificate.is_certified}")
        >>> synthetic_df = result.data
    """
    config = CertificationConfig(min_quality_score=min_quality_score)
    generator = CertifiedDataGenerator(config=config)
    return generator.generate(reference_data, n_samples)
