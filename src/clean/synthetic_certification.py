"""Synthetic Data Quality Certification.

This module provides end-to-end certification for synthetic data,
validating quality, privacy, and utility with audit trails.

Example:
    >>> from clean.synthetic_certification import SyntheticCertifier
    >>>
    >>> certifier = SyntheticCertifier()
    >>> certificate = certifier.certify(real_data, synthetic_data)
    >>> if certificate.is_certified:
    ...     print(f"Certified! ID: {certificate.certificate_id}")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CertificationStatus(Enum):
    """Status of certification."""

    CERTIFIED = "certified"
    CONDITIONALLY_CERTIFIED = "conditionally_certified"
    FAILED = "failed"
    PENDING = "pending"


class QualityDimension(Enum):
    """Dimensions of synthetic data quality."""

    FIDELITY = "fidelity"
    PRIVACY = "privacy"
    UTILITY = "utility"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"


class PrivacyRisk(Enum):
    """Privacy risk levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DimensionScore:
    """Score for a quality dimension."""

    dimension: QualityDimension
    score: float  # 0-100
    passed: bool
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
            "warnings": self.warnings,
        }


@dataclass
class PrivacyAssessment:
    """Privacy assessment result."""

    risk_level: PrivacyRisk
    k_anonymity: int | None
    l_diversity: float | None
    memorization_rate: float
    nearest_neighbor_distance: float
    reidentification_risk: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "k_anonymity": self.k_anonymity,
            "l_diversity": self.l_diversity,
            "memorization_rate": self.memorization_rate,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
            "reidentification_risk": self.reidentification_risk,
            "passed": self.passed,
        }


@dataclass
class QualityCertificate:
    """Certificate for synthetic data quality."""

    certificate_id: str
    issued_at: datetime
    expires_at: datetime | None

    status: CertificationStatus
    is_certified: bool

    # Scores
    overall_score: float
    dimension_scores: list[DimensionScore]
    privacy_assessment: PrivacyAssessment

    # Metadata
    real_data_hash: str
    synthetic_data_hash: str
    n_real_samples: int
    n_synthetic_samples: int

    # Audit trail
    audit_log: list[dict[str, Any]]

    # Conditions/warnings
    conditions: list[str]
    warnings: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status_icon = {
            CertificationStatus.CERTIFIED: "✅",
            CertificationStatus.CONDITIONALLY_CERTIFIED: "⚠️",
            CertificationStatus.FAILED: "❌",
            CertificationStatus.PENDING: "⏳",
        }

        lines = [
            "Synthetic Data Quality Certificate",
            "=" * 50,
            "",
            f"Certificate ID: {self.certificate_id}",
            f"Status: {status_icon[self.status]} {self.status.value.upper()}",
            f"Issued: {self.issued_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Overall Score: {self.overall_score:.1f}/100",
            "",
            "Dimension Scores:",
        ]

        for score in self.dimension_scores:
            icon = "✓" if score.passed else "✗"
            lines.append(
                f"  {icon} {score.dimension.value}: {score.score:.1f}/100 "
                f"(threshold: {score.threshold})"
            )

        lines.append("")
        lines.append(f"Privacy Risk: {self.privacy_assessment.risk_level.value}")
        lines.append(f"Memorization Rate: {self.privacy_assessment.memorization_rate:.2%}")

        if self.conditions:
            lines.append("")
            lines.append("Conditions:")
            for cond in self.conditions:
                lines.append(f"  ⚠️ {cond}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  ⚡ {warn}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "is_certified": self.is_certified,
            "overall_score": self.overall_score,
            "dimension_scores": [s.to_dict() for s in self.dimension_scores],
            "privacy_assessment": self.privacy_assessment.to_dict(),
            "real_data_hash": self.real_data_hash,
            "synthetic_data_hash": self.synthetic_data_hash,
            "n_real_samples": self.n_real_samples,
            "n_synthetic_samples": self.n_synthetic_samples,
            "conditions": self.conditions,
            "warnings": self.warnings,
        }

    def export_json(self, path: str | Path) -> None:
        """Export certificate to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def export_pdf(self, path: str | Path) -> None:
        """Export certificate to PDF (requires reportlab)."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

            doc = SimpleDocTemplate(str(path), pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            # Title
            elements.append(Paragraph(
                "Synthetic Data Quality Certificate",
                styles["Heading1"],
            ))
            elements.append(Spacer(1, 20))

            # Status
            status_text = f"Status: {self.status.value.upper()}"
            elements.append(Paragraph(status_text, styles["Heading2"]))
            elements.append(Spacer(1, 10))

            # Details
            details = [
                ["Certificate ID", self.certificate_id],
                ["Issued", self.issued_at.strftime("%Y-%m-%d %H:%M:%S")],
                ["Overall Score", f"{self.overall_score:.1f}/100"],
                ["Privacy Risk", self.privacy_assessment.risk_level.value],
            ]

            table = Table(details)
            elements.append(table)

            doc.build(elements)
            logger.info(f"Certificate exported to {path}")

        except ImportError:
            logger.warning("reportlab not available, saving as text instead")
            with open(str(path).replace(".pdf", ".txt"), "w") as f:
                f.write(self.summary())


@dataclass
class CertificationConfig:
    """Configuration for synthetic data certification."""

    # Thresholds (0-100)
    fidelity_threshold: float = 70.0
    privacy_threshold: float = 80.0
    utility_threshold: float = 70.0
    diversity_threshold: float = 60.0
    coherence_threshold: float = 70.0

    # Privacy settings
    max_memorization_rate: float = 0.01  # 1%
    min_nearest_neighbor_distance: float = 0.1
    max_reidentification_risk: float = 0.05

    # Weights for overall score
    fidelity_weight: float = 0.25
    privacy_weight: float = 0.30
    utility_weight: float = 0.25
    diversity_weight: float = 0.10
    coherence_weight: float = 0.10

    # Certificate validity
    certificate_validity_days: int = 90

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fidelity_threshold": self.fidelity_threshold,
            "privacy_threshold": self.privacy_threshold,
            "utility_threshold": self.utility_threshold,
            "diversity_threshold": self.diversity_threshold,
            "coherence_threshold": self.coherence_threshold,
            "max_memorization_rate": self.max_memorization_rate,
        }


class FidelityEvaluator:
    """Evaluate fidelity of synthetic data to real data."""

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> DimensionScore:
        """Evaluate fidelity.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data

        Returns:
            DimensionScore for fidelity
        """
        details = {}
        warnings = []

        # Column-wise distribution similarity
        column_scores = []
        for col in real_data.columns:
            if col not in synthetic_data.columns:
                warnings.append(f"Column '{col}' missing from synthetic data")
                continue

            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()

            if len(real_col) == 0 or len(synth_col) == 0:
                continue

            if real_data[col].dtype in [np.float64, np.int64, float, int]:
                # Numerical: KS test
                stat, p_value = stats.ks_2samp(real_col, synth_col)
                column_scores.append(1 - stat)
                details[f"{col}_ks_stat"] = float(stat)
            else:
                # Categorical: Chi-squared or overlap
                real_dist = real_col.value_counts(normalize=True)
                synth_dist = synth_col.value_counts(normalize=True)

                all_vals = set(real_dist.index) | set(synth_dist.index)
                overlap = sum(
                    min(real_dist.get(v, 0), synth_dist.get(v, 0))
                    for v in all_vals
                )
                column_scores.append(overlap)
                details[f"{col}_overlap"] = float(overlap)

        # Overall fidelity score
        if column_scores:
            fidelity_score = np.mean(column_scores) * 100
        else:
            fidelity_score = 0.0
            warnings.append("Could not compute column-wise fidelity")

        # Correlation preservation
        real_numeric = real_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])

        if len(real_numeric.columns) > 1:
            real_corr = real_numeric.corr().values
            synth_corr = synth_numeric[real_numeric.columns].corr().values

            corr_diff = np.abs(real_corr - synth_corr).mean()
            details["correlation_mae"] = float(corr_diff)

            # Adjust score based on correlation preservation
            corr_score = max(0, 1 - corr_diff) * 100
            fidelity_score = (fidelity_score + corr_score) / 2

        return DimensionScore(
            dimension=QualityDimension.FIDELITY,
            score=float(fidelity_score),
            passed=fidelity_score >= 70,
            threshold=70.0,
            details=details,
            warnings=warnings,
        )


class PrivacyEvaluator:
    """Evaluate privacy of synthetic data."""

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        config: CertificationConfig,
    ) -> PrivacyAssessment:
        """Evaluate privacy.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            config: Certification configuration

        Returns:
            PrivacyAssessment
        """
        from sklearn.neighbors import NearestNeighbors

        # Prepare numeric data
        real_numeric = real_data.select_dtypes(include=[np.number]).values
        synth_numeric = synthetic_data.select_dtypes(include=[np.number]).values

        # Memorization check using nearest neighbor distance
        if len(real_numeric) > 0 and len(synth_numeric) > 0:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(real_numeric)

            distances, _ = nn.kneighbors(synth_numeric)
            min_distances = distances.flatten()

            # Memorization: samples very close to real data
            memorization_threshold = np.std(real_numeric) * 0.01
            memorization_rate = (min_distances < memorization_threshold).mean()

            # Average nearest neighbor distance
            avg_nn_distance = float(min_distances.mean())
        else:
            memorization_rate = 0.0
            avg_nn_distance = 1.0

        # Re-identification risk (simplified)
        # Higher risk if synthetic records are too close to real records
        reidentification_risk = min(1.0, memorization_rate * 10)

        # Determine risk level
        if memorization_rate > config.max_memorization_rate:
            risk_level = PrivacyRisk.HIGH
        elif reidentification_risk > config.max_reidentification_risk:
            risk_level = PrivacyRisk.MEDIUM
        elif memorization_rate > 0:
            risk_level = PrivacyRisk.LOW
        else:
            risk_level = PrivacyRisk.NONE

        # K-anonymity estimation (simplified)
        k_anonymity = self._estimate_k_anonymity(synthetic_data)

        # L-diversity estimation
        l_diversity = self._estimate_l_diversity(synthetic_data)

        passed = (
            risk_level in (PrivacyRisk.NONE, PrivacyRisk.LOW) and
            memorization_rate <= config.max_memorization_rate
        )

        return PrivacyAssessment(
            risk_level=risk_level,
            k_anonymity=k_anonymity,
            l_diversity=l_diversity,
            memorization_rate=float(memorization_rate),
            nearest_neighbor_distance=avg_nn_distance,
            reidentification_risk=float(reidentification_risk),
            passed=passed,
        )

    def _estimate_k_anonymity(self, data: pd.DataFrame) -> int:
        """Estimate k-anonymity of dataset."""
        # Group by all columns and find minimum group size
        try:
            group_sizes = data.groupby(list(data.columns)).size()
            return int(group_sizes.min())
        except Exception:
            return 1

    def _estimate_l_diversity(self, data: pd.DataFrame) -> float:
        """Estimate l-diversity."""
        # Simplified: average number of unique values per column
        diversities = []
        for col in data.columns:
            n_unique = data[col].nunique()
            n_total = len(data)
            diversities.append(n_unique / n_total if n_total > 0 else 0)

        return float(np.mean(diversities)) if diversities else 0.0


class UtilityEvaluator:
    """Evaluate utility of synthetic data for downstream tasks."""

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: str | None = None,
    ) -> DimensionScore:
        """Evaluate utility.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            target_column: Optional target column for ML utility

        Returns:
            DimensionScore for utility
        """
        details = {}
        warnings = []

        scores = []

        # Statistical utility: similar summary statistics
        for col in real_data.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_data.columns:
                continue

            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()

            if len(real_col) == 0 or len(synth_col) == 0:
                continue

            # Compare means
            mean_diff = abs(real_col.mean() - synth_col.mean()) / (real_col.std() + 1e-10)
            mean_score = max(0, 1 - mean_diff)

            # Compare std
            std_diff = abs(real_col.std() - synth_col.std()) / (real_col.std() + 1e-10)
            std_score = max(0, 1 - std_diff)

            col_score = (mean_score + std_score) / 2
            scores.append(col_score)
            details[f"{col}_utility"] = float(col_score)

        # ML utility if target column specified
        if target_column and target_column in real_data.columns:
            ml_score = self._evaluate_ml_utility(
                real_data, synthetic_data, target_column
            )
            if ml_score is not None:
                scores.append(ml_score)
                details["ml_utility"] = float(ml_score)

        utility_score = np.mean(scores) * 100 if scores else 50.0

        return DimensionScore(
            dimension=QualityDimension.UTILITY,
            score=float(utility_score),
            passed=utility_score >= 70,
            threshold=70.0,
            details=details,
            warnings=warnings,
        )

    def _evaluate_ml_utility(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: str,
    ) -> float | None:
        """Evaluate ML utility by training on synthetic, testing on real."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        try:
            # Prepare data
            feature_cols = [
                c for c in real_data.columns
                if c != target_column and real_data[c].dtype in [np.float64, np.int64]
            ]

            if not feature_cols:
                return None

            X_real = real_data[feature_cols].fillna(0).values
            y_real = real_data[target_column]

            X_synth = synthetic_data[feature_cols].fillna(0).values
            y_synth = synthetic_data[target_column]

            # Encode labels
            le = LabelEncoder()
            le.fit(pd.concat([y_real, y_synth]))
            y_real_enc = le.transform(y_real)
            y_synth_enc = le.transform(y_synth)

            # Train on synthetic, test on real
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_synth, y_synth_enc)
            synth_to_real = clf.score(X_real, y_real_enc)

            # Train on real, test on real (baseline)
            clf.fit(X_real, y_real_enc)
            real_to_real = clf.score(X_real, y_real_enc)

            # ML utility: how close is synth-trained to real-trained
            if real_to_real > 0:
                return synth_to_real / real_to_real
            return 0.5

        except Exception as e:
            logger.warning(f"ML utility evaluation failed: {e}")
            return None


class SyntheticCertifier:
    """Certify synthetic data quality with audit trail.

    Performs comprehensive validation of fidelity, privacy,
    and utility with certificate generation.
    """

    def __init__(
        self,
        config: CertificationConfig | None = None,
    ):
        """Initialize certifier.

        Args:
            config: Certification configuration
        """
        self.config = config or CertificationConfig()

        self.fidelity_eval = FidelityEvaluator()
        self.privacy_eval = PrivacyEvaluator()
        self.utility_eval = UtilityEvaluator()

        self._audit_log: list[dict[str, Any]] = []

    def certify(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: str | None = None,
    ) -> QualityCertificate:
        """Certify synthetic data quality.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            target_column: Optional target column for ML utility

        Returns:
            QualityCertificate
        """
        self._audit_log = []
        self._log("certification_started", {"timestamp": datetime.now().isoformat()})

        # Generate certificate ID
        cert_id = hashlib.sha256(
            f"{datetime.now().isoformat()}_{len(synthetic_data)}".encode()
        ).hexdigest()[:16]

        # Compute data hashes
        real_hash = hashlib.sha256(
            real_data.to_json().encode()
        ).hexdigest()[:16]
        synth_hash = hashlib.sha256(
            synthetic_data.to_json().encode()
        ).hexdigest()[:16]

        self._log("data_hashes_computed", {
            "real_hash": real_hash,
            "synthetic_hash": synth_hash,
        })

        # Evaluate all dimensions
        dimension_scores = []
        warnings = []
        conditions = []

        # 1. Fidelity
        fidelity = self.fidelity_eval.evaluate(real_data, synthetic_data)
        fidelity.threshold = self.config.fidelity_threshold
        fidelity.passed = fidelity.score >= self.config.fidelity_threshold
        dimension_scores.append(fidelity)
        warnings.extend(fidelity.warnings)
        self._log("fidelity_evaluated", {"score": fidelity.score})

        # 2. Privacy
        privacy = self.privacy_eval.evaluate(
            real_data, synthetic_data, self.config
        )
        self._log("privacy_evaluated", {
            "risk_level": privacy.risk_level.value,
            "memorization_rate": privacy.memorization_rate,
        })

        # Convert privacy to dimension score
        privacy_score = DimensionScore(
            dimension=QualityDimension.PRIVACY,
            score=self._privacy_to_score(privacy),
            passed=privacy.passed,
            threshold=self.config.privacy_threshold,
            details=privacy.to_dict(),
        )
        dimension_scores.append(privacy_score)

        # 3. Utility
        utility = self.utility_eval.evaluate(
            real_data, synthetic_data, target_column
        )
        utility.threshold = self.config.utility_threshold
        utility.passed = utility.score >= self.config.utility_threshold
        dimension_scores.append(utility)
        warnings.extend(utility.warnings)
        self._log("utility_evaluated", {"score": utility.score})

        # 4. Diversity
        diversity = self._evaluate_diversity(synthetic_data)
        dimension_scores.append(diversity)
        self._log("diversity_evaluated", {"score": diversity.score})

        # 5. Coherence
        coherence = self._evaluate_coherence(synthetic_data)
        dimension_scores.append(coherence)
        self._log("coherence_evaluated", {"score": coherence.score})

        # Calculate overall score
        overall_score = (
            fidelity.score * self.config.fidelity_weight +
            privacy_score.score * self.config.privacy_weight +
            utility.score * self.config.utility_weight +
            diversity.score * self.config.diversity_weight +
            coherence.score * self.config.coherence_weight
        )

        # Determine certification status
        all_passed = all(s.passed for s in dimension_scores) and privacy.passed

        if all_passed:
            status = CertificationStatus.CERTIFIED
        elif privacy.passed and overall_score >= 60:
            status = CertificationStatus.CONDITIONALLY_CERTIFIED
            conditions.append("Some quality thresholds not met - review recommended")
        else:
            status = CertificationStatus.FAILED
            if not privacy.passed:
                conditions.append("Privacy requirements not met")

        self._log("certification_completed", {
            "status": status.value,
            "overall_score": overall_score,
        })

        # Create certificate
        from datetime import timedelta

        return QualityCertificate(
            certificate_id=cert_id,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=self.config.certificate_validity_days),
            status=status,
            is_certified=status in (
                CertificationStatus.CERTIFIED,
                CertificationStatus.CONDITIONALLY_CERTIFIED,
            ),
            overall_score=float(overall_score),
            dimension_scores=dimension_scores,
            privacy_assessment=privacy,
            real_data_hash=real_hash,
            synthetic_data_hash=synth_hash,
            n_real_samples=len(real_data),
            n_synthetic_samples=len(synthetic_data),
            audit_log=self._audit_log.copy(),
            conditions=conditions,
            warnings=warnings,
        )

    def _privacy_to_score(self, privacy: PrivacyAssessment) -> float:
        """Convert privacy assessment to 0-100 score."""
        risk_scores = {
            PrivacyRisk.NONE: 100,
            PrivacyRisk.LOW: 85,
            PrivacyRisk.MEDIUM: 60,
            PrivacyRisk.HIGH: 30,
            PrivacyRisk.CRITICAL: 0,
        }

        base_score = risk_scores[privacy.risk_level]

        # Adjust based on memorization rate
        memorization_penalty = privacy.memorization_rate * 100
        base_score -= memorization_penalty

        return max(0, base_score)

    def _evaluate_diversity(self, data: pd.DataFrame) -> DimensionScore:
        """Evaluate diversity of synthetic data."""
        diversities = []

        for col in data.columns:
            n_unique = data[col].nunique()
            n_total = len(data)
            diversity = n_unique / n_total if n_total > 0 else 0
            diversities.append(diversity)

        diversity_score = np.mean(diversities) * 100 if diversities else 50

        return DimensionScore(
            dimension=QualityDimension.DIVERSITY,
            score=float(diversity_score),
            passed=diversity_score >= self.config.diversity_threshold,
            threshold=self.config.diversity_threshold,
            details={"mean_diversity": float(np.mean(diversities)) if diversities else 0},
        )

    def _evaluate_coherence(self, data: pd.DataFrame) -> DimensionScore:
        """Evaluate coherence of synthetic data."""
        # Check for impossible value combinations
        coherence_issues = 0
        total_checks = 0

        # Check numeric columns for reasonable ranges
        for col in data.select_dtypes(include=[np.number]).columns:
            total_checks += 1
            # Check for extreme outliers (more than 5 std from mean)
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:
                extreme = ((data[col] - mean).abs() > 5 * std).sum()
                if extreme > len(data) * 0.05:  # More than 5% extreme
                    coherence_issues += 1

        coherence_score = 100 - (coherence_issues / max(total_checks, 1)) * 100

        return DimensionScore(
            dimension=QualityDimension.COHERENCE,
            score=float(coherence_score),
            passed=coherence_score >= self.config.coherence_threshold,
            threshold=self.config.coherence_threshold,
            details={"coherence_issues": coherence_issues},
        )

    def _log(self, event: str, data: dict[str, Any]) -> None:
        """Add entry to audit log."""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data,
        })


def certify_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    config: CertificationConfig | None = None,
    **kwargs: Any,
) -> QualityCertificate:
    """Convenience function to certify synthetic data.

    Args:
        real_data: Original real data
        synthetic_data: Generated synthetic data
        config: Certification configuration
        **kwargs: Additional arguments

    Returns:
        QualityCertificate
    """
    certifier = SyntheticCertifier(config=config)
    return certifier.certify(real_data, synthetic_data, **kwargs)


def create_certifier(
    config: CertificationConfig | None = None,
) -> SyntheticCertifier:
    """Create a synthetic data certifier.

    Args:
        config: Certification configuration

    Returns:
        SyntheticCertifier
    """
    return SyntheticCertifier(config=config)
