"""Quality scoring metrics."""

from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Weights for quality score components."""

    label_errors: float = 0.35
    duplicates: float = 0.20
    outliers: float = 0.15
    imbalance: float = 0.15
    bias: float = 0.15

    def normalize(self) -> "ScoringWeights":
        """Normalize weights to sum to 1.0."""
        total = (
            self.label_errors
            + self.duplicates
            + self.outliers
            + self.imbalance
            + self.bias
        )
        if total == 0:
            return self
        return ScoringWeights(
            label_errors=self.label_errors / total,
            duplicates=self.duplicates / total,
            outliers=self.outliers / total,
            imbalance=self.imbalance / total,
            bias=self.bias / total,
        )


def compute_label_quality_score(
    n_samples: int,
    n_errors: int,
    error_confidence_avg: float = 0.0,
) -> float:
    """Compute label quality score.

    Args:
        n_samples: Total number of samples
        n_errors: Number of detected label errors
        error_confidence_avg: Average confidence of errors

    Returns:
        Score from 0 to 100
    """
    if n_samples == 0:
        return 100.0

    error_rate = n_errors / n_samples

    # Base penalty from error rate
    if error_rate == 0:
        return 100.0

    # Penalize more heavily at higher error rates
    if error_rate < 0.01:
        base_score = 95.0
    elif error_rate < 0.03:
        base_score = 85.0
    elif error_rate < 0.05:
        base_score = 75.0
    elif error_rate < 0.10:
        base_score = 60.0
    elif error_rate < 0.20:
        base_score = 40.0
    else:
        base_score = 20.0

    # Additional penalty based on error confidence
    confidence_penalty = error_confidence_avg * 10

    return max(0.0, base_score - confidence_penalty)


def compute_duplicate_quality_score(
    n_samples: int,
    n_duplicate_pairs: int,
    n_exact: int = 0,
) -> float:
    """Compute duplicate quality score.

    Args:
        n_samples: Total number of samples
        n_duplicate_pairs: Number of duplicate pairs found
        n_exact: Number of exact duplicates

    Returns:
        Score from 0 to 100
    """
    if n_samples == 0:
        return 100.0

    if n_duplicate_pairs == 0:
        return 100.0

    # Estimate duplicate ratio (pairs to affected samples)
    # Rough estimate: each pair affects ~1.5 samples on average
    affected_estimate = min(n_duplicate_pairs * 1.5, n_samples)
    dup_rate = affected_estimate / n_samples

    # Exact duplicates are more severe
    exact_rate = n_exact * 2 / n_samples if n_samples > 0 else 0

    combined_rate = dup_rate + exact_rate

    if combined_rate < 0.01:
        return 98.0
    elif combined_rate < 0.05:
        return 90.0
    elif combined_rate < 0.10:
        return 80.0
    elif combined_rate < 0.20:
        return 65.0
    else:
        return 50.0


def compute_outlier_quality_score(
    n_samples: int,
    n_outliers: int,
    expected_contamination: float = 0.1,
) -> float:
    """Compute outlier quality score.

    Args:
        n_samples: Total number of samples
        n_outliers: Number of detected outliers
        expected_contamination: Expected outlier rate

    Returns:
        Score from 0 to 100
    """
    if n_samples == 0:
        return 100.0

    if n_outliers == 0:
        return 100.0

    outlier_rate = n_outliers / n_samples

    # Compare to expected contamination
    # If within expected range, not too bad
    if outlier_rate <= expected_contamination:
        return max(85.0, 100.0 - outlier_rate * 100)

    # Exceeding expected is concerning
    excess = outlier_rate - expected_contamination
    if excess < 0.05:
        return 75.0
    elif excess < 0.10:
        return 60.0
    elif excess < 0.20:
        return 45.0
    else:
        return 30.0


def compute_imbalance_quality_score(
    imbalance_ratio: float,
    n_classes: int,
) -> float:
    """Compute class imbalance quality score.

    Args:
        imbalance_ratio: Ratio of majority to minority class
        n_classes: Number of classes

    Returns:
        Score from 0 to 100
    """
    if imbalance_ratio <= 1.5:
        return 100.0
    elif imbalance_ratio <= 3.0:
        return 90.0
    elif imbalance_ratio <= 5.0:
        return 80.0
    elif imbalance_ratio <= 10.0:
        return 65.0
    elif imbalance_ratio <= 20.0:
        return 50.0
    elif imbalance_ratio <= 50.0:
        return 35.0
    else:
        return 20.0


def compute_bias_quality_score(
    n_bias_issues: int,
    max_demographic_parity_diff: float,
    has_correlation_issues: bool,
) -> float:
    """Compute bias quality score.

    Args:
        n_bias_issues: Number of bias issues detected
        max_demographic_parity_diff: Maximum demographic parity difference
        has_correlation_issues: Whether correlation issues were found

    Returns:
        Score from 0 to 100
    """
    if n_bias_issues == 0:
        return 100.0

    base_score = 100.0

    # Penalty for demographic parity issues
    if max_demographic_parity_diff > 0.3:
        base_score -= 30
    elif max_demographic_parity_diff > 0.2:
        base_score -= 20
    elif max_demographic_parity_diff > 0.1:
        base_score -= 10

    # Penalty for correlation issues
    if has_correlation_issues:
        base_score -= 15

    # Penalty per issue
    base_score -= min(n_bias_issues * 5, 30)

    return max(20.0, base_score)


def severity_from_score(score: float) -> str:
    """Get severity level from score.

    Args:
        score: Quality score (0-100)

    Returns:
        Severity string
    """
    if score >= 90:
        return "excellent"
    elif score >= 75:
        return "good"
    elif score >= 60:
        return "moderate"
    elif score >= 40:
        return "poor"
    else:
        return "critical"
