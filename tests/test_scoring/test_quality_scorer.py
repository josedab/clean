"""Tests for quality scoring."""


from clean.core.types import DuplicatePair, LabelError, Outlier
from clean.detection.base import DetectorResult
from clean.scoring import QualityScorer, ScoringWeights
from clean.scoring.metrics import (
    compute_bias_quality_score,
    compute_duplicate_quality_score,
    compute_imbalance_quality_score,
    compute_label_quality_score,
    compute_outlier_quality_score,
    severity_from_score,
)


class TestScoringMetrics:
    """Tests for individual scoring metrics."""

    def test_label_quality_no_errors(self):
        """Test label quality with no errors."""
        score = compute_label_quality_score(100, 0)
        assert score == 100.0

    def test_label_quality_with_errors(self):
        """Test label quality with various error rates."""
        # Low error rate
        score_low = compute_label_quality_score(1000, 5)  # 0.5%
        assert score_low >= 90

        # Medium error rate
        score_med = compute_label_quality_score(1000, 50)  # 5%
        assert 60 <= score_med < 90

        # High error rate
        score_high = compute_label_quality_score(1000, 200)  # 20%
        assert score_high < 50

    def test_duplicate_quality_no_duplicates(self):
        """Test duplicate quality with no duplicates."""
        score = compute_duplicate_quality_score(100, 0)
        assert score == 100.0

    def test_duplicate_quality_with_duplicates(self):
        """Test duplicate quality with duplicates."""
        # Few duplicates
        score_low = compute_duplicate_quality_score(1000, 5, n_exact=2)
        assert score_low >= 90

        # Many duplicates
        score_high = compute_duplicate_quality_score(1000, 200, n_exact=50)
        assert score_high < 70

    def test_outlier_quality_no_outliers(self):
        """Test outlier quality with no outliers."""
        score = compute_outlier_quality_score(100, 0)
        assert score == 100.0

    def test_outlier_quality_within_expected(self):
        """Test outlier quality within expected contamination."""
        # Within expected 10% contamination
        score = compute_outlier_quality_score(100, 5, expected_contamination=0.1)
        assert score >= 85

    def test_outlier_quality_exceeds_expected(self):
        """Test outlier quality exceeding expected contamination."""
        # Way over expected
        score = compute_outlier_quality_score(100, 30, expected_contamination=0.1)
        assert score < 60

    def test_imbalance_quality_balanced(self):
        """Test imbalance quality with balanced classes."""
        score = compute_imbalance_quality_score(1.0, 2)
        assert score == 100.0

    def test_imbalance_quality_imbalanced(self):
        """Test imbalance quality with imbalanced classes."""
        # Moderate imbalance
        score_mod = compute_imbalance_quality_score(5.0, 2)
        assert 70 <= score_mod < 90

        # Severe imbalance
        score_severe = compute_imbalance_quality_score(50.0, 2)
        assert score_severe < 50

    def test_bias_quality_no_issues(self):
        """Test bias quality with no issues."""
        score = compute_bias_quality_score(0, 0.0, False)
        assert score == 100.0

    def test_bias_quality_with_issues(self):
        """Test bias quality with issues."""
        score = compute_bias_quality_score(
            n_bias_issues=3,
            max_demographic_parity_diff=0.25,
            has_correlation_issues=True,
        )
        assert score < 80

    def test_severity_from_score(self):
        """Test severity level from score."""
        assert severity_from_score(95) == "excellent"
        assert severity_from_score(80) == "good"
        assert severity_from_score(65) == "moderate"
        assert severity_from_score(45) == "poor"
        assert severity_from_score(30) == "critical"


class TestScoringWeights:
    """Tests for ScoringWeights."""

    def test_default_weights(self):
        """Test default weights."""
        weights = ScoringWeights()

        # Should sum to 1.0
        total = (
            weights.label_errors +
            weights.duplicates +
            weights.outliers +
            weights.imbalance +
            weights.bias
        )
        assert abs(total - 1.0) < 0.001

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = ScoringWeights(
            label_errors=2.0,
            duplicates=1.0,
            outliers=1.0,
            imbalance=1.0,
            bias=1.0,
        )
        normalized = weights.normalize()

        total = (
            normalized.label_errors +
            normalized.duplicates +
            normalized.outliers +
            normalized.imbalance +
            normalized.bias
        )
        assert abs(total - 1.0) < 0.001


class TestQualityScorer:
    """Tests for QualityScorer."""

    def test_compute_score_basic(self):
        """Test basic score computation."""
        scorer = QualityScorer()

        # Create mock results
        label_result = DetectorResult(
            issues=[
                LabelError(index=0, given_label=0, predicted_label=1, confidence=0.9),
                LabelError(index=1, given_label=1, predicted_label=0, confidence=0.8),
            ],
            metadata={"error_rate": 0.02},
        )

        duplicate_result = DetectorResult(
            issues=[
                DuplicatePair(index1=0, index2=5, similarity=1.0, is_exact=True),
            ],
            metadata={"n_exact": 1},
        )

        outlier_result = DetectorResult(
            issues=[
                Outlier(index=10, score=0.8, method="isolation_forest"),
            ],
            metadata={"contamination": 0.1},
        )

        score = scorer.compute_score(
            n_samples=100,
            label_result=label_result,
            duplicate_result=duplicate_result,
            outlier_result=outlier_result,
        )

        assert 0 <= score.overall <= 100
        assert 0 <= score.label_quality <= 100
        assert 0 <= score.duplicate_quality <= 100
        assert 0 <= score.outlier_quality <= 100

    def test_compute_score_no_issues(self):
        """Test score with no issues."""
        scorer = QualityScorer()

        score = scorer.compute_score(
            n_samples=100,
            label_result=DetectorResult(issues=[], metadata={}),
            duplicate_result=DetectorResult(issues=[], metadata={}),
            outlier_result=DetectorResult(issues=[], metadata={}),
        )

        # All components should be 100
        assert score.label_quality == 100.0
        assert score.duplicate_quality == 100.0
        assert score.outlier_quality == 100.0
        assert score.overall == 100.0

    def test_get_severity(self):
        """Test severity level."""
        scorer = QualityScorer()
        scorer.compute_score(
            n_samples=100,
            label_result=DetectorResult(issues=[], metadata={}),
        )

        severity = scorer.get_severity()
        assert severity == "excellent"

    def test_get_recommendations(self):
        """Test recommendations generation."""
        scorer = QualityScorer()

        # Create result with issues
        label_result = DetectorResult(
            issues=[LabelError(index=i, given_label=0, predicted_label=1, confidence=0.9)
                    for i in range(20)],
            metadata={},
        )

        scorer.compute_score(
            n_samples=100,
            label_result=label_result,
        )

        recommendations = scorer.get_recommendations()

        # Should have recommendation for labels
        assert len(recommendations) > 0
        assert any(r["category"] == "labels" for r in recommendations)

    def test_custom_weights(self):
        """Test with custom weights."""
        weights = ScoringWeights(
            label_errors=0.5,
            duplicates=0.1,
            outliers=0.1,
            imbalance=0.2,
            bias=0.1,
        )
        scorer = QualityScorer(weights=weights)

        score = scorer.compute_score(
            n_samples=100,
            label_result=DetectorResult(issues=[], metadata={}),
        )

        assert score is not None
