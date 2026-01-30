"""Tests for annotation quality analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clean.annotation import (
    AnnotationAnalyzer,
    AnnotationQualityReport,
    analyze_annotations,
)


class TestAnnotationAnalyzer:
    """Tests for AnnotationAnalyzer class."""

    @pytest.fixture
    def simple_annotations(self) -> pd.DataFrame:
        """Create simple annotation data."""
        return pd.DataFrame({
            "sample_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "annotator_id": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "label": ["cat", "cat", "dog", "dog", "cat", "dog", "cat", "cat"],
        })

    @pytest.fixture
    def multi_annotator_data(self) -> pd.DataFrame:
        """Create data with multiple annotators."""
        np.random.seed(42)
        n_samples = 50
        n_annotators = 5

        rows = []
        for sample_id in range(n_samples):
            true_label = np.random.choice(["cat", "dog", "bird"])
            for annotator_id in range(n_annotators):
                # Add some noise - 80% accuracy
                if np.random.random() < 0.8:
                    label = true_label
                else:
                    label = np.random.choice(["cat", "dog", "bird"])
                rows.append({
                    "sample_id": sample_id,
                    "annotator_id": f"annotator_{annotator_id}",
                    "label": label,
                    "ground_truth": true_label,
                })

        return pd.DataFrame(rows)

    def test_analyzer_init(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        assert analyzer is not None

    def test_analyzer_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with pytest.raises(ValueError, match="Column"):
            AnnotationAnalyzer(df, sample_id_column="missing")

    def test_analyze_returns_report(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert isinstance(report, AnnotationQualityReport)
        assert report.n_samples == 4
        assert report.n_annotators == 2
        assert report.n_annotations == 8

    def test_agreement_metrics_computed(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert report.agreement_metrics is not None
        assert 0 <= report.agreement_metrics.percent_agreement <= 100
        assert -1 <= report.agreement_metrics.krippendorff_alpha <= 1

    def test_cohen_kappa_for_two_annotators(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        # Cohen's kappa should be computed for 2 annotators
        assert report.agreement_metrics.cohen_kappa is not None

    def test_fleiss_kappa_for_multi_annotators(
        self, multi_annotator_data: pd.DataFrame
    ) -> None:
        analyzer = AnnotationAnalyzer(
            multi_annotator_data,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert report.agreement_metrics.fleiss_kappa is not None
        assert -1 <= report.agreement_metrics.fleiss_kappa <= 1

    def test_annotator_metrics(self, multi_annotator_data: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            multi_annotator_data,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert len(report.annotator_metrics) == 5
        for annotator_id, metrics in report.annotator_metrics.items():
            assert metrics.n_annotations > 0
            assert 0 <= metrics.agreement_rate <= 1

    def test_ground_truth_accuracy(self, multi_annotator_data: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            multi_annotator_data,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
            ground_truth_column="ground_truth",
        )
        report = analyzer.analyze()

        # At least one annotator should have accuracy computed
        has_accuracy = any(
            m.accuracy is not None for m in report.annotator_metrics.values()
        )
        assert has_accuracy

    def test_problematic_annotators_detected(self) -> None:
        # Create data with one bad annotator
        df = pd.DataFrame({
            "sample_id": list(range(10)) * 3,
            "annotator_id": ["A"] * 10 + ["B"] * 10 + ["bad"] * 10,
            "label": (
                ["cat"] * 5 + ["dog"] * 5 +  # A is correct
                ["cat"] * 5 + ["dog"] * 5 +  # B agrees with A
                ["bird"] * 10  # bad always says bird
            ),
        })

        analyzer = AnnotationAnalyzer(
            df,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
            agreement_threshold=0.5,
        )
        report = analyzer.analyze()

        # "bad" annotator should be flagged
        assert "bad" in report.problematic_annotators

    def test_report_summary(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()
        summary = report.summary()

        assert "Annotation Quality Report" in summary
        assert "Total Samples" in summary
        assert "Agreement" in summary

    def test_report_to_dict(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()
        d = report.to_dict()

        assert "n_samples" in d
        assert "agreement_metrics" in d
        assert "annotator_metrics" in d

    def test_get_annotator_ranking(self, multi_annotator_data: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            multi_annotator_data,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()
        ranking = report.get_annotator_ranking()

        assert isinstance(ranking, pd.DataFrame)
        assert "annotator_id" in ranking.columns
        assert "agreement_rate" in ranking.columns
        assert len(ranking) == 5

    def test_get_review_queue(self, simple_annotations: pd.DataFrame) -> None:
        analyzer = AnnotationAnalyzer(
            simple_annotations,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        queue = analyzer.get_review_queue(max_items=10)

        assert isinstance(queue, pd.DataFrame)
        if len(queue) > 0:
            assert "sample_id" in queue.columns
            assert "agreement_ratio" in queue.columns


class TestAnalyzeAnnotations:
    """Tests for analyze_annotations convenience function."""

    def test_analyze_annotations_function(self) -> None:
        df = pd.DataFrame({
            "sample_id": [1, 1, 2, 2],
            "annotator_id": ["A", "B", "A", "B"],
            "label": ["x", "x", "y", "y"],
        })

        report = analyze_annotations(
            df,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )

        assert isinstance(report, AnnotationQualityReport)
        assert report.agreement_metrics.percent_agreement == 100.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_annotator(self) -> None:
        df = pd.DataFrame({
            "sample_id": [1, 2, 3],
            "annotator_id": ["A", "A", "A"],
            "label": ["x", "y", "z"],
        })

        analyzer = AnnotationAnalyzer(
            df,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert report.n_annotators == 1
        # With single annotator, agreement is perfect by default
        assert report.annotator_metrics["A"].agreement_rate == 1.0

    def test_no_overlap_between_annotators(self) -> None:
        df = pd.DataFrame({
            "sample_id": [1, 2, 3, 4],
            "annotator_id": ["A", "A", "B", "B"],
            "label": ["x", "y", "z", "w"],
        })

        analyzer = AnnotationAnalyzer(
            df,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert report.n_annotators == 2

    def test_empty_after_filter(self) -> None:
        df = pd.DataFrame({
            "sample_id": [1],
            "annotator_id": ["A"],
            "label": ["x"],
        })

        analyzer = AnnotationAnalyzer(
            df,
            sample_id_column="sample_id",
            annotator_column="annotator_id",
            label_column="label",
        )
        report = analyzer.analyze()

        assert report.n_samples == 1
