"""Annotation quality analysis for labeling quality assessment.

This module provides tools for analyzing annotator quality, including:
- Per-annotator accuracy and agreement metrics
- Inter-annotator agreement (Krippendorff's alpha, Cohen's kappa)
- Annotator confusion matrices
- Quality trend tracking over time

Example:
    >>> from clean.annotation import AnnotationAnalyzer
    >>>
    >>> analyzer = AnnotationAnalyzer(
    ...     annotations_df,
    ...     sample_id_column='sample_id',
    ...     annotator_column='annotator_id',
    ...     label_column='label',
    ... )
    >>> report = analyzer.analyze()
    >>> print(report.summary())
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AnnotatorMetrics:
    """Quality metrics for a single annotator."""

    annotator_id: str
    n_annotations: int
    accuracy: float | None  # If ground truth available
    agreement_rate: float  # Agreement with other annotators
    self_consistency: float  # Consistency when same sample labeled twice
    avg_confidence: float | None  # If confidence scores available
    label_distribution: dict[Any, float] = field(default_factory=dict)
    confusion_matrix: dict[tuple[Any, Any], int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotator_id": self.annotator_id,
            "n_annotations": self.n_annotations,
            "accuracy": self.accuracy,
            "agreement_rate": self.agreement_rate,
            "self_consistency": self.self_consistency,
            "avg_confidence": self.avg_confidence,
            "label_distribution": self.label_distribution,
        }


@dataclass
class AgreementMetrics:
    """Inter-annotator agreement metrics."""

    krippendorff_alpha: float
    fleiss_kappa: float | None  # For multi-rater
    cohen_kappa: float | None  # For pair-wise
    percent_agreement: float
    n_samples_with_disagreement: int
    disagreement_samples: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "krippendorff_alpha": self.krippendorff_alpha,
            "fleiss_kappa": self.fleiss_kappa,
            "cohen_kappa": self.cohen_kappa,
            "percent_agreement": self.percent_agreement,
            "n_samples_with_disagreement": self.n_samples_with_disagreement,
        }


@dataclass
class AnnotationQualityReport:
    """Complete annotation quality report."""

    n_samples: int
    n_annotators: int
    n_annotations: int
    avg_annotations_per_sample: float
    agreement_metrics: AgreementMetrics
    annotator_metrics: dict[str, AnnotatorMetrics]
    problematic_annotators: list[str]
    problematic_samples: list[int]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "Annotation Quality Report",
            "=" * 50,
            f"Total Samples: {self.n_samples:,}",
            f"Total Annotators: {self.n_annotators}",
            f"Total Annotations: {self.n_annotations:,}",
            f"Avg Annotations per Sample: {self.avg_annotations_per_sample:.2f}",
            "",
            "Agreement Metrics:",
            f"  Krippendorff's Alpha: {self.agreement_metrics.krippendorff_alpha:.3f}",
            f"  Percent Agreement: {self.agreement_metrics.percent_agreement:.1f}%",
            f"  Samples with Disagreement: {self.agreement_metrics.n_samples_with_disagreement}",
        ]

        if self.agreement_metrics.fleiss_kappa is not None:
            lines.append(f"  Fleiss' Kappa: {self.agreement_metrics.fleiss_kappa:.3f}")

        lines.extend([
            "",
            "Annotator Summary:",
            "-" * 50,
        ])

        # Sort annotators by agreement rate
        sorted_annotators = sorted(
            self.annotator_metrics.values(),
            key=lambda x: x.agreement_rate,
            reverse=True,
        )

        for metrics in sorted_annotators[:10]:  # Top 10
            accuracy_str = f"{metrics.accuracy:.1%}" if metrics.accuracy else "N/A"
            lines.append(
                f"  {metrics.annotator_id}: "
                f"agreement={metrics.agreement_rate:.1%}, "
                f"accuracy={accuracy_str}, "
                f"n={metrics.n_annotations}"
            )

        if len(sorted_annotators) > 10:
            lines.append(f"  ... and {len(sorted_annotators) - 10} more annotators")

        if self.problematic_annotators:
            lines.extend([
                "",
                "⚠️ Problematic Annotators (low agreement):",
            ])
            for annotator_id in self.problematic_annotators[:5]:
                metrics = self.annotator_metrics[annotator_id]
                lines.append(f"  - {annotator_id}: {metrics.agreement_rate:.1%} agreement")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_annotators": self.n_annotators,
            "n_annotations": self.n_annotations,
            "avg_annotations_per_sample": self.avg_annotations_per_sample,
            "agreement_metrics": self.agreement_metrics.to_dict(),
            "annotator_metrics": {
                k: v.to_dict() for k, v in self.annotator_metrics.items()
            },
            "problematic_annotators": self.problematic_annotators,
            "problematic_samples": self.problematic_samples[:100],  # Limit size
        }

    def get_annotator_ranking(self) -> pd.DataFrame:
        """Get annotators ranked by quality metrics."""
        rows = []
        for annotator_id, metrics in self.annotator_metrics.items():
            rows.append({
                "annotator_id": annotator_id,
                "n_annotations": metrics.n_annotations,
                "agreement_rate": metrics.agreement_rate,
                "accuracy": metrics.accuracy,
                "self_consistency": metrics.self_consistency,
                "is_problematic": annotator_id in self.problematic_annotators,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("agreement_rate", ascending=False).reset_index(drop=True)


class AnnotationAnalyzer:
    """Analyzer for annotation quality assessment.

    Analyzes labeling quality across multiple annotators, computing
    agreement metrics and identifying problematic annotators or samples.
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        sample_id_column: str = "sample_id",
        annotator_column: str = "annotator_id",
        label_column: str = "label",
        ground_truth_column: str | None = None,
        confidence_column: str | None = None,
        timestamp_column: str | None = None,
        agreement_threshold: float = 0.7,
    ):
        """Initialize the annotation analyzer.

        Args:
            annotations: DataFrame with annotation data
            sample_id_column: Column name for sample IDs
            annotator_column: Column name for annotator IDs
            label_column: Column name for labels
            ground_truth_column: Optional column with ground truth labels
            confidence_column: Optional column with annotator confidence scores
            timestamp_column: Optional column with annotation timestamps
            agreement_threshold: Threshold below which annotators are flagged
        """
        self.annotations = annotations.copy()
        self.sample_id_col = sample_id_column
        self.annotator_col = annotator_column
        self.label_col = label_column
        self.ground_truth_col = ground_truth_column
        self.confidence_col = confidence_column
        self.timestamp_col = timestamp_column
        self.agreement_threshold = agreement_threshold

        # Validate columns
        required_cols = [sample_id_column, annotator_column, label_column]
        for col in required_cols:
            if col not in annotations.columns:
                raise ValueError(f"Column '{col}' not found in annotations DataFrame")

        # Extract unique values
        self._samples = annotations[sample_id_column].unique()
        self._annotators = annotations[annotator_column].unique()
        self._labels = annotations[label_column].unique()

    def analyze(self) -> AnnotationQualityReport:
        """Run complete annotation quality analysis.

        Returns:
            AnnotationQualityReport with all metrics
        """
        # Compute agreement metrics
        agreement_metrics = self._compute_agreement_metrics()

        # Compute per-annotator metrics
        annotator_metrics = {}
        for annotator_id in self._annotators:
            metrics = self._compute_annotator_metrics(str(annotator_id))
            annotator_metrics[str(annotator_id)] = metrics

        # Identify problematic annotators
        problematic_annotators = [
            annotator_id
            for annotator_id, metrics in annotator_metrics.items()
            if metrics.agreement_rate < self.agreement_threshold
        ]

        # Identify problematic samples (high disagreement)
        problematic_samples = self._find_problematic_samples()

        return AnnotationQualityReport(
            n_samples=len(self._samples),
            n_annotators=len(self._annotators),
            n_annotations=len(self.annotations),
            avg_annotations_per_sample=len(self.annotations) / len(self._samples),
            agreement_metrics=agreement_metrics,
            annotator_metrics=annotator_metrics,
            problematic_annotators=problematic_annotators,
            problematic_samples=problematic_samples,
        )

    def _compute_agreement_metrics(self) -> AgreementMetrics:
        """Compute inter-annotator agreement metrics."""
        # Build annotation matrix
        sample_annotations = self._build_annotation_matrix()

        # Compute Krippendorff's alpha
        alpha = self._krippendorff_alpha(sample_annotations)

        # Compute percent agreement
        percent_agreement, n_disagreements, disagreement_samples = (
            self._compute_percent_agreement(sample_annotations)
        )

        # Compute Fleiss' kappa for multi-rater
        fleiss_kappa = self._fleiss_kappa(sample_annotations)

        # Cohen's kappa only for 2 annotators
        cohen_kappa = None
        if len(self._annotators) == 2:
            cohen_kappa = self._cohen_kappa()

        return AgreementMetrics(
            krippendorff_alpha=alpha,
            fleiss_kappa=fleiss_kappa,
            cohen_kappa=cohen_kappa,
            percent_agreement=percent_agreement,
            n_samples_with_disagreement=n_disagreements,
            disagreement_samples=disagreement_samples,
        )

    def _build_annotation_matrix(self) -> dict[Any, dict[Any, Any]]:
        """Build a matrix of sample -> annotator -> label."""
        matrix: dict[Any, dict[Any, Any]] = defaultdict(dict)
        for _, row in self.annotations.iterrows():
            sample_id = row[self.sample_id_col]
            annotator_id = row[self.annotator_col]
            label = row[self.label_col]
            matrix[sample_id][annotator_id] = label
        return dict(matrix)

    def _krippendorff_alpha(
        self, sample_annotations: dict[Any, dict[Any, Any]]
    ) -> float:
        """Compute Krippendorff's alpha for inter-rater reliability.

        Uses nominal metric (for categorical data).
        """
        # Convert to reliability data matrix
        # Rows = annotators, Columns = samples
        annotator_list = list(self._annotators)
        sample_list = list(sample_annotations.keys())

        # Encode labels to integers
        label_to_int = {label: i for i, label in enumerate(self._labels)}

        # Build data matrix (annotators x samples)
        data = np.full((len(annotator_list), len(sample_list)), np.nan)
        for j, sample_id in enumerate(sample_list):
            for i, annotator_id in enumerate(annotator_list):
                if annotator_id in sample_annotations[sample_id]:
                    label = sample_annotations[sample_id][annotator_id]
                    data[i, j] = label_to_int.get(label, np.nan)

        # Compute alpha using the formula
        # alpha = 1 - Do/De where Do is observed disagreement, De is expected
        n_labels = len(self._labels)

        # Count coincidences
        observed_coincidences = np.zeros((n_labels, n_labels))
        expected_coincidences = np.zeros((n_labels, n_labels))

        total_pairs = 0
        label_counts = np.zeros(n_labels)

        for j in range(len(sample_list)):
            # Get all non-nan values for this sample
            values = data[:, j]
            values = values[~np.isnan(values)].astype(int)

            n_coders = len(values)
            if n_coders < 2:
                continue

            # Count pairs
            for v1 in values:
                label_counts[v1] += 1
                for v2 in values:
                    if v1 != v2 or n_coders > 1:
                        observed_coincidences[v1, v2] += 1
                total_pairs += n_coders - 1

        if total_pairs == 0:
            return 1.0  # Perfect agreement if no pairs to compare

        # Normalize observed coincidences
        observed_coincidences /= total_pairs

        # Compute expected coincidences
        total_labels = label_counts.sum()
        if total_labels > 1:
            for i in range(n_labels):
                for j in range(n_labels):
                    if i == j:
                        expected_coincidences[i, j] = (
                            label_counts[i] * (label_counts[i] - 1)
                        ) / (total_labels * (total_labels - 1))
                    else:
                        expected_coincidences[i, j] = (
                            2 * label_counts[i] * label_counts[j]
                        ) / (total_labels * (total_labels - 1))

        # Compute disagreement (1 - agreement for nominal data)
        # For nominal data: disagreement = 1 when labels differ, 0 when same
        do = 1 - np.trace(observed_coincidences)
        de = 1 - np.trace(expected_coincidences)

        if de == 0:
            return 1.0  # Perfect expected agreement

        alpha = 1 - (do / de)
        return float(np.clip(alpha, -1, 1))

    def _fleiss_kappa(self, sample_annotations: dict[Any, dict[Any, Any]]) -> float:
        """Compute Fleiss' kappa for multi-rater agreement."""
        # Build category count matrix
        sample_list = list(sample_annotations.keys())
        label_list = list(self._labels)
        n_categories = len(label_list)
        label_to_idx = {label: i for i, label in enumerate(label_list)}

        # Matrix: samples x categories (count of annotators choosing each category)
        counts = np.zeros((len(sample_list), n_categories))

        for i, sample_id in enumerate(sample_list):
            for label in sample_annotations[sample_id].values():
                if label in label_to_idx:
                    counts[i, label_to_idx[label]] += 1

        # Number of raters per sample
        n_raters = counts.sum(axis=1)

        # Filter samples with at least 2 raters
        valid_mask = n_raters >= 2
        if not valid_mask.any():
            return 0.0

        counts = counts[valid_mask]
        n_raters = n_raters[valid_mask]
        n_samples = len(counts)
        n = n_raters[0]  # Assume same number of raters (use first)

        # Proportion of assignments to each category
        p = counts.sum(axis=0) / counts.sum()

        # Agreement for each sample
        p_i = (counts * (counts - 1)).sum(axis=1) / (n * (n - 1))

        # Mean agreement
        p_bar = p_i.mean()

        # Expected agreement by chance
        p_e = (p ** 2).sum()

        if p_e == 1:
            return 1.0

        kappa = (p_bar - p_e) / (1 - p_e)
        return float(np.clip(kappa, -1, 1))

    def _cohen_kappa(self) -> float:
        """Compute Cohen's kappa for two annotators."""
        if len(self._annotators) != 2:
            return 0.0

        annotator1, annotator2 = self._annotators[:2]

        # Get paired annotations
        df1 = self.annotations[self.annotations[self.annotator_col] == annotator1]
        df2 = self.annotations[self.annotations[self.annotator_col] == annotator2]

        merged = pd.merge(
            df1[[self.sample_id_col, self.label_col]],
            df2[[self.sample_id_col, self.label_col]],
            on=self.sample_id_col,
            suffixes=("_1", "_2"),
        )

        if len(merged) == 0:
            return 0.0

        labels1 = merged[f"{self.label_col}_1"]
        labels2 = merged[f"{self.label_col}_2"]

        # Observed agreement
        po = (labels1 == labels2).mean()

        # Expected agreement
        label_list = list(self._labels)
        pe = 0.0
        for label in label_list:
            p1 = (labels1 == label).mean()
            p2 = (labels2 == label).mean()
            pe += p1 * p2

        if pe == 1:
            return 1.0

        kappa = (po - pe) / (1 - pe)
        return float(np.clip(kappa, -1, 1))

    def _compute_percent_agreement(
        self, sample_annotations: dict[Any, dict[Any, Any]]
    ) -> tuple[float, int, list[int]]:
        """Compute percent agreement across samples."""
        agreements = 0
        total = 0
        n_disagreements = 0
        disagreement_samples = []

        for sample_id, annotator_labels in sample_annotations.items():
            labels = list(annotator_labels.values())
            if len(labels) < 2:
                continue

            # Check if all labels agree
            if len(set(labels)) == 1:
                agreements += 1
            else:
                n_disagreements += 1
                disagreement_samples.append(sample_id)
            total += 1

        percent = (agreements / total * 100) if total > 0 else 100.0
        return percent, n_disagreements, disagreement_samples

    def _compute_annotator_metrics(self, annotator_id: str) -> AnnotatorMetrics:
        """Compute metrics for a single annotator."""
        annotator_df = self.annotations[
            self.annotations[self.annotator_col] == annotator_id
        ]

        n_annotations = len(annotator_df)

        # Compute accuracy if ground truth available
        accuracy = None
        if self.ground_truth_col and self.ground_truth_col in self.annotations.columns:
            gt_df = annotator_df[annotator_df[self.ground_truth_col].notna()]
            if len(gt_df) > 0:
                accuracy = (
                    gt_df[self.label_col] == gt_df[self.ground_truth_col]
                ).mean()

        # Compute agreement rate with other annotators
        agreement_rate = self._compute_annotator_agreement(annotator_id)

        # Compute self-consistency
        self_consistency = self._compute_self_consistency(annotator_id)

        # Average confidence if available
        avg_confidence = None
        if self.confidence_col and self.confidence_col in annotator_df.columns:
            avg_confidence = annotator_df[self.confidence_col].mean()

        # Label distribution
        label_counts = annotator_df[self.label_col].value_counts(normalize=True)
        label_distribution = label_counts.to_dict()

        return AnnotatorMetrics(
            annotator_id=annotator_id,
            n_annotations=n_annotations,
            accuracy=accuracy,
            agreement_rate=agreement_rate,
            self_consistency=self_consistency,
            avg_confidence=avg_confidence,
            label_distribution=label_distribution,
        )

    def _compute_annotator_agreement(self, annotator_id: str) -> float:
        """Compute agreement rate between an annotator and others."""
        annotator_df = self.annotations[
            self.annotations[self.annotator_col] == annotator_id
        ]
        other_df = self.annotations[
            self.annotations[self.annotator_col] != annotator_id
        ]

        # Find samples labeled by both
        merged = pd.merge(
            annotator_df[[self.sample_id_col, self.label_col]],
            other_df[[self.sample_id_col, self.label_col]],
            on=self.sample_id_col,
            suffixes=("_target", "_other"),
        )

        if len(merged) == 0:
            return 1.0  # No overlap, assume agreement

        agreements = (merged[f"{self.label_col}_target"] == merged[f"{self.label_col}_other"]).sum()
        return agreements / len(merged)

    def _compute_self_consistency(self, annotator_id: str) -> float:
        """Compute self-consistency for annotator (if they labeled same sample twice)."""
        annotator_df = self.annotations[
            self.annotations[self.annotator_col] == annotator_id
        ]

        # Find samples labeled multiple times by same annotator
        sample_counts = annotator_df[self.sample_id_col].value_counts()
        repeated_samples = sample_counts[sample_counts > 1].index

        if len(repeated_samples) == 0:
            return 1.0  # No repeated samples, assume consistent

        consistent = 0
        total = 0

        for sample_id in repeated_samples:
            labels = annotator_df[
                annotator_df[self.sample_id_col] == sample_id
            ][self.label_col].values

            # Check if all labels for this sample are the same
            if len(set(labels)) == 1:
                consistent += 1
            total += 1

        return consistent / total if total > 0 else 1.0

    def _find_problematic_samples(self) -> list[int]:
        """Find samples with high disagreement."""
        sample_annotations = self._build_annotation_matrix()
        problematic = []

        for sample_id, annotator_labels in sample_annotations.items():
            labels = list(annotator_labels.values())
            if len(labels) < 2:
                continue

            # Compute agreement ratio
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                # Multiple different labels - disagreement
                most_common_count = max(labels.count(l) for l in unique_labels)
                agreement_ratio = most_common_count / len(labels)

                if agreement_ratio < 0.6:  # Less than 60% agreement
                    problematic.append(sample_id)

        return problematic

    def get_review_queue(self, max_items: int = 100) -> pd.DataFrame:
        """Get samples prioritized for review based on disagreement.

        Args:
            max_items: Maximum number of samples to return

        Returns:
            DataFrame with samples sorted by disagreement level
        """
        sample_annotations = self._build_annotation_matrix()
        rows = []

        for sample_id, annotator_labels in sample_annotations.items():
            labels = list(annotator_labels.values())
            if len(labels) < 2:
                continue

            unique_labels = list(set(labels))
            most_common = max(labels, key=labels.count)
            agreement = labels.count(most_common) / len(labels)

            rows.append({
                "sample_id": sample_id,
                "n_annotators": len(labels),
                "n_unique_labels": len(unique_labels),
                "agreement_ratio": agreement,
                "labels": dict(zip(annotator_labels.keys(), labels)),
                "majority_label": most_common,
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("agreement_ratio").head(max_items)
        return df.reset_index(drop=True)


def analyze_annotations(
    annotations: pd.DataFrame,
    sample_id_column: str = "sample_id",
    annotator_column: str = "annotator_id",
    label_column: str = "label",
    **kwargs: Any,
) -> AnnotationQualityReport:
    """Analyze annotation quality.

    Args:
        annotations: DataFrame with annotation data
        sample_id_column: Column name for sample IDs
        annotator_column: Column name for annotator IDs
        label_column: Column name for labels
        **kwargs: Additional arguments for AnnotationAnalyzer

    Returns:
        AnnotationQualityReport with quality metrics
    """
    analyzer = AnnotationAnalyzer(
        annotations,
        sample_id_column=sample_id_column,
        annotator_column=annotator_column,
        label_column=label_column,
        **kwargs,
    )
    return analyzer.analyze()
