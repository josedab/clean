"""Quality report for data analysis results.

The QualityReport class encapsulates all findings from a data quality
analysis, including detected issues, scores, and metadata.

Example:
    >>> from clean import DatasetCleaner
    >>>
    >>> cleaner = DatasetCleaner(data=df, label_column='label')
    >>> report = cleaner.analyze()
    >>>
    >>> # Text summary
    >>> print(report.summary())
    >>>
    >>> # Access specific issues
    >>> label_errors = report.label_errors()
    >>> duplicates = report.duplicates()
    >>> outliers = report.outliers()
    >>>
    >>> # Export
    >>> report.to_json('report.json')
    >>> report.to_html('report.html')
    >>> report_dict = report.to_dict()

See Also:
    - :class:`DatasetCleaner`: Creates reports via analyze()
    - :class:`QualityScore`: Overall and component quality scores
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from clean.core.types import (
    ClassDistribution,
    DatasetInfo,
    QualityScore,
    numpy_to_list,
)
from clean.detection.base import DetectorResult


@dataclass
class QualityReport:
    """Comprehensive data quality report.

    Contains all detection results, scores, and provides methods
    for accessing and exporting the analysis.
    """

    dataset_info: DatasetInfo
    quality_score: QualityScore
    label_errors_result: DetectorResult | None = None
    duplicates_result: DetectorResult | None = None
    outliers_result: DetectorResult | None = None
    imbalance_result: DetectorResult | None = None
    bias_result: DetectorResult | None = None
    class_distribution: ClassDistribution | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a text summary of the report.

        Returns:
            Formatted summary string
        """
        lines = [
            "Data Quality Report",
            "=" * 50,
            f"Samples analyzed: {self.dataset_info.n_samples:,}",
            f"Features: {self.dataset_info.n_features}",
        ]

        if self.dataset_info.n_classes:
            lines.append(f"Classes: {self.dataset_info.n_classes}")

        lines.extend(
            [
                "",
                f"Overall Quality Score: {self.quality_score.overall}/100 "
                f"({self._get_severity_emoji()} {self._get_severity()})",
                "",
                "Component Scores:",
                f"  - Label Quality:     {self.quality_score.label_quality}/100",
                f"  - Duplicate Quality: {self.quality_score.duplicate_quality}/100",
                f"  - Outlier Quality:   {self.quality_score.outlier_quality}/100",
                f"  - Imbalance Quality: {self.quality_score.imbalance_quality}/100",
                f"  - Bias Quality:      {self.quality_score.bias_quality}/100",
                "",
                "Issues Found:",
            ]
        )

        # Label errors
        n_label_errors = self.label_errors_result.n_issues if self.label_errors_result else 0
        pct = (
            n_label_errors / self.dataset_info.n_samples * 100
            if self.dataset_info.n_samples > 0
            else 0
        )
        priority = "HIGH PRIORITY" if pct > 3 else "MEDIUM" if pct > 1 else "LOW"
        lines.append(f"  - Label errors: {n_label_errors:,} ({pct:.1f}%) - {priority}")

        # Duplicates
        n_dups = self.duplicates_result.n_issues if self.duplicates_result else 0
        if n_dups > 0:
            n_exact = (
                self.duplicates_result.metadata.get("n_exact", 0) if self.duplicates_result else 0
            )
            lines.append(f"  - Duplicate pairs: {n_dups:,} ({n_exact} exact)")

        # Outliers
        n_outliers = self.outliers_result.n_issues if self.outliers_result else 0
        pct = (
            n_outliers / self.dataset_info.n_samples * 100 if self.dataset_info.n_samples > 0 else 0
        )
        lines.append(f"  - Outliers: {n_outliers:,} ({pct:.1f}%)")

        # Imbalance
        if self.imbalance_result and self.imbalance_result.metadata:
            ratio = self.imbalance_result.metadata.get("imbalance_ratio", 1)
            lines.append(f"  - Class imbalance: {ratio:.1f}:1 ratio")

        # Bias
        n_bias = self.bias_result.n_issues if self.bias_result else 0
        if n_bias > 0:
            lines.append(f"  - Bias issues: {n_bias}")

        # Estimated impact
        if self.quality_score.overall < 80:
            lines.extend(
                [
                    "",
                    f"Estimated accuracy impact: +{self._estimate_impact()}% after cleaning",
                ]
            )

        return "\n".join(lines)

    def _get_severity(self) -> str:
        """Get severity level string."""
        score = self.quality_score.overall
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Moderate"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"

    def _get_severity_emoji(self) -> str:
        """Get severity emoji."""
        score = self.quality_score.overall
        if score >= 90:
            return "✅"
        elif score >= 75:
            return "🟢"
        elif score >= 60:
            return "🟡"
        elif score >= 40:
            return "🟠"
        else:
            return "🔴"

    def _estimate_impact(self) -> str:
        """Estimate potential accuracy improvement."""
        # Rough heuristic based on issues
        impact_low = 0
        impact_high = 0

        if self.label_errors_result:
            error_rate = self.label_errors_result.n_issues / self.dataset_info.n_samples
            impact_low += error_rate * 50
            impact_high += error_rate * 100

        if self.duplicates_result:
            dup_rate = self.duplicates_result.n_issues / self.dataset_info.n_samples
            impact_low += dup_rate * 10
            impact_high += dup_rate * 30

        return f"{int(impact_low)}-{int(impact_high)}"

    def label_errors(self) -> pd.DataFrame:
        """Get label errors as DataFrame.

        Returns:
            DataFrame with columns: index, given_label, predicted_label, confidence
        """
        if self.label_errors_result is None or self.label_errors_result.n_issues == 0:
            return pd.DataFrame(columns=["index", "given_label", "predicted_label", "confidence"])

        return pd.DataFrame([e.to_dict() for e in self.label_errors_result.issues])

    def duplicates(self) -> pd.DataFrame:
        """Get duplicate pairs as DataFrame.

        Returns:
            DataFrame with columns: index1, index2, similarity, is_exact
        """
        if self.duplicates_result is None or self.duplicates_result.n_issues == 0:
            return pd.DataFrame(columns=["index1", "index2", "similarity", "is_exact"])

        return pd.DataFrame([d.to_dict() for d in self.duplicates_result.issues])

    def outliers(self) -> pd.DataFrame:
        """Get outliers as DataFrame.

        Returns:
            DataFrame with columns: index, score, method, features_contributing
        """
        if self.outliers_result is None or self.outliers_result.n_issues == 0:
            return pd.DataFrame(columns=["index", "score", "method"])

        return pd.DataFrame([o.to_dict() for o in self.outliers_result.issues])

    def bias_issues(self) -> pd.DataFrame:
        """Get bias issues as DataFrame.

        Returns:
            DataFrame with bias issue information
        """
        if self.bias_result is None or self.bias_result.n_issues == 0:
            return pd.DataFrame(columns=["feature", "metric", "value", "threshold"])

        return pd.DataFrame([b.to_dict() for b in self.bias_result.issues])

    def get_all_issue_indices(self) -> set[int]:
        """Get all indices with any issue.

        Returns:
            Set of sample indices with issues
        """
        indices: set[int] = set()

        if self.label_errors_result:
            for e in self.label_errors_result.issues:
                indices.add(e.index)

        if self.duplicates_result:
            for d in self.duplicates_result.issues:
                indices.add(d.index1)
                indices.add(d.index2)

        if self.outliers_result:
            for o in self.outliers_result.issues:
                indices.add(o.index)

        return indices

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of report
        """
        return {
            "dataset_info": self.dataset_info.to_dict(),
            "quality_score": self.quality_score.to_dict(),
            "label_errors": {
                "count": self.label_errors_result.n_issues if self.label_errors_result else 0,
                "metadata": self.label_errors_result.metadata if self.label_errors_result else {},
            },
            "duplicates": {
                "count": self.duplicates_result.n_issues if self.duplicates_result else 0,
                "metadata": self.duplicates_result.metadata if self.duplicates_result else {},
            },
            "outliers": {
                "count": self.outliers_result.n_issues if self.outliers_result else 0,
                "metadata": self.outliers_result.metadata if self.outliers_result else {},
            },
            "imbalance": {
                "metadata": self.imbalance_result.metadata if self.imbalance_result else {},
            },
            "bias": {
                "count": self.bias_result.n_issues if self.bias_result else 0,
                "metadata": self.bias_result.metadata if self.bias_result else {},
            },
            "class_distribution": (
                self.class_distribution.to_dict() if self.class_distribution else None
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(numpy_to_list(self.to_dict()), indent=indent, default=str)

    def to_html(self) -> str:
        """Generate HTML report.

        Returns:
            HTML string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Data Quality Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2 { color: #333; }",
            ".score { font-size: 24px; font-weight: bold; }",
            ".excellent { color: #28a745; }",
            ".good { color: #5cb85c; }",
            ".moderate { color: #f0ad4e; }",
            ".poor { color: #d9534f; }",
            ".critical { color: #c9302c; }",
            "table { border-collapse: collapse; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background: #f5f5f5; }",
            "</style>",
            "</head><body>",
            "<h1>Data Quality Report</h1>",
        ]

        # Dataset info
        html_parts.extend(
            [
                "<h2>Dataset Information</h2>",
                "<table>",
                f"<tr><td>Samples</td><td>{self.dataset_info.n_samples:,}</td></tr>",
                f"<tr><td>Features</td><td>{self.dataset_info.n_features}</td></tr>",
                f"<tr><td>Classes</td><td>{self.dataset_info.n_classes or 'N/A'}</td></tr>",
                "</table>",
            ]
        )

        # Quality scores
        severity = self._get_severity().lower()
        html_parts.extend(
            [
                "<h2>Quality Scores</h2>",
                f"<p class='score {severity}'>Overall: {self.quality_score.overall}/100</p>",
                "<table>",
                f"<tr><td>Label Quality</td><td>{self.quality_score.label_quality}</td></tr>",
                f"<tr><td>Duplicate Quality</td><td>{self.quality_score.duplicate_quality}</td></tr>",
                f"<tr><td>Outlier Quality</td><td>{self.quality_score.outlier_quality}</td></tr>",
                f"<tr><td>Imbalance Quality</td><td>{self.quality_score.imbalance_quality}</td></tr>",
                f"<tr><td>Bias Quality</td><td>{self.quality_score.bias_quality}</td></tr>",
                "</table>",
            ]
        )

        # Issues summary
        html_parts.extend(
            [
                "<h2>Issues Summary</h2>",
                "<table>",
                "<tr><th>Issue Type</th><th>Count</th></tr>",
                f"<tr><td>Label Errors</td><td>{self.label_errors_result.n_issues if self.label_errors_result else 0}</td></tr>",
                f"<tr><td>Duplicate Pairs</td><td>{self.duplicates_result.n_issues if self.duplicates_result else 0}</td></tr>",
                f"<tr><td>Outliers</td><td>{self.outliers_result.n_issues if self.outliers_result else 0}</td></tr>",
                f"<tr><td>Bias Issues</td><td>{self.bias_result.n_issues if self.bias_result else 0}</td></tr>",
                "</table>",
            ]
        )

        html_parts.extend(
            [
                "</body></html>",
            ]
        )

        return "\n".join(html_parts)

    def save_html(self, path: str | Path) -> None:
        """Save HTML report to file.

        Args:
            path: Output file path
        """
        Path(path).write_text(self.to_html())

    def save_json(self, path: str | Path) -> None:
        """Save JSON report to file.

        Args:
            path: Output file path
        """
        Path(path).write_text(self.to_json())
