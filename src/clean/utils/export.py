"""Export utilities for Clean."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clean.core.report import QualityReport
from clean.core.types import numpy_to_list


def export_clean_dataset(
    features: pd.DataFrame,
    labels: np.ndarray | None = None,
    label_column: str = "label",
    path: str | Path | None = None,
    format: str = "csv",
) -> pd.DataFrame | None:
    """Export cleaned dataset to file.

    Args:
        features: Feature DataFrame
        labels: Label array
        label_column: Name for label column
        path: Output path (returns DataFrame if None)
        format: Output format ('csv', 'parquet', 'json')

    Returns:
        DataFrame if path is None
    """
    df = features.copy()

    if labels is not None:
        df[label_column] = labels

    if path is None:
        return df

    path = Path(path)

    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    return None


def export_issues(
    report: QualityReport,
    path: str | Path,
    issue_types: list[str] | None = None,
) -> None:
    """Export detected issues to JSON file.

    Args:
        report: QualityReport
        path: Output path
        issue_types: Types to export (None = all)
    """
    issues: dict[str, list[dict]] = {}

    types = issue_types or ["label_errors", "duplicates", "outliers", "bias"]

    if "label_errors" in types and report.label_errors_result:
        issues["label_errors"] = [
            e.to_dict() for e in report.label_errors_result.issues
        ]

    if "duplicates" in types and report.duplicates_result:
        issues["duplicates"] = [
            d.to_dict() for d in report.duplicates_result.issues
        ]

    if "outliers" in types and report.outliers_result:
        issues["outliers"] = [
            o.to_dict() for o in report.outliers_result.issues
        ]

    if "bias" in types and report.bias_result:
        issues["bias"] = [
            b.to_dict() for b in report.bias_result.issues
        ]

    with open(path, "w") as f:
        json.dump(numpy_to_list(issues), f, indent=2, default=str)


def export_review_queue(
    report: QualityReport,
    path: str | Path,
    max_items: int = 500,
    format: str = "csv",
) -> None:
    """Export prioritized review queue.

    Args:
        report: QualityReport
        path: Output path
        max_items: Maximum items
        format: Output format
    """
    items: list[dict[str, Any]] = []

    # Add label errors
    if report.label_errors_result:
        for e in report.label_errors_result.issues:
            items.append({
                "index": e.index,
                "issue_type": "label_error",
                "priority": e.confidence,
                "current_value": e.given_label,
                "suggested_value": e.predicted_label,
                "confidence": e.confidence,
            })

    # Add outliers
    if report.outliers_result:
        for o in report.outliers_result.issues:
            items.append({
                "index": o.index,
                "issue_type": "outlier",
                "priority": o.score * 0.5,
                "method": o.method,
                "score": o.score,
            })

    # Sort and limit
    items.sort(key=lambda x: x["priority"], reverse=True)
    items = items[:max_items]

    df = pd.DataFrame(items)

    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def export_summary_stats(
    report: QualityReport,
    path: str | Path,
) -> None:
    """Export summary statistics to JSON.

    Args:
        report: QualityReport
        path: Output path
    """
    stats = {
        "dataset": report.dataset_info.to_dict(),
        "quality_scores": report.quality_score.to_dict(),
        "issue_counts": {
            "label_errors": report.label_errors_result.n_issues if report.label_errors_result else 0,
            "duplicates": report.duplicates_result.n_issues if report.duplicates_result else 0,
            "outliers": report.outliers_result.n_issues if report.outliers_result else 0,
            "bias_issues": report.bias_result.n_issues if report.bias_result else 0,
        },
        "recommendations": [],
    }

    # Add recommendations based on scores
    if report.quality_score.label_quality < 80:
        stats["recommendations"].append("Review and correct label errors")
    if report.quality_score.duplicate_quality < 80:
        stats["recommendations"].append("Remove or deduplicate similar samples")
    if report.quality_score.imbalance_quality < 70:
        stats["recommendations"].append("Apply resampling techniques")

    with open(path, "w") as f:
        json.dump(numpy_to_list(stats), f, indent=2, default=str)
