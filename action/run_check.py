#!/usr/bin/env python3
"""GitHub Action runner script for Clean data quality checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def set_output(name: str, value: str) -> None:
    """Set a GitHub Actions output variable."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            # Handle multiline values
            if "\n" in str(value):
                delimiter = "EOF"
                f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
            else:
                f.write(f"{name}={value}\n")
    else:
        # Fallback for local testing
        print(f"::set-output name={name}::{value}")


def main() -> int:
    """Run the data quality check."""
    # Get inputs from environment
    file_path = os.environ.get("INPUT_FILE", "")
    label_column = os.environ.get("INPUT_LABEL_COLUMN", "label")
    fail_below = float(os.environ.get("INPUT_FAIL_BELOW", "0"))
    detectors_str = os.environ.get("INPUT_DETECTORS", "all")
    output_format = os.environ.get("INPUT_OUTPUT_FORMAT", "markdown")
    output_file = os.environ.get("INPUT_OUTPUT_FILE", "")

    if not file_path:
        print("::error::No input file specified")
        return 1

    path = Path(file_path)
    if not path.exists():
        print(f"::error::File not found: {file_path}")
        return 1

    # Import clean after validation
    try:
        import pandas as pd

        from clean import DatasetCleaner
    except ImportError as e:
        print(f"::error::Failed to import clean: {e}")
        return 1

    # Load data based on file extension
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".json":
            df = pd.read_json(path)
        else:
            print(f"::error::Unsupported file format: {path.suffix}")
            return 1
    except Exception as e:
        print(f"::error::Failed to load file: {e}")
        return 1

    # Check label column
    if label_column not in df.columns:
        print(f"::error::Label column '{label_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return 1

    # Parse detectors
    detect_label_errors = True
    detect_duplicates = True
    detect_outliers = True
    detect_imbalance = True
    detect_bias = True

    if detectors_str != "all":
        detectors = [d.strip() for d in detectors_str.split(",")]
        detect_label_errors = "label_errors" in detectors
        detect_duplicates = "duplicates" in detectors
        detect_outliers = "outliers" in detectors
        detect_imbalance = "imbalance" in detectors
        detect_bias = "bias" in detectors

    # Run analysis
    print(f"Analyzing {path.name}...")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Label column: {label_column}")

    try:
        cleaner = DatasetCleaner(data=df, label_column=label_column)
        report = cleaner.analyze(
            detect_label_errors=detect_label_errors,
            detect_duplicates=detect_duplicates,
            detect_outliers=detect_outliers,
            detect_imbalance=detect_imbalance,
            detect_bias=detect_bias,
            show_progress=False,
        )
    except Exception as e:
        print(f"::error::Analysis failed: {e}")
        return 1

    # Extract metrics
    quality_score = report.quality_score.overall
    label_errors = len(report.label_errors()) if report.label_errors_result else 0
    duplicates = len(report.duplicates()) if report.duplicates_result else 0
    outliers = len(report.outliers()) if report.outliers_result else 0

    # Determine pass/fail
    passed = quality_score >= fail_below

    # Set outputs
    set_output("quality_score", str(round(quality_score, 1)))
    set_output("label_errors", str(label_errors))
    set_output("duplicates", str(duplicates))
    set_output("outliers", str(outliers))
    set_output("passed", str(passed).lower())

    # Generate report
    if output_format == "json":
        report_content = json.dumps(report.to_dict(), indent=2, default=str)
    elif output_format == "markdown":
        report_content = _generate_markdown_report(report, path.name)
    else:
        report_content = report.summary()

    # Save report
    report_file = output_file or f"clean-report.{'json' if output_format == 'json' else 'md'}"
    Path(report_file).write_text(report_content)
    set_output("report_file", report_file)
    set_output("report", report_content)

    # Print summary
    print("\n" + "=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)
    print(f"Quality Score: {quality_score:.1f}/100")
    print(f"Label Errors:  {label_errors}")
    print(f"Duplicates:    {duplicates}")
    print(f"Outliers:      {outliers}")
    print("=" * 50)

    if not passed:
        print(f"\n::error::Quality score {quality_score:.1f} is below threshold {fail_below}")
        return 1

    print(f"\n✅ Quality check passed (score: {quality_score:.1f} >= {fail_below})")
    return 0


def _generate_markdown_report(report, filename: str) -> str:
    """Generate a markdown report."""
    score = report.quality_score

    # Determine status emoji
    if score.overall >= 90:
        status = "🟢 Excellent"
    elif score.overall >= 70:
        status = "🟡 Good"
    elif score.overall >= 50:
        status = "🟠 Needs Attention"
    else:
        status = "🔴 Poor"

    lines = [
        f"### Dataset: `{filename}`",
        "",
        f"**Overall Quality Score: {score.overall:.1f}/100** {status}",
        "",
        "| Metric | Score | Status |",
        "|--------|-------|--------|",
        f"| Label Quality | {score.label_quality:.1f} | {_score_emoji(score.label_quality)} |",
        f"| Duplicate Quality | {score.duplicate_quality:.1f} | {_score_emoji(score.duplicate_quality)} |",
        f"| Outlier Quality | {score.outlier_quality:.1f} | {_score_emoji(score.outlier_quality)} |",
        f"| Balance Quality | {score.imbalance_quality:.1f} | {_score_emoji(score.imbalance_quality)} |",
        f"| Bias Quality | {score.bias_quality:.1f} | {_score_emoji(score.bias_quality)} |",
        "",
    ]

    # Issues summary
    issues = []
    if report.label_errors_result:
        n = len(report.label_errors())
        if n > 0:
            issues.append(f"- **Label Errors**: {n} samples with potential mislabeling")
    if report.duplicates_result:
        n = len(report.duplicates())
        if n > 0:
            issues.append(f"- **Duplicates**: {n} duplicate pairs detected")
    if report.outliers_result:
        n = len(report.outliers())
        if n > 0:
            issues.append(f"- **Outliers**: {n} statistical outliers detected")

    if issues:
        lines.append("#### Issues Found")
        lines.extend(issues)
        lines.append("")

    # Add sample info
    lines.append(f"*Analyzed {report.dataset_info.n_samples:,} samples with {report.dataset_info.n_features} features*")

    return "\n".join(lines)


def _score_emoji(score: float) -> str:
    """Get emoji for a score."""
    if score >= 90:
        return "✅"
    elif score >= 70:
        return "⚠️"
    else:
        return "❌"


if __name__ == "__main__":
    sys.exit(main())
