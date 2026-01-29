"""Command-line interface for Clean data quality platform.

Usage:
    clean analyze data.csv --label-column label
    clean analyze data.csv --output report.json
    clean fix data.csv --output cleaned.csv
    clean benchmark --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the analyze subcommand parser."""
    parser = subparsers.add_parser(
        "analyze", help="Analyze dataset for quality issues"
    )
    parser.add_argument("file", type=str, help="Path to CSV file")
    parser.add_argument(
        "--label-column",
        "-l",
        type=str,
        default="label",
        help="Name of the label column (default: label)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (optional)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--detectors",
        "-d",
        type=str,
        nargs="+",
        choices=["label_errors", "duplicates", "outliers", "imbalance", "bias"],
        help="Specific detectors to run (default: all)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )


def _add_fix_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the fix subcommand parser."""
    parser = subparsers.add_parser(
        "fix", help="Apply automatic fixes to dataset"
    )
    parser.add_argument("file", type=str, help="Path to CSV file")
    parser.add_argument(
        "--label-column",
        "-l",
        type=str,
        default="label",
        help="Name of the label column (default: label)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path for cleaned data",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=["conservative", "aggressive", "custom"],
        default="conservative",
        help="Fix strategy (default: conservative)",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        default=True,
        help="Remove duplicate rows",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove outlier rows",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence for fixes (default: 0.8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without applying",
    )


def _add_info_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the info subcommand parser."""
    parser = subparsers.add_parser(
        "info", help="Show dataset information"
    )
    parser.add_argument("file", type=str, help="Path to CSV file")


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the benchmark subcommand parser."""
    parser = subparsers.add_parser(
        "benchmark", help="Run detection accuracy benchmarks"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for benchmark results",
    )


def _add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the serve subcommand parser."""
    parser = subparsers.add_parser(
        "serve", help="Start the REST API server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 to expose to network)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )


def _add_dashboard_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the dashboard subcommand parser."""
    parser = subparsers.add_parser(
        "dashboard", help="Start the web dashboard"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 to expose to network)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Clean - Data Quality Dashboard",
        help="Dashboard title",
    )


def _add_check_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the check subcommand parser for CI/CD integration."""
    parser = subparsers.add_parser(
        "check", help="Run quality check with pass/fail (for CI/CD)"
    )
    parser.add_argument("file", type=str, help="Path to dataset file")
    parser.add_argument(
        "--label-column",
        "-l",
        type=str,
        default="label",
        help="Name of the label column (default: label)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0,
        help="Fail if quality score is below this threshold (0-100)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for report",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Output in GitHub Actions format",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="clean",
        description="Clean: AI-powered data quality platform for ML datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clean analyze data.csv --label-column target
  clean analyze data.csv --output report.json --format json
  clean fix data.csv --output cleaned.csv --strategy conservative
  clean info data.csv
  clean benchmark --quick

For more information, visit: https://github.com/yourusername/clean
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add all subcommand parsers
    _add_analyze_parser(subparsers)
    _add_fix_parser(subparsers)
    _add_info_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_dashboard_parser(subparsers)
    _add_check_parser(subparsers)

    return parser


def _get_version() -> str:
    """Get the package version."""
    try:
        from clean import __version__

        return __version__
    except ImportError:
        return "unknown"


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run the analyze command."""
    from clean import DatasetCleaner

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Analyzing {file_path}...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    if args.label_column not in df.columns:
        print(
            f"Error: Label column '{args.label_column}' not found. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    try:
        cleaner = DatasetCleaner(data=df, label_column=args.label_column)
        report = cleaner.analyze()
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1

    # Format output
    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2, default=str)
    elif args.format == "html":
        output = report.to_html()
    else:
        output = report.summary()

    # Write or print output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output)
        if not args.quiet:
            print(f"Report saved to {output_path}")
    else:
        print(output)

    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    """Run the fix command."""
    from clean import DatasetCleaner, FixConfig, FixEngine

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    print(f"Loading {file_path}...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    if args.label_column not in df.columns:
        print(
            f"Error: Label column '{args.label_column}' not found.",
            file=sys.stderr,
        )
        return 1

    print("Analyzing dataset...")
    cleaner = DatasetCleaner(data=df, label_column=args.label_column)
    report = cleaner.analyze()

    # Configure fixes based on strategy
    if args.strategy == "aggressive":
        config = FixConfig(
            auto_relabel=True,
            label_error_threshold=0.7,
            outlier_action="remove" if args.remove_outliers else "flag",
        )
    else:  # conservative
        config = FixConfig(
            auto_relabel=False,
            label_error_threshold=args.min_confidence,
            outlier_action="flag",
        )

    # Get features (all columns except label)
    features = df.drop(columns=[args.label_column])
    labels = df[args.label_column].values

    engine = FixEngine(report=report, features=features, labels=labels, config=config)

    if args.dry_run:
        print("\n--- Dry Run ---")
        fixes = engine.suggest_fixes()
        print(f"Found {len(fixes)} suggested fixes:")
        for fix in fixes[:10]:
            print(f"  - {fix.issue_type}: index {fix.issue_index} ({fix.fix_type})")
        if len(fixes) > 10:
            print(f"  ... and {len(fixes) - 10} more")
        return 0

    print("Applying fixes...")
    fixes = engine.suggest_fixes()
    result = engine.apply_fixes(fixes)

    # Combine features and labels back into DataFrame
    cleaned_df = result.features.copy()
    if result.labels is not None:
        cleaned_df[args.label_column] = result.labels

    # Save cleaned data
    output_path = Path(args.output)
    cleaned_df.to_csv(output_path, index=False)

    print("\n--- Fix Summary ---")
    print(f"Applied: {result.n_applied} fixes")
    print(f"Skipped: {result.n_skipped} fixes")
    print(f"Errors: {result.n_errors} fixes")
    print(f"Cleaned data saved to: {output_path}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Run the info command."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    print(f"Dataset: {file_path.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn Types:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        print(f"  {col}: {dtype} ({unique_count} unique, {null_count} nulls)")

    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the benchmark command."""
    try:
        # Add benchmarks directory to path
        import sys
        from pathlib import Path

        benchmarks_dir = Path(__file__).parent.parent.parent.parent / "benchmarks"
        if benchmarks_dir.exists():
            sys.path.insert(0, str(benchmarks_dir))
            from run_benchmarks import run_all_benchmarks, run_quick_benchmark

            if args.quick:
                print("Running quick benchmark...")
                results = run_quick_benchmark()
            else:
                print("Running full benchmark suite...")
                results = run_all_benchmarks()

            output = json.dumps(results, indent=2)

            if args.output:
                Path(args.output).write_text(output)
                print(f"Results saved to {args.output}")
            else:
                print(output)

            return 0
        else:
            print("Benchmark suite not found. Running inline benchmark...")
            _run_inline_benchmark()
            return 0
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        return 1


def _run_inline_benchmark() -> None:
    """Run a simple inline benchmark if benchmark module not found."""
    import numpy as np
    from sklearn.datasets import make_classification

    from clean import DatasetCleaner

    print("Generating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5)

    # Inject some errors
    error_indices = np.random.choice(len(y), size=50, replace=False)
    y_noisy = y.copy()
    y_noisy[error_indices] = (y_noisy[error_indices] + 1) % 3

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y_noisy

    print("Running analysis...")
    cleaner = DatasetCleaner(data=df, label_column="label")
    report = cleaner.analyze()

    print("\n--- Benchmark Results ---")
    print(f"Quality Score: {report.quality_score.overall:.1f}")
    print(f"Label Errors Found: {len(report.label_errors())}")
    print(f"True Errors: {len(error_indices)}")


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the serve command."""
    try:
        from clean.api import run_server

        print(f"Starting API server on {args.host}:{args.port}...")
        print(f"API docs available at http://{args.host}:{args.port}/docs")
        run_server(host=args.host, port=args.port)
        return 0
    except ImportError as e:
        print(
            f"Error: API dependencies not installed. "
            f"Install with: pip install clean[api]\n{e}",
            file=sys.stderr,
        )
        return 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Run the dashboard command."""
    try:
        from clean.dashboard import run_dashboard

        print(f"Starting dashboard on http://{args.host}:{args.port}...")
        run_dashboard(host=args.host, port=args.port, title=args.title)
        return 0
    except ImportError as e:
        print(
            f"Error: Dashboard dependencies not installed. "
            f"Install with: pip install fastapi uvicorn python-multipart\n{e}",
            file=sys.stderr,
        )
        return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Run the check command for CI/CD integration."""
    from clean import DatasetCleaner

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    # Load data based on extension
    try:
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    if args.label_column not in df.columns:
        print(
            f"Error: Label column '{args.label_column}' not found.",
            file=sys.stderr,
        )
        return 1

    try:
        cleaner = DatasetCleaner(data=df, label_column=args.label_column)
        report = cleaner.analyze(show_progress=False)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1

    score = report.quality_score.overall
    passed = score >= args.fail_below

    # GitHub Actions output
    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"quality_score={score:.1f}\n")
                f.write(f"passed={str(passed).lower()}\n")
                f.write(f"label_errors={len(report.label_errors())}\n")
                f.write(f"duplicates={len(report.duplicates())}\n")
                f.write(f"outliers={len(report.outliers())}\n")

    # Format output
    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2, default=str)
    elif args.format == "markdown":
        output = _generate_markdown_report(report, file_path.name, passed)
    else:
        output = f"""Data Quality Check
==================
File: {file_path.name}
Quality Score: {score:.1f}/100
Threshold: {args.fail_below}
Status: {'PASSED ✓' if passed else 'FAILED ✗'}

Issues:
  Label Errors: {len(report.label_errors())}
  Duplicates: {len(report.duplicates())}
  Outliers: {len(report.outliers())}
"""

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)

    return 0 if passed else 1


def _generate_markdown_report(report, filename: str, passed: bool) -> str:
    """Generate a markdown report for CI/CD."""
    score = report.quality_score
    status = "✅ PASSED" if passed else "❌ FAILED"

    return f"""## Data Quality Report: `{filename}`

**Status: {status}**

| Metric | Score |
|--------|-------|
| Overall | {score.overall:.1f}/100 |
| Label Quality | {score.label_quality:.1f} |
| Duplicate Quality | {score.duplicate_quality:.1f} |
| Outlier Quality | {score.outlier_quality:.1f} |

### Issues Found
- Label Errors: {len(report.label_errors())}
- Duplicates: {len(report.duplicates())}
- Outliers: {len(report.outliers())}
"""


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "analyze": cmd_analyze,
        "fix": cmd_fix,
        "info": cmd_info,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
        "check": cmd_check,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
