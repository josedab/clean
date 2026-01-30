"""Tests for CLI module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.cli import cmd_analyze, cmd_check, cmd_fix, cmd_info, create_parser, main


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_created(self) -> None:
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "clean"

    def test_analyze_command(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["analyze", "test.csv"])
        assert args.command == "analyze"
        assert args.file == "test.csv"
        assert args.label_column == "label"

    def test_analyze_with_options(self) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "analyze", "test.csv",
            "--label-column", "target",
            "--output", "report.json",
            "--format", "json",
        ])
        assert args.label_column == "target"
        assert args.output == "report.json"
        assert args.format == "json"

    def test_fix_command(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["fix", "test.csv", "--output", "cleaned.csv"])
        assert args.command == "fix"
        assert args.file == "test.csv"
        assert args.output == "cleaned.csv"

    def test_fix_with_options(self) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "fix", "test.csv",
            "--output", "cleaned.csv",
            "--strategy", "aggressive",
            "--min-confidence", "0.9",
        ])
        assert args.strategy == "aggressive"
        assert args.min_confidence == 0.9

    def test_info_command(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["info", "test.csv"])
        assert args.command == "info"
        assert args.file == "test.csv"

    def test_benchmark_command(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["benchmark", "--quick"])
        assert args.command == "benchmark"
        assert args.quick is True

    def test_serve_command(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["serve", "--port", "9000"])
        assert args.command == "serve"
        assert args.port == 9000


class TestCmdInfo:
    """Tests for info command."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, np.nan],
            "feature2": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_info_success(self, sample_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
        parser = create_parser()
        args = parser.parse_args(["info", str(sample_csv)])
        result = cmd_info(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Rows: 4" in captured.out
        assert "Columns: 3" in captured.out

    def test_info_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = create_parser()
        args = parser.parse_args(["info", "/nonexistent/file.csv"])
        result = cmd_info(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: File not found" in captured.err


class TestCmdAnalyze:
    """Tests for analyze command."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1, 2], 100),
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_analyze_success(self, sample_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
        parser = create_parser()
        args = parser.parse_args(["analyze", str(sample_csv), "--quiet"])
        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Quality Score" in captured.out or "quality" in captured.out.lower()

    def test_analyze_json_output(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "report.json"
        parser = create_parser()
        args = parser.parse_args([
            "analyze", str(sample_csv),
            "--output", str(output_path),
            "--format", "json",
            "--quiet",
        ])
        result = cmd_analyze(args)

        assert result == 0
        assert output_path.exists()

        import json
        with open(output_path) as f:
            report = json.load(f)
        assert "quality_score" in report

    def test_analyze_missing_label_column(self, sample_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "analyze", str(sample_csv),
            "--label-column", "nonexistent",
        ])
        result = cmd_analyze(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err


class TestCmdFix:
    """Tests for fix command."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV with issues."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1, 2], 100),
        })
        # Add duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_fix_success(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "cleaned.csv"
        parser = create_parser()
        args = parser.parse_args([
            "fix", str(sample_csv),
            "--output", str(output_path),
        ])
        result = cmd_fix(args)

        assert result == 0
        assert output_path.exists()

        cleaned = pd.read_csv(output_path)
        original = pd.read_csv(sample_csv)
        assert len(cleaned) <= len(original)

    def test_fix_dry_run(self, sample_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        output_path = tmp_path / "cleaned.csv"
        parser = create_parser()
        args = parser.parse_args([
            "fix", str(sample_csv),
            "--output", str(output_path),
            "--dry-run",
        ])
        result = cmd_fix(args)

        assert result == 0
        assert not output_path.exists()  # Should not create file in dry run
        captured = capsys.readouterr()
        assert "Dry Run" in captured.out


class TestMain:
    """Tests for main entry point."""

    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "clean" in captured.out

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        # argparse exits with 0 for --version
        assert exc_info.value.code == 0


class TestCmdCheck:
    """Tests for check command (CI/CD integration)."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1, 2], 100),
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_check_command_parsed(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["check", "test.csv", "--fail-below", "80"])
        assert args.command == "check"
        assert args.file == "test.csv"
        assert args.fail_below == 80

    def test_check_passes_low_threshold(self, sample_csv: Path) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "check", str(sample_csv),
            "--fail-below", "0",
        ])
        result = cmd_check(args)
        assert result == 0  # Should pass with no threshold

    def test_check_fails_high_threshold(self, sample_csv: Path) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "check", str(sample_csv),
            "--fail-below", "100",
        ])
        result = cmd_check(args)
        assert result == 1  # Should fail with 100% threshold

    def test_check_json_output(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "report.json"
        parser = create_parser()
        args = parser.parse_args([
            "check", str(sample_csv),
            "--output", str(output_path),
            "--format", "json",
        ])
        result = cmd_check(args)

        assert result == 0
        assert output_path.exists()
        import json
        with open(output_path) as f:
            report = json.load(f)
        assert "quality_score" in report

    def test_check_markdown_output(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "report.md"
        parser = create_parser()
        args = parser.parse_args([
            "check", str(sample_csv),
            "--output", str(output_path),
            "--format", "markdown",
        ])
        result = cmd_check(args)

        assert result == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "Data Quality Report" in content
        assert "PASSED" in content or "FAILED" in content
