# Command-Line Interface

Clean provides a command-line interface for quick data quality checks.

## Installation

The CLI is installed automatically with the package:

```bash
pip install clean-data-quality
```

## Commands

### analyze

Analyze a dataset for quality issues:

```bash
# Basic usage
clean analyze data.csv --label-column target

# Output to JSON
clean analyze data.csv -o report.json --format json

# Quiet mode (no progress output)
clean analyze data.csv -q
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--label-column` | `-l` | Label column name (default: label) |
| `--output` | `-o` | Output file path |
| `--format` | `-f` | Output format: text, json, html |
| `--detectors` | `-d` | Specific detectors to run |
| `--quiet` | `-q` | Suppress progress output |

### fix

Apply automatic fixes to a dataset:

```bash
# Conservative fixes (default)
clean fix data.csv --output cleaned.csv

# Aggressive fixes
clean fix data.csv --output cleaned.csv --strategy aggressive

# Dry run (preview changes)
clean fix data.csv --output cleaned.csv --dry-run
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file for cleaned data (required) |
| `--strategy` | `-s` | Fix strategy: conservative, aggressive |
| `--remove-duplicates` | | Remove duplicate rows (default: true) |
| `--remove-outliers` | | Remove outlier rows |
| `--min-confidence` | | Minimum confidence for fixes (0-1) |
| `--dry-run` | | Preview fixes without applying |

### info

Show dataset information:

```bash
clean info data.csv
```

Output:
```
Dataset: data.csv
Rows: 10,000
Columns: 15

Column Types:
  feature1: float64 (9850 unique, 150 nulls)
  feature2: object (25 unique, 0 nulls)
  label: int64 (3 unique, 0 nulls)

Memory Usage: 1.25 MB
```

### benchmark

Run detection accuracy benchmarks:

```bash
# Quick benchmark
clean benchmark --quick

# Full benchmark with output
clean benchmark --output results.json
```

### serve

Start the REST API server:

```bash
# Default settings
clean serve

# Custom host and port
clean serve --host 0.0.0.0 --port 9000
```

## Examples

### Typical Workflow

```bash
# 1. Check dataset info
clean info mydata.csv

# 2. Analyze for issues
clean analyze mydata.csv -l target -o report.json -f json

# 3. Preview fixes
clean fix mydata.csv -o cleaned.csv --dry-run

# 4. Apply fixes
clean fix mydata.csv -o cleaned.csv

# 5. Verify cleaned data
clean analyze cleaned.csv -l target
```

### Scripting

```bash
#!/bin/bash

# Analyze all CSV files in directory
for file in data/*.csv; do
    echo "Analyzing $file..."
    clean analyze "$file" -l label -o "reports/$(basename "$file" .csv).json" -f json -q
done
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (file not found, invalid arguments, etc.) |
