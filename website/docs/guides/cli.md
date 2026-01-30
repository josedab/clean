---
sidebar_position: 5
title: CLI
---

# Command-Line Interface

Use Clean from the terminal for quick analysis and scripting.

## Installation

The CLI is included with the package:

```bash
pip install clean-data-quality
```

Verify installation:

```bash
clean --version
```

## Commands

### analyze

Analyze a dataset for quality issues:

```bash
clean analyze data.csv --label-column target
```

Output:
```
Data Quality Report
==================
Samples analyzed: 10,000
Quality Score: 82.5/100

Issues Found:
  - Label errors: 347 (3.5%)
  - Duplicates: 234 pairs
  - Outliers: 156 (1.6%)
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--label-column` | `-l` | Label column name (required) |
| `--output` | `-o` | Output file path |
| `--format` | `-f` | Output format: text, json, html |
| `--detectors` | `-d` | Specific detectors to run |
| `--quiet` | `-q` | Suppress progress output |

### fix

Apply automatic fixes:

```bash
clean fix data.csv --output cleaned.csv
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--label-column` | `-l` | Label column name |
| `--output` | `-o` | Output file (required) |
| `--strategy` | `-s` | Fix strategy: conservative, aggressive |
| `--dry-run` | | Preview without applying |

### info

Show dataset information:

```bash
clean info data.csv
```

### serve

Start the REST API server:

```bash
clean serve --port 8000
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error |
