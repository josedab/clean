# CI/CD Integration

Automate data quality checks in your CI/CD pipelines.

## Overview

Clean provides both a GitHub Action and CLI command for integrating data quality gates into automated workflows. Fail builds when data quality drops below thresholds.

## GitHub Action

### Basic Usage

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on:
  push:
    paths:
      - 'data/**'
  pull_request:
    paths:
      - 'data/**'

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: clean-data/clean-action@v1
        with:
          file: data/training.csv
          label-column: label
          fail-below: 80
```

### Action Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `file` | Yes | - | Path to data file (CSV, Parquet, JSON) |
| `label-column` | Yes | - | Name of the label column |
| `fail-below` | No | `70` | Minimum quality score (0-100) |
| `detectors` | No | `all` | Comma-separated list of detectors |
| `output-format` | No | `text` | Output format: text, json, markdown |

### Action Outputs

| Output | Description |
|--------|-------------|
| `quality_score` | Overall quality score (0-100) |
| `passed` | Whether check passed (`true`/`false`) |
| `label_errors` | Number of label errors found |
| `duplicates` | Number of duplicates found |
| `outliers` | Number of outliers found |

### Using Outputs

```yaml
jobs:
  quality-check:
    runs-on: ubuntu-latest
    outputs:
      quality_score: ${{ steps.check.outputs.quality_score }}
      passed: ${{ steps.check.outputs.passed }}
    steps:
      - uses: actions/checkout@v4
      
      - id: check
        uses: clean-data/clean-action@v1
        with:
          file: data/training.csv
          label-column: label
          fail-below: 80
      
      - name: Report Results
        run: |
          echo "Quality Score: ${{ steps.check.outputs.quality_score }}"
          echo "Passed: ${{ steps.check.outputs.passed }}"
```

### Matrix Testing

```yaml
jobs:
  quality-check:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        dataset:
          - data/train.csv
          - data/validation.csv
          - data/test.csv
    steps:
      - uses: actions/checkout@v4
      
      - uses: clean-data/clean-action@v1
        with:
          file: ${{ matrix.dataset }}
          label-column: label
          fail-below: 80
```

## CLI Command

### Basic Usage

```bash
# Run quality check
clean check data.csv --label-column label --fail-below 80

# Exit codes:
# 0 = passed
# 1 = failed (below threshold)
# 2 = error
```

### Options

```bash
clean check --help

Usage: clean check [OPTIONS] FILE

Options:
  -l, --label-column TEXT   Name of the label column (required)
  --fail-below FLOAT        Minimum quality score to pass (default: 70)
  -o, --output TEXT         Output file for report
  -f, --format TEXT         Output format: text, json, markdown
  --github-output           Output in GitHub Actions format
  --help                    Show this message and exit
```

### GitHub Actions Format

```bash
# Enable GitHub Actions output
clean check data.csv -l label --github-output

# This writes to $GITHUB_OUTPUT:
# quality_score=85.2
# passed=true
# label_errors=23
# duplicates=45
# outliers=12
```

### JSON Output

```bash
clean check data.csv -l label --format json --output report.json
```

```json
{
  "quality_score": {
    "overall": 85.2,
    "label_quality": 92.1,
    "duplicate_quality": 78.3,
    "outlier_quality": 88.5
  },
  "issues": {
    "label_errors": 23,
    "duplicates": 45,
    "outliers": 12
  },
  "passed": true,
  "threshold": 80
}
```

### Markdown Output

```bash
clean check data.csv -l label --format markdown
```

```markdown
# Data Quality Report

**File:** data.csv
**Quality Score:** 85.2/100
**Status:** ✓ PASSED

| Metric | Score |
|--------|-------|
| Overall | 85.2/100 |
| Label Quality | 92.1 |
| Duplicate Quality | 78.3 |
| Outlier Quality | 88.5 |

### Issues Found
- Label Errors: 23
- Duplicates: 45
- Outliers: 12
```

## Integration Examples

### GitLab CI

```yaml
# .gitlab-ci.yml
data-quality:
  image: python:3.10
  stage: test
  script:
    - pip install clean-data-quality
    - clean check data/training.csv -l label --fail-below 80
  rules:
    - changes:
        - data/**/*
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Data Quality') {
            steps {
                sh 'pip install clean-data-quality'
                sh 'clean check data/training.csv -l label --fail-below 80'
            }
        }
    }
}
```

### CircleCI

```yaml
# .circleci/config.yml
version: 2.1
jobs:
  data-quality:
    docker:
      - image: python:3.10
    steps:
      - checkout
      - run:
          name: Install Clean
          command: pip install clean-data-quality
      - run:
          name: Check Data Quality
          command: clean check data/training.csv -l label --fail-below 80
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: data-quality
        name: Data Quality Check
        entry: clean check
        args: ['--label-column', 'label', '--fail-below', '80']
        language: system
        files: '^data/.*\.csv$'
        pass_filenames: true
```

## Best Practices

1. **Set appropriate thresholds**: Start lenient (70) and tighten over time
2. **Check on data changes**: Trigger only when data files change
3. **Use matrix testing**: Check all datasets (train/val/test)
4. **Archive reports**: Save JSON/Markdown reports as artifacts
5. **Block merges**: Require quality checks to pass before merge
6. **Monitor trends**: Track quality scores over time

## Troubleshooting

### Exit Code 1: Quality Below Threshold

```bash
Error: Quality score 65.2 is below threshold 80
```

**Solution:** Review the report, fix issues, or adjust threshold.

### Exit Code 2: Analysis Error

```bash
Error: Label column 'target' not found
```

**Solution:** Check column name matches your data.

### Timeout Issues

For large files, increase timeout:

```yaml
- uses: clean-data/clean-action@v1
  timeout-minutes: 30
  with:
    file: large_data.csv
```
