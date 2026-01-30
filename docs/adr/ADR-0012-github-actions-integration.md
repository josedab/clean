# ADR-0012: GitHub Actions Integration for CI/CD Quality Gates

## Status

Accepted

## Context

Data quality should be enforced continuously, not just during manual analysis. Modern ML pipelines use CI/CD for:

- Model training triggered by data updates
- Automated testing of data transformations
- Quality gates before production deployment
- Pull request checks for dataset changes

Teams wanted to:
1. Fail builds when data quality drops below threshold
2. See quality reports in pull request comments
3. Track quality metrics over time
4. Integrate with existing GitHub workflows

Options considered:

1. **Documentation only**: "Run `clean check` in your CI" - requires user setup
2. **Docker image**: Publish image, users reference in workflows - flexible but verbose
3. **GitHub Action**: First-class integration, simple YAML - chosen approach
4. **GitHub App**: More powerful but complex approval process

## Decision

We created a **custom GitHub Action** (composite action) that wraps the Clean CLI.

```yaml
# action/action.yml
name: 'Clean Data Quality Check'
description: 'AI-powered data quality analysis for ML datasets'

inputs:
  file:
    description: 'Path to the dataset file'
    required: true
  label-column:
    description: 'Name of the label column'
    default: 'label'
  fail-below:
    description: 'Fail if quality score is below this (0-100)'
    default: '0'
  detectors:
    description: 'Comma-separated detectors to run'
    default: 'all'
  output-format:
    description: 'Output format (text, json, markdown)'
    default: 'markdown'

outputs:
  quality-score:
    description: 'Overall data quality score (0-100)'
  label-errors:
    description: 'Number of label errors detected'
  passed:
    description: 'Whether the quality check passed'

runs:
  using: 'composite'
  steps:
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}

    - name: Install Clean
      shell: bash
      run: pip install clean-data-quality

    - name: Run Analysis
      id: analyze
      shell: bash
      run: python ${{ github.action_path }}/run_check.py

    - name: Add PR Comment
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          // Post quality report as PR comment
          await github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: context.issue.number,
            body: `## ${passed ? '✅' : '❌'} Data Quality Report\n\n${report}`
          });
```

Users add to their workflows:

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check
on:
  push:
    paths: ['data/**']
  pull_request:
    paths: ['data/**']

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: clean-data/clean-action@v1
        with:
          file: data/training.csv
          label-column: target
          fail-below: 80
```

The `run_check.py` script handles:
- Loading the dataset
- Running configured detectors
- Formatting output for GitHub
- Setting exit code based on threshold
- Writing outputs for subsequent steps

## Consequences

### Positive

- **Zero config for users**: Single YAML block enables quality gates
- **PR integration**: Quality reports appear as comments automatically
- **Threshold enforcement**: Builds fail when quality drops
- **GitHub-native**: Works with existing GitHub workflows
- **Marketplace presence**: Discoverable in GitHub Actions marketplace

### Negative

- **GitHub lock-in**: Only works with GitHub Actions (not GitLab CI, Jenkins)
- **Maintenance burden**: Action versioning separate from library
- **Limited customization**: Complex analysis may need direct CLI usage
- **Cold start**: Python install + pip install adds ~30-60 seconds

### Neutral

- **Composite action**: Uses bash steps rather than JavaScript/Docker
- **Version coupling**: Action version should match library version

## CLI Integration

The action uses the CLI's `check` command:

```bash
# CLI command that powers the action
clean check data.csv \
  --label-column target \
  --fail-below 80 \
  --format markdown \
  --github-output
```

The `--github-output` flag formats output for GitHub Actions:

```bash
echo "quality_score=85.5" >> $GITHUB_OUTPUT
echo "passed=true" >> $GITHUB_OUTPUT
```

## Example PR Comment

When the action runs on a pull request:

```markdown
## ✅ Data Quality Report

**Quality Score: 85.5/100**

| Issue Type | Count | Severity |
|------------|-------|----------|
| Label Errors | 23 | ⚠️ Medium |
| Duplicates | 5 | 🔵 Low |
| Outliers | 12 | 🔵 Low |

<details>
<summary>Top Label Errors</summary>

| Index | Given | Predicted | Confidence |
|-------|-------|-----------|------------|
| 42 | cat | dog | 0.94 |
| 187 | bird | cat | 0.91 |
...
</details>
```

## Related Decisions

- ADR-0004 (Facade Pattern): Action uses `DatasetCleaner` internally
- ADR-0008 (FastAPI): Alternative integration for non-GitHub environments
