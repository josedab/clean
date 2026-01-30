# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do NOT create public GitHub issues for security vulnerabilities.**

Instead, please email security concerns to: **team@clean.dev**

Include as much detail as possible:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Timeline**: Depends on severity, typically 30-90 days

### After Reporting

1. We will acknowledge receipt of your report
2. We will investigate and validate the issue
3. We will work on a fix and coordinate disclosure
4. We will credit you in the release notes (unless you prefer anonymity)

## Security Best Practices for Users

When using Clean in production:

1. **Network Binding**: Use `--host 127.0.0.1` (default) instead of `0.0.0.0` unless you need network access
2. **API Keys**: If using the cloud service, rotate API keys regularly
3. **Data Privacy**: Be aware that Clean processes your data locally by default
4. **Dependencies**: Keep Clean and its dependencies up to date

## Scope

This security policy applies to:
- The Clean Python package (`clean-data-quality`)
- The Clean CLI tool
- The Clean REST API
- The Clean GitHub Action

Third-party dependencies are outside our direct control but we monitor for known vulnerabilities.
