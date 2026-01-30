---
sidebar_position: 15
title: Privacy Vault
---

# Data Privacy Vault

Detect, anonymize, and protect sensitive information in your datasets.

## Why Privacy Matters

ML training data often contains PII (Personally Identifiable Information):
- Emails, phone numbers, addresses
- Social security numbers, credit cards
- Names, dates of birth

This creates compliance risks (GDPR, CCPA, HIPAA) and data breach exposure. The Privacy Vault helps you:

1. **Scan** - Find PII in your data
2. **Anonymize** - Replace PII with safe alternatives
3. **Audit** - Track all privacy operations

## Quick Start

```python
from clean.privacy import PrivacyVault

# Initialize vault
vault = PrivacyVault(
    pii_types=["email", "phone", "ssn", "name"],
)

# Scan for PII
scan = vault.scan(df)

print(f"Columns with PII: {scan.columns_with_pii}")
print(f"Total PII found: {scan.total_pii_count}")
print(f"Risk level: {scan.risk_level}")

# Anonymize
safe_df = vault.anonymize(df)
```

Example output:
```
Columns with PII: ['customer_email', 'phone', 'notes']
Total PII found: 15,234
Risk level: high

PII breakdown:
  customer_email: 5,000 emails
  phone: 4,892 phone numbers
  notes: 2,342 names, 1,000 SSNs (in free text)
```

## PII Detection

### Supported PII Types

| Type | Examples | Detection Method |
|------|----------|------------------|
| `email` | user@example.com | Regex |
| `phone` | +1-555-123-4567 | Regex + validation |
| `ssn` | 123-45-6789 | Regex + checksum |
| `credit_card` | 4111-1111-1111-1111 | Regex + Luhn |
| `name` | John Smith | NER + patterns |
| `address` | 123 Main St | NER + patterns |
| `ip_address` | 192.168.1.1 | Regex |
| `date_of_birth` | 01/15/1990 | Regex + context |

### Scanning Options

```python
vault = PrivacyVault(
    pii_types=["email", "phone", "ssn"],  # Specific types
    # or
    pii_types="all",  # All supported types
    
    confidence_threshold=0.8,  # Minimum detection confidence
)

# Full scan
scan = vault.scan(df)

# Scan specific columns
scan = vault.scan(df, columns=["notes", "description"])

# Sample scan for large datasets
scan = vault.scan(df, sample_size=10000)
```

### Scan Results

```python
scan = vault.scan(df)

# Summary
scan.columns_scanned      # 15
scan.columns_with_pii     # ['email', 'notes']
scan.total_pii_count      # 15234
scan.risk_level           # "low", "medium", "high", "critical"

# Per-column breakdown
for col, findings in scan.findings.items():
    print(f"\n{col}:")
    for pii_type, instances in findings.items():
        print(f"  {pii_type}: {len(instances)} instances")

# Export report
scan.to_html("pii_report.html")
scan.to_json("pii_report.json")
```

## Anonymization Methods

### Pseudonymization

Replace PII with consistent fake values:

```python
vault = PrivacyVault(anonymization_method="pseudonymize")
safe_df = vault.anonymize(df)

# Same input always produces same output (deterministic)
# "john.doe@email.com" → "user_7f3a9c@example.com"
# "john.doe@email.com" → "user_7f3a9c@example.com" (same)
```

### Masking

Replace characters with mask symbols:

```python
vault = PrivacyVault(
    anonymization_method="mask",
    mask_char="*",
    mask_fraction=0.6,  # Mask 60% of characters
)

# "555-123-4567" → "***-***-4567"
# "john@email.com" → "****@*****.com"
```

### Generalization

Reduce precision of values:

```python
vault = PrivacyVault(
    anonymization_method="generalize",
    generalization_rules={
        "age": {"type": "range", "width": 10},    # 25 → "20-30"
        "zip": {"type": "truncate", "keep": 3},   # 12345 → "123**"
        "salary": {"type": "bucket", "buckets": [0, 50000, 100000, 150000]},
    }
)
```

### Synthetic Replacement

Replace with realistic but fake values:

```python
vault = PrivacyVault(
    anonymization_method="synthetic",
    locale="en_US",  # Locale for generated data
)

# Generates realistic names, emails, addresses, etc.
# "John Smith" → "Michael Johnson"
# "john@gmail.com" → "sarah.wilson@email.com"
```

### Differential Privacy

Add noise for statistical privacy:

```python
vault = PrivacyVault(
    anonymization_method="differential_privacy",
    epsilon=1.0,  # Privacy budget (lower = more private)
)

# Adds calibrated noise to numeric values
# Suitable for aggregate statistics, not individual records
```

## Column-Specific Settings

```python
vault = PrivacyVault(
    column_settings={
        "email": {"method": "pseudonymize"},
        "phone": {"method": "mask", "keep_last": 4},
        "name": {"method": "synthetic"},
        "age": {"method": "generalize", "width": 5},
    }
)
```

## Encryption

### Column-Level Encryption

```python
from clean.privacy import PrivacyVault, EncryptionConfig

vault = PrivacyVault(
    encryption_config=EncryptionConfig(
        method="aes256",
        key_source="env:ENCRYPTION_KEY",  # From environment variable
    )
)

# Encrypt sensitive columns
encrypted_df = vault.encrypt(df, columns=["ssn", "credit_card"])

# Decrypt when needed
decrypted_df = vault.decrypt(encrypted_df, columns=["ssn", "credit_card"])
```

### Format-Preserving Encryption

Encrypt while maintaining format:

```python
vault = PrivacyVault(
    encryption_config=EncryptionConfig(method="fpe")
)

# SSN stays in SSN format but is encrypted
# "123-45-6789" → "847-92-3156"
```

## Audit Logging

Track all privacy operations for compliance:

```python
vault = PrivacyVault(
    audit_log_path="privacy_audit.log",
    audit_level="detailed",  # "minimal", "standard", "detailed"
)

# All operations are logged
vault.scan(df)
vault.anonymize(df)

# View audit log
for entry in vault.get_audit_log():
    print(f"{entry.timestamp}: {entry.operation}")
    print(f"  User: {entry.user}")
    print(f"  Rows affected: {entry.affected_rows}")
    print(f"  Columns: {entry.affected_columns}")
```

## Verification

Verify anonymization was successful:

```python
# Anonymize
safe_df = vault.anonymize(df)

# Verify
verification = vault.verify_anonymization(
    original=df,
    anonymized=safe_df,
)

print(f"Verification passed: {verification.success}")
print(f"PII remaining: {verification.remaining_pii_count}")

if not verification.success:
    print(f"Issues: {verification.issues}")
```

## Compliance Integration

### GDPR

```python
from clean.privacy import GDPRCompliance

gdpr = GDPRCompliance(vault)

# Assess compliance
assessment = gdpr.assess(df)
print(f"GDPR Ready: {assessment.is_compliant}")

# Generate DPIA
gdpr.generate_dpia(df, output="dpia_report.pdf")
```

### HIPAA

```python
from clean.privacy import HIPAACompliance

hipaa = HIPAACompliance(vault)

# Safe Harbor de-identification
safe_df = hipaa.safe_harbor_deidentify(df)

# Check PHI
phi_report = hipaa.scan_phi(df)
```

## Convenience Functions

```python
from clean.privacy import scan_for_pii, anonymize_dataframe

# Quick scan
findings = scan_for_pii(df, pii_types=["email", "phone"])

# Quick anonymize
safe_df = anonymize_dataframe(df, method="pseudonymize")
```

## Integration with Clean

Use privacy scanning as part of data quality:

```python
from clean import DatasetCleaner
from clean.privacy import PrivacyVault

# Scan for PII first
vault = PrivacyVault()
pii_scan = vault.scan(df)

if pii_scan.risk_level in ["high", "critical"]:
    print(f"⚠️ High PII risk detected: {pii_scan.total_pii_count} instances")
    
    # Anonymize before analysis
    safe_df = vault.anonymize(df)
else:
    safe_df = df

# Run quality analysis on safe data
cleaner = DatasetCleaner(data=safe_df, label_column="label")
report = cleaner.analyze()
```

## Best Practices

1. **Scan before sharing**: Always check for PII before distributing data
2. **Use appropriate method**: Pseudonymization for analytics, encryption for storage
3. **Enable audit logging**: Required for most compliance frameworks
4. **Test reversibility**: Ensure you can decrypt when legitimately needed
5. **Re-scan after transforms**: PII can be created through data transformations
6. **Document your process**: Maintain records of privacy procedures

## Next Steps

- [Collaborative Review](/docs/guides/collaboration) - Secure team-based review
- [Compliance Reports](/docs/guides/privacy) - Generate compliance documentation
- [API Reference](/docs/guides/privacy) - Full API documentation
