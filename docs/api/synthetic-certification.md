# Synthetic Data Certification

Quality certification for synthetic datasets.

## Quick Example

```python
from clean.synthetic_certification import SyntheticCertifier

# Certify synthetic data
certifier = SyntheticCertifier()
certificate = certifier.certify(
    real_data=real_df,
    synthetic_data=synthetic_df,
    target_column="label"
)

print(f"Certified: {certificate.is_certified}")
print(f"Overall score: {certificate.overall_score:.2f}")
print(f"Privacy score: {certificate.dimension_scores['privacy']:.2f}")
```

## API Reference

### CertificationConfig

Configuration for certification.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fidelity_threshold` | float | `0.7` | Min fidelity score required |
| `privacy_threshold` | float | `0.8` | Min privacy score required |
| `utility_threshold` | float | `0.7` | Min utility score required |
| `diversity_threshold` | float | `0.6` | Min diversity score required |
| `coherence_threshold` | float | `0.7` | Min coherence score required |
| `max_memorization_rate` | float | `0.01` | Max memorization rate |
| `min_nearest_neighbor_distance` | float | `0.1` | Min distance to real samples |
| `max_reidentification_risk` | float | `0.05` | Max re-identification risk |
| `fidelity_weight` | float | `0.25` | Weight in overall score |
| `privacy_weight` | float | `0.30` | Weight in overall score |
| `utility_weight` | float | `0.20` | Weight in overall score |
| `diversity_weight` | float | `0.15` | Weight in overall score |
| `coherence_weight` | float | `0.10` | Weight in overall score |
| `certificate_validity_days` | int | `90` | Certificate validity period |

### QualityDimension (Enum)

Certification quality dimensions.

| Value | Description |
|-------|-------------|
| `FIDELITY` | Statistical similarity to real data |
| `PRIVACY` | Privacy preservation (no memorization) |
| `UTILITY` | Usefulness for downstream tasks |
| `DIVERSITY` | Coverage of real data distribution |
| `COHERENCE` | Internal consistency of synthetic data |

### CertificationStatus (Enum)

Certification status values.

| Value | Description |
|-------|-------------|
| `CERTIFIED` | Passed all thresholds |
| `CONDITIONAL` | Passed with conditions/warnings |
| `FAILED` | Did not meet requirements |

### SyntheticCertifier

Main certifier class.

#### `__init__(config: CertificationConfig | None = None)`

Initialize with optional configuration.

#### `certify(real_data, synthetic_data, target_column=None) -> QualityCertificate`

Certify synthetic data against real data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_data` | `pd.DataFrame` | Original real dataset |
| `synthetic_data` | `pd.DataFrame` | Generated synthetic dataset |
| `target_column` | `str \| None` | Target/label column |

### QualityCertificate

Certification result dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `certificate_id` | str | Unique certificate ID |
| `issued_at` | datetime | Issue timestamp |
| `expires_at` | datetime | Expiration timestamp |
| `status` | CertificationStatus | Certification status |
| `is_certified` | bool | Quick check for certification |
| `overall_score` | float | Weighted overall score |
| `dimension_scores` | dict[str, float] | Scores by dimension |
| `privacy_assessment` | dict | Detailed privacy assessment |
| `real_data_hash` | str | Hash of real data |
| `synthetic_data_hash` | str | Hash of synthetic data |
| `n_real_samples` | int | Real sample count |
| `n_synthetic_samples` | int | Synthetic sample count |
| `audit_log` | list | Certification audit trail |
| `conditions` | list[str] | Conditions for conditional cert |
| `warnings` | list[str] | Certification warnings |

#### `to_dict() -> dict`

Convert certificate to dictionary.

## Example Workflows

### Basic Certification

```python
from clean.synthetic_certification import SyntheticCertifier

certifier = SyntheticCertifier()
certificate = certifier.certify(real_df, synthetic_df)

if certificate.is_certified:
    print(f"✓ Data certified (ID: {certificate.certificate_id})")
    print(f"  Expires: {certificate.expires_at}")
else:
    print("✗ Certification failed")
    for warning in certificate.warnings:
        print(f"  - {warning}")
```

### Custom Thresholds

```python
from clean.synthetic_certification import SyntheticCertifier, CertificationConfig

config = CertificationConfig(
    privacy_threshold=0.9,  # Stricter privacy
    fidelity_threshold=0.8,
    max_memorization_rate=0.005
)

certifier = SyntheticCertifier(config=config)
certificate = certifier.certify(real_df, synthetic_df)
```

### Detailed Assessment

```python
certificate = certifier.certify(real_df, synthetic_df, target_column="label")

print("Dimension Scores:")
for dim, score in certificate.dimension_scores.items():
    status = "✓" if score >= 0.7 else "✗"
    print(f"  {status} {dim}: {score:.2f}")

print(f"\nPrivacy Assessment:")
for key, value in certificate.privacy_assessment.items():
    print(f"  {key}: {value}")
```

### CI/CD Integration

```python
from clean.synthetic_certification import SyntheticCertifier, CertificationConfig

# Strict configuration for production
config = CertificationConfig(
    privacy_threshold=0.9,
    fidelity_threshold=0.8,
    utility_threshold=0.8
)

certifier = SyntheticCertifier(config=config)
certificate = certifier.certify(real_df, synthetic_df)

# Fail if not certified
assert certificate.is_certified, \
    f"Synthetic data failed certification: {certificate.warnings}"

# Save certificate for audit
import json
with open("certificate.json", "w") as f:
    json.dump(certificate.to_dict(), f, default=str)
```

### Monitoring Over Time

```python
from clean.synthetic_certification import SyntheticCertifier
import json
from datetime import datetime

certifier = SyntheticCertifier()

# Track certificates over time
certificates = []
for version, synthetic_df in synthetic_versions.items():
    cert = certifier.certify(real_df, synthetic_df)
    certificates.append({
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "certified": cert.is_certified,
        "overall_score": cert.overall_score,
        "dimension_scores": cert.dimension_scores
    })

# Analyze trends
scores_over_time = [c["overall_score"] for c in certificates]
```

### Enterprise Compliance

```python
from clean.synthetic_certification import SyntheticCertifier, CertificationConfig

# GDPR-focused configuration
config = CertificationConfig(
    privacy_threshold=0.95,
    max_memorization_rate=0.001,
    max_reidentification_risk=0.01,
    certificate_validity_days=30  # Short validity for compliance
)

certifier = SyntheticCertifier(config=config)
certificate = certifier.certify(
    real_df, 
    synthetic_df,
    target_column="outcome"
)

# Generate compliance report
compliance_report = {
    "certificate_id": certificate.certificate_id,
    "privacy_score": certificate.dimension_scores.get("privacy", 0),
    "memorization_risk": certificate.privacy_assessment.get("memorization_rate", "N/A"),
    "reidentification_risk": certificate.privacy_assessment.get("reidentification_risk", "N/A"),
    "valid_until": certificate.expires_at.isoformat(),
    "audit_log": certificate.audit_log
}
```
