"""Tests for synthetic_certification module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestSyntheticCertificationModule:
    """Tests for synthetic data certification module."""

    def test_imports(self) -> None:
        from clean.synthetic_certification import (
            SyntheticCertifier,
            QualityCertificate,
            QualityDimension,
            CertificationConfig,
        )
        assert SyntheticCertifier is not None
        assert QualityCertificate is not None
        assert QualityDimension is not None
        assert CertificationConfig is not None

    def test_quality_dimensions(self) -> None:
        from clean.synthetic_certification import QualityDimension
        
        assert QualityDimension.FIDELITY is not None
        assert QualityDimension.PRIVACY is not None
        assert QualityDimension.UTILITY is not None
        assert QualityDimension.DIVERSITY is not None
        assert QualityDimension.COHERENCE is not None

    def test_config_defaults(self) -> None:
        from clean.synthetic_certification import CertificationConfig
        
        config = CertificationConfig()
        assert config.fidelity_threshold > 0
        assert config.privacy_threshold > 0
        assert config.utility_threshold > 0
        assert config.diversity_threshold > 0
        assert config.coherence_threshold > 0

    def test_certifier_init(self) -> None:
        from clean.synthetic_certification import SyntheticCertifier, CertificationConfig
        
        certifier = SyntheticCertifier()
        assert certifier is not None
        
        config = CertificationConfig(fidelity_threshold=0.9)
        certifier_with_config = SyntheticCertifier(config=config)
        assert certifier_with_config is not None

    def test_certify_basic(self) -> None:
        from clean.synthetic_certification import SyntheticCertifier, QualityCertificate
        
        np.random.seed(42)
        
        # Create real data
        real_data = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })
        
        # Create synthetic data (similar distribution)
        synthetic_data = pd.DataFrame({
            "feature_0": np.random.randn(100) + 0.1,  # Slight shift
            "feature_1": np.random.randn(100) + 0.1,
            "label": np.random.choice([0, 1], 100),
        })
        
        certifier = SyntheticCertifier()
        certificate = certifier.certify(real_data, synthetic_data, target_column="label")
        
        assert isinstance(certificate, QualityCertificate)
        assert certificate.certificate_id is not None
        assert certificate.issued_at is not None
        assert certificate.overall_score >= 0
        assert len(certificate.dimension_scores) > 0

    def test_certificate_fields(self) -> None:
        from clean.synthetic_certification import SyntheticCertifier
        
        np.random.seed(42)
        
        real_data = pd.DataFrame({
            "feature_0": np.random.randn(50),
            "label": np.random.choice([0, 1], 50),
        })
        
        synthetic_data = pd.DataFrame({
            "feature_0": np.random.randn(50),
            "label": np.random.choice([0, 1], 50),
        })
        
        certifier = SyntheticCertifier()
        cert = certifier.certify(real_data, synthetic_data, target_column="label")
        
        # Check that basic fields exist
        assert cert.certificate_id is not None
        assert cert.is_certified is not None
        assert cert.overall_score >= 0
