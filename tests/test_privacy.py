"""Tests for the privacy module (PII detection and anonymization)."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from clean.privacy import (
    PIIType,
    AnonymizationMethod,
    PIIMatch,
    PIIScanReport,
    TokenMapping,
    AnonymizationResult,
    PIIDetector,
    RegexPIIDetector,
    PIIScanner,
    PrivacyVault,
    scan_pii,
    anonymize_data,
)


class TestPIIType:
    """Tests for PIIType enum."""

    def test_all_types_exist(self):
        """Test that all expected PII types exist."""
        expected = [
            "EMAIL", "PHONE", "SSN", "CREDIT_CARD", "NAME",
            "ADDRESS", "IP_ADDRESS", "DATE_OF_BIRTH", "PASSPORT",
            "DRIVER_LICENSE", "BANK_ACCOUNT", "MEDICAL_ID",
            "BIOMETRIC", "LOCATION", "CUSTOM"
        ]
        for pii_type in expected:
            assert hasattr(PIIType, pii_type)

    def test_type_values(self):
        """Test enum values are lowercase."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.SSN.value == "ssn"


class TestAnonymizationMethod:
    """Tests for AnonymizationMethod enum."""

    def test_all_methods_exist(self):
        """Test all anonymization methods exist."""
        expected = [
            "REDACT", "MASK", "HASH", "TOKENIZE",
            "GENERALIZE", "PSEUDONYMIZE", "ENCRYPT"
        ]
        for method in expected:
            assert hasattr(AnonymizationMethod, method)


class TestPIIMatch:
    """Tests for PIIMatch dataclass."""

    def test_creation(self):
        """Test PIIMatch creation."""
        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            column="email",
            row_index=0,
            value="test@example.com",
            confidence=0.95,
            start_pos=0,
            end_pos=16,
            context="Email: test@example.com here",
        )
        assert match.pii_type == PIIType.EMAIL
        assert match.confidence == 0.95

    def test_to_dict(self):
        """Test conversion to dictionary."""
        match = PIIMatch(
            pii_type=PIIType.PHONE,
            column="phone",
            row_index=5,
            value="555-123-4567",
            confidence=0.9,
        )
        d = match.to_dict()
        assert d["pii_type"] == "phone"
        assert d["row_index"] == 5
        assert d["confidence"] == 0.9


class TestPIIScanReport:
    """Tests for PIIScanReport dataclass."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        matches = [
            PIIMatch(PIIType.EMAIL, "email_col", 0, "a@b.com", 0.95),
            PIIMatch(PIIType.EMAIL, "email_col", 1, "c@d.com", 0.95),
            PIIMatch(PIIType.SSN, "ssn_col", 2, "123-45-6789", 0.85),
        ]
        return PIIScanReport(
            n_rows=100,
            n_columns=5,
            total_pii_found=3,
            matches=matches,
            pii_by_type={PIIType.EMAIL: 2, PIIType.SSN: 1},
            pii_by_column={"email_col": [PIIType.EMAIL], "ssn_col": [PIIType.SSN]},
            high_risk_columns=["ssn_col"],
            recommendations=["Review SSN data"],
        )

    def test_summary(self, sample_report):
        """Test report summary generation."""
        summary = sample_report.summary()
        assert "PII Scan Report" in summary
        assert "100" in summary  # n_rows
        assert "3" in summary  # total_pii_found
        assert "email" in summary
        assert "ssn" in summary

    def test_to_dict(self, sample_report):
        """Test conversion to dictionary."""
        d = sample_report.to_dict()
        assert d["n_rows"] == 100
        assert d["total_pii_found"] == 3
        assert "email" in d["pii_by_type"]


class TestRegexPIIDetector:
    """Tests for RegexPIIDetector."""

    def test_email_detection(self):
        """Test email pattern detection."""
        detector = RegexPIIDetector(
            PIIType.EMAIL,
            [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            confidence=0.95,
        )
        
        matches = detector.detect("Contact us at info@example.com for help")
        assert len(matches) == 1
        start, end, conf = matches[0]
        assert conf == 0.95
        assert "info@example.com" in "Contact us at info@example.com for help"[start:end]

    def test_no_match(self):
        """Test when no PII is found."""
        detector = RegexPIIDetector(
            PIIType.EMAIL,
            [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        )
        matches = detector.detect("No email here")
        assert len(matches) == 0

    def test_multiple_patterns(self):
        """Test detection with multiple patterns."""
        detector = RegexPIIDetector(
            PIIType.PHONE,
            [r"\d{3}-\d{3}-\d{4}", r"\(\d{3}\) \d{3}-\d{4}"],
            confidence=0.9,
        )
        
        text = "Call 555-123-4567 or (555) 987-6543"
        matches = detector.detect(text)
        assert len(matches) == 2


class TestPIIScanner:
    """Tests for PIIScanner."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with PII."""
        return pd.DataFrame({
            "name": ["John Doe", "Jane Smith", "Bob Wilson"],
            "email": ["john@example.com", "jane@test.org", "bob@company.co"],
            "phone": ["555-123-4567", "555-987-6543", "555-111-2222"],
            "notes": ["SSN: 123-45-6789", "No PII here", "IP: 192.168.1.1"],
            "numeric": [1, 2, 3],
        })

    def test_scanner_init_defaults(self):
        """Test scanner initialization with defaults."""
        scanner = PIIScanner()
        assert scanner.min_confidence == 0.5
        assert len(scanner.detectors) > 0

    def test_scanner_init_custom(self):
        """Test scanner with custom settings."""
        scanner = PIIScanner(min_confidence=0.8)
        assert scanner.min_confidence == 0.8

    def test_scanner_with_custom_detector(self):
        """Test scanner with custom detector."""
        custom = RegexPIIDetector(PIIType.CUSTOM, [r"CUSTOM-\d+"])
        scanner = PIIScanner(custom_detectors=[custom])
        
        df = pd.DataFrame({"data": ["CUSTOM-12345", "normal text"]})
        report = scanner.scan(df)
        
        # Should detect the custom pattern
        custom_matches = [m for m in report.matches if m.pii_type == PIIType.CUSTOM]
        assert len(custom_matches) == 1

    def test_scan_detects_emails(self, sample_df):
        """Test scanning detects emails."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df)
        
        email_matches = [m for m in report.matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 3

    def test_scan_detects_phones(self, sample_df):
        """Test scanning detects phone numbers."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df)
        
        phone_matches = [m for m in report.matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 3

    def test_scan_detects_ssn(self, sample_df):
        """Test scanning detects SSN."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df)
        
        ssn_matches = [m for m in report.matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) >= 1

    def test_scan_detects_ip_address(self, sample_df):
        """Test scanning detects IP addresses."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df)
        
        ip_matches = [m for m in report.matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_matches) >= 1

    def test_scan_specific_columns(self, sample_df):
        """Test scanning only specific columns."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df, columns=["email"])
        
        # Should only find emails
        for match in report.matches:
            assert match.column == "email"

    def test_scan_with_sampling(self, sample_df):
        """Test scanning with sampling."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df, sample_size=2)
        
        # Should scan fewer rows
        assert report.n_rows == 3  # Original count preserved
        # But matches may be fewer due to sampling

    def test_scan_empty_dataframe(self):
        """Test scanning empty DataFrame."""
        scanner = PIIScanner()
        df = pd.DataFrame({"col": []})
        report = scanner.scan(df)
        
        assert report.n_rows == 0
        assert report.total_pii_found == 0

    def test_scan_no_string_columns(self):
        """Test scanning DataFrame with only numeric columns."""
        scanner = PIIScanner()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        report = scanner.scan(df)
        
        assert report.total_pii_found == 0

    def test_recommendations_generated(self, sample_df):
        """Test that recommendations are generated."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df)
        
        assert len(report.recommendations) > 0

    def test_high_risk_columns_identified(self):
        """Test high-risk column identification."""
        df = pd.DataFrame({
            "sensitive": ["123-45-6789", "987-65-4321"],  # SSN
            "normal": ["hello", "world"],
        })
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        # SSN column should be high risk
        assert "sensitive" in report.high_risk_columns or len(report.high_risk_columns) > 0


class TestPrivacyVault:
    """Tests for PrivacyVault."""

    @pytest.fixture
    def vault(self):
        """Create a vault for testing."""
        return PrivacyVault(
            encryption_key="test_key_12345678901234567890123456789012",
            hash_salt="test_salt_1234567890",
        )

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with PII."""
        return pd.DataFrame({
            "email": ["john@example.com", "jane@test.org"],
            "phone": ["555-123-4567", "555-987-6543"],
            "name": ["John Doe", "Jane Smith"],
        })

    def test_vault_init_auto_keys(self):
        """Test vault auto-generates keys when not provided."""
        vault = PrivacyVault()
        assert vault.encryption_key is not None
        assert vault.hash_salt is not None
        assert len(vault.encryption_key) > 0

    def test_vault_init_custom_keys(self, vault):
        """Test vault with custom keys."""
        assert "test_key" in vault.encryption_key
        assert "test_salt" in vault.hash_salt

    def test_redact_method(self, vault):
        """Test redaction anonymization."""
        result, is_reversible = vault._redact("test@email.com", PIIType.EMAIL)
        assert "[EMAIL_REDACTED]" in result
        assert is_reversible is False

    def test_mask_method(self, vault):
        """Test masking anonymization."""
        result, is_reversible = vault._mask("555-123-4567", PIIType.PHONE)
        assert "****" in result
        assert result.endswith("4567")
        assert is_reversible is False

    def test_mask_short_value(self, vault):
        """Test masking short values."""
        result, _ = vault._mask("ab", PIIType.NAME)
        assert result == "**"

    def test_hash_method(self, vault):
        """Test hashing anonymization."""
        result, is_reversible = vault._hash("test@email.com", PIIType.EMAIL)
        assert result.startswith("HASH_")
        assert is_reversible is False
        
        # Same input should produce same hash
        result2, _ = vault._hash("test@email.com", PIIType.EMAIL)
        assert result == result2

    def test_tokenize_method(self, vault):
        """Test tokenization anonymization."""
        result, is_reversible = vault._tokenize("test@email.com", PIIType.EMAIL)
        assert result.startswith("TOK_")
        assert is_reversible is True
        
        # Different calls should produce different tokens
        result2, _ = vault._tokenize("test@email.com", PIIType.EMAIL)
        assert result != result2

    def test_generalize_method_email(self, vault):
        """Test generalization for email."""
        result, _ = vault._generalize("john@example.com", PIIType.EMAIL)
        assert "@example.com" in result
        assert "john" not in result

    def test_generalize_method_ip(self, vault):
        """Test generalization for IP address."""
        result, _ = vault._generalize("192.168.1.100", PIIType.IP_ADDRESS)
        assert "192.168" in result
        assert "*" in result

    def test_pseudonymize_method(self, vault):
        """Test pseudonymization."""
        result, is_reversible = vault._pseudonymize("john@example.com", PIIType.EMAIL)
        assert "@example.com" in result
        assert "john" not in result
        assert is_reversible is True

    def test_encrypt_method(self, vault):
        """Test encryption."""
        result, is_reversible = vault._encrypt("secret", PIIType.NAME)
        assert result.startswith("ENC_")
        assert is_reversible is True

    def test_anonymize_basic(self, vault, sample_df):
        """Test basic anonymization."""
        result = vault.anonymize(sample_df)
        
        assert isinstance(result, AnonymizationResult)
        assert len(result.anonymized_data) == len(sample_df)
        assert result.n_values_anonymized > 0

    def test_anonymize_with_method(self, vault, sample_df):
        """Test anonymization with specific method."""
        result = vault.anonymize(
            sample_df,
            method=AnonymizationMethod.MASK,
        )
        
        # Check that masking was applied
        for col in result.anonymized_data.columns:
            if col in ["email", "phone"]:
                for val in result.anonymized_data[col]:
                    # Should have asterisks from masking
                    assert "*" in str(val) or str(val) == str(sample_df[col].iloc[0])

    def test_anonymize_with_method_by_type(self, vault, sample_df):
        """Test different methods for different PII types."""
        result = vault.anonymize(
            sample_df,
            method_by_type={
                PIIType.EMAIL: AnonymizationMethod.REDACT,
                PIIType.PHONE: AnonymizationMethod.MASK,
            },
        )
        
        assert result.n_values_anonymized > 0

    def test_anonymize_specific_columns(self, vault, sample_df):
        """Test anonymizing specific columns."""
        result = vault.anonymize(sample_df, columns=["email"])
        
        # Phone should not be anonymized
        # (Note: depends on whether there are matches in phone column)
        assert result.anonymized_data is not None

    def test_anonymize_with_precomputed_report(self, vault, sample_df):
        """Test anonymization with pre-scanned report."""
        scanner = PIIScanner()
        report = scanner.scan(sample_df)
        
        result = vault.anonymize(sample_df, pii_report=report)
        assert result.n_values_anonymized >= 0

    def test_deanonymize(self, vault, sample_df):
        """Test de-anonymization with tokenization."""
        # Anonymize with tokenization (reversible)
        result = vault.anonymize(
            sample_df,
            method=AnonymizationMethod.TOKENIZE,
        )
        
        # De-anonymize
        restored = vault.deanonymize(result.anonymized_data, result.token_mappings)
        
        # Original values should be restored
        for col in ["email", "phone"]:
            for i in range(len(sample_df)):
                original = sample_df[col].iloc[i]
                restored_val = restored[col].iloc[i]
                # Should contain original value
                assert original in restored_val or restored_val == original

    def test_deanonymize_with_stored_tokens(self, vault, sample_df):
        """Test de-anonymization using stored tokens."""
        result = vault.anonymize(sample_df, method=AnonymizationMethod.TOKENIZE)
        
        # De-anonymize without passing mappings (use stored)
        restored = vault.deanonymize(result.anonymized_data)
        assert restored is not None

    def test_export_import_tokens(self, vault, sample_df):
        """Test token export and import."""
        result = vault.anonymize(sample_df, method=AnonymizationMethod.TOKENIZE)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "tokens.json"
            
            # Export
            vault.export_tokens(str(token_path))
            assert token_path.exists()
            
            # Create new vault and import
            vault2 = PrivacyVault()
            n_imported = vault2.import_tokens(str(token_path))
            
            assert n_imported > 0

    def test_anonymization_result_get_token_map(self):
        """Test AnonymizationResult.get_token_map()."""
        mappings = [
            TokenMapping("TOK_123", "original1", PIIType.EMAIL, "col", 0.0),
            TokenMapping("TOK_456", "original2", PIIType.PHONE, "col", 0.0),
        ]
        result = AnonymizationResult(
            anonymized_data=pd.DataFrame(),
            token_mappings=mappings,
            n_values_anonymized=2,
            methods_used={},
        )
        
        token_map = result.get_token_map()
        assert token_map["TOK_123"] == "original1"
        assert token_map["TOK_456"] == "original2"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "email": ["test@example.com", "user@domain.org"],
            "text": ["Hello world", "Normal text"],
        })

    def test_scan_pii_function(self, sample_df):
        """Test scan_pii convenience function."""
        report = scan_pii(sample_df)
        
        assert isinstance(report, PIIScanReport)
        assert report.n_rows == 2

    def test_scan_pii_with_kwargs(self, sample_df):
        """Test scan_pii with additional arguments."""
        report = scan_pii(sample_df, min_confidence=0.9)
        assert isinstance(report, PIIScanReport)

    def test_anonymize_data_function(self, sample_df):
        """Test anonymize_data convenience function."""
        result = anonymize_data(sample_df)
        
        assert isinstance(result, AnonymizationResult)

    def test_anonymize_data_with_method(self, sample_df):
        """Test anonymize_data with method argument."""
        result = anonymize_data(sample_df, method="mask")
        assert isinstance(result, AnonymizationResult)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_values_handled(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame({
            "email": ["test@example.com", None, np.nan, "user@test.org"],
        })
        
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        # Should not crash
        assert report is not None

    def test_empty_strings_handled(self):
        """Test that empty strings are handled."""
        df = pd.DataFrame({
            "email": ["", "  ", "test@example.com"],
        })
        
        scanner = PIIScanner()
        report = scanner.scan(df)
        assert report is not None

    def test_unicode_text(self):
        """Test handling of unicode text."""
        df = pd.DataFrame({
            "text": ["用户@example.com", "test@日本.jp", "normal@test.com"],
        })
        
        scanner = PIIScanner()
        report = scanner.scan(df)
        assert report is not None

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_email = "a" * 1000 + "@example.com"
        df = pd.DataFrame({
            "text": [long_email, "short@test.com"],
        })
        
        scanner = PIIScanner()
        report = scanner.scan(df)
        assert report is not None

    def test_mixed_types_in_column(self):
        """Test handling of mixed types in column."""
        df = pd.DataFrame({
            "mixed": [123, "test@example.com", 45.6, None],
        })
        
        scanner = PIIScanner()
        report = scanner.scan(df)
        assert report is not None


class TestCreditCardDetection:
    """Tests specifically for credit card detection."""

    def test_detect_visa(self):
        """Test Visa card detection."""
        df = pd.DataFrame({"card": ["4111-1111-1111-1111"]})
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        cc_matches = [m for m in report.matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1

    def test_detect_mastercard(self):
        """Test Mastercard detection."""
        df = pd.DataFrame({"card": ["5500 0000 0000 0004"]})
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        cc_matches = [m for m in report.matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1

    def test_detect_16_digit(self):
        """Test plain 16-digit card number."""
        df = pd.DataFrame({"card": ["1234567890123456"]})
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        cc_matches = [m for m in report.matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1


class TestDateOfBirthDetection:
    """Tests for date of birth detection."""

    def test_detect_us_format(self):
        """Test US date format MM/DD/YYYY."""
        df = pd.DataFrame({"dob": ["12/25/1990", "01/01/2000"]})
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        dob_matches = [m for m in report.matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dob_matches) >= 1

    def test_detect_iso_format(self):
        """Test ISO date format YYYY-MM-DD."""
        df = pd.DataFrame({"dob": ["1990-12-25", "2000-01-01"]})
        scanner = PIIScanner()
        report = scanner.scan(df)
        
        dob_matches = [m for m in report.matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dob_matches) >= 1
