"""Data Privacy Vault - PII detection and anonymization.

This module provides comprehensive PII detection and data anonymization
with reversible tokenization for compliance with privacy regulations.

Example:
    >>> from clean.privacy import PrivacyVault, PIIScanner
    >>>
    >>> vault = PrivacyVault(encryption_key="...")
    >>> scanner = PIIScanner()
    >>>
    >>> pii_report = scanner.scan(df)
    >>> anonymized_df, tokens = vault.anonymize(df, pii_report)
    >>> original_df = vault.deanonymize(anonymized_df, tokens)
"""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


class PIIType(Enum):
    """Types of Personally Identifiable Information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_ID = "medical_id"
    BIOMETRIC = "biometric"
    LOCATION = "location"
    CUSTOM = "custom"


class AnonymizationMethod(Enum):
    """Methods for anonymizing PII."""

    REDACT = "redact"  # Replace with [REDACTED]
    MASK = "mask"  # Partial masking (e.g., ****1234)
    HASH = "hash"  # One-way hash
    TOKENIZE = "tokenize"  # Reversible tokenization
    GENERALIZE = "generalize"  # Replace with broader category
    PSEUDONYMIZE = "pseudonymize"  # Replace with fake but realistic data
    ENCRYPT = "encrypt"  # Reversible encryption


@dataclass
class PIIMatch:
    """A detected PII match in the data."""

    pii_type: PIIType
    column: str
    row_index: int
    value: str
    confidence: float
    start_pos: int | None = None
    end_pos: int | None = None
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pii_type": self.pii_type.value,
            "column": self.column,
            "row_index": self.row_index,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        }


@dataclass
class PIIScanReport:
    """Report from PII scanning."""

    n_rows: int
    n_columns: int
    total_pii_found: int
    matches: list[PIIMatch]
    pii_by_type: dict[PIIType, int]
    pii_by_column: dict[str, list[PIIType]]
    high_risk_columns: list[str]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "PII Scan Report",
            "=" * 50,
            "",
            f"Dataset: {self.n_rows:,} rows × {self.n_columns} columns",
            f"Total PII instances found: {self.total_pii_found:,}",
            "",
        ]

        if self.pii_by_type:
            lines.append("PII by Type:")
            for pii_type, count in sorted(
                self.pii_by_type.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  • {pii_type.value}: {count:,}")
            lines.append("")

        if self.high_risk_columns:
            lines.append(f"High-Risk Columns: {', '.join(self.high_risk_columns)}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "total_pii_found": self.total_pii_found,
            "pii_by_type": {k.value: v for k, v in self.pii_by_type.items()},
            "high_risk_columns": self.high_risk_columns,
        }


@dataclass
class TokenMapping:
    """Mapping between original value and token."""

    token: str
    original_value: str
    pii_type: PIIType
    column: str
    created_at: float


@dataclass
class AnonymizationResult:
    """Result of data anonymization."""

    anonymized_data: pd.DataFrame
    token_mappings: list[TokenMapping]
    n_values_anonymized: int
    methods_used: dict[PIIType, AnonymizationMethod]

    def get_token_map(self) -> dict[str, str]:
        """Get token to original value mapping."""
        return {t.token: t.original_value for t in self.token_mappings}


class PIIDetector(ABC):
    """Abstract base class for PII detectors."""

    pii_type: PIIType

    @abstractmethod
    def detect(self, text: str) -> list[tuple[int, int, float]]:
        """Detect PII in text.

        Args:
            text: Text to scan

        Returns:
            List of (start, end, confidence) tuples
        """
        pass


class RegexPIIDetector(PIIDetector):
    """Regex-based PII detector."""

    def __init__(
        self,
        pii_type: PIIType,
        patterns: list[str],
        confidence: float = 0.9,
    ):
        """Initialize regex detector.

        Args:
            pii_type: Type of PII to detect
            patterns: Regex patterns
            confidence: Confidence score for matches
        """
        self.pii_type = pii_type
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.confidence = confidence

    def detect(self, text: str) -> list[tuple[int, int, float]]:
        """Detect PII using regex patterns."""
        matches = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), self.confidence))

        return matches


class PIIScanner:
    """Scanner for detecting PII in datasets.

    Uses a combination of regex patterns and heuristics to detect
    various types of PII.
    """

    def __init__(
        self,
        custom_detectors: list[PIIDetector] | None = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the scanner.

        Args:
            custom_detectors: Additional custom detectors
            min_confidence: Minimum confidence to report
        """
        self.min_confidence = min_confidence

        # Initialize built-in detectors
        self.detectors = self._create_default_detectors()

        if custom_detectors:
            self.detectors.extend(custom_detectors)

    def _create_default_detectors(self) -> list[PIIDetector]:
        """Create default PII detectors."""
        return [
            # Email
            RegexPIIDetector(
                PIIType.EMAIL,
                [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                confidence=0.95,
            ),
            # Phone (US)
            RegexPIIDetector(
                PIIType.PHONE,
                [
                    r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                    r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
                ],
                confidence=0.9,
            ),
            # SSN
            RegexPIIDetector(
                PIIType.SSN,
                [r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"],
                confidence=0.85,
            ),
            # Credit Card
            RegexPIIDetector(
                PIIType.CREDIT_CARD,
                [
                    r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                    r"\b\d{16}\b",
                ],
                confidence=0.9,
            ),
            # IP Address
            RegexPIIDetector(
                PIIType.IP_ADDRESS,
                [r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"],
                confidence=0.95,
            ),
            # Date of Birth (various formats)
            RegexPIIDetector(
                PIIType.DATE_OF_BIRTH,
                [
                    r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
                    r"\b(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])\b",
                ],
                confidence=0.7,
            ),
            # Passport (generic)
            RegexPIIDetector(
                PIIType.PASSPORT,
                [r"\b[A-Z]{1,2}\d{6,9}\b"],
                confidence=0.6,
            ),
            # Bank Account (generic)
            RegexPIIDetector(
                PIIType.BANK_ACCOUNT,
                [
                    r"\b\d{8,17}\b",  # Account numbers
                    r"\b\d{9}\b",  # Routing numbers
                ],
                confidence=0.5,
            ),
        ]

    def scan(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        sample_size: int | None = None,
    ) -> PIIScanReport:
        """Scan a DataFrame for PII.

        Args:
            data: DataFrame to scan
            columns: Specific columns to scan (default: all string columns)
            sample_size: Number of rows to sample (default: all)

        Returns:
            PIIScanReport with findings
        """
        if columns is None:
            columns = data.select_dtypes(include=["object", "string"]).columns.tolist()

        if sample_size and sample_size < len(data):
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            data_to_scan = data.iloc[sample_indices]
        else:
            data_to_scan = data
            sample_indices = np.arange(len(data))

        matches = []
        pii_by_type: dict[PIIType, int] = {}
        pii_by_column: dict[str, list[PIIType]] = {}

        for col in columns:
            if col not in data_to_scan.columns:
                continue

            column_pii_types = set()

            for i, (idx, value) in enumerate(
                zip(sample_indices, data_to_scan[col].astype(str))
            ):
                if pd.isna(value) or value == "nan":
                    continue

                for detector in self.detectors:
                    detections = detector.detect(str(value))

                    for start, end, confidence in detections:
                        if confidence < self.min_confidence:
                            continue

                        matches.append(PIIMatch(
                            pii_type=detector.pii_type,
                            column=col,
                            row_index=int(idx),
                            value=str(value)[start:end],
                            confidence=confidence,
                            start_pos=start,
                            end_pos=end,
                            context=str(value)[max(0, start - 20):end + 20],
                        ))

                        pii_by_type[detector.pii_type] = (
                            pii_by_type.get(detector.pii_type, 0) + 1
                        )
                        column_pii_types.add(detector.pii_type)

            if column_pii_types:
                pii_by_column[col] = list(column_pii_types)

        # Identify high-risk columns
        high_risk = [
            col for col, types in pii_by_column.items()
            if len(types) >= 2 or PIIType.SSN in types or PIIType.CREDIT_CARD in types
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pii_by_type, pii_by_column, high_risk
        )

        return PIIScanReport(
            n_rows=len(data),
            n_columns=len(columns),
            total_pii_found=len(matches),
            matches=matches,
            pii_by_type=pii_by_type,
            pii_by_column=pii_by_column,
            high_risk_columns=high_risk,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        pii_by_type: dict[PIIType, int],
        pii_by_column: dict[str, list[PIIType]],
        high_risk: list[str],
    ) -> list[str]:
        """Generate recommendations based on scan results."""
        recommendations = []

        if high_risk:
            recommendations.append(
                f"High-risk columns ({', '.join(high_risk)}) should be anonymized or encrypted."
            )

        if PIIType.SSN in pii_by_type:
            recommendations.append(
                "Social Security Numbers detected. Consider tokenization with secure vault storage."
            )

        if PIIType.CREDIT_CARD in pii_by_type:
            recommendations.append(
                "Credit card numbers detected. Ensure PCI-DSS compliance through encryption."
            )

        if PIIType.EMAIL in pii_by_type:
            recommendations.append(
                "Email addresses can be pseudonymized while preserving domain for analysis."
            )

        if pii_by_column:
            recommendations.append(
                f"Found PII in {len(pii_by_column)} columns. Review data retention policies."
            )

        if not recommendations:
            recommendations.append("No significant PII detected.")

        return recommendations


class PrivacyVault:
    """Secure vault for data anonymization with reversible tokenization.

    Provides encryption, tokenization, and secure storage for PII
    with support for reversible de-anonymization.
    """

    def __init__(
        self,
        encryption_key: str | None = None,
        hash_salt: str | None = None,
    ):
        """Initialize the privacy vault.

        Args:
            encryption_key: Key for encryption (auto-generated if not provided)
            hash_salt: Salt for hashing (auto-generated if not provided)
        """
        self.encryption_key = encryption_key or secrets.token_hex(32)
        self.hash_salt = hash_salt or secrets.token_hex(16)

        self._token_store: dict[str, TokenMapping] = {}
        self._method_handlers = self._create_method_handlers()

    def _create_method_handlers(
        self,
    ) -> dict[AnonymizationMethod, Callable]:
        """Create handlers for each anonymization method."""
        return {
            AnonymizationMethod.REDACT: self._redact,
            AnonymizationMethod.MASK: self._mask,
            AnonymizationMethod.HASH: self._hash,
            AnonymizationMethod.TOKENIZE: self._tokenize,
            AnonymizationMethod.GENERALIZE: self._generalize,
            AnonymizationMethod.PSEUDONYMIZE: self._pseudonymize,
            AnonymizationMethod.ENCRYPT: self._encrypt,
        }

    def anonymize(
        self,
        data: pd.DataFrame,
        pii_report: PIIScanReport | None = None,
        method: AnonymizationMethod = AnonymizationMethod.TOKENIZE,
        method_by_type: dict[PIIType, AnonymizationMethod] | None = None,
        columns: list[str] | None = None,
    ) -> AnonymizationResult:
        """Anonymize PII in a DataFrame.

        Args:
            data: DataFrame to anonymize
            pii_report: Optional pre-computed PII scan report
            method: Default anonymization method
            method_by_type: Method overrides by PII type
            columns: Specific columns to anonymize

        Returns:
            AnonymizationResult with anonymized data and token mappings
        """
        import time

        if pii_report is None:
            scanner = PIIScanner()
            pii_report = scanner.scan(data, columns)

        # Determine which columns and PII to anonymize
        if columns is None:
            columns = list(pii_report.pii_by_column.keys())

        method_by_type = method_by_type or {}

        # Create a copy of the data
        anonymized = data.copy()
        token_mappings = []
        n_anonymized = 0
        methods_used = {}

        # Group matches by (column, row_index)
        matches_by_location: dict[tuple[str, int], list[PIIMatch]] = {}
        for match in pii_report.matches:
            key = (match.column, match.row_index)
            if key not in matches_by_location:
                matches_by_location[key] = []
            matches_by_location[key].append(match)

        # Process each location
        for (col, row_idx), matches in matches_by_location.items():
            if col not in anonymized.columns:
                continue

            original_value = str(anonymized.at[row_idx, col])

            # Sort matches by position (reverse) to handle replacements correctly
            matches.sort(key=lambda m: m.start_pos or 0, reverse=True)

            new_value = original_value

            for match in matches:
                pii_method = method_by_type.get(match.pii_type, method)
                handler = self._method_handlers[pii_method]

                # Get the substring to anonymize
                start = match.start_pos or 0
                end = match.end_pos or len(original_value)
                pii_value = original_value[start:end]

                # Anonymize
                anonymized_value, token = handler(pii_value, match.pii_type)

                # Replace in the value
                new_value = new_value[:start] + anonymized_value + new_value[end:]

                # Store token mapping if reversible
                if token:
                    mapping = TokenMapping(
                        token=anonymized_value,
                        original_value=pii_value,
                        pii_type=match.pii_type,
                        column=col,
                        created_at=time.time(),
                    )
                    token_mappings.append(mapping)
                    self._token_store[anonymized_value] = mapping

                methods_used[match.pii_type] = pii_method
                n_anonymized += 1

            anonymized.at[row_idx, col] = new_value

        return AnonymizationResult(
            anonymized_data=anonymized,
            token_mappings=token_mappings,
            n_values_anonymized=n_anonymized,
            methods_used=methods_used,
        )

    def deanonymize(
        self,
        data: pd.DataFrame,
        token_mappings: list[TokenMapping] | None = None,
    ) -> pd.DataFrame:
        """Reverse anonymization using stored token mappings.

        Args:
            data: Anonymized DataFrame
            token_mappings: Token mappings (uses stored if not provided)

        Returns:
            De-anonymized DataFrame
        """
        result = data.copy()

        # Build token map
        if token_mappings:
            token_map = {t.token: t.original_value for t in token_mappings}
        else:
            token_map = {t.token: t.original_value for t in self._token_store.values()}

        # Replace tokens with original values
        for col in result.columns:
            if result[col].dtype == object:
                for token, original in token_map.items():
                    result[col] = result[col].str.replace(
                        token, original, regex=False
                    )

        return result

    def _redact(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """Redact value completely."""
        return f"[{pii_type.value.upper()}_REDACTED]", False

    def _mask(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """Partially mask value."""
        if len(value) <= 4:
            return "*" * len(value), False

        visible = 4
        masked = "*" * (len(value) - visible) + value[-visible:]
        return masked, False

    def _hash(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """One-way hash value."""
        salted = f"{self.hash_salt}{value}"
        hash_value = hashlib.sha256(salted.encode()).hexdigest()[:16]
        return f"HASH_{hash_value}", False

    def _tokenize(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """Tokenize value (reversible)."""
        token = f"TOK_{uuid.uuid4().hex[:12]}"
        return token, True

    def _generalize(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """Generalize value to broader category."""
        generalizations = {
            PIIType.EMAIL: lambda v: f"*@{v.split('@')[1]}" if "@" in v else "[EMAIL]",
            PIIType.PHONE: lambda v: f"({v[:3]}) ***-****" if len(v) >= 3 else "[PHONE]",
            PIIType.DATE_OF_BIRTH: lambda v: f"{v[-4:]}/**/**" if len(v) >= 4 else "[DOB]",
            PIIType.IP_ADDRESS: lambda v: ".".join(v.split(".")[:2] + ["*", "*"]),
        }

        handler = generalizations.get(pii_type, lambda v: f"[{pii_type.value.upper()}]")
        return handler(value), False

    def _pseudonymize(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """Replace with fake but realistic data."""
        pseudonyms = {
            PIIType.EMAIL: lambda: f"user{secrets.randbelow(10000)}@example.com",
            PIIType.PHONE: lambda: f"555-{secrets.randbelow(1000):03d}-{secrets.randbelow(10000):04d}",
            PIIType.NAME: lambda: f"Person_{secrets.randbelow(10000)}",
            PIIType.SSN: lambda: f"000-00-{secrets.randbelow(10000):04d}",
            PIIType.IP_ADDRESS: lambda: f"10.0.{secrets.randbelow(256)}.{secrets.randbelow(256)}",
        }

        handler = pseudonyms.get(pii_type, lambda: f"PSEUDO_{secrets.token_hex(6)}")
        return handler(), True

    def _encrypt(
        self,
        value: str,
        pii_type: PIIType,
    ) -> tuple[str, bool]:
        """Encrypt value (reversible)."""
        # Simple XOR encryption for demonstration
        # In production, use proper encryption (Fernet, AES, etc.)
        key_bytes = self.encryption_key.encode()[:len(value)]
        value_bytes = value.encode()

        encrypted = bytes(
            v ^ k for v, k in zip(value_bytes, key_bytes * (len(value_bytes) // len(key_bytes) + 1))
        )
        encoded = base64.b64encode(encrypted).decode()
        return f"ENC_{encoded}", True

    def export_tokens(self, output_path: str) -> None:
        """Export token mappings to file.

        Args:
            output_path: Path for output file
        """
        import json
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "token": m.token,
                "original_value": m.original_value,
                "pii_type": m.pii_type.value,
                "column": m.column,
                "created_at": m.created_at,
            }
            for m in self._token_store.values()
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def import_tokens(self, input_path: str) -> int:
        """Import token mappings from file.

        Args:
            input_path: Path to token file

        Returns:
            Number of tokens imported
        """
        import json
        from pathlib import Path

        with open(Path(input_path)) as f:
            data = json.load(f)

        for item in data:
            mapping = TokenMapping(
                token=item["token"],
                original_value=item["original_value"],
                pii_type=PIIType(item["pii_type"]),
                column=item["column"],
                created_at=item.get("created_at", 0),
            )
            self._token_store[mapping.token] = mapping

        return len(data)


def scan_pii(
    data: pd.DataFrame,
    **kwargs: Any,
) -> PIIScanReport:
    """Convenience function for PII scanning.

    Args:
        data: DataFrame to scan
        **kwargs: Additional arguments for PIIScanner

    Returns:
        PIIScanReport with findings
    """
    scanner = PIIScanner(**kwargs)
    return scanner.scan(data)


def anonymize_data(
    data: pd.DataFrame,
    method: str = "tokenize",
    **kwargs: Any,
) -> AnonymizationResult:
    """Convenience function for data anonymization.

    Args:
        data: DataFrame to anonymize
        method: Anonymization method
        **kwargs: Additional arguments

    Returns:
        AnonymizationResult with anonymized data
    """
    vault = PrivacyVault()
    return vault.anonymize(
        data,
        method=AnonymizationMethod(method),
        **kwargs,
    )
