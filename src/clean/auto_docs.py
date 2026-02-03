"""Automated Data Documentation Generator.

This module uses LLMs to automatically generate comprehensive documentation
for datasets, including column descriptions, data quality notes, and
compliance documentation.

Example:
    >>> from clean.auto_docs import DataDocumenter, generate_docs
    >>>
    >>> # Quick documentation
    >>> docs = generate_docs(df)
    >>> print(docs.to_markdown())
    >>>
    >>> # Detailed documentation with LLM
    >>> documenter = DataDocumenter(api_key="sk-...")
    >>> docs = documenter.document(df, name="customer_data")
    >>> docs.save("docs/customer_data.md")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from clean.exceptions import CleanError, ConfigurationError, DependencyError

logger = logging.getLogger(__name__)


class DocumentationLevel(Enum):
    """Level of documentation detail."""

    BASIC = "basic"  # Column names, types, basic stats
    STANDARD = "standard"  # + descriptions, patterns, quality notes
    COMPREHENSIVE = "comprehensive"  # + LLM analysis, relationships, compliance


@dataclass
class ColumnDocumentation:
    """Documentation for a single column."""

    name: str
    dtype: str
    description: str = ""
    semantic_type: str = ""  # e.g., "email", "phone", "name", "id"
    example_values: list[Any] = field(default_factory=list)
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    min_value: Any = None
    max_value: Any = None
    mean_value: float | None = None
    patterns: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)
    pii_detected: bool = False
    pii_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "semantic_type": self.semantic_type,
            "example_values": self.example_values,
            "null_count": self.null_count,
            "null_percentage": self.null_percentage,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "patterns": self.patterns,
            "quality_notes": self.quality_notes,
            "pii_detected": self.pii_detected,
            "pii_type": self.pii_type,
        }


@dataclass
class DatasetDocumentation:
    """Complete documentation for a dataset."""

    name: str
    description: str
    generated_at: str
    n_rows: int
    n_columns: int
    columns: list[ColumnDocumentation]
    quality_summary: str = ""
    usage_notes: list[str] = field(default_factory=list)
    compliance_notes: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "1.0"
    data_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "generated_at": self.generated_at,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": [c.to_dict() for c in self.columns],
            "quality_summary": self.quality_summary,
            "usage_notes": self.usage_notes,
            "compliance_notes": self.compliance_notes,
            "relationships": self.relationships,
            "schema_version": self.schema_version,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Generate Markdown documentation."""
        lines = [
            f"# {self.name}",
            "",
            f"_{self.description}_",
            "",
            f"**Generated**: {self.generated_at}",
            f"**Rows**: {self.n_rows:,}",
            f"**Columns**: {self.n_columns}",
            "",
        ]

        if self.data_hash:
            lines.append(f"**Data Hash**: `{self.data_hash[:12]}`")
            lines.append("")

        # Quality summary
        if self.quality_summary:
            lines.extend([
                "## Quality Summary",
                "",
                self.quality_summary,
                "",
            ])

        # Column documentation
        lines.extend([
            "## Schema",
            "",
            "| Column | Type | Description | Nulls | Unique |",
            "|--------|------|-------------|-------|--------|",
        ])

        for col in self.columns:
            null_pct = f"{col.null_percentage:.1f}%" if col.null_percentage > 0 else "0%"
            unique_pct = f"{col.unique_percentage:.1f}%"
            desc = col.description[:50] + "..." if len(col.description) > 50 else col.description
            pii_marker = " 🔒" if col.pii_detected else ""
            lines.append(
                f"| {col.name}{pii_marker} | `{col.dtype}` | {desc} | {null_pct} | {unique_pct} |"
            )

        lines.append("")

        # Detailed column docs
        lines.extend(["## Column Details", ""])

        for col in self.columns:
            lines.append(f"### {col.name}")
            lines.append("")

            if col.description:
                lines.append(f"**Description**: {col.description}")
                lines.append("")

            if col.semantic_type:
                lines.append(f"**Semantic Type**: {col.semantic_type}")

            lines.append(f"**Data Type**: `{col.dtype}`")

            if col.pii_detected:
                lines.append(f"**⚠️ PII Detected**: {col.pii_type or 'Unknown type'}")

            lines.append("")
            lines.append("**Statistics**:")
            lines.append(f"- Null values: {col.null_count:,} ({col.null_percentage:.1f}%)")
            lines.append(f"- Unique values: {col.unique_count:,} ({col.unique_percentage:.1f}%)")

            if col.min_value is not None:
                lines.append(f"- Min: {col.min_value}")
            if col.max_value is not None:
                lines.append(f"- Max: {col.max_value}")
            if col.mean_value is not None:
                lines.append(f"- Mean: {col.mean_value:.2f}")

            if col.example_values:
                examples = ", ".join(f"`{v}`" for v in col.example_values[:5])
                lines.append(f"- Examples: {examples}")

            if col.patterns:
                lines.append("")
                lines.append("**Detected Patterns**:")
                for pattern in col.patterns:
                    lines.append(f"- {pattern}")

            if col.quality_notes:
                lines.append("")
                lines.append("**Quality Notes**:")
                for note in col.quality_notes:
                    lines.append(f"- {note}")

            lines.append("")

        # Usage notes
        if self.usage_notes:
            lines.extend(["## Usage Notes", ""])
            for note in self.usage_notes:
                lines.append(f"- {note}")
            lines.append("")

        # Compliance notes
        if self.compliance_notes:
            lines.extend(["## Compliance Notes", ""])
            for note in self.compliance_notes:
                lines.append(f"- {note}")
            lines.append("")

        # Relationships
        if self.relationships:
            lines.extend(["## Relationships", ""])
            for rel in self.relationships:
                lines.append(f"- {rel.get('description', str(rel))}")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str | Path, format: str = "auto") -> None:
        """Save documentation to file.

        Args:
            path: Output path
            format: 'markdown', 'json', or 'auto' (detect from extension)
        """
        path = Path(path)

        if format == "auto":
            if path.suffix in (".md", ".markdown"):
                format = "markdown"
            elif path.suffix == ".json":
                format = "json"
            else:
                format = "markdown"

        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            path.write_text(self.to_markdown())
        else:
            path.write_text(self.to_json())

        logger.info(f"Documentation saved to {path}")


# PII detection patterns
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
    "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
}

# Semantic type detection patterns
SEMANTIC_PATTERNS = {
    "email": (r"email|e-mail|mail", r"@.*\."),
    "phone": (r"phone|tel|mobile|cell", r"^\+?\d[\d\s\-\(\)]{7,}$"),
    "name": (r"name|first.*name|last.*name|full.*name", None),
    "address": (r"address|street|city|state|zip|postal", None),
    "date": (r"date|time|created|updated|timestamp", None),
    "id": (r"_id$|^id$|identifier|uuid|guid", None),
    "url": (r"url|link|website|href", r"^https?://"),
    "currency": (r"price|cost|amount|total|salary", r"^\$?\d+\.?\d*$"),
    "percentage": (r"percent|pct|rate|ratio", r"^\d+\.?\d*%?$"),
}


class DataDocumenter:
    """Generate comprehensive documentation for datasets.

    Example:
        >>> documenter = DataDocumenter()
        >>> docs = documenter.document(df, name="my_dataset")
        >>> print(docs.to_markdown())
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        level: DocumentationLevel = DocumentationLevel.STANDARD,
        detect_pii: bool = True,
        max_sample_size: int = 1000,
    ):
        """Initialize documenter.

        Args:
            api_key: OpenAI API key (optional, for LLM-powered docs)
            model: LLM model to use
            level: Documentation detail level
            detect_pii: Whether to detect PII in data
            max_sample_size: Maximum samples for analysis
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.level = level
        self.detect_pii = detect_pii
        self.max_sample_size = max_sample_size
        self._openai = None

    def _get_openai(self) -> Any:
        """Lazy load OpenAI client."""
        if self._openai is None:
            try:
                import openai
            except ImportError as e:
                raise DependencyError(
                    "openai", "pip install openai", "LLM documentation"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "OpenAI API key required for LLM documentation. "
                    "Set OPENAI_API_KEY or pass api_key parameter."
                )

            self._openai = openai.OpenAI(api_key=self.api_key)

        return self._openai

    def document(
        self,
        data: pd.DataFrame,
        name: str = "dataset",
        description: str | None = None,
    ) -> DatasetDocumentation:
        """Generate documentation for a dataset.

        Args:
            data: DataFrame to document
            name: Dataset name
            description: Optional dataset description

        Returns:
            DatasetDocumentation object
        """
        logger.info(f"Generating documentation for {name} ({len(data)} rows)")

        # Calculate data hash
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data).values.tobytes()
        ).hexdigest()

        # Document each column
        columns = []
        for col_name in data.columns:
            col_doc = self._document_column(data, col_name)
            columns.append(col_doc)

        # Generate dataset description
        if description is None:
            description = self._generate_description(data, columns)

        # Generate quality summary
        quality_summary = self._generate_quality_summary(data, columns)

        # Detect relationships
        relationships = self._detect_relationships(data, columns)

        # Generate compliance notes
        compliance_notes = self._generate_compliance_notes(columns)

        # Generate usage notes
        usage_notes = self._generate_usage_notes(data, columns)

        # Use LLM for comprehensive docs
        if self.level == DocumentationLevel.COMPREHENSIVE and self.api_key:
            self._enhance_with_llm(columns, data)

        return DatasetDocumentation(
            name=name,
            description=description,
            generated_at=datetime.now().isoformat(),
            n_rows=len(data),
            n_columns=len(data.columns),
            columns=columns,
            quality_summary=quality_summary,
            usage_notes=usage_notes,
            compliance_notes=compliance_notes,
            relationships=relationships,
            data_hash=data_hash,
        )

    def _document_column(
        self, data: pd.DataFrame, col_name: str
    ) -> ColumnDocumentation:
        """Document a single column."""
        col = data[col_name]
        n_rows = len(data)

        # Basic stats
        null_count = int(col.isna().sum())
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0
        unique_count = int(col.nunique())
        unique_pct = (unique_count / n_rows * 100) if n_rows > 0 else 0

        # Type info
        dtype = str(col.dtype)

        # Examples (non-null values)
        non_null = col.dropna()
        examples = non_null.head(5).tolist() if len(non_null) > 0 else []

        # Numeric stats
        min_val = max_val = mean_val = None
        if pd.api.types.is_numeric_dtype(col):
            if len(non_null) > 0:
                min_val = float(non_null.min())
                max_val = float(non_null.max())
                mean_val = float(non_null.mean())

        # Detect semantic type
        semantic_type = self._detect_semantic_type(col_name, col)

        # Detect PII
        pii_detected = False
        pii_type = None
        if self.detect_pii:
            pii_detected, pii_type = self._detect_pii(col)

        # Detect patterns
        patterns = self._detect_patterns(col)

        # Quality notes
        quality_notes = self._generate_column_quality_notes(col, null_pct, unique_pct)

        # Generate description
        description = self._generate_column_description(
            col_name, dtype, semantic_type, unique_count, n_rows
        )

        return ColumnDocumentation(
            name=col_name,
            dtype=dtype,
            description=description,
            semantic_type=semantic_type,
            example_values=examples,
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            patterns=patterns,
            quality_notes=quality_notes,
            pii_detected=pii_detected,
            pii_type=pii_type,
        )

    def _detect_semantic_type(self, col_name: str, col: pd.Series) -> str:
        """Detect semantic type of a column."""
        col_lower = col_name.lower()

        for sem_type, (name_pattern, value_pattern) in SEMANTIC_PATTERNS.items():
            # Check column name
            if re.search(name_pattern, col_lower, re.IGNORECASE):
                return sem_type

            # Check values if pattern provided
            if value_pattern:
                sample = col.dropna().head(100).astype(str)
                if len(sample) > 0:
                    match_rate = sample.str.match(value_pattern, na=False).mean()
                    if match_rate > 0.8:
                        return sem_type

        # Infer from dtype
        if pd.api.types.is_datetime64_any_dtype(col):
            return "datetime"
        if pd.api.types.is_bool_dtype(col):
            return "boolean"
        if pd.api.types.is_categorical_dtype(col):
            return "category"

        return ""

    def _detect_pii(self, col: pd.Series) -> tuple[bool, str | None]:
        """Detect if column contains PII."""
        sample = col.dropna().head(100).astype(str)
        if len(sample) == 0:
            return False, None

        for pii_type, pattern in PII_PATTERNS.items():
            match_rate = sample.str.match(pattern, na=False).mean()
            if match_rate > 0.5:
                return True, pii_type

        return False, None

    def _detect_patterns(self, col: pd.Series) -> list[str]:
        """Detect common patterns in column values."""
        patterns = []
        sample = col.dropna().head(100)

        if len(sample) == 0:
            return patterns

        # Check for constant values
        if col.nunique() == 1:
            patterns.append(f"Constant value: {col.iloc[0]}")

        # Check for sequential values
        if pd.api.types.is_numeric_dtype(col):
            non_null = col.dropna()
            if len(non_null) > 10:
                diffs = non_null.diff().dropna()
                if diffs.nunique() == 1 and diffs.iloc[0] != 0:
                    patterns.append(f"Sequential (step={diffs.iloc[0]})")

        # Check for string patterns
        if pd.api.types.is_string_dtype(col) or col.dtype == object:
            str_sample = sample.astype(str)

            # Check for common prefixes
            if len(str_sample) >= 10:
                prefix_len = 0
                first = str_sample.iloc[0]
                for i in range(1, min(len(first), 20)):
                    prefix = first[:i]
                    if str_sample.str.startswith(prefix).mean() > 0.9:
                        prefix_len = i
                    else:
                        break
                if prefix_len >= 3:
                    patterns.append(f"Common prefix: '{first[:prefix_len]}'")

            # Check string lengths
            lengths = str_sample.str.len()
            if lengths.std() == 0 and lengths.iloc[0] > 0:
                patterns.append(f"Fixed length: {int(lengths.iloc[0])}")

        return patterns

    def _generate_column_quality_notes(
        self, col: pd.Series, null_pct: float, unique_pct: float
    ) -> list[str]:
        """Generate quality notes for a column."""
        notes = []

        if null_pct > 50:
            notes.append(f"⚠️ High null rate ({null_pct:.1f}%)")
        elif null_pct > 20:
            notes.append(f"Moderate null rate ({null_pct:.1f}%)")

        if unique_pct == 100 and len(col) > 10:
            notes.append("All values unique (potential ID column)")
        elif unique_pct < 1 and len(col) > 100:
            notes.append("Very low cardinality")

        # Check for mixed types
        if col.dtype == object:
            types = col.dropna().apply(type).unique()
            if len(types) > 1:
                notes.append(f"Mixed types detected: {[t.__name__ for t in types]}")

        return notes

    def _generate_column_description(
        self,
        col_name: str,
        dtype: str,
        semantic_type: str,
        unique_count: int,
        n_rows: int,
    ) -> str:
        """Generate a description for a column."""
        if semantic_type:
            type_desc = semantic_type.replace("_", " ").title()
        elif "int" in dtype:
            type_desc = "Integer"
        elif "float" in dtype:
            type_desc = "Numeric"
        elif "bool" in dtype:
            type_desc = "Boolean flag"
        elif "datetime" in dtype:
            type_desc = "Timestamp"
        else:
            type_desc = "Text"

        # Cardinality hint
        if unique_count == n_rows and n_rows > 10:
            card_hint = " (unique identifier)"
        elif unique_count <= 10:
            card_hint = f" ({unique_count} categories)"
        else:
            card_hint = ""

        return f"{type_desc} column{card_hint}"

    def _generate_description(
        self, data: pd.DataFrame, columns: list[ColumnDocumentation]
    ) -> str:
        """Generate dataset description."""
        # Count column types
        num_cols = sum(1 for c in columns if "int" in c.dtype or "float" in c.dtype)
        cat_cols = sum(1 for c in columns if c.unique_count <= 20)
        pii_cols = sum(1 for c in columns if c.pii_detected)

        parts = [f"Dataset with {len(data):,} rows and {len(columns)} columns"]

        if num_cols > 0:
            parts.append(f"{num_cols} numeric")
        if cat_cols > 0:
            parts.append(f"{cat_cols} categorical")
        if pii_cols > 0:
            parts.append(f"{pii_cols} PII columns detected")

        return ". ".join(parts) + "."

    def _generate_quality_summary(
        self, data: pd.DataFrame, columns: list[ColumnDocumentation]
    ) -> str:
        """Generate quality summary."""
        total_nulls = sum(c.null_count for c in columns)
        total_cells = len(data) * len(columns)
        null_pct = (total_nulls / total_cells * 100) if total_cells > 0 else 0

        high_null_cols = [c.name for c in columns if c.null_percentage > 50]
        pii_cols = [c.name for c in columns if c.pii_detected]

        lines = []
        lines.append(f"Overall completeness: {100 - null_pct:.1f}%")

        if high_null_cols:
            lines.append(f"Columns with >50% nulls: {', '.join(high_null_cols)}")

        if pii_cols:
            lines.append(f"⚠️ PII detected in: {', '.join(pii_cols)}")

        return " ".join(lines)

    def _detect_relationships(
        self, data: pd.DataFrame, columns: list[ColumnDocumentation]
    ) -> list[dict[str, Any]]:
        """Detect potential relationships between columns."""
        relationships = []

        # Find potential foreign keys (ID columns)
        id_cols = [c for c in columns if c.semantic_type == "id" or "_id" in c.name.lower()]

        for col in id_cols:
            # Check if this matches another column name pattern
            base_name = col.name.replace("_id", "").replace("Id", "")
            for other in columns:
                if other.name.lower() == base_name.lower() + "_name":
                    relationships.append({
                        "type": "foreign_key",
                        "from": col.name,
                        "to": other.name,
                        "description": f"{col.name} likely references {other.name}",
                    })

        return relationships

    def _generate_compliance_notes(
        self, columns: list[ColumnDocumentation]
    ) -> list[str]:
        """Generate compliance-related notes."""
        notes = []

        pii_cols = [c for c in columns if c.pii_detected]
        if pii_cols:
            notes.append(
                f"GDPR/CCPA: {len(pii_cols)} columns contain PII - "
                "ensure proper consent and data handling"
            )

            email_cols = [c for c in pii_cols if c.pii_type == "email"]
            if email_cols:
                notes.append(
                    f"Email data in {[c.name for c in email_cols]} - "
                    "verify opt-in status for marketing use"
                )

        return notes

    def _generate_usage_notes(
        self, data: pd.DataFrame, columns: list[ColumnDocumentation]
    ) -> list[str]:
        """Generate usage notes."""
        notes = []

        # Check for imbalanced categorical columns
        for col in columns:
            if col.unique_count <= 10 and col.unique_count > 1:
                series = data[col.name]
                value_counts = series.value_counts(normalize=True)
                if len(value_counts) > 0 and value_counts.iloc[0] > 0.9:
                    notes.append(
                        f"Column '{col.name}' is highly imbalanced - "
                        f"consider stratified sampling"
                    )

        # Check for potential target columns
        binary_cols = [
            c for c in columns
            if c.unique_count == 2 and c.null_percentage < 5
        ]
        if binary_cols:
            notes.append(
                f"Potential target columns (binary): {[c.name for c in binary_cols]}"
            )

        return notes

    def _enhance_with_llm(
        self, columns: list[ColumnDocumentation], data: pd.DataFrame
    ) -> None:
        """Enhance documentation with LLM analysis."""
        client = self._get_openai()

        # Prepare context
        col_summaries = []
        for col in columns:
            summary = (
                f"- {col.name}: {col.dtype}, "
                f"{col.unique_count} unique values, "
                f"{col.null_percentage:.1f}% null"
            )
            if col.example_values:
                examples = ", ".join(str(v) for v in col.example_values[:3])
                summary += f", examples: [{examples}]"
            col_summaries.append(summary)

        prompt = f"""Analyze this dataset schema and provide improved column descriptions.

Dataset: {len(data)} rows, {len(columns)} columns

Columns:
{chr(10).join(col_summaries)}

For each column, provide a clear 1-sentence description of what the data represents.
Return as JSON: {{"column_name": "description", ...}}"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Update column descriptions
            for col in columns:
                if col.name in result:
                    col.description = result[col.name]

            logger.info("Enhanced documentation with LLM analysis")

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")


def generate_docs(
    data: pd.DataFrame,
    name: str = "dataset",
    description: str | None = None,
    level: DocumentationLevel | str = DocumentationLevel.STANDARD,
    detect_pii: bool = True,
    api_key: str | None = None,
) -> DatasetDocumentation:
    """Generate documentation for a dataset.

    This is the main entry point for quick documentation generation.

    Args:
        data: DataFrame to document
        name: Dataset name
        description: Optional description
        level: Documentation level ('basic', 'standard', 'comprehensive')
        detect_pii: Whether to detect PII
        api_key: OpenAI API key (required for 'comprehensive' level)

    Returns:
        DatasetDocumentation object

    Example:
        >>> docs = generate_docs(df, name="customers")
        >>> print(docs.to_markdown())
        >>> docs.save("docs/customers.md")
    """
    if isinstance(level, str):
        level = DocumentationLevel(level.lower())

    documenter = DataDocumenter(
        api_key=api_key,
        level=level,
        detect_pii=detect_pii,
    )

    return documenter.document(data, name=name, description=description)


__all__ = [
    "DataDocumenter",
    "DatasetDocumentation",
    "ColumnDocumentation",
    "DocumentationLevel",
    "generate_docs",
]
