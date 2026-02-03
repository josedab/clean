"""Tests for automated documentation module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.auto_docs import (
    ColumnDocumentation,
    DataDocumenter,
    DatasetDocumentation,
    DocumentationLevel,
    generate_docs,
)


class TestColumnDocumentation:
    """Tests for ColumnDocumentation dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        col = ColumnDocumentation(
            name="test_col",
            dtype="int64",
            description="Test column",
            null_count=5,
            null_percentage=5.0,
            unique_count=95,
            unique_percentage=95.0,
        )
        result = col.to_dict()

        assert result["name"] == "test_col"
        assert result["dtype"] == "int64"
        assert result["null_count"] == 5

    def test_pii_fields(self):
        """Test PII detection fields."""
        col = ColumnDocumentation(
            name="email",
            dtype="object",
            pii_detected=True,
            pii_type="email",
        )

        assert col.pii_detected is True
        assert col.pii_type == "email"


class TestDatasetDocumentation:
    """Tests for DatasetDocumentation dataclass."""

    @pytest.fixture
    def sample_docs(self):
        """Create sample documentation."""
        return DatasetDocumentation(
            name="test_dataset",
            description="A test dataset",
            generated_at="2024-01-01T00:00:00",
            n_rows=100,
            n_columns=3,
            columns=[
                ColumnDocumentation(
                    name="id",
                    dtype="int64",
                    description="Unique identifier",
                    unique_count=100,
                    unique_percentage=100.0,
                ),
                ColumnDocumentation(
                    name="name",
                    dtype="object",
                    description="Customer name",
                    null_count=5,
                    null_percentage=5.0,
                    unique_count=95,
                    unique_percentage=95.0,
                ),
                ColumnDocumentation(
                    name="email",
                    dtype="object",
                    description="Email address",
                    pii_detected=True,
                    pii_type="email",
                    unique_count=100,
                    unique_percentage=100.0,
                ),
            ],
            quality_summary="Overall completeness: 98.3%",
            compliance_notes=["GDPR: 1 column contains PII"],
        )

    def test_to_dict(self, sample_docs):
        """Test conversion to dictionary."""
        result = sample_docs.to_dict()

        assert result["name"] == "test_dataset"
        assert result["n_rows"] == 100
        assert len(result["columns"]) == 3

    def test_to_json(self, sample_docs):
        """Test conversion to JSON."""
        json_str = sample_docs.to_json()
        parsed = json.loads(json_str)

        assert parsed["name"] == "test_dataset"
        assert parsed["n_columns"] == 3

    def test_to_markdown(self, sample_docs):
        """Test conversion to Markdown."""
        md = sample_docs.to_markdown()

        assert "# test_dataset" in md
        assert "A test dataset" in md
        assert "| id |" in md
        assert "🔒" in md  # PII marker
        assert "## Quality Summary" in md

    def test_save_markdown(self, sample_docs):
        """Test saving as Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.md"
            sample_docs.save(path)

            assert path.exists()
            content = path.read_text()
            assert "# test_dataset" in content

    def test_save_json(self, sample_docs):
        """Test saving as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.json"
            sample_docs.save(path)

            assert path.exists()
            content = json.loads(path.read_text())
            assert content["name"] == "test_dataset"


class TestDataDocumenter:
    """Tests for DataDocumenter class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "user_id": range(100),
            "name": [f"User {i}" for i in range(100)],
            "email": [f"user{i}@example.com" for i in range(100)],
            "age": np.random.randint(18, 80, 100),
            "score": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "is_active": np.random.choice([True, False], 100),
        })

    def test_document_basic(self, sample_data):
        """Test basic documentation."""
        documenter = DataDocumenter(level=DocumentationLevel.BASIC)
        docs = documenter.document(sample_data, name="users")

        assert docs.name == "users"
        assert docs.n_rows == 100
        assert docs.n_columns == 7
        assert len(docs.columns) == 7

    def test_document_detects_types(self, sample_data):
        """Test that documenter detects column types."""
        documenter = DataDocumenter()
        docs = documenter.document(sample_data)

        # Find columns by name
        col_map = {c.name: c for c in docs.columns}

        assert "int" in col_map["user_id"].dtype
        assert col_map["email"].pii_detected is True
        assert col_map["email"].pii_type == "email"

    def test_document_detects_semantic_types(self, sample_data):
        """Test semantic type detection."""
        documenter = DataDocumenter()
        docs = documenter.document(sample_data)

        col_map = {c.name: c for c in docs.columns}

        assert col_map["email"].semantic_type == "email"
        assert col_map["user_id"].semantic_type == "id"

    def test_document_calculates_stats(self, sample_data):
        """Test that stats are calculated correctly."""
        # Add some nulls
        sample_data.loc[0:9, "name"] = None

        documenter = DataDocumenter()
        docs = documenter.document(sample_data)

        col_map = {c.name: c for c in docs.columns}
        name_col = col_map["name"]

        assert name_col.null_count == 10
        assert name_col.null_percentage == 10.0

    def test_document_generates_quality_notes(self, sample_data):
        """Test quality note generation."""
        # Create column with high null rate
        sample_data["sparse"] = None
        sample_data.loc[0:10, "sparse"] = "value"

        documenter = DataDocumenter()
        docs = documenter.document(sample_data)

        col_map = {c.name: c for c in docs.columns}
        sparse_col = col_map["sparse"]

        assert any("null" in note.lower() for note in sparse_col.quality_notes)

    def test_document_detects_patterns(self):
        """Test pattern detection."""
        data = pd.DataFrame({
            "code": ["ABC001", "ABC002", "ABC003", "ABC004", "ABC005"] * 20,
        })

        documenter = DataDocumenter()
        docs = documenter.document(data)

        code_col = docs.columns[0]
        assert any("prefix" in p.lower() for p in code_col.patterns)

    def test_document_data_hash(self, sample_data):
        """Test that data hash is generated."""
        documenter = DataDocumenter()
        docs = documenter.document(sample_data)

        assert docs.data_hash
        assert len(docs.data_hash) == 32  # MD5 hash


class TestGenerateDocs:
    """Tests for generate_docs function."""

    def test_generate_docs_basic(self):
        """Test basic documentation generation."""
        data = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        docs = generate_docs(data, name="test")

        assert docs.name == "test"
        assert docs.n_rows == 3
        assert docs.n_columns == 2

    def test_generate_docs_with_description(self):
        """Test with custom description."""
        data = pd.DataFrame({"a": [1, 2, 3]})
        docs = generate_docs(data, description="Custom description")

        assert docs.description == "Custom description"

    def test_generate_docs_level_string(self):
        """Test with string level."""
        data = pd.DataFrame({"a": [1, 2, 3]})
        docs = generate_docs(data, level="basic")

        assert docs is not None

    def test_generate_docs_pii_detection(self):
        """Test PII detection."""
        data = pd.DataFrame({
            "email": ["test@example.com", "user@test.org"],
            "ssn": ["123-45-6789", "987-65-4321"],
        })

        docs = generate_docs(data, detect_pii=True)

        col_map = {c.name: c for c in docs.columns}
        assert col_map["email"].pii_detected is True
        assert col_map["ssn"].pii_detected is True

    def test_generate_docs_no_pii_detection(self):
        """Test with PII detection disabled."""
        data = pd.DataFrame({
            "email": ["test@example.com"],
        })

        docs = generate_docs(data, detect_pii=False)

        col_map = {c.name: c for c in docs.columns}
        assert col_map["email"].pii_detected is False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame()
        docs = generate_docs(data)

        assert docs.n_rows == 0
        assert docs.n_columns == 0

    def test_single_column(self):
        """Test with single column."""
        data = pd.DataFrame({"only_col": [1, 2, 3]})
        docs = generate_docs(data)

        assert docs.n_columns == 1
        assert docs.columns[0].name == "only_col"

    def test_all_null_column(self):
        """Test with all-null column."""
        data = pd.DataFrame({
            "null_col": [None, None, None],
            "normal": [1, 2, 3],
        })
        docs = generate_docs(data)

        col_map = {c.name: c for c in docs.columns}
        assert col_map["null_col"].null_percentage == 100.0

    def test_mixed_types(self):
        """Test with mixed types in object column."""
        data = pd.DataFrame({
            "mixed": [1, "two", 3.0, None],
        })
        docs = generate_docs(data)

        col_map = {c.name: c for c in docs.columns}
        # Should note mixed types
        assert col_map["mixed"].dtype == "object"

    def test_datetime_column(self):
        """Test with datetime column."""
        data = pd.DataFrame({
            "created_at": pd.date_range("2024-01-01", periods=10),
        })
        docs = generate_docs(data)

        col_map = {c.name: c for c in docs.columns}
        assert "datetime" in col_map["created_at"].dtype.lower()
