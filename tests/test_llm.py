"""Tests for the LLM data quality analyzer."""

import pandas as pd
import pytest

from clean.llm import (
    InstructionIssue,
    LLMDataCleaner,
)


@pytest.fixture
def instruction_dataset():
    """Create a sample instruction-tuning dataset."""
    return pd.DataFrame({
        "instruction": [
            "What is the capital of France?",
            "Explain quantum computing",
            "What is the capital of France?",  # Duplicate
            "Write a poem about nature",
            "Translate to Spanish: Hello",
            "What is 2+2?",
            "Tell me about AI",
        ],
        "response": [
            "The capital of France is Paris.",
            "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously...",
            "Paris is the capital of France.",  # Different response, duplicate instruction
            "",  # Empty response
            "Hola",  # Too short
            "4",  # Very short
            "I cannot help with that as an AI language model...",  # Refusal
        ],
    })


@pytest.fixture
def rag_dataset():
    """Create a sample RAG document dataset."""
    return pd.DataFrame({
        "document": [
            "This is a comprehensive guide to machine learning algorithms...",
            "Python is a popular programming language used for data science.",
            "This is a comprehensive guide to machine learning algorithms...",  # Duplicate
            "Hi",  # Too short
            "Deep learning is a subset of machine learning that uses neural networks.",
        ],
        "source": ["doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.txt", "doc5.pdf"],
    })


class TestLLMDataCleaner:
    """Tests for LLMDataCleaner."""

    def test_init_instruction_mode(self, instruction_dataset):
        """Test initialization in instruction mode."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        assert cleaner.instruction_column == "instruction"
        assert cleaner.response_column == "response"

    def test_init_rag_mode(self, rag_dataset):
        """Test initialization in RAG mode."""
        cleaner = LLMDataCleaner(
            data=rag_dataset,
            text_column="document",
        )
        assert cleaner.text_column == "document"

    def test_analyze_requires_columns(self):
        """Test that analyze requires proper columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        cleaner = LLMDataCleaner(data=df)

        with pytest.raises(ValueError):
            cleaner.analyze()

    def test_detect_duplicate_instructions(self, instruction_dataset):
        """Test detection of duplicate instructions."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        report = cleaner.analyze(
            check_duplicates=True,
            check_length=False,
            check_coherence=False,
            check_format=False,
        )

        duplicates = report.get_issues_by_type("duplicate")
        assert len(duplicates) == 1
        assert duplicates[0].index == 2  # Third row is duplicate

    def test_detect_short_responses(self, instruction_dataset):
        """Test detection of too-short responses."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
            min_response_length=5,
        )
        report = cleaner.analyze(
            check_duplicates=False,
            check_length=True,
            check_coherence=False,
            check_format=False,
        )

        short = report.get_issues_by_type("too_short")
        # Should find: empty response, "Hola" (4 chars), "4" (1 char)
        assert len(short) >= 2

    def test_detect_refusals(self, instruction_dataset):
        """Test detection of refusal patterns."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        report = cleaner.analyze(
            check_duplicates=False,
            check_length=False,
            check_coherence=True,
            check_format=False,
        )

        refusals = report.get_issues_by_type("refusal")
        assert len(refusals) == 1
        assert refusals[0].index == 6  # "I cannot help..." response

    def test_detect_incoherent_empty(self, instruction_dataset):
        """Test detection of empty responses."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        report = cleaner.analyze(
            check_duplicates=False,
            check_length=False,
            check_coherence=True,
            check_format=False,
        )

        incoherent = report.get_issues_by_type("incoherent")
        # Empty response should be flagged
        assert any(i.index == 3 for i in incoherent)

    def test_detect_document_duplicates(self, rag_dataset):
        """Test detection of duplicate documents in RAG mode."""
        cleaner = LLMDataCleaner(
            data=rag_dataset,
            text_column="document",
        )
        report = cleaner.analyze(
            check_duplicates=True,
            check_length=False,
        )

        duplicates = report.get_issues_by_type("duplicate")
        assert len(duplicates) == 1
        assert duplicates[0].index == 2

    def test_detect_short_documents(self, rag_dataset):
        """Test detection of too-short documents in RAG mode."""
        cleaner = LLMDataCleaner(
            data=rag_dataset,
            text_column="document",
            min_response_length=10,
        )
        report = cleaner.analyze(
            check_duplicates=False,
            check_length=True,
        )

        short = report.get_issues_by_type("too_short")
        assert len(short) == 1
        assert short[0].index == 3  # "Hi"

    def test_full_analysis(self, instruction_dataset):
        """Test running full analysis."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        report = cleaner.analyze()

        assert report.n_samples == 7
        assert report.n_issues > 0
        assert 0 <= report.quality_score <= 1
        assert 0 <= report.duplicate_rate <= 1
        assert 0 <= report.coherence_score <= 1


class TestLLMQualityReport:
    """Tests for LLMQualityReport."""

    def test_summary(self, instruction_dataset):
        """Test report summary generation."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        report = cleaner.analyze()
        summary = report.summary()

        assert "LLM Data Quality Report" in summary
        assert "Samples analyzed: 7" in summary
        assert "quality score" in summary.lower()

    def test_to_dataframe(self, instruction_dataset):
        """Test converting issues to DataFrame."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        report = cleaner.analyze()
        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "index" in df.columns
            assert "issue_type" in df.columns
            assert "confidence" in df.columns


class TestGetCleanData:
    """Tests for get_clean_data method."""

    def test_remove_duplicates(self, instruction_dataset):
        """Test removing duplicates."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
        )
        clean_df = cleaner.get_clean_data(
            remove_duplicates=True,
            remove_short=False,
            remove_incoherent=False,
        )

        # Should have removed 1 duplicate
        assert len(clean_df) == len(instruction_dataset) - 1

    def test_remove_short(self, instruction_dataset):
        """Test removing short responses."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
            min_response_length=5,
        )
        clean_df = cleaner.get_clean_data(
            remove_duplicates=False,
            remove_short=True,
            remove_incoherent=False,
        )

        # Should have fewer rows
        assert len(clean_df) < len(instruction_dataset)

    def test_remove_multiple_issues(self, instruction_dataset):
        """Test removing multiple issue types."""
        cleaner = LLMDataCleaner(
            data=instruction_dataset,
            instruction_column="instruction",
            response_column="response",
            min_response_length=5,
        )
        clean_df = cleaner.get_clean_data(
            remove_duplicates=True,
            remove_short=True,
            remove_incoherent=True,
        )

        # Should be significantly smaller
        assert len(clean_df) < len(instruction_dataset)


class TestInstructionIssue:
    """Tests for InstructionIssue dataclass."""

    def test_create_issue(self):
        """Test creating an issue."""
        issue = InstructionIssue(
            index=42,
            issue_type="duplicate",
            severity="medium",
            description="Duplicate instruction found",
            confidence=1.0,
            metadata={"original_index": 10},
        )

        assert issue.index == 42
        assert issue.issue_type == "duplicate"
        assert issue.confidence == 1.0
        assert issue.metadata["original_index"] == 10
