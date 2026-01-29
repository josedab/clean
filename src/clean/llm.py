"""LLM Data Quality Analyzer.

This module provides specialized quality analysis for LLM datasets including:
- Instruction-response pairs (fine-tuning data)
- RAG document collections
- Prompt templates

Example:
    >>> from clean.llm import LLMDataCleaner
    >>>
    >>> # Analyze instruction-tuning dataset
    >>> cleaner = LLMDataCleaner(
    ...     data=df,
    ...     instruction_column="instruction",
    ...     response_column="response",
    ... )
    >>> report = cleaner.analyze()
    >>> print(report.summary())
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.constants import (
    DEFAULT_MIN_TEXT_LENGTH,
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_SIMILARITY_THRESHOLD,
    SHORT_CONTENT_THRESHOLD,
)

if TYPE_CHECKING:
    pass


@dataclass
class InstructionIssue:
    """Issue detected in an instruction-response pair."""

    index: int
    issue_type: str  # 'incoherent', 'too_short', 'too_long', 'duplicate', 'low_quality'
    severity: str  # 'low', 'medium', 'high'
    description: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGDocumentIssue:
    """Issue detected in a RAG document."""

    index: int
    issue_type: str  # 'duplicate', 'low_relevance', 'poor_chunking', 'missing_context'
    severity: str
    description: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMQualityReport:
    """Quality report for LLM datasets."""

    n_samples: int
    issues: list[InstructionIssue | RAGDocumentIssue]
    quality_score: float
    duplicate_rate: float
    short_response_rate: float
    coherence_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "LLM Data Quality Report",
            "=" * 50,
            f"Samples analyzed: {self.n_samples:,}",
            f"Overall quality score: {self.quality_score:.2f}",
            "",
            "Issue Summary:",
            f"  Total issues: {self.n_issues}",
            f"  Duplicate rate: {self.duplicate_rate:.1%}",
            f"  Short response rate: {self.short_response_rate:.1%}",
            f"  Coherence score: {self.coherence_score:.2f}",
            "",
        ]

        # Group issues by type
        issue_counts: dict[str, int] = {}
        for issue in self.issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1

        if issue_counts:
            lines.append("Issues by type:")
            for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {issue_type}: {count}")

        return "\n".join(lines)

    def get_issues_by_type(self, issue_type: str) -> list[InstructionIssue | RAGDocumentIssue]:
        """Get issues filtered by type."""
        return [i for i in self.issues if i.issue_type == issue_type]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to DataFrame."""
        if not self.issues:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "index": i.index,
                "issue_type": i.issue_type,
                "severity": i.severity,
                "confidence": i.confidence,
                "description": i.description,
            }
            for i in self.issues
        ])


class LLMDataCleaner:
    """Quality analyzer for LLM datasets.

    Specialized for detecting issues in:
    - Instruction-tuning datasets (instruction/response pairs)
    - RLHF preference datasets
    - RAG document collections

    Example:
        >>> cleaner = LLMDataCleaner(
        ...     data=df,
        ...     instruction_column="instruction",
        ...     response_column="response",
        ... )
        >>> report = cleaner.analyze()
        >>> print(report.summary())

        >>> # Get specific issues
        >>> duplicates = report.get_issues_by_type("duplicate")
        >>> low_quality = report.get_issues_by_type("low_quality")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        instruction_column: str | None = None,
        response_column: str | None = None,
        text_column: str | None = None,  # For RAG documents
        embedding_column: str | None = None,
        min_response_length: int = DEFAULT_MIN_TEXT_LENGTH,
        max_response_length: int = DEFAULT_MAX_TEXT_LENGTH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """Initialize the LLM data cleaner.

        Args:
            data: DataFrame with LLM data
            instruction_column: Column with instructions/prompts
            response_column: Column with responses/completions
            text_column: Column with text (for RAG documents)
            embedding_column: Column with pre-computed embeddings
            min_response_length: Minimum acceptable response length
            max_response_length: Maximum acceptable response length
            similarity_threshold: Threshold for duplicate detection
        """
        self.data = data.copy()
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.text_column = text_column
        self.embedding_column = embedding_column
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.similarity_threshold = similarity_threshold

        self._embeddings: np.ndarray | None = None

    def analyze(
        self,
        check_duplicates: bool = True,
        check_length: bool = True,
        check_coherence: bool = True,
        check_format: bool = True,
        show_progress: bool = True,
    ) -> LLMQualityReport:
        """Run quality analysis on the LLM dataset.

        Args:
            check_duplicates: Check for duplicate instructions/responses
            check_length: Check for too short/long responses
            check_coherence: Check instruction-response coherence
            check_format: Check for formatting issues
            show_progress: Show progress bar

        Returns:
            LLMQualityReport with detected issues
        """
        issues: list[InstructionIssue] = []

        # Determine mode
        is_instruction_mode = self.instruction_column and self.response_column
        is_rag_mode = self.text_column is not None

        if not is_instruction_mode and not is_rag_mode:
            raise ValueError(
                "Must provide either instruction_column+response_column or text_column"
            )

        if is_instruction_mode:
            if check_duplicates:
                issues.extend(self._check_instruction_duplicates())

            if check_length:
                issues.extend(self._check_response_length())

            if check_coherence:
                issues.extend(self._check_coherence())

            if check_format:
                issues.extend(self._check_format())

        elif is_rag_mode:
            if check_duplicates:
                issues.extend(self._check_document_duplicates())

            if check_length:
                issues.extend(self._check_document_length())

        # Calculate metrics
        n_samples = len(self.data)
        duplicate_count = len([i for i in issues if i.issue_type == "duplicate"])
        short_count = len([i for i in issues if i.issue_type == "too_short"])

        duplicate_rate = duplicate_count / n_samples if n_samples > 0 else 0
        short_rate = short_count / n_samples if n_samples > 0 else 0

        # Coherence score (1 = all coherent, 0 = all incoherent)
        incoherent_count = len([i for i in issues if i.issue_type == "incoherent"])
        coherence_score = 1 - (incoherent_count / n_samples) if n_samples > 0 else 1

        # Overall quality score
        issue_rate = len(issues) / n_samples if n_samples > 0 else 0
        quality_score = max(0, 1 - issue_rate)

        return LLMQualityReport(
            n_samples=n_samples,
            issues=issues,
            quality_score=quality_score,
            duplicate_rate=duplicate_rate,
            short_response_rate=short_rate,
            coherence_score=coherence_score,
            metadata={
                "instruction_column": self.instruction_column,
                "response_column": self.response_column,
                "text_column": self.text_column,
            },
        )

    def _check_instruction_duplicates(self) -> list[InstructionIssue]:
        """Check for duplicate instructions."""
        issues = []

        if self.instruction_column is None:
            return issues

        # Hash-based exact duplicate detection
        instruction_hashes: dict[str, int] = {}

        for idx, row in self.data.iterrows():
            instruction = str(row[self.instruction_column])
            inst_hash = hashlib.md5(instruction.encode()).hexdigest()

            if inst_hash in instruction_hashes:
                original_idx = instruction_hashes[inst_hash]
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="duplicate",
                    severity="medium",
                    description=f"Duplicate instruction (same as index {original_idx})",
                    confidence=1.0,
                    metadata={"original_index": original_idx, "type": "exact"},
                ))
            else:
                instruction_hashes[inst_hash] = int(idx)

        return issues

    def _check_response_length(self) -> list[InstructionIssue]:
        """Check for too short or too long responses."""
        issues = []

        if self.response_column is None:
            return issues

        for idx, row in self.data.iterrows():
            response = str(row[self.response_column])
            length = len(response)

            if length < self.min_response_length:
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="too_short",
                    severity="high" if length < SHORT_CONTENT_THRESHOLD else "medium",
                    description=f"Response too short ({length} chars, min {self.min_response_length})",
                    confidence=1.0,
                    metadata={"length": length, "min": self.min_response_length},
                ))
            elif length > self.max_response_length:
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="too_long",
                    severity="low",
                    description=f"Response very long ({length} chars, max {self.max_response_length})",
                    confidence=0.8,
                    metadata={"length": length, "max": self.max_response_length},
                ))

        return issues

    def _check_coherence(self) -> list[InstructionIssue]:
        """Check instruction-response coherence using heuristics."""
        issues = []

        if self.instruction_column is None or self.response_column is None:
            return issues

        for idx, row in self.data.iterrows():
            instruction = str(row[self.instruction_column]).lower()
            response = str(row[self.response_column]).lower()

            # Heuristic 1: Response that just repeats instruction
            if len(response) > 0 and response in instruction:
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="incoherent",
                    severity="high",
                    description="Response appears to just repeat part of the instruction",
                    confidence=0.9,
                    metadata={"reason": "repetition"},
                ))
                continue

            # Heuristic 2: Response starts with "I cannot" or similar refusals
            refusal_patterns = [
                "i cannot", "i can't", "i'm not able", "i am not able",
                "as an ai", "as a language model", "i don't have",
            ]
            if any(response.strip().startswith(p) for p in refusal_patterns):
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="refusal",
                    severity="medium",
                    description="Response appears to be a refusal/limitation statement",
                    confidence=0.85,
                    metadata={"reason": "refusal_pattern"},
                ))
                continue

            # Heuristic 3: Empty or near-empty response
            if len(response.strip()) < 3:
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="incoherent",
                    severity="high",
                    description="Response is essentially empty",
                    confidence=1.0,
                    metadata={"reason": "empty"},
                ))

        return issues

    def _check_format(self) -> list[InstructionIssue]:
        """Check for formatting issues in responses."""
        issues = []

        if self.response_column is None:
            return issues

        for idx, row in self.data.iterrows():
            response = str(row[self.response_column])

            # Check for potential truncation
            if response.rstrip().endswith(("...", "…")):
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="truncated",
                    severity="medium",
                    description="Response appears to be truncated",
                    confidence=0.7,
                    metadata={"ending": response[-10:]},
                ))

            # Check for code blocks without language tags
            if "```" in response:
                # Simple check for unclosed code blocks
                if response.count("```") % 2 != 0:
                    issues.append(InstructionIssue(
                        index=int(idx),
                        issue_type="format_error",
                        severity="low",
                        description="Unclosed code block detected",
                        confidence=0.8,
                        metadata={"reason": "unclosed_code_block"},
                    ))

        return issues

    def _check_document_duplicates(self) -> list[InstructionIssue]:
        """Check for duplicate documents (RAG mode)."""
        issues = []

        if self.text_column is None:
            return issues

        # Hash-based exact duplicate detection
        doc_hashes: dict[str, int] = {}

        for idx, row in self.data.iterrows():
            text = str(row[self.text_column])
            doc_hash = hashlib.md5(text.encode()).hexdigest()

            if doc_hash in doc_hashes:
                original_idx = doc_hashes[doc_hash]
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="duplicate",
                    severity="medium",
                    description=f"Duplicate document (same as index {original_idx})",
                    confidence=1.0,
                    metadata={"original_index": original_idx},
                ))
            else:
                doc_hashes[doc_hash] = int(idx)

        return issues

    def _check_document_length(self) -> list[InstructionIssue]:
        """Check document lengths (RAG mode)."""
        issues = []

        if self.text_column is None:
            return issues

        for idx, row in self.data.iterrows():
            text = str(row[self.text_column])
            length = len(text)

            if length < self.min_response_length:
                issues.append(InstructionIssue(
                    index=int(idx),
                    issue_type="too_short",
                    severity="medium",
                    description=f"Document too short ({length} chars)",
                    confidence=1.0,
                    metadata={"length": length},
                ))

        return issues

    def get_clean_data(
        self,
        remove_duplicates: bool = True,
        remove_short: bool = True,
        remove_incoherent: bool = False,
    ) -> pd.DataFrame:
        """Get cleaned dataset with issues removed.

        Args:
            remove_duplicates: Remove duplicate instructions
            remove_short: Remove too-short responses
            remove_incoherent: Remove incoherent pairs

        Returns:
            Cleaned DataFrame
        """
        report = self.analyze(show_progress=False)

        indices_to_remove = set()

        for issue in report.issues:
            if remove_duplicates and issue.issue_type == "duplicate":
                indices_to_remove.add(issue.index)
            if remove_short and issue.issue_type == "too_short":
                indices_to_remove.add(issue.index)
            if remove_incoherent and issue.issue_type == "incoherent":
                indices_to_remove.add(issue.index)

        return self.data.drop(indices_to_remove).reset_index(drop=True)


__all__ = [
    "LLMDataCleaner",
    "LLMQualityReport",
    "InstructionIssue",
    "RAGDocumentIssue",
]
