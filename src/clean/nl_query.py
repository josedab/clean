"""Natural Language Query Interface for data quality exploration.

This module provides an LLM-powered chat interface for exploring data
quality issues using natural language queries.

Example:
    >>> from clean.nl_query import NLQueryEngine
    >>>
    >>> engine = NLQueryEngine(report=quality_report)
    >>> result = engine.query("show me label errors with confidence > 0.9")
    >>> print(result.answer)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clean.core.report import QualityReport

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classified intent of a natural language query."""

    LIST_ISSUES = "list_issues"
    FILTER_ISSUES = "filter_issues"
    GET_SUMMARY = "get_summary"
    GET_STATISTICS = "get_statistics"
    COMPARE = "compare"
    EXPLAIN = "explain"
    SUGGEST_FIX = "suggest_fix"
    EXPORT = "export"
    UNKNOWN = "unknown"


class IssueType(Enum):
    """Types of data quality issues."""

    LABEL_ERROR = "label_error"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    BIAS = "bias"
    IMBALANCE = "imbalance"
    MISSING = "missing"
    ALL = "all"


@dataclass
class ParsedQuery:
    """Parsed representation of a natural language query."""

    original_query: str
    intent: QueryIntent
    issue_type: IssueType | None
    filters: dict[str, Any]
    limit: int | None
    sort_by: str | None
    sort_order: str
    class_filter: str | None
    confidence_threshold: float | None
    columns: list[str] | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "issue_type": self.issue_type.value if self.issue_type else None,
            "filters": self.filters,
            "limit": self.limit,
            "sort_by": self.sort_by,
            "class_filter": self.class_filter,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class QueryResult:
    """Result of a natural language query."""

    query: ParsedQuery
    answer: str
    data: pd.DataFrame | None
    statistics: dict[str, Any] | None
    suggestions: list[str]
    execution_time_ms: float
    confidence: float  # How confident we are in the interpretation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "has_data": self.data is not None,
            "data_rows": len(self.data) if self.data is not None else 0,
            "statistics": self.statistics,
            "suggestions": self.suggestions,
            "execution_time_ms": self.execution_time_ms,
            "confidence": self.confidence,
        }

    def show(self) -> None:
        """Display result in console."""
        print(self.answer)
        if self.data is not None and len(self.data) > 0:
            print("\n" + self.data.to_string())
        if self.suggestions:
            print("\nSuggestions:")
            for s in self.suggestions:
                print(f"  • {s}")


@dataclass
class ConversationContext:
    """Context for multi-turn conversations."""

    history: list[tuple[str, QueryResult]] = field(default_factory=list)
    last_issue_type: IssueType | None = None
    last_filters: dict[str, Any] = field(default_factory=dict)
    last_data: pd.DataFrame | None = None

    def add_turn(self, query: str, result: QueryResult) -> None:
        """Add a conversation turn."""
        self.history.append((query, result))
        if result.query.issue_type:
            self.last_issue_type = result.query.issue_type
        if result.query.filters:
            self.last_filters = result.query.filters
        if result.data is not None:
            self.last_data = result.data

    def get_recent_context(self, n: int = 3) -> str:
        """Get recent conversation context as string."""
        if not self.history:
            return ""

        context_parts = []
        for query, result in self.history[-n:]:
            context_parts.append(f"Q: {query}")
            context_parts.append(f"A: {result.answer[:200]}...")

        return "\n".join(context_parts)


class QueryParser:
    """Parse natural language queries into structured form."""

    ISSUE_KEYWORDS = {
        "label": IssueType.LABEL_ERROR,
        "label error": IssueType.LABEL_ERROR,
        "mislabel": IssueType.LABEL_ERROR,
        "wrong label": IssueType.LABEL_ERROR,
        "duplicate": IssueType.DUPLICATE,
        "near-duplicate": IssueType.DUPLICATE,
        "similar": IssueType.DUPLICATE,
        "outlier": IssueType.OUTLIER,
        "anomaly": IssueType.OUTLIER,
        "anomalies": IssueType.OUTLIER,
        "bias": IssueType.BIAS,
        "fairness": IssueType.BIAS,
        "imbalance": IssueType.IMBALANCE,
        "class imbalance": IssueType.IMBALANCE,
        "missing": IssueType.MISSING,
        "null": IssueType.MISSING,
        "all": IssueType.ALL,
        "everything": IssueType.ALL,
    }

    INTENT_PATTERNS = {
        QueryIntent.GET_SUMMARY: [
            r"summary", r"overview", r"summarize", r"report",
            r"how.*quality", r"overall", r"total",
        ],
        QueryIntent.LIST_ISSUES: [
            r"show", r"list", r"get", r"find", r"display",
            r"what are", r"which", r"give me",
        ],
        QueryIntent.FILTER_ISSUES: [
            r"where", r"with", r"filter", r"confidence.*>",
            r"class.*=", r"in category", r"only",
        ],
        QueryIntent.GET_STATISTICS: [
            r"how many", r"count", r"statistics", r"stats",
            r"number of", r"percentage", r"distribution",
        ],
        QueryIntent.COMPARE: [
            r"compare", r"difference", r"versus", r"vs",
            r"between.*and", r"contrast",
        ],
        QueryIntent.EXPLAIN: [
            r"why", r"explain", r"reason", r"cause",
            r"what.*wrong", r"understand",
        ],
        QueryIntent.SUGGEST_FIX: [
            r"fix", r"correct", r"resolve", r"suggestion",
            r"recommend", r"how.*to.*fix", r"what.*should",
        ],
        QueryIntent.EXPORT: [
            r"export", r"save", r"download", r"csv",
            r"json", r"file",
        ],
    }

    def parse(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> ParsedQuery:
        """Parse a natural language query.

        Args:
            query: Natural language query string
            context: Optional conversation context

        Returns:
            ParsedQuery with structured representation
        """
        query_lower = query.lower().strip()

        # Detect intent
        intent = self._detect_intent(query_lower)

        # Detect issue type
        issue_type = self._detect_issue_type(query_lower)
        if issue_type is None and context and context.last_issue_type:
            issue_type = context.last_issue_type

        # Extract filters
        filters = self._extract_filters(query_lower, context)

        # Extract limit
        limit = self._extract_limit(query_lower)

        # Extract sort
        sort_by, sort_order = self._extract_sort(query_lower)

        # Extract class filter
        class_filter = self._extract_class_filter(query_lower)

        # Extract confidence threshold
        confidence_threshold = self._extract_confidence(query_lower)

        # Extract columns
        columns = self._extract_columns(query_lower)

        return ParsedQuery(
            original_query=query,
            intent=intent,
            issue_type=issue_type,
            filters=filters,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            class_filter=class_filter,
            confidence_threshold=confidence_threshold,
            columns=columns,
        )

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of the query."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        return QueryIntent.UNKNOWN

    def _detect_issue_type(self, query: str) -> IssueType | None:
        """Detect which issue type the query refers to."""
        for keyword, issue_type in self.ISSUE_KEYWORDS.items():
            if keyword in query:
                return issue_type
        return None

    def _extract_filters(
        self,
        query: str,
        context: ConversationContext | None,
    ) -> dict[str, Any]:
        """Extract filter conditions from query."""
        filters = {}

        # Inherit from context if using "those" or "them"
        if context and any(w in query for w in ["those", "them", "these"]):
            filters.update(context.last_filters)

        return filters

    def _extract_limit(self, query: str) -> int | None:
        """Extract result limit from query."""
        # Match patterns like "top 10", "first 5", "10 results"
        patterns = [
            r"top\s+(\d+)",
            r"first\s+(\d+)",
            r"(\d+)\s+results?",
            r"(\d+)\s+samples?",
            r"(\d+)\s+items?",
            r"show\s+(\d+)",
            r"limit\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return int(match.group(1))

        return None

    def _extract_sort(self, query: str) -> tuple[str | None, str]:
        """Extract sort field and order."""
        sort_by = None
        sort_order = "desc"

        if "confidence" in query:
            sort_by = "confidence"
        elif "score" in query:
            sort_by = "score"

        if "ascending" in query or "lowest" in query or "least" in query:
            sort_order = "asc"

        return sort_by, sort_order

    def _extract_class_filter(self, query: str) -> str | None:
        """Extract class/category filter."""
        # Match patterns like "class 'cat'", "category=dog", "in class X"
        patterns = [
            r"class\s*[=:]\s*['\"]?(\w+)['\"]?",
            r"category\s*[=:]\s*['\"]?(\w+)['\"]?",
            r"in\s+class\s+['\"]?(\w+)['\"]?",
            r"for\s+['\"]?(\w+)['\"]?\s+class",
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    def _extract_confidence(self, query: str) -> float | None:
        """Extract confidence threshold from query."""
        # Match patterns like "confidence > 0.9", "confidence above 90%"
        patterns = [
            r"confidence\s*[>>=]\s*(\d+\.?\d*)\s*%?",
            r"confidence\s+above\s+(\d+\.?\d*)\s*%?",
            r"confidence\s+over\s+(\d+\.?\d*)\s*%?",
            r"high\s+confidence",
        ]

        for pattern in patterns:
            if pattern == r"high\s+confidence":
                if re.search(pattern, query):
                    return 0.9
            else:
                match = re.search(pattern, query)
                if match:
                    value = float(match.group(1))
                    return value / 100 if value > 1 else value

        return None

    def _extract_columns(self, query: str) -> list[str] | None:
        """Extract specific columns to return."""
        if "only" not in query and "just" not in query:
            return None

        columns = []
        possible_cols = ["index", "confidence", "label", "predicted", "score"]
        for col in possible_cols:
            if col in query:
                columns.append(col)

        return columns if columns else None


class QueryExecutor:
    """Execute parsed queries against quality reports."""

    def __init__(
        self,
        report: QualityReport,
        data: pd.DataFrame | None = None,
    ):
        """Initialize executor.

        Args:
            report: Quality report to query
            data: Optional original dataset
        """
        self.report = report
        self.data = data

    def execute(self, parsed: ParsedQuery) -> QueryResult:
        """Execute a parsed query.

        Args:
            parsed: Parsed query

        Returns:
            QueryResult with answer and data
        """
        import time

        start = time.perf_counter()

        handlers = {
            QueryIntent.GET_SUMMARY: self._handle_summary,
            QueryIntent.LIST_ISSUES: self._handle_list,
            QueryIntent.FILTER_ISSUES: self._handle_filter,
            QueryIntent.GET_STATISTICS: self._handle_statistics,
            QueryIntent.COMPARE: self._handle_compare,
            QueryIntent.EXPLAIN: self._handle_explain,
            QueryIntent.SUGGEST_FIX: self._handle_suggest_fix,
            QueryIntent.EXPORT: self._handle_export,
            QueryIntent.UNKNOWN: self._handle_unknown,
        }

        handler = handlers.get(parsed.intent, self._handle_unknown)
        answer, data, stats, suggestions, confidence = handler(parsed)

        execution_time = (time.perf_counter() - start) * 1000

        return QueryResult(
            query=parsed,
            answer=answer,
            data=data,
            statistics=stats,
            suggestions=suggestions,
            execution_time_ms=execution_time,
            confidence=confidence,
        )

    def _handle_summary(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle summary queries."""
        summary = self.report.summary()
        stats = {
            "quality_score": self.report.quality_score,
            "n_samples": self.report.n_samples,
            "n_issues": self.report.total_issues,
        }

        suggestions = []
        if self.report.total_issues > 0:
            suggestions.append("Try 'show label errors' to see mislabeled samples")
            suggestions.append("Try 'how many duplicates?' for duplicate count")

        return summary, None, stats, suggestions, 0.95

    def _handle_list(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle list queries."""
        issue_type = parsed.issue_type or IssueType.ALL

        if issue_type == IssueType.LABEL_ERROR:
            return self._list_label_errors(parsed)
        elif issue_type == IssueType.DUPLICATE:
            return self._list_duplicates(parsed)
        elif issue_type == IssueType.OUTLIER:
            return self._list_outliers(parsed)
        elif issue_type == IssueType.ALL:
            return self._list_all_issues(parsed)
        else:
            return self._list_all_issues(parsed)

    def _list_label_errors(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """List label errors."""
        errors = self.report.label_errors()

        if errors is None or len(errors) == 0:
            return "No label errors detected.", None, {"count": 0}, [], 0.9

        # Apply confidence filter
        if parsed.confidence_threshold:
            errors = errors[errors["confidence"] >= parsed.confidence_threshold]

        # Apply class filter
        if parsed.class_filter:
            if "given_label" in errors.columns:
                errors = errors[errors["given_label"] == parsed.class_filter]

        # Sort
        if parsed.sort_by == "confidence" and "confidence" in errors.columns:
            errors = errors.sort_values(
                "confidence",
                ascending=(parsed.sort_order == "asc"),
            )

        # Limit
        if parsed.limit:
            errors = errors.head(parsed.limit)

        n_total = len(self.report.label_errors()) if self.report.label_errors() is not None else 0
        n_shown = len(errors)

        answer = f"Found {n_total} label errors"
        if parsed.confidence_threshold:
            answer += f" with confidence ≥ {parsed.confidence_threshold:.0%}"
        if parsed.class_filter:
            answer += f" for class '{parsed.class_filter}'"
        answer += f". Showing {n_shown}."

        suggestions = [
            "Try 'explain why sample X is a label error'",
            "Try 'suggest fixes for these label errors'",
        ]

        return answer, errors, {"count": n_total, "shown": n_shown}, suggestions, 0.9

    def _list_duplicates(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """List duplicates."""
        duplicates = self.report.duplicates()

        if duplicates is None or len(duplicates) == 0:
            return "No duplicates detected.", None, {"count": 0}, [], 0.9

        if parsed.limit:
            duplicates = duplicates.head(parsed.limit)

        n_total = len(self.report.duplicates()) if self.report.duplicates() is not None else 0

        answer = f"Found {n_total} duplicate pairs. Showing {len(duplicates)}."

        suggestions = [
            "Try 'suggest how to handle duplicates'",
        ]

        return answer, duplicates, {"count": n_total}, suggestions, 0.85

    def _list_outliers(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """List outliers."""
        outliers = self.report.outliers()

        if outliers is None or len(outliers) == 0:
            return "No outliers detected.", None, {"count": 0}, [], 0.9

        if parsed.limit:
            outliers = outliers[:parsed.limit]

        outlier_df = pd.DataFrame({"index": outliers})

        answer = f"Found {len(outliers)} outliers."

        suggestions = [
            "Try 'explain why these are outliers'",
        ]

        return answer, outlier_df, {"count": len(outliers)}, suggestions, 0.85

    def _list_all_issues(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """List all issues."""
        issues = []

        label_errors = self.report.label_errors()
        if label_errors is not None:
            issues.append(f"Label errors: {len(label_errors)}")

        duplicates = self.report.duplicates()
        if duplicates is not None:
            issues.append(f"Duplicates: {len(duplicates)} pairs")

        outliers = self.report.outliers()
        if outliers is not None:
            issues.append(f"Outliers: {len(outliers)}")

        answer = f"Issues found:\n" + "\n".join(f"  • {i}" for i in issues)

        suggestions = [
            "Try 'show label errors' for details",
            "Try 'get statistics for all issues'",
        ]

        return answer, None, {"total_issues": self.report.total_issues}, suggestions, 0.9

    def _handle_filter(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle filter queries."""
        return self._handle_list(parsed)

    def _handle_statistics(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle statistics queries."""
        stats = {
            "quality_score": self.report.quality_score,
            "n_samples": self.report.n_samples,
            "total_issues": self.report.total_issues,
        }

        # Add issue-specific counts
        issue_type = parsed.issue_type

        if issue_type in (IssueType.LABEL_ERROR, IssueType.ALL, None):
            errors = self.report.label_errors()
            stats["label_errors"] = len(errors) if errors is not None else 0

        if issue_type in (IssueType.DUPLICATE, IssueType.ALL, None):
            dupes = self.report.duplicates()
            stats["duplicates"] = len(dupes) if dupes is not None else 0

        if issue_type in (IssueType.OUTLIER, IssueType.ALL, None):
            outliers = self.report.outliers()
            stats["outliers"] = len(outliers) if outliers is not None else 0

        # Format answer
        lines = ["Data Quality Statistics:"]
        lines.append(f"  Quality Score: {stats['quality_score']:.1f}/100")
        lines.append(f"  Samples: {stats['n_samples']:,}")
        lines.append(f"  Total Issues: {stats['total_issues']}")

        if "label_errors" in stats:
            rate = stats["label_errors"] / stats["n_samples"] * 100
            lines.append(f"  Label Errors: {stats['label_errors']} ({rate:.1f}%)")

        if "duplicates" in stats:
            lines.append(f"  Duplicates: {stats['duplicates']} pairs")

        if "outliers" in stats:
            rate = stats["outliers"] / stats["n_samples"] * 100
            lines.append(f"  Outliers: {stats['outliers']} ({rate:.1f}%)")

        return "\n".join(lines), None, stats, [], 0.95

    def _handle_compare(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle compare queries."""
        answer = "Comparison feature requires multiple reports. Please provide another report to compare."
        suggestions = ["Use 'analyze second_dataset.csv' first"]
        return answer, None, None, suggestions, 0.5

    def _handle_explain(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle explain queries."""
        issue_type = parsed.issue_type or IssueType.LABEL_ERROR

        explanations = {
            IssueType.LABEL_ERROR: (
                "Label errors are detected using confident learning. "
                "These samples have high model confidence that the label is wrong. "
                "The model predicts a different class with high probability."
            ),
            IssueType.DUPLICATE: (
                "Duplicates are found using exact matching and semantic similarity. "
                "Near-duplicates have very similar content even if not exactly the same. "
                "This can cause data leakage if duplicates appear in train and test sets."
            ),
            IssueType.OUTLIER: (
                "Outliers are samples that differ significantly from the rest. "
                "They may be errors, rare cases, or genuinely unusual samples. "
                "Methods used: Isolation Forest, Local Outlier Factor, Z-score."
            ),
        }

        answer = explanations.get(
            issue_type,
            "Data quality issues can affect model performance. "
            "Clean helps identify and fix these issues automatically.",
        )

        return answer, None, None, [], 0.8

    def _handle_suggest_fix(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle fix suggestion queries."""
        issue_type = parsed.issue_type or IssueType.ALL

        fixes = []

        if issue_type in (IssueType.LABEL_ERROR, IssueType.ALL):
            errors = self.report.label_errors()
            if errors is not None and len(errors) > 0:
                fixes.append(
                    f"Label Errors ({len(errors)}):\n"
                    "  1. Review high-confidence errors (confidence > 0.9) first\n"
                    "  2. Consider auto-correcting errors with confidence > 0.95\n"
                    "  3. Flag medium-confidence errors for manual review"
                )

        if issue_type in (IssueType.DUPLICATE, IssueType.ALL):
            dupes = self.report.duplicates()
            if dupes is not None and len(dupes) > 0:
                fixes.append(
                    f"Duplicates ({len(dupes)} pairs):\n"
                    "  1. Remove exact duplicates (keep first occurrence)\n"
                    "  2. Review near-duplicates manually\n"
                    "  3. Ensure no duplicates across train/test split"
                )

        if issue_type in (IssueType.OUTLIER, IssueType.ALL):
            outliers = self.report.outliers()
            if outliers is not None and len(outliers) > 0:
                fixes.append(
                    f"Outliers ({len(outliers)}):\n"
                    "  1. Review outliers - they may be valuable edge cases\n"
                    "  2. Remove only clear data entry errors\n"
                    "  3. Consider keeping outliers if they're valid rare cases"
                )

        if not fixes:
            answer = "No major issues found that need fixing!"
        else:
            answer = "Fix Recommendations:\n\n" + "\n\n".join(fixes)

        return answer, None, None, [], 0.85

    def _handle_export(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle export queries."""
        answer = (
            "To export data, use:\n"
            "  • report.label_errors().to_csv('label_errors.csv')\n"
            "  • report.to_json('report.json')\n"
            "  • clean fix data.csv --output cleaned.csv"
        )
        return answer, None, None, [], 0.9

    def _handle_unknown(
        self,
        parsed: ParsedQuery,
    ) -> tuple[str, pd.DataFrame | None, dict | None, list[str], float]:
        """Handle unknown queries."""
        suggestions = [
            "Try 'show me the summary'",
            "Try 'list label errors'",
            "Try 'how many duplicates?'",
            "Try 'suggest fixes'",
        ]

        answer = (
            "I'm not sure how to answer that. Here are some things you can ask:\n"
            + "\n".join(f"  • {s}" for s in suggestions)
        )

        return answer, None, None, suggestions, 0.3


class NLQueryEngine:
    """Natural language query engine for data quality exploration.

    Supports queries like:
    - "show me label errors with confidence > 0.9"
    - "how many duplicates are there?"
    - "explain why sample 42 is an outlier"
    - "suggest fixes for label errors in class 'cat'"
    """

    def __init__(
        self,
        report: QualityReport,
        data: pd.DataFrame | None = None,
        llm_provider: Any | None = None,
    ):
        """Initialize query engine.

        Args:
            report: Quality report to query
            data: Optional original dataset
            llm_provider: Optional LLM provider for advanced queries
        """
        self.report = report
        self.data = data
        self.llm_provider = llm_provider
        self.parser = QueryParser()
        self.executor = QueryExecutor(report, data)
        self.context = ConversationContext()

    def query(self, text: str) -> QueryResult:
        """Execute a natural language query.

        Args:
            text: Natural language query

        Returns:
            QueryResult with answer and data
        """
        # Parse query
        parsed = self.parser.parse(text, self.context)

        # If we have an LLM and parsing confidence is low, use LLM
        if self.llm_provider and parsed.intent == QueryIntent.UNKNOWN:
            result = self._query_with_llm(text, parsed)
        else:
            result = self.executor.execute(parsed)

        # Update context
        self.context.add_turn(text, result)

        return result

    def _query_with_llm(self, text: str, parsed: ParsedQuery) -> QueryResult:
        """Use LLM to interpret and answer query."""
        # This would integrate with the llm_judge module for advanced queries
        # For now, fall back to regular execution
        return self.executor.execute(parsed)

    def chat(self) -> None:
        """Start interactive chat session."""
        print("Data Quality Chat")
        print("=" * 50)
        print("Ask questions about your data quality report.")
        print("Type 'quit' or 'exit' to end.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "bye"):
                print("Goodbye!")
                break

            result = self.query(user_input)
            print(f"\nAssistant: {result.answer}\n")

            if result.data is not None and len(result.data) > 0:
                print(result.data.head(10).to_string())
                if len(result.data) > 10:
                    print(f"... ({len(result.data) - 10} more rows)")
                print()

    def get_suggestions(self) -> list[str]:
        """Get suggested queries based on report."""
        suggestions = []

        if self.report.total_issues > 0:
            suggestions.append("What issues were found?")

        errors = self.report.label_errors()
        if errors is not None and len(errors) > 0:
            suggestions.append(f"Show me the {len(errors)} label errors")
            suggestions.append("Show label errors with confidence > 0.9")

        dupes = self.report.duplicates()
        if dupes is not None and len(dupes) > 0:
            suggestions.append("How many duplicates are there?")

        suggestions.append("Give me a summary of the quality report")
        suggestions.append("Suggest how to fix the issues")

        return suggestions

    def reset_context(self) -> None:
        """Reset conversation context."""
        self.context = ConversationContext()


def create_query_engine(
    report: QualityReport,
    data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> NLQueryEngine:
    """Create a natural language query engine.

    Args:
        report: Quality report to query
        data: Optional original dataset
        **kwargs: Additional arguments

    Returns:
        NLQueryEngine instance
    """
    return NLQueryEngine(report=report, data=data, **kwargs)


def query_report(
    report: QualityReport,
    query: str,
    **kwargs: Any,
) -> QueryResult:
    """Convenience function to query a report.

    Args:
        report: Quality report
        query: Natural language query

    Returns:
        QueryResult
    """
    engine = NLQueryEngine(report=report, **kwargs)
    return engine.query(query)
