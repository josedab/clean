"""Data Quality Copilot - Natural Language Interface.

This module provides a natural language interface for exploring data quality
issues and generating fix scripts. Users can ask questions like:

- "Show me the worst label errors"
- "What are the most common issues in images of cats?"
- "Remove all duplicates with similarity > 0.95"

Example:
    >>> from clean.copilot import DataQualityCopilot, OpenAIProvider
    >>>
    >>> # Create copilot with your data
    >>> copilot = DataQualityCopilot(
    ...     data=df,
    ...     label_column="label",
    ...     provider=OpenAIProvider(api_key="sk-..."),
    ... )
    >>>
    >>> # Ask questions
    >>> result = copilot.ask("Show me the top 10 label errors")
    >>> print(result.answer)
    >>> print(result.data)
    >>>
    >>> # Generate fix scripts
    >>> script = copilot.generate_fix("Remove all outliers with z-score > 3")
    >>> print(script.code)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.core.report import QualityReport
from clean.exceptions import CleanError, ConfigurationError, DependencyError

if TYPE_CHECKING:
    from clean.llm_judge import LLMProvider

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Intent types for user queries."""

    SHOW_ISSUES = "show_issues"
    COUNT_ISSUES = "count_issues"
    FILTER_DATA = "filter_data"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    FIX_DATA = "fix_data"
    EXPORT = "export"
    VISUALIZE = "visualize"
    COMPARE = "compare"
    UNKNOWN = "unknown"


class IssueCategory(Enum):
    """Categories of issues users can query."""

    LABEL_ERRORS = "label_errors"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    BIAS = "bias"
    IMBALANCE = "imbalance"
    SHORT_RESPONSES = "short_responses"
    ALL = "all"


@dataclass
class QueryResult:
    """Result from a copilot query."""

    query: str
    intent: QueryIntent
    answer: str
    data: pd.DataFrame | None = None
    visualization: Any = None
    code: str | None = None
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def display(self) -> str:
        """Generate display string."""
        lines = [self.answer]

        if self.data is not None and len(self.data) > 0:
            lines.append("")
            lines.append("Data Preview:")
            lines.append(self.data.head(10).to_string())

        if self.code:
            lines.append("")
            lines.append("Generated Code:")
            lines.append("```python")
            lines.append(self.code)
            lines.append("```")

        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for s in self.suggestions:
                lines.append(f"  • {s}")

        return "\n".join(lines)


@dataclass
class FixScript:
    """Generated fix script from copilot."""

    description: str
    code: str
    estimated_impact: str
    warnings: list[str] = field(default_factory=list)
    dry_run_result: pd.DataFrame | None = None

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the fix script on data.

        WARNING: This executes generated code. Use with caution.
        """
        local_vars: dict[str, Any] = {"df": data.copy(), "pd": pd, "np": np}
        exec(self.code, {"pd": pd, "np": np}, local_vars)  # noqa: S102
        return local_vars.get("result", local_vars.get("df", data))


# Query patterns for intent detection
QUERY_PATTERNS = {
    QueryIntent.SHOW_ISSUES: [
        r"show\s+(me\s+)?(the\s+)?(all\s+)?(.+?)?(label\s*errors?|duplicates?|outliers?|issues?)",
        r"(what|which)\s+(are\s+)?(the\s+)?(.+?)?(errors?|duplicates?|outliers?|issues?)",
        r"(list|get|find)\s+(all\s+)?(.+?)?(errors?|duplicates?|outliers?|issues?)",
        r"(top|worst|best)\s+\d+\s+(label\s*)?errors?",
    ],
    QueryIntent.COUNT_ISSUES: [
        r"how\s+many\s+(.+?)?(errors?|duplicates?|outliers?|issues?)",
        r"count\s+(the\s+)?(.+?)?(errors?|duplicates?|outliers?|issues?)",
        r"(number|total)\s+of\s+(.+?)?(errors?|duplicates?|outliers?)",
    ],
    QueryIntent.SUMMARIZE: [
        r"summar(y|ize)",
        r"(give|show)\s+(me\s+)?(a\s+)?overview",
        r"(what|how)\s+(is\s+)?(the\s+)?(data\s+)?quality",
        r"quality\s+(score|report|summary)",
    ],
    QueryIntent.EXPLAIN: [
        r"(why|explain|what\s+causes?)",
        r"(tell|help)\s+(me\s+)?(understand|about)",
    ],
    QueryIntent.FIX_DATA: [
        r"(remove|delete|drop)\s+",
        r"(fix|correct|clean)\s+",
        r"(relabel|change\s+label)",
        r"(filter|keep)\s+only",
    ],
    QueryIntent.EXPORT: [
        r"(export|save|write|download)",
        r"(to|as)\s+(csv|json|parquet)",
    ],
    QueryIntent.VISUALIZE: [
        r"(plot|chart|graph|visualize|show\s+me\s+a\s+chart)",
        r"(distribution|histogram)",
    ],
    QueryIntent.FILTER_DATA: [
        r"(filter|where|with)\s+.*(confidence|score|threshold)",
        r"samples?\s+(with|where|having)",
    ],
}

ISSUE_PATTERNS = {
    IssueCategory.LABEL_ERRORS: [r"label\s*errors?", r"mislabel", r"wrong\s+label"],
    IssueCategory.DUPLICATES: [r"duplicates?", r"duplicate\s+pairs?", r"near.?duplicates?"],
    IssueCategory.OUTLIERS: [r"outliers?", r"anomal(y|ies)", r"unusual"],
    IssueCategory.BIAS: [r"bias", r"fairness", r"discrimination"],
    IssueCategory.IMBALANCE: [r"imbalance", r"class\s+distribution", r"skew"],
    IssueCategory.SHORT_RESPONSES: [r"short", r"too\s+short", r"brief"],
}


SYSTEM_PROMPT = """You are a data quality assistant helping users explore and fix issues in their ML datasets.

You have access to a quality report with the following information:
{report_summary}

When the user asks a question, analyze it and respond with a JSON object containing:
{{
  "intent": "show_issues|count_issues|filter_data|summarize|explain|fix_data|export|visualize",
  "issue_type": "label_errors|duplicates|outliers|bias|imbalance|all",
  "filters": {{"confidence_min": 0.0, "confidence_max": 1.0, "limit": 10, "class": null}},
  "answer": "Natural language answer to the question",
  "code": "Optional Python code to execute (using variable 'df' for data and 'report' for quality report)",
  "suggestions": ["Follow-up questions the user might want to ask"]
}}

Be helpful and specific. If the user asks about something not in the data, explain what's available."""


class DataQualityCopilot:
    """Natural language interface for data quality exploration.

    Allows users to ask questions about their data quality in plain English
    and generates appropriate visualizations, summaries, and fix scripts.

    Example:
        >>> copilot = DataQualityCopilot(data=df, label_column="label")
        >>> result = copilot.ask("What are the top 10 label errors?")
        >>> print(result.answer)
        >>> print(result.data)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        report: QualityReport | None = None,
        provider: "LLMProvider | None" = None,
        auto_analyze: bool = True,
    ):
        """Initialize the copilot.

        Args:
            data: DataFrame to analyze
            label_column: Column containing labels
            report: Pre-computed quality report (optional)
            provider: LLM provider for advanced queries (optional)
            auto_analyze: Automatically run analysis if no report provided
        """
        self.data = data.copy()
        self.label_column = label_column
        self.provider = provider

        if report is not None:
            self.report = report
        elif auto_analyze:
            cleaner = DatasetCleaner(data=data, label_column=label_column)
            self.report = cleaner.analyze()
        else:
            self.report = None

        self._query_history: list[QueryResult] = []

    def ask(self, query: str) -> QueryResult:
        """Ask a natural language question about data quality.

        Args:
            query: Natural language question

        Returns:
            QueryResult with answer, data, and suggestions

        Example:
            >>> result = copilot.ask("Show me label errors with confidence > 0.9")
            >>> print(result.answer)
        """
        query = query.strip()

        # Detect intent
        intent = self._detect_intent(query)
        issue_category = self._detect_issue_category(query)

        # Handle query based on intent
        if self.provider is not None:
            result = self._handle_with_llm(query, intent, issue_category)
        else:
            result = self._handle_with_rules(query, intent, issue_category)

        self._query_history.append(result)
        return result

    def generate_fix(self, description: str) -> FixScript:
        """Generate a fix script from natural language description.

        Args:
            description: Description of the fix to apply

        Returns:
            FixScript with code and metadata

        Example:
            >>> script = copilot.generate_fix("Remove all duplicates")
            >>> clean_df = script.execute(df)
        """
        # Parse the fix description
        fix_type, params = self._parse_fix_description(description)

        code_templates = {
            "remove_duplicates": """
# Remove duplicates
duplicate_indices = {duplicate_indices}
result = df.drop(index=duplicate_indices).reset_index(drop=True)
print(f"Removed {{len(duplicate_indices)}} duplicate samples")
""",
            "remove_outliers": """
# Remove outliers
outlier_indices = {outlier_indices}
result = df.drop(index=outlier_indices).reset_index(drop=True)
print(f"Removed {{len(outlier_indices)}} outliers")
""",
            "remove_label_errors": """
# Remove high-confidence label errors
error_indices = {error_indices}
result = df.drop(index=error_indices).reset_index(drop=True)
print(f"Removed {{len(error_indices)}} label errors")
""",
            "filter_confidence": """
# Filter by confidence threshold
threshold = {threshold}
keep_indices = [i for i in range(len(df)) if i not in {low_confidence_indices}]
result = df.iloc[keep_indices].reset_index(drop=True)
print(f"Kept {{len(keep_indices)}} samples with confidence >= {{threshold}}")
""",
        }

        # Get indices based on fix type
        if fix_type == "remove_duplicates":
            if self.report and hasattr(self.report, "duplicates"):
                dup_df = self.report.duplicates()
                indices = list(dup_df["index_2"].unique()) if "index_2" in dup_df.columns else []
            else:
                indices = []
            code = code_templates["remove_duplicates"].format(duplicate_indices=indices)
            impact = f"Will remove {len(indices)} duplicate samples"

        elif fix_type == "remove_outliers":
            if self.report and hasattr(self.report, "outliers"):
                out_df = self.report.outliers()
                indices = list(out_df["index"]) if "index" in out_df.columns else list(out_df.index)
            else:
                indices = []
            code = code_templates["remove_outliers"].format(outlier_indices=indices)
            impact = f"Will remove {len(indices)} outliers"

        elif fix_type == "remove_label_errors":
            threshold = params.get("threshold", 0.9)
            if self.report and hasattr(self.report, "label_errors"):
                err_df = self.report.label_errors()
                if "confidence" in err_df.columns:
                    high_conf = err_df[err_df["confidence"] >= threshold]
                    indices = list(high_conf["index"]) if "index" in high_conf.columns else list(high_conf.index)
                else:
                    indices = list(err_df.index)[:10]  # Limit to top 10
            else:
                indices = []
            code = code_templates["remove_label_errors"].format(error_indices=indices)
            impact = f"Will remove {len(indices)} label errors with confidence >= {threshold}"

        else:
            code = "# No specific fix identified\nresult = df.copy()"
            impact = "No changes will be made"

        warnings = []
        if len(self.data) > 0:
            removal_pct = len(indices) / len(self.data) * 100 if "indices" in dir() else 0
            if removal_pct > 10:
                warnings.append(f"This will remove {removal_pct:.1f}% of your data")

        return FixScript(
            description=description,
            code=code.strip(),
            estimated_impact=impact,
            warnings=warnings,
        )

    def suggest_queries(self) -> list[str]:
        """Suggest useful queries based on the data.

        Returns:
            List of suggested natural language queries
        """
        suggestions = [
            "Give me a quality summary",
            "What are the top 10 label errors?",
            "How many duplicates are there?",
            "Show me outliers with high confidence",
        ]

        if self.report:
            if hasattr(self.report, "label_errors"):
                err_df = self.report.label_errors()
                if len(err_df) > 0:
                    suggestions.append("Which classes have the most label errors?")

            if hasattr(self.report, "duplicates"):
                dup_df = self.report.duplicates()
                if len(dup_df) > 0:
                    suggestions.append("Show me the most similar duplicate pairs")

        return suggestions

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent using pattern matching."""
        query_lower = query.lower()

        for intent, patterns in QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return QueryIntent.UNKNOWN

    def _detect_issue_category(self, query: str) -> IssueCategory:
        """Detect which issue category the query is about."""
        query_lower = query.lower()

        for category, patterns in ISSUE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return category

        return IssueCategory.ALL

    def _handle_with_rules(
        self,
        query: str,
        intent: QueryIntent,
        issue_category: IssueCategory,
    ) -> QueryResult:
        """Handle query using rule-based logic."""
        # Extract limit from query
        limit_match = re.search(r"(top|first|worst)\s+(\d+)", query.lower())
        limit = int(limit_match.group(2)) if limit_match else 10

        # Extract confidence threshold
        conf_match = re.search(r"confidence\s*[>>=]+\s*([\d.]+)", query.lower())
        min_confidence = float(conf_match.group(1)) if conf_match else None

        data_result = None
        answer = ""
        suggestions = []

        if intent == QueryIntent.SUMMARIZE:
            if self.report:
                answer = self.report.summary()
                suggestions = [
                    "Show me the label errors",
                    "What are the worst duplicates?",
                    "How can I improve data quality?",
                ]
            else:
                answer = "No quality report available. Run analysis first."

        elif intent == QueryIntent.SHOW_ISSUES:
            if issue_category == IssueCategory.LABEL_ERRORS:
                if self.report and hasattr(self.report, "label_errors"):
                    data_result = self.report.label_errors()
                    if min_confidence and "confidence" in data_result.columns:
                        data_result = data_result[data_result["confidence"] >= min_confidence]
                    data_result = data_result.head(limit)
                    answer = f"Found {len(self.report.label_errors())} label errors. Showing top {limit}."
                else:
                    answer = "No label errors found in the report."

            elif issue_category == IssueCategory.DUPLICATES:
                if self.report and hasattr(self.report, "duplicates"):
                    data_result = self.report.duplicates().head(limit)
                    answer = f"Found {len(self.report.duplicates())} duplicate pairs. Showing top {limit}."
                else:
                    answer = "No duplicates found in the report."

            elif issue_category == IssueCategory.OUTLIERS:
                if self.report and hasattr(self.report, "outliers"):
                    data_result = self.report.outliers().head(limit)
                    answer = f"Found {len(self.report.outliers())} outliers. Showing top {limit}."
                else:
                    answer = "No outliers found in the report."

            else:
                answer = "Showing all detected issues."
                suggestions = [
                    "Show me label errors",
                    "Show me duplicates",
                    "Show me outliers",
                ]

        elif intent == QueryIntent.COUNT_ISSUES:
            counts = {}
            if self.report:
                if hasattr(self.report, "label_errors"):
                    counts["label_errors"] = len(self.report.label_errors())
                if hasattr(self.report, "duplicates"):
                    counts["duplicates"] = len(self.report.duplicates())
                if hasattr(self.report, "outliers"):
                    counts["outliers"] = len(self.report.outliers())

            answer = "Issue counts:\n" + "\n".join(
                f"  - {k}: {v}" for k, v in counts.items()
            )

        elif intent == QueryIntent.FIX_DATA:
            fix_script = self.generate_fix(query)
            answer = f"Generated fix script:\n\n{fix_script.description}\n\nEstimated impact: {fix_script.estimated_impact}"
            return QueryResult(
                query=query,
                intent=intent,
                answer=answer,
                code=fix_script.code,
                suggestions=["Preview the fix with dry_run=True", "Execute with script.execute(df)"],
            )

        else:
            answer = f"I understood your query as: {intent.value}. "
            if self.report:
                answer += f"Your dataset has {len(self.data)} samples with quality score {self.report.score.overall:.2f}."
            suggestions = self.suggest_queries()

        return QueryResult(
            query=query,
            intent=intent,
            answer=answer,
            data=data_result,
            suggestions=suggestions,
        )

    def _handle_with_llm(
        self,
        query: str,
        intent: QueryIntent,
        issue_category: IssueCategory,
    ) -> QueryResult:
        """Handle query using LLM for more sophisticated understanding."""
        if self.provider is None:
            return self._handle_with_rules(query, intent, issue_category)

        # Build context
        report_summary = self.report.summary() if self.report else "No report available"

        prompt = f"""User query: {query}

Based on the quality report, provide a helpful response.

Quality Report Summary:
{report_summary}

Respond with JSON containing:
- answer: Your natural language response
- issue_type: label_errors, duplicates, outliers, or all
- filters: {{limit: 10, min_confidence: null}}
- code: Optional Python code using 'df' and 'report' variables
- suggestions: List of follow-up questions
"""

        try:
            response_text, _ = self.provider.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT.format(report_summary=report_summary),
                temperature=0.0,
                max_tokens=1024,
            )

            parsed = self._parse_llm_response(response_text)

            # Execute any code if provided
            data_result = None
            if parsed.get("code") and self.report:
                try:
                    local_vars = {"df": self.data, "report": self.report, "pd": pd, "np": np}
                    exec(parsed["code"], {"pd": pd, "np": np}, local_vars)  # noqa: S102
                    if "result" in local_vars:
                        data_result = local_vars["result"]
                        if isinstance(data_result, pd.DataFrame):
                            data_result = data_result.head(20)
                except Exception as e:
                    logger.warning(f"Code execution failed: {e}")

            return QueryResult(
                query=query,
                intent=intent,
                answer=parsed.get("answer", "I couldn't understand that query."),
                data=data_result,
                code=parsed.get("code"),
                suggestions=parsed.get("suggestions", []),
            )

        except Exception as e:
            logger.warning(f"LLM query failed: {e}. Falling back to rules.")
            return self._handle_with_rules(query, intent, issue_category)

    def _parse_llm_response(self, text: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        text = text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"answer": text}

    def _parse_fix_description(self, description: str) -> tuple[str, dict[str, Any]]:
        """Parse fix description to determine fix type and parameters."""
        desc_lower = description.lower()
        params: dict[str, Any] = {}

        if "duplicate" in desc_lower:
            fix_type = "remove_duplicates"
            # Extract similarity threshold if mentioned
            sim_match = re.search(r"similarity\s*[>>=]+\s*([\d.]+)", desc_lower)
            if sim_match:
                params["similarity_threshold"] = float(sim_match.group(1))

        elif "outlier" in desc_lower:
            fix_type = "remove_outliers"
            # Extract z-score threshold if mentioned
            z_match = re.search(r"z.?score\s*[>>=]+\s*([\d.]+)", desc_lower)
            if z_match:
                params["z_threshold"] = float(z_match.group(1))

        elif "label" in desc_lower and ("error" in desc_lower or "wrong" in desc_lower):
            fix_type = "remove_label_errors"
            # Extract confidence threshold
            conf_match = re.search(r"confidence\s*[>>=]+\s*([\d.]+)", desc_lower)
            params["threshold"] = float(conf_match.group(1)) if conf_match else 0.9

        elif "filter" in desc_lower or "keep" in desc_lower:
            fix_type = "filter_confidence"
            conf_match = re.search(r"confidence\s*[>>=]+\s*([\d.]+)", desc_lower)
            params["threshold"] = float(conf_match.group(1)) if conf_match else 0.5

        else:
            fix_type = "unknown"

        return fix_type, params

    @property
    def history(self) -> list[QueryResult]:
        """Get query history."""
        return self._query_history.copy()

    def clear_history(self) -> None:
        """Clear query history."""
        self._query_history.clear()


def create_copilot(
    data: pd.DataFrame,
    label_column: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> DataQualityCopilot:
    """Create a Data Quality Copilot instance.

    Args:
        data: DataFrame to analyze
        label_column: Column containing labels
        provider: LLM provider name ("openai", "anthropic", "ollama")
        api_key: API key for the provider
        **kwargs: Additional arguments

    Returns:
        DataQualityCopilot instance

    Example:
        >>> copilot = create_copilot(df, label_column="label", provider="openai", api_key="sk-...")
        >>> result = copilot.ask("Show me the worst label errors")
    """
    llm_provider = None

    if provider:
        from clean.llm_judge import create_provider
        llm_provider = create_provider(provider, api_key)

    return DataQualityCopilot(
        data=data,
        label_column=label_column,
        provider=llm_provider,
        **kwargs,
    )


__all__ = [
    "DataQualityCopilot",
    "QueryResult",
    "FixScript",
    "QueryIntent",
    "IssueCategory",
    "create_copilot",
]
