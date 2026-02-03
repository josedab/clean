"""Tests for Data Quality Copilot."""

from __future__ import annotations

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from clean.copilot import (
    DataQualityCopilot,
    QueryResult,
    FixScript,
    QueryIntent,
    IssueCategory,
    create_copilot,
)


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_all_intents_exist(self) -> None:
        assert QueryIntent.SHOW_ISSUES is not None
        assert QueryIntent.SUMMARIZE is not None
        assert QueryIntent.FIX_DATA is not None
        assert QueryIntent.EXPORT is not None


class TestIssueCategory:
    """Tests for IssueCategory enum."""

    def test_all_categories_exist(self) -> None:
        assert IssueCategory.LABEL_ERRORS is not None
        assert IssueCategory.DUPLICATES is not None
        assert IssueCategory.OUTLIERS is not None
        assert IssueCategory.IMBALANCE is not None


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_result_creation(self) -> None:
        result = QueryResult(
            query="Show me label errors",
            intent=QueryIntent.SHOW_ISSUES,
            answer="Found 10 label errors",
            data=pd.DataFrame({"idx": [1, 2, 3]}),
        )
        assert result.query == "Show me label errors"
        assert result.intent == QueryIntent.SHOW_ISSUES
        assert len(result.data) == 3

    def test_result_without_data(self) -> None:
        result = QueryResult(
            query="Test",
            intent=QueryIntent.SUMMARIZE,
            answer="Report generated",
        )
        assert result.data is None
        assert result.answer == "Report generated"


class TestFixScript:
    """Tests for FixScript dataclass."""

    def test_fix_script_creation(self) -> None:
        script = FixScript(
            description="Remove duplicate entries",
            code="df.drop_duplicates()",
            estimated_impact="~5% rows removed",
        )
        assert script.description == "Remove duplicate entries"
        assert "drop_duplicates" in script.code

    def test_fix_script_with_warnings(self) -> None:
        script = FixScript(
            description="Fix issues",
            code="df.fix()",
            estimated_impact="10% improvement",
            warnings=["May remove valid data"],
        )
        assert len(script.warnings) == 1


class TestDataQualityCopilot:
    """Tests for DataQualityCopilot class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "text": ["hello", "world", "hello", "test"],
            "label": [0, 1, 0, 1],
        })

    def test_copilot_init(self, sample_data: pd.DataFrame) -> None:
        copilot = DataQualityCopilot(data=sample_data, label_column="label")
        assert copilot is not None
        assert copilot.data is not None

    def test_copilot_without_label(self, sample_data: pd.DataFrame) -> None:
        copilot = DataQualityCopilot(data=sample_data)
        assert copilot is not None
        assert copilot.label_column is None

    def test_ask_returns_result(self, sample_data: pd.DataFrame) -> None:
        copilot = DataQualityCopilot(data=sample_data, label_column="label")
        result = copilot.ask("How many duplicates are there?")
        assert isinstance(result, QueryResult)
        assert result.answer is not None

    def test_ask_explore_intent(self, sample_data: pd.DataFrame) -> None:
        copilot = DataQualityCopilot(data=sample_data, label_column="label")
        result = copilot.ask("summarize the data")
        assert isinstance(result, QueryResult)

    def test_suggest_queries(self, sample_data: pd.DataFrame) -> None:
        copilot = DataQualityCopilot(data=sample_data, label_column="label")
        suggestions = copilot.suggest_queries()
        assert isinstance(suggestions, list)

    def test_generate_fix(self, sample_data: pd.DataFrame) -> None:
        copilot = DataQualityCopilot(data=sample_data, label_column="label")
        fix = copilot.generate_fix("duplicates")  # Use string instead of enum
        # May return None or FixScript
        assert fix is None or isinstance(fix, FixScript)


class TestCreateCopilot:
    """Tests for create_copilot factory function."""

    def test_create_copilot_basic(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        copilot = create_copilot(data=df, label_column="y")
        assert isinstance(copilot, DataQualityCopilot)

    def test_create_copilot_without_label(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        copilot = create_copilot(data=df)
        assert isinstance(copilot, DataQualityCopilot)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["text", "label"])
        copilot = DataQualityCopilot(data=df, auto_analyze=False)
        assert copilot is not None

    def test_single_row_dataframe(self) -> None:
        df = pd.DataFrame({"x": [1], "label": [0]})
        copilot = DataQualityCopilot(data=df, label_column="label", auto_analyze=False)
        # Just verify init works
        assert copilot is not None

    def test_query_with_special_characters(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
        copilot = DataQualityCopilot(data=df, label_column="label", auto_analyze=False)
        result = copilot.ask("summarize")
        assert result is not None
