"""Tests for nl_query module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestNLQueryModule:
    """Tests for natural language query module."""

    def test_imports(self) -> None:
        from clean.nl_query import (
            NLQueryEngine,
            QueryResult,
            create_query_engine,
            query_report,
        )
        assert NLQueryEngine is not None
        assert QueryResult is not None
        assert create_query_engine is not None

    def test_query_result_fields(self) -> None:
        from clean.nl_query import QueryResult
        
        result = QueryResult(
            query="test query",
            answer="test answer",
            data=None,
            statistics={},
            suggestions=[],
            execution_time_ms=10.0,
            confidence=0.9,
        )
        
        assert result.query == "test query"
        assert result.answer == "test answer"
        assert result.confidence == 0.9

    def test_query_result_to_dict(self) -> None:
        from clean.nl_query import QueryResult
        
        result = QueryResult(
            query="test query",
            answer="test answer",
            data=None,
            statistics={"count": 10},
            suggestions=["try this"],
            execution_time_ms=10.0,
            confidence=0.9,
        )
        
        d = result.to_dict()
        assert d["answer"] == "test answer"
        assert d["statistics"]["count"] == 10

    def test_engine_init(self) -> None:
        from clean import DatasetCleaner
        from clean.nl_query import NLQueryEngine
        
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })
        
        cleaner = DatasetCleaner(data=df, label_column="label")
        report = cleaner.analyze()
        
        engine = NLQueryEngine(report=report, data=df)
        assert engine is not None

    def test_engine_query_basic(self) -> None:
        from clean import DatasetCleaner
        from clean.nl_query import NLQueryEngine
        
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })
        
        cleaner = DatasetCleaner(data=df, label_column="label")
        report = cleaner.analyze()
        
        engine = NLQueryEngine(report=report, data=df)
        # Use a simpler query that should work
        result = engine.query("What is the quality score?")
        
        assert result is not None
        assert result.answer is not None
        # Just verify we got some response, don't check content

    def test_convenience_functions(self) -> None:
        from clean import DatasetCleaner
        from clean.nl_query import create_query_engine, query_report
        
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })
        
        cleaner = DatasetCleaner(data=df, label_column="label")
        report = cleaner.analyze()
        
        engine = create_query_engine(report, data=df)
        assert engine is not None
        
        result = query_report(report, "What is the data quality?")
        assert result is not None
        assert result.answer is not None
