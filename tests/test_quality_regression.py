"""Tests for quality_regression module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestQualityRegressionModule:
    """Tests for quality regression testing module."""

    def test_imports(self) -> None:
        from clean.quality_regression import (
            QualityRegressionTester,
            QualitySnapshot,
            RegressionTestConfig,
            QualityTestResult,
            MetricThreshold,
        )
        assert QualityRegressionTester is not None
        assert QualitySnapshot is not None
        assert RegressionTestConfig is not None

    def test_config_defaults(self) -> None:
        from clean.quality_regression import RegressionTestConfig
        
        config = RegressionTestConfig()
        assert config.quality_score_warning > 0
        assert config.quality_score_critical > 0

    def test_snapshot_creation(self) -> None:
        from clean.quality_regression import QualitySnapshot
        from datetime import datetime
        
        snapshot = QualitySnapshot(
            id="test-snapshot-1",
            timestamp=datetime.now(),
            dataset_name="test_dataset",
            n_samples=1000,
            metrics={"quality_score": 0.85, "label_error_rate": 0.05},
            metadata={"version": "1.0"},
        )
        
        assert snapshot.id == "test-snapshot-1"
        assert snapshot.n_samples == 1000
        assert snapshot.metrics["quality_score"] == 0.85

    def test_tester_init(self) -> None:
        from clean.quality_regression import QualityRegressionTester, RegressionTestConfig
        
        tester = QualityRegressionTester()
        assert tester is not None
        
        config = RegressionTestConfig(quality_score_warning=0.05)
        tester_with_config = QualityRegressionTester(config=config)
        assert tester_with_config is not None

    def test_set_baseline_with_snapshot(self) -> None:
        from clean.quality_regression import QualityRegressionTester, QualitySnapshot
        from datetime import datetime
        
        baseline = QualitySnapshot(
            id="baseline-1",
            timestamp=datetime.now(),
            dataset_name="baseline",
            n_samples=1000,
            metrics={"quality_score": 0.9, "label_error_rate": 0.02},
            metadata={},
        )
        
        tester = QualityRegressionTester()
        tester.set_baseline(baseline)
        assert tester._baseline is not None

    def test_test_with_snapshot(self) -> None:
        from clean.quality_regression import QualityRegressionTester, QualitySnapshot, QualityTestResult
        from datetime import datetime
        
        baseline = QualitySnapshot(
            id="baseline",
            timestamp=datetime.now(),
            dataset_name="baseline",
            n_samples=1000,
            metrics={"quality_score": 0.9, "label_error_rate": 0.02},
            metadata={},
        )
        
        current = QualitySnapshot(
            id="current",
            timestamp=datetime.now(),
            dataset_name="current",
            n_samples=1000,
            metrics={"quality_score": 0.85, "label_error_rate": 0.03},
            metadata={},
        )
        
        tester = QualityRegressionTester()
        tester.set_baseline(baseline)
        
        result = tester.test(current)
        
        assert isinstance(result, QualityTestResult)
        assert result.overall_passed is not None
