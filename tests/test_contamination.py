"""Tests for contamination module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestContaminationModule:
    """Tests for cross-dataset contamination detection module."""

    def test_imports(self) -> None:
        from clean.contamination import (
            ContaminationDetector,
            ContaminationReport,
            SeverityLevel,
            ContaminationConfig,
            detect_contamination,
        )
        assert ContaminationDetector is not None
        assert ContaminationReport is not None
        assert SeverityLevel is not None

    def test_severity_levels(self) -> None:
        from clean.contamination import SeverityLevel
        
        assert SeverityLevel.CRITICAL is not None
        assert SeverityLevel.HIGH is not None
        assert SeverityLevel.MEDIUM is not None
        assert SeverityLevel.LOW is not None

    def test_config_defaults(self) -> None:
        from clean.contamination import ContaminationConfig
        
        config = ContaminationConfig()
        assert config is not None

    def test_detector_init(self) -> None:
        from clean.contamination import ContaminationDetector, ContaminationConfig
        
        detector = ContaminationDetector()
        assert detector is not None
        
        config = ContaminationConfig()
        detector_with_config = ContaminationDetector(config=config)
        assert detector_with_config is not None

    def test_detect_basic(self) -> None:
        from clean.contamination import ContaminationDetector, ContaminationReport
        
        np.random.seed(42)
        # Create two datasets with some overlap
        train_data = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
        })
        
        # Test has some samples from train (contamination)
        test_data = pd.concat([
            train_data.iloc[:10],  # Leaked samples
            pd.DataFrame({
                "feature_0": np.random.randn(40),
                "feature_1": np.random.randn(40),
            })
        ], ignore_index=True)
        
        detector = ContaminationDetector()
        report = detector.detect(train_data, test_data)
        
        assert isinstance(report, ContaminationReport)
        # datasets_compared is a list of dataset names
        assert len(report.datasets_compared) == 2
        assert report.timestamp is not None

    def test_convenience_function(self) -> None:
        from clean.contamination import detect_contamination, ContaminationReport
        
        np.random.seed(42)
        train_data = pd.DataFrame({
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
        })
        
        test_data = pd.concat([
            train_data.iloc[:5],  # Leaked samples
            pd.DataFrame({
                "feature_0": np.random.randn(45),
                "feature_1": np.random.randn(45),
            })
        ], ignore_index=True)
        
        report = detect_contamination(train_data, test_data)
        
        assert isinstance(report, ContaminationReport)
        assert len(report.datasets_compared) == 2

    def test_report_fields(self) -> None:
        from clean.contamination import ContaminationReport
        from datetime import datetime
        
        # ContaminationReport doesn't have to_dict, check fields directly
        report = ContaminationReport(
            timestamp=datetime.now(),
            datasets_compared=["train", "test"],  # It's a list of names
            n_contaminated=5,
            contamination_rate=0.05,
            contaminated_pairs=[],
            contamination_by_type={},
            contamination_by_severity={},
            recommendations=[],
        )
        
        assert len(report.datasets_compared) == 2
        assert report.n_contaminated == 5
        assert report.contamination_rate == 0.05

    def test_detect_with_text_columns(self) -> None:
        from clean.contamination import ContaminationDetector
        
        # Create data with consistent rows
        train_data = pd.DataFrame({
            "text": ["hello world"] * 10 + ["foo bar"] * 10 + ["test text"] * 10,
            "label": [0] * 10 + [1] * 10 + [0] * 10,
        })
        
        test_data = pd.DataFrame({
            "text": ["hello world"] * 5 + ["new text"] * 5 + ["different"] * 5,
            "label": [0] * 5 + [1] * 5 + [1] * 5,
        })
        
        detector = ContaminationDetector()
        report = detector.detect(train_data, test_data, text_columns=["text"])
        
        assert report is not None
        assert len(report.datasets_compared) == 2
