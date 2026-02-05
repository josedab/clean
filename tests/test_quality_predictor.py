"""Tests for quality_predictor module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestQualityPredictorModule:
    """Tests for quality predictor module."""

    def test_imports(self) -> None:
        from clean.quality_predictor import (
            QualityPredictor,
            PredictorConfig,
            QualityPrediction,
            predict_quality,
        )
        assert QualityPredictor is not None
        assert PredictorConfig is not None
        assert QualityPrediction is not None

    def test_config_defaults(self) -> None:
        from clean.quality_predictor import PredictorConfig, PredictionModel
        config = PredictorConfig()
        assert config.model_type == PredictionModel.GRADIENT_BOOSTING
        assert config.n_estimators > 0
        assert config.min_samples_for_training > 0

    def test_predictor_init(self) -> None:
        from clean.quality_predictor import QualityPredictor, PredictorConfig
        predictor = QualityPredictor()
        assert predictor is not None

        config = PredictorConfig(n_estimators=50)
        predictor_with_config = QualityPredictor(config=config)
        assert predictor_with_config is not None

    def test_predictor_fit(self) -> None:
        from clean.quality_predictor import QualityPredictor, PredictorConfig
        
        np.random.seed(42)
        # Need at least 10 datasets for training
        datasets = [
            pd.DataFrame({
                "feature_0": np.random.randn(100),
                "feature_1": np.random.randn(100),
                "label": np.random.choice([0, 1], 100),
            })
            for _ in range(10)
        ]
        quality_scores = [0.7, 0.8, 0.6, 0.9, 0.75, 0.72, 0.85, 0.65, 0.88, 0.78]
        
        predictor = QualityPredictor()
        predictor.fit(datasets, quality_scores, label_columns=["label"] * 10)
        assert predictor._fitted is True

    def test_predictor_predict(self) -> None:
        from clean.quality_predictor import QualityPredictor, QualityPrediction, ConfidenceLevel
        
        np.random.seed(42)
        # Need at least 10 datasets for training
        datasets = [
            pd.DataFrame({
                "feature_0": np.random.randn(100),
                "feature_1": np.random.randn(100),
                "label": np.random.choice([0, 1], 100),
            })
            for _ in range(10)
        ]
        quality_scores = [0.7, 0.8, 0.6, 0.9, 0.75, 0.72, 0.85, 0.65, 0.88, 0.78]
        
        predictor = QualityPredictor()
        predictor.fit(datasets, quality_scores, label_columns=["label"] * 10)
        
        test_data = pd.DataFrame({
            "feature_0": np.random.randn(50),
            "feature_1": np.random.randn(50),
            "label": np.random.choice([0, 1], 50),
        })
        
        prediction = predictor.predict(test_data, label_column="label")
        assert isinstance(prediction, QualityPrediction)
        assert 0 <= prediction.quality_score <= 1
        assert prediction.confidence >= 0
        assert prediction.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
        assert len(prediction.confidence_interval) == 2

    def test_prediction_to_dict(self) -> None:
        from clean.quality_predictor import QualityPrediction, ConfidenceLevel
        
        prediction = QualityPrediction(
            quality_score=0.85,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            confidence_interval=(0.8, 0.9),
            prediction_time_ms=10.5,
            features_used=["f1", "f2"],
            feature_importances={"f1": 0.6, "f2": 0.4},
            warnings=[],
        )
        
        result = prediction.to_dict()
        assert result["quality_score"] == 0.85
        assert result["confidence"] == 0.9
        assert result["confidence_level"] == "high"

    def test_convenience_function(self) -> None:
        from clean.quality_predictor import QualityPredictor, predict_quality
        
        np.random.seed(42)
        # Need at least 10 datasets for training
        datasets = [
            pd.DataFrame({
                "feature_0": np.random.randn(100),
                "feature_1": np.random.randn(100),
                "label": np.random.choice([0, 1], 100),
            })
            for _ in range(10)
        ]
        quality_scores = [0.7, 0.8, 0.6, 0.9, 0.75, 0.72, 0.85, 0.65, 0.88, 0.78]
        
        predictor = QualityPredictor()
        predictor.fit(datasets, quality_scores, label_columns=["label"] * 10)
        
        test_data = pd.DataFrame({
            "feature_0": np.random.randn(50),
            "feature_1": np.random.randn(50),
            "label": np.random.choice([0, 1], 50),
        })
        
        prediction = predict_quality(test_data, predictor, label_column="label")
        assert prediction is not None
        assert 0 <= prediction.quality_score <= 1
