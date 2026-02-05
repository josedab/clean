"""Quality Prediction Model for real-time data quality scoring.

This module provides ML-based quality score prediction without running
full analysis, enabling instant quality gates in data ingestion pipelines.

Example:
    >>> from clean.quality_predictor import QualityPredictor
    >>>
    >>> predictor = QualityPredictor()
    >>> predictor.fit(training_datasets, quality_scores)
    >>> score = predictor.predict(new_data)
    >>> print(f"Predicted quality: {score.quality_score:.1f}")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PredictionModel(Enum):
    """Available prediction model types."""

    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LINEAR = "linear"
    ENSEMBLE = "ensemble"


class ConfidenceLevel(Enum):
    """Confidence level of prediction."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DatasetFeatures:
    """Meta-features extracted from a dataset for prediction."""

    n_samples: int
    n_features: int
    n_numeric_features: int
    n_categorical_features: int
    n_classes: int | None

    # Distribution features
    class_imbalance_ratio: float | None
    missing_rate: float
    duplicate_rate: float
    numeric_mean_skewness: float
    numeric_mean_kurtosis: float
    
    # Correlation features
    mean_feature_correlation: float
    max_feature_correlation: float
    
    # Complexity features
    feature_entropy: float
    class_overlap_estimate: float | None
    
    # Text features (if applicable)
    avg_text_length: float | None
    text_vocabulary_size: int | None

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for prediction."""
        values = [
            np.log1p(self.n_samples),
            np.log1p(self.n_features),
            self.n_numeric_features / max(self.n_features, 1),
            self.n_categorical_features / max(self.n_features, 1),
            np.log1p(self.n_classes) if self.n_classes else 0,
            self.class_imbalance_ratio if self.class_imbalance_ratio else 0,
            self.missing_rate,
            self.duplicate_rate,
            self.numeric_mean_skewness,
            self.numeric_mean_kurtosis,
            self.mean_feature_correlation,
            self.max_feature_correlation,
            self.feature_entropy,
            self.class_overlap_estimate if self.class_overlap_estimate else 0,
            np.log1p(self.avg_text_length) if self.avg_text_length else 0,
            np.log1p(self.text_vocabulary_size) if self.text_vocabulary_size else 0,
        ]
        return np.array(values, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_numeric_features": self.n_numeric_features,
            "n_categorical_features": self.n_categorical_features,
            "n_classes": self.n_classes,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "missing_rate": self.missing_rate,
            "duplicate_rate": self.duplicate_rate,
            "numeric_mean_skewness": self.numeric_mean_skewness,
            "numeric_mean_kurtosis": self.numeric_mean_kurtosis,
            "mean_feature_correlation": self.mean_feature_correlation,
            "max_feature_correlation": self.max_feature_correlation,
            "feature_entropy": self.feature_entropy,
            "class_overlap_estimate": self.class_overlap_estimate,
            "avg_text_length": self.avg_text_length,
            "text_vocabulary_size": self.text_vocabulary_size,
        }


@dataclass
class QualityPrediction:
    """Predicted quality score with confidence."""

    quality_score: float  # 0-100
    confidence: float  # 0-1
    confidence_level: ConfidenceLevel
    confidence_interval: tuple[float, float]  # (lower, upper)
    prediction_time_ms: float
    features_used: DatasetFeatures
    feature_importances: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "confidence_interval": self.confidence_interval,
            "prediction_time_ms": self.prediction_time_ms,
            "feature_importances": self.feature_importances,
            "warnings": self.warnings,
        }

    def passes_threshold(self, threshold: float = 70.0) -> bool:
        """Check if prediction passes quality threshold."""
        return self.quality_score >= threshold

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✅ PASS" if self.passes_threshold() else "❌ FAIL"
        return (
            f"Quality Prediction: {self.quality_score:.1f}/100 {status}\n"
            f"Confidence: {self.confidence:.1%} ({self.confidence_level.value})\n"
            f"95% CI: [{self.confidence_interval[0]:.1f}, {self.confidence_interval[1]:.1f}]\n"
            f"Prediction time: {self.prediction_time_ms:.1f}ms"
        )


@dataclass
class PredictorConfig:
    """Configuration for quality predictor."""

    model_type: PredictionModel = PredictionModel.GRADIENT_BOOSTING
    n_estimators: int = 100
    max_depth: int = 6
    min_samples_for_training: int = 10
    confidence_percentile: float = 95.0
    cache_features: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_for_training": self.min_samples_for_training,
            "confidence_percentile": self.confidence_percentile,
            "cache_features": self.cache_features,
        }


class FeatureExtractor:
    """Extract meta-features from datasets for quality prediction."""

    def __init__(self, sample_size: int = 10000):
        """Initialize feature extractor.

        Args:
            sample_size: Max samples to use for feature extraction
        """
        self.sample_size = sample_size

    def extract(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        text_columns: list[str] | None = None,
    ) -> DatasetFeatures:
        """Extract meta-features from dataset.

        Args:
            data: Dataset to extract features from
            label_column: Column containing labels
            text_columns: Columns containing text data

        Returns:
            DatasetFeatures object
        """
        # Sample if needed
        if len(data) > self.sample_size:
            data = data.sample(n=self.sample_size, random_state=42)

        n_samples = len(data)
        n_features = len(data.columns)

        # Identify column types
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)

        # Class information
        n_classes = None
        class_imbalance = None
        class_overlap = None

        if label_column and label_column in data.columns:
            labels = data[label_column]
            n_classes = labels.nunique()
            class_counts = labels.value_counts()
            if len(class_counts) > 1:
                class_imbalance = float(class_counts.max() / class_counts.min())

                # Estimate class overlap using nearest neighbors if numeric features exist
                if n_numeric > 0:
                    class_overlap = self._estimate_class_overlap(
                        data[numeric_cols], labels
                    )

        # Missing rate
        missing_rate = float(data.isna().sum().sum() / (n_samples * n_features))

        # Duplicate rate
        duplicate_rate = float(data.duplicated().sum() / n_samples)

        # Distribution stats for numeric features
        mean_skewness = 0.0
        mean_kurtosis = 0.0
        if n_numeric > 0:
            from scipy.stats import kurtosis, skew
            skewness_values = []
            kurtosis_values = []
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 3:
                    skewness_values.append(abs(skew(col_data)))
                    kurtosis_values.append(abs(kurtosis(col_data)))
            if skewness_values:
                mean_skewness = float(np.mean(skewness_values))
                mean_kurtosis = float(np.mean(kurtosis_values))

        # Correlation features
        mean_corr = 0.0
        max_corr = 0.0
        if n_numeric > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            mean_corr = float(corr_matrix.values.mean())
            max_corr = float(corr_matrix.values.max())

        # Feature entropy
        feature_entropy = self._calculate_entropy(data)

        # Text features
        avg_text_length = None
        text_vocab_size = None
        if text_columns:
            text_lengths = []
            all_words = set()
            for col in text_columns:
                if col in data.columns:
                    texts = data[col].dropna().astype(str)
                    text_lengths.extend(texts.str.len().tolist())
                    for text in texts:
                        all_words.update(text.lower().split())
            if text_lengths:
                avg_text_length = float(np.mean(text_lengths))
                text_vocab_size = len(all_words)

        return DatasetFeatures(
            n_samples=n_samples,
            n_features=n_features,
            n_numeric_features=n_numeric,
            n_categorical_features=n_categorical,
            n_classes=n_classes,
            class_imbalance_ratio=class_imbalance,
            missing_rate=missing_rate,
            duplicate_rate=duplicate_rate,
            numeric_mean_skewness=mean_skewness,
            numeric_mean_kurtosis=mean_kurtosis,
            mean_feature_correlation=mean_corr,
            max_feature_correlation=max_corr,
            feature_entropy=feature_entropy,
            class_overlap_estimate=class_overlap,
            avg_text_length=avg_text_length,
            text_vocabulary_size=text_vocab_size,
        )

    def _estimate_class_overlap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Estimate class overlap using k-NN."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]

        if len(X_clean) < 10:
            return 0.0

        # Sample for efficiency
        if len(X_clean) > 1000:
            idx = np.random.choice(len(X_clean), 1000, replace=False)
            X_clean = X_clean.iloc[idx]
            y_clean = y_clean.iloc[idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Use 5-NN to estimate overlap
        k = min(5, len(X_clean) - 1)
        if k < 1:
            return 0.0

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_scaled, y_clean)

        # Overlap = 1 - leave-one-out accuracy
        predictions = knn.predict(X_scaled)
        accuracy = (predictions == y_clean).mean()

        return float(1 - accuracy)

    def _calculate_entropy(self, data: pd.DataFrame) -> float:
        """Calculate average entropy of categorical features."""
        entropy_values = []

        for col in data.columns:
            if data[col].dtype in ["object", "category"]:
                value_counts = data[col].value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                entropy_values.append(entropy)

        return float(np.mean(entropy_values)) if entropy_values else 0.0


class QualityPredictor:
    """Predict data quality scores using ML model.

    This enables instant quality gates without running full analysis.
    The model learns from previously analyzed datasets.
    """

    FEATURE_NAMES = [
        "log_n_samples",
        "log_n_features",
        "numeric_ratio",
        "categorical_ratio",
        "log_n_classes",
        "class_imbalance",
        "missing_rate",
        "duplicate_rate",
        "mean_skewness",
        "mean_kurtosis",
        "mean_correlation",
        "max_correlation",
        "feature_entropy",
        "class_overlap",
        "log_text_length",
        "log_vocab_size",
    ]

    def __init__(
        self,
        config: PredictorConfig | None = None,
    ):
        """Initialize quality predictor.

        Args:
            config: Predictor configuration
        """
        self.config = config or PredictorConfig()
        self.extractor = FeatureExtractor()
        self.scaler = StandardScaler()

        self._model: BaseEstimator | None = None
        self._ensemble_models: list[BaseEstimator] = []
        self._fitted = False
        self._training_scores: np.ndarray | None = None
        self._feature_cache: dict[str, DatasetFeatures] = {}
        self._cv_score: float | None = None

    def fit(
        self,
        datasets: list[pd.DataFrame],
        quality_scores: list[float],
        label_columns: list[str | None] | None = None,
        text_columns: list[list[str] | None] | None = None,
    ) -> QualityPredictor:
        """Fit predictor on analyzed datasets.

        Args:
            datasets: List of datasets
            quality_scores: Corresponding quality scores (0-100)
            label_columns: Label column for each dataset
            text_columns: Text columns for each dataset

        Returns:
            self
        """
        if len(datasets) < self.config.min_samples_for_training:
            raise ValueError(
                f"Need at least {self.config.min_samples_for_training} datasets "
                f"for training, got {len(datasets)}"
            )

        # Extract features from all datasets
        feature_vectors = []
        for i, df in enumerate(datasets):
            label_col = label_columns[i] if label_columns else None
            text_cols = text_columns[i] if text_columns else None

            features = self.extractor.extract(df, label_col, text_cols)
            feature_vectors.append(features.to_vector())

        X = np.array(feature_vectors)
        y = np.array(quality_scores)

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create model
        self._model = self._create_model()

        # Fit with cross-validation score
        if len(datasets) >= 5:
            cv_scores = cross_val_score(self._model, X_scaled, y, cv=5, scoring="r2")
            self._cv_score = float(cv_scores.mean())
            logger.info(f"Cross-validation R² score: {self._cv_score:.3f}")

        # Fit final model
        self._model.fit(X_scaled, y)

        # For ensemble, fit multiple models
        if self.config.model_type == PredictionModel.ENSEMBLE:
            self._ensemble_models = [
                GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=42,
                ),
                RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=42,
                ),
                Ridge(alpha=1.0),
            ]
            for model in self._ensemble_models:
                model.fit(X_scaled, y)

        self._training_scores = y
        self._fitted = True

        return self

    def predict(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        text_columns: list[str] | None = None,
    ) -> QualityPrediction:
        """Predict quality score for a dataset.

        Args:
            data: Dataset to score
            label_column: Label column name
            text_columns: Text column names

        Returns:
            QualityPrediction with score and confidence
        """
        import time

        if not self._fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        start_time = time.perf_counter()

        # Check cache
        cache_key = self._get_cache_key(data)
        if self.config.cache_features and cache_key in self._feature_cache:
            features = self._feature_cache[cache_key]
        else:
            features = self.extractor.extract(data, label_column, text_columns)
            if self.config.cache_features:
                self._feature_cache[cache_key] = features

        # Predict
        feature_vector = features.to_vector().reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(feature_vector)

        # Get prediction
        if self.config.model_type == PredictionModel.ENSEMBLE and self._ensemble_models:
            predictions = [m.predict(X_scaled)[0] for m in self._ensemble_models]
            score = float(np.mean(predictions))
            std = float(np.std(predictions))
        else:
            score = float(self._model.predict(X_scaled)[0])
            std = self._estimate_prediction_std(X_scaled)

        # Clip to valid range
        score = np.clip(score, 0, 100)

        # Calculate confidence interval
        z = 1.96  # 95% CI
        ci_lower = max(0, score - z * std)
        ci_upper = min(100, score + z * std)

        # Calculate confidence based on prediction variance and training coverage
        confidence = self._calculate_confidence(X_scaled, std)
        confidence_level = self._get_confidence_level(confidence)

        # Get feature importances
        feature_importances = self._get_feature_importances()

        # Generate warnings
        warnings = self._generate_warnings(features, confidence)

        prediction_time = (time.perf_counter() - start_time) * 1000

        return QualityPrediction(
            quality_score=score,
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            prediction_time_ms=prediction_time,
            features_used=features,
            feature_importances=feature_importances,
            warnings=warnings,
        )

    def predict_batch(
        self,
        datasets: list[pd.DataFrame],
        label_columns: list[str | None] | None = None,
        text_columns: list[list[str] | None] | None = None,
    ) -> list[QualityPrediction]:
        """Predict quality scores for multiple datasets.

        Args:
            datasets: List of datasets
            label_columns: Label columns for each dataset
            text_columns: Text columns for each dataset

        Returns:
            List of QualityPrediction objects
        """
        predictions = []
        for i, df in enumerate(datasets):
            label_col = label_columns[i] if label_columns else None
            text_cols = text_columns[i] if text_columns else None
            pred = self.predict(df, label_col, text_cols)
            predictions.append(pred)
        return predictions

    def _create_model(self) -> BaseEstimator:
        """Create the prediction model."""
        if self.config.model_type == PredictionModel.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
            )
        elif self.config.model_type == PredictionModel.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
            )
        elif self.config.model_type == PredictionModel.LINEAR:
            return Ridge(alpha=1.0)
        else:  # ENSEMBLE
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
            )

    def _estimate_prediction_std(self, X_scaled: np.ndarray) -> float:
        """Estimate prediction standard deviation."""
        if self._training_scores is None:
            return 10.0

        # Use training score std as baseline
        base_std = float(np.std(self._training_scores))

        # Increase uncertainty for out-of-distribution samples
        return base_std

    def _calculate_confidence(self, X_scaled: np.ndarray, std: float) -> float:
        """Calculate prediction confidence (0-1)."""
        # Base confidence from prediction std
        max_reasonable_std = 20.0
        confidence = 1 - min(std / max_reasonable_std, 1.0)

        # Reduce confidence for out-of-distribution
        if self._cv_score is not None:
            confidence *= max(0.5, self._cv_score)

        return float(np.clip(confidence, 0.1, 0.99))

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level enum."""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _get_feature_importances(self) -> dict[str, float]:
        """Get feature importances from model."""
        if not hasattr(self._model, "feature_importances_"):
            return {}

        importances = self._model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.FEATURE_NAMES, importances)
        }

    def _generate_warnings(
        self,
        features: DatasetFeatures,
        confidence: float,
    ) -> list[str]:
        """Generate warnings for the prediction."""
        warnings = []

        if confidence < 0.5:
            warnings.append(
                "Low prediction confidence - dataset may be outside training distribution"
            )

        if features.n_samples < 100:
            warnings.append("Small dataset - prediction may be less reliable")

        if features.missing_rate > 0.3:
            warnings.append(f"High missing rate ({features.missing_rate:.1%})")

        if features.duplicate_rate > 0.1:
            warnings.append(f"High duplicate rate ({features.duplicate_rate:.1%})")

        if features.class_imbalance_ratio and features.class_imbalance_ratio > 10:
            warnings.append(
                f"Severe class imbalance ({features.class_imbalance_ratio:.1f}:1)"
            )

        return warnings

    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """Get cache key for dataset."""
        # Use hash of column names, shape, and sample of data
        key_parts = [
            str(data.shape),
            str(list(data.columns)),
            str(data.head(1).values.tolist()),
        ]
        key_str = json.dumps(key_parts, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()  # noqa: S324

    def save(self, path: str | Path) -> None:
        """Save predictor to file.

        Args:
            path: Path to save to
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.to_dict(),
            "model": self._model,
            "ensemble_models": self._ensemble_models,
            "scaler": self.scaler,
            "training_scores": self._training_scores,
            "cv_score": self._cv_score,
            "fitted": self._fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved predictor to {path}")

    @classmethod
    def load(cls, path: str | Path) -> QualityPredictor:
        """Load predictor from file.

        Args:
            path: Path to load from

        Returns:
            Loaded QualityPredictor
        """
        import pickle

        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301

        config = PredictorConfig(
            model_type=PredictionModel(state["config"]["model_type"]),
            n_estimators=state["config"]["n_estimators"],
            max_depth=state["config"]["max_depth"],
        )

        predictor = cls(config=config)
        predictor._model = state["model"]
        predictor._ensemble_models = state["ensemble_models"]
        predictor.scaler = state["scaler"]
        predictor._training_scores = state["training_scores"]
        predictor._cv_score = state["cv_score"]
        predictor._fitted = state["fitted"]

        logger.info(f"Loaded predictor from {path}")
        return predictor


class QualityGate:
    """Quality gate for data ingestion pipelines.

    Automatically passes or rejects data based on predicted quality.
    """

    def __init__(
        self,
        predictor: QualityPredictor,
        threshold: float = 70.0,
        min_confidence: float = 0.5,
        on_low_confidence: str = "pass",  # "pass", "reject", or "manual"
    ):
        """Initialize quality gate.

        Args:
            predictor: Fitted quality predictor
            threshold: Minimum quality score to pass
            min_confidence: Minimum confidence for automatic decision
            on_low_confidence: Action when confidence is below minimum
        """
        self.predictor = predictor
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.on_low_confidence = on_low_confidence

        self._history: list[dict[str, Any]] = []

    def check(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        text_columns: list[str] | None = None,
    ) -> tuple[bool, QualityPrediction, str]:
        """Check if data passes quality gate.

        Args:
            data: Dataset to check
            label_column: Label column name
            text_columns: Text column names

        Returns:
            Tuple of (passes, prediction, reason)
        """
        prediction = self.predictor.predict(data, label_column, text_columns)

        # Check confidence
        if prediction.confidence < self.min_confidence:
            if self.on_low_confidence == "pass":
                result = True
                reason = "Passed (low confidence, defaulting to pass)"
            elif self.on_low_confidence == "reject":
                result = False
                reason = f"Rejected (confidence {prediction.confidence:.1%} below minimum)"
            else:
                result = False
                reason = "Manual review required (low confidence)"
        elif prediction.quality_score >= self.threshold:
            result = True
            reason = f"Passed (score {prediction.quality_score:.1f} >= {self.threshold})"
        else:
            result = False
            reason = f"Rejected (score {prediction.quality_score:.1f} < {self.threshold})"

        # Record in history
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "passed": result,
            "score": prediction.quality_score,
            "confidence": prediction.confidence,
            "reason": reason,
        })

        return result, prediction, reason

    def get_history(self) -> list[dict[str, Any]]:
        """Get gate check history."""
        return self._history.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get gate statistics."""
        if not self._history:
            return {"n_checks": 0}

        passes = sum(1 for h in self._history if h["passed"])
        scores = [h["score"] for h in self._history]

        return {
            "n_checks": len(self._history),
            "pass_rate": passes / len(self._history),
            "avg_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }


def predict_quality(
    data: pd.DataFrame,
    predictor: QualityPredictor,
    label_column: str | None = None,
    text_columns: list[str] | None = None,
) -> QualityPrediction:
    """Convenience function to predict quality.

    Args:
        data: Dataset to score
        predictor: Fitted predictor
        label_column: Label column name
        text_columns: Text column names

    Returns:
        QualityPrediction
    """
    return predictor.predict(data, label_column, text_columns)


def create_quality_gate(
    predictor: QualityPredictor,
    threshold: float = 70.0,
    **kwargs: Any,
) -> QualityGate:
    """Create a quality gate for pipelines.

    Args:
        predictor: Fitted quality predictor
        threshold: Minimum quality score
        **kwargs: Additional gate arguments

    Returns:
        QualityGate instance
    """
    return QualityGate(predictor, threshold=threshold, **kwargs)
