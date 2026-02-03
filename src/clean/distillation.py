"""Custom Model Distillation - Lightweight Detector Training.

This module provides functionality to distill complex quality detection
models into lightweight versions suitable for on-premise deployment.

Example:
    >>> from clean.distillation import ModelDistiller, DistillationConfig
    >>>
    >>> # Create distiller
    >>> distiller = ModelDistiller()
    >>>
    >>> # Train lightweight model from existing detector
    >>> student = distiller.distill(
    ...     teacher_predictions=predictions,
    ...     training_data=data,
    ...     config=DistillationConfig(target_size_mb=5)
    ... )
    >>>
    >>> # Export for deployment
    >>> distiller.export(student, "my_detector.onnx")
"""

from __future__ import annotations

import json
import logging
import pickle
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.exceptions import CleanError, ConfigurationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Export format for distilled models."""

    PICKLE = "pickle"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    TORCHSCRIPT = "torchscript"


class CompressionLevel(Enum):
    """Model compression level."""

    NONE = "none"
    LOW = "low"  # ~20% size reduction
    MEDIUM = "medium"  # ~50% size reduction
    HIGH = "high"  # ~80% size reduction
    EXTREME = "extreme"  # Maximum compression


@dataclass
class DistillationConfig:
    """Configuration for model distillation."""

    target_size_mb: float = 10.0  # Target model size
    compression: CompressionLevel = CompressionLevel.MEDIUM
    temperature: float = 2.0  # Softmax temperature for knowledge distillation
    alpha: float = 0.5  # Weight for soft labels vs hard labels
    max_depth: int | None = None  # Max depth for tree-based models
    n_estimators: int = 100  # Number of estimators
    quantize: bool = False  # Apply quantization
    prune: bool = False  # Apply pruning


@dataclass
class DistillationResult:
    """Result of model distillation."""

    student_model: Any
    teacher_accuracy: float
    student_accuracy: float
    size_reduction: float  # Percentage
    original_size_mb: float
    final_size_mb: float
    compression_ratio: float
    training_time: float
    config: DistillationConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Distillation Result:\n"
            f"  Teacher Accuracy: {self.teacher_accuracy:.4f}\n"
            f"  Student Accuracy: {self.student_accuracy:.4f}\n"
            f"  Accuracy Loss: {(self.teacher_accuracy - self.student_accuracy) * 100:.2f}%\n"
            f"  Size: {self.original_size_mb:.2f}MB → {self.final_size_mb:.2f}MB\n"
            f"  Compression: {self.compression_ratio:.1f}x\n"
            f"  Training Time: {self.training_time:.1f}s"
        )


@dataclass
class ExportResult:
    """Result of model export."""

    path: Path
    format: ModelFormat
    size_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LightweightDetector:
    """Lightweight quality issue detector.

    A simple, fast detector trained through knowledge distillation
    from a more complex teacher model.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        threshold: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize detector.

        Args:
            model: Underlying model (sklearn-compatible)
            feature_names: Expected feature names
            threshold: Classification threshold
            metadata: Additional metadata
        """
        self.model = model
        self.feature_names = feature_names or []
        self.threshold = threshold
        self.metadata = metadata or {}

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict quality issues.

        Args:
            X: Features

        Returns:
            Binary predictions (1 = issue, 0 = no issue)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict issue probabilities.

        Args:
            X: Features

        Returns:
            Probability of issue for each sample
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if len(probs.shape) > 1:
                return probs[:, 1]
            return probs
        else:
            # Fall back to decision function
            return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        """Save detector to file."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LightweightDetector":
        """Load detector from file."""
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)


class ModelDistiller:
    """Distills complex models into lightweight versions.

    Uses knowledge distillation to train smaller, faster models
    that approximate the behavior of larger teacher models.

    Example:
        >>> distiller = ModelDistiller()
        >>> result = distiller.distill(
        ...     teacher_predictions=teacher.predict_proba(X),
        ...     training_data=(X_train, y_train),
        ...     config=DistillationConfig(target_size_mb=5)
        ... )
    """

    def __init__(self):
        """Initialize distiller."""
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Check if sklearn is available."""
        try:
            import sklearn
            return True
        except ImportError:
            return False

    def distill(
        self,
        teacher_predictions: np.ndarray,
        training_data: tuple[np.ndarray, np.ndarray],
        config: DistillationConfig | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        feature_names: list[str] | None = None,
    ) -> DistillationResult:
        """Distill a teacher model into a lightweight student.

        Args:
            teacher_predictions: Soft predictions from teacher model
            training_data: (X_train, y_train) tuple
            config: Distillation configuration
            validation_data: Optional (X_val, y_val) for evaluation
            feature_names: Feature names for the model

        Returns:
            DistillationResult with trained student model
        """
        import time
        start_time = time.time()

        config = config or DistillationConfig()
        X_train, y_train = training_data

        if validation_data is None:
            # Use portion of training data for validation
            split_idx = int(len(X_train) * 0.8)
            X_val, y_val = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]
            teacher_predictions = teacher_predictions[:split_idx]
        else:
            X_val, y_val = validation_data

        # Create soft labels using temperature scaling
        soft_labels = self._create_soft_labels(
            teacher_predictions, config.temperature
        )

        # Combine soft and hard labels
        combined_labels = self._combine_labels(
            soft_labels, y_train[:len(soft_labels)], config.alpha
        )

        # Train student model
        student = self._train_student(
            X_train[:len(combined_labels)],
            combined_labels,
            config,
        )

        # Evaluate
        teacher_accuracy = self._evaluate_accuracy(
            teacher_predictions, y_train[:len(teacher_predictions)]
        )
        student_predictions = student.predict(X_val)
        student_accuracy = self._evaluate_accuracy(student_predictions, y_val)

        # Calculate sizes
        original_size = self._estimate_model_size(len(teacher_predictions) * 4)  # 4 bytes per float
        final_size = self._get_model_size(student.model)

        # Apply compression if requested
        if config.compression != CompressionLevel.NONE:
            student = self._compress_model(student, config.compression)
            final_size = self._get_model_size(student.model)

        # Wrap in lightweight detector
        detector = LightweightDetector(
            model=student.model if hasattr(student, "model") else student,
            feature_names=feature_names,
            threshold=0.5,
            metadata={
                "distillation_config": config.__dict__,
                "teacher_accuracy": teacher_accuracy,
                "student_accuracy": student_accuracy,
            },
        )

        return DistillationResult(
            student_model=detector,
            teacher_accuracy=teacher_accuracy,
            student_accuracy=student_accuracy,
            size_reduction=(1 - final_size / original_size) * 100 if original_size > 0 else 0,
            original_size_mb=original_size / (1024 * 1024),
            final_size_mb=final_size / (1024 * 1024),
            compression_ratio=original_size / final_size if final_size > 0 else 1,
            training_time=time.time() - start_time,
            config=config,
        )

    def _create_soft_labels(
        self,
        predictions: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Create soft labels using temperature scaling."""
        # Apply softmax with temperature
        if len(predictions.shape) == 1:
            # Convert to 2D for consistency
            predictions = np.column_stack([1 - predictions, predictions])

        scaled = predictions / temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        soft = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        return soft[:, 1]  # Return positive class probability

    def _combine_labels(
        self,
        soft_labels: np.ndarray,
        hard_labels: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Combine soft and hard labels."""
        return alpha * soft_labels + (1 - alpha) * hard_labels.astype(float)

    def _train_student(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: DistillationConfig,
    ) -> Any:
        """Train student model."""
        if not self._sklearn_available:
            raise CleanError(
                "sklearn required for distillation. Install with: pip install scikit-learn"
            )

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        # Choose model based on compression level
        if config.compression in (CompressionLevel.HIGH, CompressionLevel.EXTREME):
            # Use simple decision tree for maximum compression
            max_depth = config.max_depth or (3 if config.compression == CompressionLevel.EXTREME else 5)
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=5,
            )
        elif config.compression == CompressionLevel.MEDIUM:
            # Use random forest with limited trees
            model = RandomForestClassifier(
                n_estimators=min(config.n_estimators, 50),
                max_depth=config.max_depth or 10,
                min_samples_leaf=3,
                n_jobs=-1,
            )
        else:
            # Use gradient boosting for best accuracy
            model = GradientBoostingClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth or 5,
                min_samples_leaf=2,
            )

        # Convert continuous labels to binary for classification
        y_binary = (y >= 0.5).astype(int)
        model.fit(X, y_binary)

        return LightweightDetector(model=model)

    def _compress_model(
        self,
        detector: LightweightDetector,
        level: CompressionLevel,
    ) -> LightweightDetector:
        """Apply additional compression to model."""
        # For tree-based models, prune based on compression level
        model = detector.model

        if hasattr(model, "estimators_"):
            # Random forest - reduce estimators
            reduction = {
                CompressionLevel.LOW: 0.8,
                CompressionLevel.MEDIUM: 0.5,
                CompressionLevel.HIGH: 0.3,
                CompressionLevel.EXTREME: 0.1,
            }.get(level, 1.0)

            n_keep = max(1, int(len(model.estimators_) * reduction))
            model.estimators_ = model.estimators_[:n_keep]
            model.n_estimators = n_keep

        return detector

    def _evaluate_accuracy(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Evaluate prediction accuracy."""
        if len(predictions.shape) > 1:
            predictions = predictions[:, 1]
        binary_preds = (predictions >= 0.5).astype(int)
        return (binary_preds == labels).mean()

    def _estimate_model_size(self, n_bytes: int) -> int:
        """Estimate model size in bytes."""
        return n_bytes

    def _get_model_size(self, model: Any) -> int:
        """Get actual model size in bytes."""
        with tempfile.NamedTemporaryFile(delete=True) as f:
            pickle.dump(model, f)
            f.flush()
            return f.tell()

    def export(
        self,
        model: LightweightDetector | DistillationResult,
        path: str | Path,
        format: ModelFormat = ModelFormat.PICKLE,
    ) -> ExportResult:
        """Export distilled model.

        Args:
            model: Model to export
            path: Output path
            format: Export format

        Returns:
            ExportResult with export details
        """
        path = Path(path)

        if isinstance(model, DistillationResult):
            detector = model.student_model
        else:
            detector = model

        if format == ModelFormat.PICKLE:
            detector.save(path)
            size_mb = path.stat().st_size / (1024 * 1024)

        elif format == ModelFormat.ONNX:
            self._export_onnx(detector, path)
            size_mb = path.stat().st_size / (1024 * 1024)

        elif format == ModelFormat.SKLEARN:
            # Export just the sklearn model
            with open(path, "wb") as f:
                pickle.dump(detector.model, f)
            size_mb = path.stat().st_size / (1024 * 1024)

        else:
            raise CleanError(f"Unsupported export format: {format}")

        return ExportResult(
            path=path,
            format=format,
            size_mb=size_mb,
            metadata={"threshold": detector.threshold},
        )

    def _export_onnx(self, detector: LightweightDetector, path: Path) -> None:
        """Export model to ONNX format."""
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as e:
            raise CleanError(
                "skl2onnx required for ONNX export. Install with: pip install skl2onnx"
            ) from e

        # Determine input shape
        n_features = getattr(detector.model, "n_features_in_", 10)
        initial_type = [("input", FloatTensorType([None, n_features]))]

        onnx_model = convert_sklearn(detector.model, initial_types=initial_type)

        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())


class DistillationPipeline:
    """End-to-end pipeline for model distillation.

    Automates the process of distilling existing detectors
    into lightweight versions.

    Example:
        >>> pipeline = DistillationPipeline()
        >>> result = pipeline.distill_from_cleaner(
        ...     cleaner=my_cleaner,
        ...     training_data=df,
        ...     target_size_mb=5
        ... )
    """

    def __init__(self):
        """Initialize pipeline."""
        self.distiller = ModelDistiller()

    def distill_from_cleaner(
        self,
        cleaner: Any,
        training_data: pd.DataFrame,
        label_column: str | None = None,
        target_size_mb: float = 10.0,
        issue_type: str = "label_errors",
    ) -> DistillationResult:
        """Distill from an existing DatasetCleaner.

        Args:
            cleaner: Trained DatasetCleaner instance
            training_data: Data to train on
            label_column: Label column name
            target_size_mb: Target model size
            issue_type: Type of issue to detect

        Returns:
            DistillationResult
        """
        # Get teacher predictions
        report = cleaner.analyze()

        if issue_type == "label_errors" and hasattr(report, "label_errors"):
            issues = report.label_errors()
            issue_indices = set(issues.index.tolist())
        elif issue_type == "duplicates" and hasattr(report, "duplicates"):
            issues = report.duplicates()
            issue_indices = set(issues.index.tolist())
        elif issue_type == "outliers" and hasattr(report, "outliers"):
            issues = report.outliers()
            issue_indices = set(issues.index.tolist())
        else:
            raise CleanError(f"Unknown issue type: {issue_type}")

        # Create teacher predictions
        teacher_predictions = np.array([
            1.0 if i in issue_indices else 0.0
            for i in range(len(training_data))
        ])

        # Prepare features
        numeric_cols = training_data.select_dtypes(include=[np.number]).columns
        X = training_data[numeric_cols].fillna(0).values
        y = (np.array(range(len(training_data)))[:, None] == np.array(list(issue_indices))).any(axis=1).astype(int)

        config = DistillationConfig(
            target_size_mb=target_size_mb,
            compression=CompressionLevel.MEDIUM,
        )

        return self.distiller.distill(
            teacher_predictions=teacher_predictions,
            training_data=(X, y),
            config=config,
            feature_names=list(numeric_cols),
        )

    def distill_from_predictions(
        self,
        X: np.ndarray,
        teacher_predictions: np.ndarray,
        y_true: np.ndarray | None = None,
        config: DistillationConfig | None = None,
    ) -> DistillationResult:
        """Distill from raw predictions.

        Args:
            X: Feature matrix
            teacher_predictions: Teacher model predictions
            y_true: Ground truth labels (optional)
            config: Distillation config

        Returns:
            DistillationResult
        """
        if y_true is None:
            # Use teacher predictions as pseudo-labels
            y_true = (teacher_predictions >= 0.5).astype(int)

        return self.distiller.distill(
            teacher_predictions=teacher_predictions,
            training_data=(X, y_true),
            config=config,
        )


def create_distiller() -> ModelDistiller:
    """Create a ModelDistiller instance.

    Returns:
        Configured ModelDistiller
    """
    return ModelDistiller()


def create_distillation_pipeline() -> DistillationPipeline:
    """Create a DistillationPipeline instance.

    Returns:
        Configured DistillationPipeline
    """
    return DistillationPipeline()


__all__ = [
    # Core classes
    "ModelDistiller",
    "DistillationPipeline",
    "LightweightDetector",
    # Config and results
    "DistillationConfig",
    "DistillationResult",
    "ExportResult",
    # Enums
    "ModelFormat",
    "CompressionLevel",
    # Functions
    "create_distiller",
    "create_distillation_pipeline",
]
