"""Active learning integration for efficient labeling workflows.

This module provides tools for active learning-based sample selection:
- Uncertainty sampling strategies
- Label Studio integration
- CVAT integration
- Export to labeling platforms

Example:
    >>> from clean.active_learning import ActiveLearner, LabelStudioExporter
    >>>
    >>> learner = ActiveLearner(strategy="uncertainty")
    >>> samples = learner.select_samples(X, n_samples=100)
    >>>
    >>> exporter = LabelStudioExporter()
    >>> exporter.export(df.iloc[samples], "tasks.json")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


class SamplingStrategy(Enum):
    """Active learning sampling strategies."""

    UNCERTAINTY = "uncertainty"  # Least confident predictions
    MARGIN = "margin"  # Smallest margin between top 2 predictions
    ENTROPY = "entropy"  # Highest prediction entropy
    RANDOM = "random"  # Random baseline
    DIVERSITY = "diversity"  # Maximize sample diversity
    BADGE = "badge"  # Batch Active learning by Diverse Gradient Embeddings
    COMBINED = "combined"  # Uncertainty + diversity


@dataclass
class SampleSelection:
    """Result of active learning sample selection."""

    indices: list[int]
    scores: list[float]
    strategy: SamplingStrategy
    n_selected: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get selected samples as DataFrame."""
        df = data.iloc[self.indices].copy()
        df["al_score"] = self.scores
        df["al_rank"] = range(1, len(self.indices) + 1)
        return df


class ActiveLearner:
    """Active learning sample selector.

    Selects the most informative samples for labeling based on
    model uncertainty or diversity criteria.
    """

    def __init__(
        self,
        strategy: SamplingStrategy | str = SamplingStrategy.UNCERTAINTY,
        classifier: Any | None = None,
        n_jobs: int = -1,
    ):
        """Initialize the active learner.

        Args:
            strategy: Sampling strategy to use
            classifier: Sklearn classifier for uncertainty estimation
            n_jobs: Number of parallel jobs
        """
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)

        self.strategy = strategy
        self.classifier = classifier or LogisticRegression(
            max_iter=1000, n_jobs=n_jobs, random_state=42
        )
        self.n_jobs = n_jobs

        self._pred_probs: np.ndarray | None = None

    def select_samples(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        n_samples: int = 100,
        exclude_indices: list[int] | None = None,
    ) -> SampleSelection:
        """Select samples for labeling.

        Args:
            X: Feature data
            y: Labels (required for uncertainty-based strategies)
            n_samples: Number of samples to select
            exclude_indices: Indices to exclude from selection

        Returns:
            SampleSelection with selected indices and scores
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            X_arr = np.asarray(X)

        n_total = len(X_arr)
        n_samples = min(n_samples, n_total)

        # Create mask for valid indices
        valid_mask = np.ones(n_total, dtype=bool)
        if exclude_indices:
            valid_mask[exclude_indices] = False

        valid_indices = np.where(valid_mask)[0]

        if self.strategy == SamplingStrategy.RANDOM:
            return self._random_sampling(valid_indices, n_samples)

        elif self.strategy == SamplingStrategy.UNCERTAINTY:
            return self._uncertainty_sampling(X_arr, y, valid_indices, n_samples)

        elif self.strategy == SamplingStrategy.MARGIN:
            return self._margin_sampling(X_arr, y, valid_indices, n_samples)

        elif self.strategy == SamplingStrategy.ENTROPY:
            return self._entropy_sampling(X_arr, y, valid_indices, n_samples)

        elif self.strategy == SamplingStrategy.DIVERSITY:
            return self._diversity_sampling(X_arr, valid_indices, n_samples)

        elif self.strategy == SamplingStrategy.COMBINED:
            return self._combined_sampling(X_arr, y, valid_indices, n_samples)

        else:
            return self._random_sampling(valid_indices, n_samples)

    def _get_pred_probs(
        self, X: np.ndarray, y: np.ndarray | None
    ) -> np.ndarray:
        """Get prediction probabilities using cross-validation."""
        if y is None:
            raise ValueError("Labels required for uncertainty-based sampling")

        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)

        # Cross-validated predictions
        cv_folds = min(5, len(np.unique(y)))
        self._pred_probs = cross_val_predict(
            self.classifier, X, y, cv=cv_folds, method="predict_proba", n_jobs=self.n_jobs
        )
        return self._pred_probs

    def _random_sampling(
        self, valid_indices: np.ndarray, n_samples: int
    ) -> SampleSelection:
        """Random sampling baseline."""
        selected = np.random.choice(valid_indices, size=n_samples, replace=False)
        scores = [1.0] * n_samples

        return SampleSelection(
            indices=selected.tolist(),
            scores=scores,
            strategy=SamplingStrategy.RANDOM,
            n_selected=n_samples,
        )

    def _uncertainty_sampling(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        valid_indices: np.ndarray,
        n_samples: int,
    ) -> SampleSelection:
        """Select samples with lowest prediction confidence."""
        pred_probs = self._get_pred_probs(X, y)

        # Confidence = max probability
        confidence = pred_probs.max(axis=1)

        # Uncertainty = 1 - confidence
        uncertainty = 1 - confidence

        # Get top uncertain samples from valid indices
        valid_uncertainty = uncertainty[valid_indices]
        top_indices = np.argsort(valid_uncertainty)[::-1][:n_samples]

        selected = valid_indices[top_indices]
        scores = valid_uncertainty[top_indices]

        return SampleSelection(
            indices=selected.tolist(),
            scores=scores.tolist(),
            strategy=SamplingStrategy.UNCERTAINTY,
            n_selected=n_samples,
            metadata={"mean_uncertainty": float(np.mean(scores))},
        )

    def _margin_sampling(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        valid_indices: np.ndarray,
        n_samples: int,
    ) -> SampleSelection:
        """Select samples with smallest margin between top 2 predictions."""
        pred_probs = self._get_pred_probs(X, y)

        # Sort probabilities for each sample
        sorted_probs = np.sort(pred_probs, axis=1)

        # Margin = difference between top 2
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Select smallest margins (most ambiguous)
        valid_margin = margin[valid_indices]
        top_indices = np.argsort(valid_margin)[:n_samples]

        selected = valid_indices[top_indices]
        scores = (1 - valid_margin[top_indices])  # Invert so higher = more informative

        return SampleSelection(
            indices=selected.tolist(),
            scores=scores.tolist(),
            strategy=SamplingStrategy.MARGIN,
            n_selected=n_samples,
        )

    def _entropy_sampling(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        valid_indices: np.ndarray,
        n_samples: int,
    ) -> SampleSelection:
        """Select samples with highest prediction entropy."""
        pred_probs = self._get_pred_probs(X, y)

        # Compute entropy
        epsilon = 1e-10
        entropy = -np.sum(pred_probs * np.log(pred_probs + epsilon), axis=1)

        # Select highest entropy
        valid_entropy = entropy[valid_indices]
        top_indices = np.argsort(valid_entropy)[::-1][:n_samples]

        selected = valid_indices[top_indices]
        scores = valid_entropy[top_indices]

        return SampleSelection(
            indices=selected.tolist(),
            scores=scores.tolist(),
            strategy=SamplingStrategy.ENTROPY,
            n_selected=n_samples,
        )

    def _diversity_sampling(
        self,
        X: np.ndarray,
        valid_indices: np.ndarray,
        n_samples: int,
    ) -> SampleSelection:
        """Select diverse samples using k-means++ initialization."""
        from scipy.spatial.distance import cdist

        X_valid = X[valid_indices]

        # Normalize features
        X_norm = (X_valid - X_valid.mean(axis=0)) / (X_valid.std(axis=0) + 1e-10)

        # K-means++ style selection
        selected_local = [np.random.randint(len(X_norm))]
        scores = [1.0]

        for _ in range(n_samples - 1):
            # Compute distances to nearest selected
            dists = cdist(X_norm, X_norm[selected_local], metric='euclidean')
            min_dists = dists.min(axis=1)

            # Exclude already selected
            min_dists[selected_local] = 0

            # Select furthest point
            next_idx = np.argmax(min_dists)
            selected_local.append(next_idx)
            scores.append(float(min_dists[next_idx]))

        selected = valid_indices[selected_local]

        return SampleSelection(
            indices=selected.tolist(),
            scores=scores,
            strategy=SamplingStrategy.DIVERSITY,
            n_selected=n_samples,
        )

    def _combined_sampling(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        valid_indices: np.ndarray,
        n_samples: int,
    ) -> SampleSelection:
        """Combined uncertainty and diversity sampling."""
        # Get uncertainty scores
        pred_probs = self._get_pred_probs(X, y)
        uncertainty = 1 - pred_probs.max(axis=1)
        valid_uncertainty = uncertainty[valid_indices]

        # Select top 2*n uncertain samples first
        n_uncertain = min(n_samples * 2, len(valid_indices))
        uncertain_local = np.argsort(valid_uncertainty)[::-1][:n_uncertain]

        # Then apply diversity sampling on uncertain subset
        X_uncertain = X[valid_indices[uncertain_local]]
        X_norm = (X_uncertain - X_uncertain.mean(axis=0)) / (X_uncertain.std(axis=0) + 1e-10)

        from scipy.spatial.distance import cdist

        selected_in_uncertain = [0]  # Start with most uncertain
        for _ in range(min(n_samples - 1, len(X_norm) - 1)):
            dists = cdist(X_norm, X_norm[selected_in_uncertain], metric='euclidean')
            min_dists = dists.min(axis=1)
            min_dists[selected_in_uncertain] = 0
            next_idx = np.argmax(min_dists)
            selected_in_uncertain.append(next_idx)

        # Map back to original indices
        selected_local = uncertain_local[selected_in_uncertain]
        selected = valid_indices[selected_local]
        scores = valid_uncertainty[selected_local]

        return SampleSelection(
            indices=selected.tolist(),
            scores=scores.tolist(),
            strategy=SamplingStrategy.COMBINED,
            n_selected=len(selected),
        )


class BaseLabelingExporter(ABC):
    """Base class for labeling platform exporters."""

    @abstractmethod
    def export(
        self,
        data: pd.DataFrame,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Export data for labeling platform."""
        pass

    @abstractmethod
    def create_tasks(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[dict]:
        """Create labeling tasks from data."""
        pass


class LabelStudioExporter(BaseLabelingExporter):
    """Exporter for Label Studio format.

    Creates JSON task files compatible with Label Studio.
    """

    def __init__(
        self,
        text_column: str | None = None,
        image_column: str | None = None,
        id_column: str | None = None,
    ):
        """Initialize the exporter.

        Args:
            text_column: Column containing text data
            image_column: Column containing image paths/URLs
            id_column: Column to use as task ID
        """
        self.text_column = text_column
        self.image_column = image_column
        self.id_column = id_column

    def export(
        self,
        data: pd.DataFrame,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Export data to Label Studio JSON format.

        Args:
            data: DataFrame with samples to export
            output_path: Path for output JSON file
            **kwargs: Additional arguments
        """
        tasks = self.create_tasks(data, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2, default=str)

    def create_tasks(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[dict]:
        """Create Label Studio tasks from DataFrame.

        Args:
            data: DataFrame with samples
            **kwargs: Additional task metadata

        Returns:
            List of Label Studio task dictionaries
        """
        tasks = []

        for idx, row in data.iterrows():
            task: dict[str, Any] = {"id": idx if self.id_column is None else row.get(self.id_column, idx)}
            task_data: dict[str, Any] = {}

            # Add text if present
            if self.text_column and self.text_column in data.columns:
                task_data["text"] = str(row[self.text_column])

            # Add image if present
            if self.image_column and self.image_column in data.columns:
                task_data["image"] = str(row[self.image_column])

            # Add all columns as data if no specific columns specified
            if not task_data:
                task_data = row.to_dict()

            task["data"] = task_data

            # Add metadata
            if "al_score" in row:
                task["meta"] = {"al_score": float(row["al_score"])}

            tasks.append(task)

        return tasks

    def create_config(
        self,
        task_type: str = "classification",
        labels: list[str] | None = None,
    ) -> str:
        """Create Label Studio labeling config.

        Args:
            task_type: Type of labeling task
            labels: List of class labels

        Returns:
            XML labeling configuration
        """
        labels = labels or ["Label1", "Label2"]

        if task_type == "classification":
            choices = "\n".join([f'    <Choice value="{label}"/>' for label in labels])
            return f"""<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text">
{choices}
  </Choices>
</View>"""

        elif task_type == "ner":
            labels_xml = "\n".join([f'    <Label value="{label}"/>' for label in labels])
            return f"""<View>
  <Labels name="label" toName="text">
{labels_xml}
  </Labels>
  <Text name="text" value="$text"/>
</View>"""

        else:
            return """<View>
  <Text name="text" value="$text"/>
</View>"""


class CVATExporter(BaseLabelingExporter):
    """Exporter for CVAT format.

    Creates XML annotation files compatible with CVAT.
    """

    def __init__(
        self,
        image_column: str = "image_path",
        id_column: str | None = None,
    ):
        """Initialize the exporter.

        Args:
            image_column: Column containing image paths
            id_column: Column to use as task ID
        """
        self.image_column = image_column
        self.id_column = id_column

    def export(
        self,
        data: pd.DataFrame,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Export data to CVAT XML format.

        Args:
            data: DataFrame with samples to export
            output_path: Path for output XML file
            **kwargs: Additional arguments including 'labels' list
        """
        tasks = self.create_tasks(data, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        labels = kwargs.get("labels", ["object"])
        xml_content = self._create_cvat_xml(tasks, labels)

        with open(output_path, "w") as f:
            f.write(xml_content)

    def create_tasks(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[dict]:
        """Create CVAT tasks from DataFrame.

        Args:
            data: DataFrame with samples
            **kwargs: Additional task metadata

        Returns:
            List of task dictionaries
        """
        tasks = []

        for idx, row in data.iterrows():
            task = {
                "id": idx if self.id_column is None else row.get(self.id_column, idx),
                "name": str(row.get(self.image_column, f"image_{idx}")),
            }

            if self.image_column in data.columns:
                task["image_path"] = str(row[self.image_column])

            tasks.append(task)

        return tasks

    def _create_cvat_xml(
        self,
        tasks: list[dict],
        labels: list[str],
    ) -> str:
        """Create CVAT XML content."""
        labels_xml = "\n".join([
            f'    <label><name>{label}</name></label>'
            for label in labels
        ])

        images_xml = "\n".join([
            f'  <image id="{task["id"]}" name="{task["name"]}"/>'
            for task in tasks
        ])

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
{labels_xml}
      </labels>
    </task>
  </meta>
{images_xml}
</annotations>"""


class ProdigyExporter(BaseLabelingExporter):
    """Exporter for Prodigy format.

    Creates JSONL files compatible with Prodigy.
    """

    def __init__(
        self,
        text_column: str = "text",
        id_column: str | None = None,
    ):
        """Initialize the exporter.

        Args:
            text_column: Column containing text data
            id_column: Column to use as example ID
        """
        self.text_column = text_column
        self.id_column = id_column

    def export(
        self,
        data: pd.DataFrame,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Export data to Prodigy JSONL format.

        Args:
            data: DataFrame with samples to export
            output_path: Path for output JSONL file
            **kwargs: Additional arguments
        """
        tasks = self.create_tasks(data, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task, default=str) + "\n")

    def create_tasks(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[dict]:
        """Create Prodigy examples from DataFrame.

        Args:
            data: DataFrame with samples
            **kwargs: Additional example metadata

        Returns:
            List of Prodigy example dictionaries
        """
        examples = []

        for idx, row in data.iterrows():
            example: dict[str, Any] = {}

            # Add text
            if self.text_column in data.columns:
                example["text"] = str(row[self.text_column])

            # Add ID
            if self.id_column and self.id_column in data.columns:
                example["_input_hash"] = hash(str(row[self.id_column]))
            else:
                example["_input_hash"] = hash(str(idx))

            # Add metadata
            example["meta"] = {"index": int(idx)}
            if "al_score" in row:
                example["meta"]["al_score"] = float(row["al_score"])

            examples.append(example)

        return examples


def select_for_labeling(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray | None = None,
    n_samples: int = 100,
    strategy: str = "uncertainty",
    **kwargs: Any,
) -> SampleSelection:
    """Select samples for labeling using active learning.

    Args:
        X: Feature data
        y: Labels (if available)
        n_samples: Number of samples to select
        strategy: Sampling strategy
        **kwargs: Additional arguments for ActiveLearner

    Returns:
        SampleSelection with selected indices
    """
    learner = ActiveLearner(strategy=strategy, **kwargs)
    return learner.select_samples(X, y, n_samples)


@dataclass
class CorrectionFeedback:
    """Feedback from human correction of a label."""

    index: int
    original_label: Any
    corrected_label: Any
    confidence: float  # Human's confidence in correction
    annotator_id: str | None = None
    timestamp: float | None = None
    comment: str | None = None


@dataclass
class LearningSession:
    """State of an interactive learning session."""

    session_id: str
    total_reviewed: int
    total_corrected: int
    corrections: list[CorrectionFeedback]
    model_accuracy_history: list[float]
    selection_history: list[list[int]]


class IntelligentSampler:
    """Intelligent sampling with human-in-the-loop learning.

    Learns from human corrections to improve future sampling decisions.
    """

    def __init__(
        self,
        base_learner: ActiveLearner | None = None,
        correction_weight: float = 2.0,
        memory_size: int = 1000,
    ):
        """Initialize the intelligent sampler.

        Args:
            base_learner: Base active learner for initial selection
            correction_weight: Weight multiplier for corrected samples
            memory_size: Maximum corrections to remember
        """
        self.base_learner = base_learner or ActiveLearner(strategy="combined")
        self.correction_weight = correction_weight
        self.memory_size = memory_size

        self._corrections: list[CorrectionFeedback] = []
        self._correction_patterns: dict[tuple, float] = {}
        self._session: LearningSession | None = None

    def start_session(self, session_id: str | None = None) -> LearningSession:
        """Start a new labeling session.

        Args:
            session_id: Optional session identifier

        Returns:
            New LearningSession
        """
        import uuid

        self._session = LearningSession(
            session_id=session_id or str(uuid.uuid4()),
            total_reviewed=0,
            total_corrected=0,
            corrections=[],
            model_accuracy_history=[],
            selection_history=[],
        )
        return self._session

    def select_samples(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        n_samples: int = 100,
        exclude_indices: list[int] | None = None,
    ) -> SampleSelection:
        """Select samples intelligently based on past corrections.

        Args:
            X: Feature data
            y: Labels (if available)
            n_samples: Number of samples to select
            exclude_indices: Indices to exclude

        Returns:
            SampleSelection with selected indices
        """
        # Get base selection
        base_selection = self.base_learner.select_samples(
            X, y, n_samples * 2, exclude_indices
        )

        if not self._corrections:
            # No corrections yet, use base selection
            selected = base_selection.indices[:n_samples]
            scores = base_selection.scores[:n_samples]
        else:
            # Adjust scores based on correction patterns
            adjusted_scores = self._adjust_scores(
                X, base_selection.indices, base_selection.scores, y
            )

            # Sort by adjusted scores and select top
            sorted_indices = np.argsort(adjusted_scores)[::-1][:n_samples]
            selected = [base_selection.indices[i] for i in sorted_indices]
            scores = [adjusted_scores[i] for i in sorted_indices]

        if self._session:
            self._session.selection_history.append(selected)

        return SampleSelection(
            indices=selected,
            scores=scores,
            strategy=self.base_learner.strategy,
            n_selected=len(selected),
            metadata={"correction_informed": bool(self._corrections)},
        )

    def record_correction(
        self,
        index: int,
        original_label: Any,
        corrected_label: Any,
        confidence: float = 1.0,
        annotator_id: str | None = None,
        comment: str | None = None,
    ) -> CorrectionFeedback:
        """Record a human correction.

        Args:
            index: Sample index
            original_label: Original/predicted label
            corrected_label: Human-corrected label
            confidence: Human's confidence (0-1)
            annotator_id: Optional annotator identifier
            comment: Optional comment

        Returns:
            CorrectionFeedback record
        """
        import time

        feedback = CorrectionFeedback(
            index=index,
            original_label=original_label,
            corrected_label=corrected_label,
            confidence=confidence,
            annotator_id=annotator_id,
            timestamp=time.time(),
            comment=comment,
        )

        self._corrections.append(feedback)

        # Update correction patterns
        pattern_key = (original_label, corrected_label)
        self._correction_patterns[pattern_key] = (
            self._correction_patterns.get(pattern_key, 0) + 1
        )

        # Trim memory if needed
        if len(self._corrections) > self.memory_size:
            self._corrections = self._corrections[-self.memory_size:]

        if self._session:
            self._session.corrections.append(feedback)
            self._session.total_corrected += 1

        return feedback

    def record_review(
        self,
        index: int,
        was_correct: bool,
    ) -> None:
        """Record that a sample was reviewed (correct or not).

        Args:
            index: Sample index
            was_correct: Whether the label was correct
        """
        if self._session:
            self._session.total_reviewed += 1

    def _adjust_scores(
        self,
        X: pd.DataFrame | np.ndarray,
        indices: list[int],
        scores: list[float],
        y: np.ndarray | None,
    ) -> np.ndarray:
        """Adjust scores based on correction history."""
        adjusted = np.array(scores, dtype=float)

        if y is None or not self._corrections:
            return adjusted

        # Boost samples similar to corrected ones
        if isinstance(X, pd.DataFrame):
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            X_arr = np.asarray(X)

        # Get features of corrected samples
        correction_indices = [c.index for c in self._corrections[-100:]]
        valid_indices = [i for i in correction_indices if i < len(X_arr)]

        if not valid_indices:
            return adjusted

        corrected_features = X_arr[valid_indices]

        # Calculate similarity to corrected samples
        for i, idx in enumerate(indices):
            if idx >= len(X_arr):
                continue

            sample = X_arr[idx]

            # Simple similarity based on Euclidean distance
            distances = np.linalg.norm(corrected_features - sample, axis=1)
            min_distance = distances.min() if len(distances) > 0 else float('inf')

            # Boost score for samples similar to corrected ones
            if min_distance < np.median(distances) if len(distances) > 1 else float('inf'):
                adjusted[i] *= self.correction_weight

        # Also boost samples with labels in common correction patterns
        if y is not None:
            for i, idx in enumerate(indices):
                if idx >= len(y):
                    continue

                label = y[idx]
                # Check if this label is commonly corrected
                pattern_count = sum(
                    count for (orig, _), count in self._correction_patterns.items()
                    if orig == label
                )
                if pattern_count > 0:
                    adjusted[i] *= (1 + 0.1 * min(pattern_count, 10))

        return adjusted

    def get_correction_summary(self) -> dict[str, Any]:
        """Get summary of corrections.

        Returns:
            Dictionary with correction statistics
        """
        if not self._corrections:
            return {"total_corrections": 0}

        return {
            "total_corrections": len(self._corrections),
            "correction_patterns": dict(self._correction_patterns),
            "unique_annotators": len(set(
                c.annotator_id for c in self._corrections if c.annotator_id
            )),
            "avg_confidence": np.mean([c.confidence for c in self._corrections]),
            "recent_corrections": len(self._corrections[-10:]),
        }

    def export_corrections(self, output_path: str | Path) -> None:
        """Export corrections to file.

        Args:
            output_path: Path for output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "index": c.index,
                "original_label": c.original_label,
                "corrected_label": c.corrected_label,
                "confidence": c.confidence,
                "annotator_id": c.annotator_id,
                "timestamp": c.timestamp,
                "comment": c.comment,
            }
            for c in self._corrections
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def import_corrections(self, input_path: str | Path) -> int:
        """Import corrections from file.

        Args:
            input_path: Path to JSON file with corrections

        Returns:
            Number of corrections imported
        """
        input_path = Path(input_path)

        with open(input_path) as f:
            data = json.load(f)

        for item in data:
            self.record_correction(
                index=item["index"],
                original_label=item["original_label"],
                corrected_label=item["corrected_label"],
                confidence=item.get("confidence", 1.0),
                annotator_id=item.get("annotator_id"),
                comment=item.get("comment"),
            )

        return len(data)


class QueryByCommittee:
    """Query by Committee active learning strategy.

    Uses disagreement among a committee of models to select samples.
    """

    def __init__(
        self,
        models: list[Any] | None = None,
        n_models: int = 5,
    ):
        """Initialize the committee.

        Args:
            models: List of sklearn classifiers
            n_models: Number of models if not provided
        """
        if models is None:
            from sklearn.tree import DecisionTreeClassifier

            models = [
                DecisionTreeClassifier(random_state=i, max_depth=10)
                for i in range(n_models)
            ]

        self.models = models
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QueryByCommittee':
        """Fit all committee members.

        Args:
            X: Feature data
            y: Labels

        Returns:
            self
        """
        from sklearn.utils import resample

        for i, model in enumerate(self.models):
            # Bootstrap sample for each model
            X_boot, y_boot = resample(X, y, random_state=i)
            model.fit(X_boot, y_boot)

        self._fitted = True
        return self

    def select_samples(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        exclude_indices: list[int] | None = None,
    ) -> SampleSelection:
        """Select samples based on committee disagreement.

        Args:
            X: Feature data
            n_samples: Number of samples to select
            exclude_indices: Indices to exclude

        Returns:
            SampleSelection with selected indices
        """
        if not self._fitted:
            raise RuntimeError("Committee not fitted. Call fit() first.")

        # Get predictions from all models
        predictions = np.array([m.predict(X) for m in self.models])

        # Calculate disagreement (vote entropy)
        n_samples_total = X.shape[0]
        disagreement = np.zeros(n_samples_total)

        for i in range(n_samples_total):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            probs = counts / len(self.models)
            disagreement[i] = -np.sum(probs * np.log(probs + 1e-10))

        # Apply exclusions
        if exclude_indices:
            disagreement[exclude_indices] = -np.inf

        # Select top disagreement samples
        top_indices = np.argsort(disagreement)[::-1][:n_samples]

        return SampleSelection(
            indices=top_indices.tolist(),
            scores=disagreement[top_indices].tolist(),
            strategy=SamplingStrategy.ENTROPY,  # Closest match
            n_selected=len(top_indices),
            metadata={"method": "query_by_committee", "n_models": len(self.models)},
        )


class ExpectedModelChange:
    """Expected Model Change active learning strategy.

    Selects samples that would cause the largest change to the model
    if labeled.
    """

    def __init__(
        self,
        model: Any | None = None,
        n_gradient_samples: int = 10,
    ):
        """Initialize the EMC sampler.

        Args:
            model: Sklearn classifier with partial_fit
            n_gradient_samples: Samples for gradient estimation
        """
        from sklearn.linear_model import SGDClassifier

        self.model = model or SGDClassifier(random_state=42)
        self.n_gradient_samples = n_gradient_samples
        self._fitted = False
        self._classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExpectedModelChange':
        """Fit the model.

        Args:
            X: Feature data
            y: Labels

        Returns:
            self
        """
        self._classes = np.unique(y)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def select_samples(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        exclude_indices: list[int] | None = None,
    ) -> SampleSelection:
        """Select samples based on expected model change.

        Args:
            X: Feature data
            n_samples: Number of samples to select
            exclude_indices: Indices to exclude

        Returns:
            SampleSelection with selected indices
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


        n_total = X.shape[0]
        emc_scores = np.zeros(n_total)

        # Get current predictions
        pred_proba = self.model.predict_proba(X)

        for i in range(n_total):
            if exclude_indices and i in exclude_indices:
                emc_scores[i] = -np.inf
                continue

            # Estimate expected model change
            sample = X[i:i+1]
            change = 0.0

            for c, label in enumerate(self._classes):
                # Weight by probability of this being the true label
                prob = pred_proba[i, c]

                # Estimate gradient magnitude
                # Simplified: use prediction confidence as proxy
                gradient_magnitude = 1 - prob

                change += prob * gradient_magnitude

            emc_scores[i] = change

        # Select top EMC samples
        top_indices = np.argsort(emc_scores)[::-1][:n_samples]

        return SampleSelection(
            indices=top_indices.tolist(),
            scores=emc_scores[top_indices].tolist(),
            strategy=SamplingStrategy.UNCERTAINTY,
            n_selected=len(top_indices),
            metadata={"method": "expected_model_change"},
        )
