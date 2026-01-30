"""Tests for active learning integration module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean.active_learning import (
    ActiveLearner,
    CVATExporter,
    LabelStudioExporter,
    ProdigyExporter,
    SampleSelection,
    SamplingStrategy,
    select_for_labeling,
)


class TestActiveLearner:
    """Tests for ActiveLearner class."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample classification data."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.choice([0, 1, 2], 200)
        return X, y

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "label": np.random.choice([0, 1, 2], 200),
        })

    def test_learner_init(self) -> None:
        learner = ActiveLearner()
        assert learner is not None
        assert learner.strategy == SamplingStrategy.UNCERTAINTY

    def test_learner_with_strategy_string(self) -> None:
        learner = ActiveLearner(strategy="entropy")
        assert learner.strategy == SamplingStrategy.ENTROPY

    def test_random_sampling(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.RANDOM)
        selection = learner.select_samples(X, y, n_samples=50)

        assert isinstance(selection, SampleSelection)
        assert len(selection.indices) == 50
        assert selection.strategy == SamplingStrategy.RANDOM

    def test_uncertainty_sampling(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.UNCERTAINTY)
        selection = learner.select_samples(X, y, n_samples=50)

        assert len(selection.indices) == 50
        assert selection.strategy == SamplingStrategy.UNCERTAINTY
        # Scores should be uncertainty values (higher = more uncertain)
        assert all(0 <= s <= 1 for s in selection.scores)

    def test_margin_sampling(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.MARGIN)
        selection = learner.select_samples(X, y, n_samples=50)

        assert len(selection.indices) == 50
        assert selection.strategy == SamplingStrategy.MARGIN

    def test_entropy_sampling(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.ENTROPY)
        selection = learner.select_samples(X, y, n_samples=50)

        assert len(selection.indices) == 50
        assert selection.strategy == SamplingStrategy.ENTROPY

    def test_diversity_sampling(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.DIVERSITY)
        selection = learner.select_samples(X, n_samples=50)  # No labels needed

        assert len(selection.indices) == 50
        assert selection.strategy == SamplingStrategy.DIVERSITY

    def test_combined_sampling(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.COMBINED)
        selection = learner.select_samples(X, y, n_samples=50)

        assert len(selection.indices) <= 50
        assert selection.strategy == SamplingStrategy.COMBINED

    def test_exclude_indices(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.RANDOM)

        # Exclude first 50 indices
        exclude = list(range(50))
        selection = learner.select_samples(X, y, n_samples=20, exclude_indices=exclude)

        # No selected index should be in excluded
        assert all(idx not in exclude for idx in selection.indices)

    def test_with_dataframe(self, sample_df: pd.DataFrame) -> None:
        X = sample_df[["feature1", "feature2"]]
        y = sample_df["label"].values

        learner = ActiveLearner(strategy=SamplingStrategy.UNCERTAINTY)
        selection = learner.select_samples(X, y, n_samples=30)

        assert len(selection.indices) == 30

    def test_selection_to_dataframe(self, sample_df: pd.DataFrame) -> None:
        X = sample_df[["feature1", "feature2"]]
        y = sample_df["label"].values

        learner = ActiveLearner(strategy=SamplingStrategy.UNCERTAINTY)
        selection = learner.select_samples(X, y, n_samples=30)

        result_df = selection.to_dataframe(sample_df)
        assert len(result_df) == 30
        assert "al_score" in result_df.columns
        assert "al_rank" in result_df.columns

    def test_n_samples_larger_than_data(self, sample_data: tuple) -> None:
        X, y = sample_data
        learner = ActiveLearner(strategy=SamplingStrategy.RANDOM)
        selection = learner.select_samples(X, y, n_samples=1000)

        # Should return all available samples
        assert len(selection.indices) == len(X)


class TestLabelStudioExporter:
    """Tests for LabelStudioExporter class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for export."""
        return pd.DataFrame({
            "text": ["This is text 1", "This is text 2", "This is text 3"],
            "label": [0, 1, 0],
            "al_score": [0.9, 0.8, 0.7],
        })

    def test_exporter_init(self) -> None:
        exporter = LabelStudioExporter(text_column="text")
        assert exporter is not None

    def test_create_tasks(self, sample_df: pd.DataFrame) -> None:
        exporter = LabelStudioExporter(text_column="text")
        tasks = exporter.create_tasks(sample_df)

        assert len(tasks) == 3
        assert "data" in tasks[0]
        assert "text" in tasks[0]["data"]

    def test_export_to_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        exporter = LabelStudioExporter(text_column="text")
        output_path = tmp_path / "tasks.json"

        exporter.export(sample_df, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            tasks = json.load(f)
        assert len(tasks) == 3

    def test_create_config_classification(self) -> None:
        exporter = LabelStudioExporter()
        config = exporter.create_config(
            task_type="classification",
            labels=["positive", "negative"],
        )

        assert "<Choices" in config
        assert "positive" in config
        assert "negative" in config

    def test_create_config_ner(self) -> None:
        exporter = LabelStudioExporter()
        config = exporter.create_config(
            task_type="ner",
            labels=["PER", "ORG", "LOC"],
        )

        assert "<Labels" in config
        assert "PER" in config


class TestCVATExporter:
    """Tests for CVATExporter class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for export."""
        return pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "label": [0, 1, 0],
        })

    def test_exporter_init(self) -> None:
        exporter = CVATExporter(image_column="image_path")
        assert exporter is not None

    def test_create_tasks(self, sample_df: pd.DataFrame) -> None:
        exporter = CVATExporter(image_column="image_path")
        tasks = exporter.create_tasks(sample_df)

        assert len(tasks) == 3
        assert "image_path" in tasks[0]

    def test_export_to_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        exporter = CVATExporter(image_column="image_path")
        output_path = tmp_path / "annotations.xml"

        exporter.export(sample_df, output_path, labels=["car", "person"])

        assert output_path.exists()
        content = output_path.read_text()
        assert "<annotations>" in content
        assert "car" in content


class TestProdigyExporter:
    """Tests for ProdigyExporter class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for export."""
        return pd.DataFrame({
            "text": ["Text 1", "Text 2", "Text 3"],
            "al_score": [0.9, 0.8, 0.7],
        })

    def test_exporter_init(self) -> None:
        exporter = ProdigyExporter(text_column="text")
        assert exporter is not None

    def test_create_tasks(self, sample_df: pd.DataFrame) -> None:
        exporter = ProdigyExporter(text_column="text")
        examples = exporter.create_tasks(sample_df)

        assert len(examples) == 3
        assert "text" in examples[0]
        assert "meta" in examples[0]

    def test_export_to_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        exporter = ProdigyExporter(text_column="text")
        output_path = tmp_path / "examples.jsonl"

        exporter.export(sample_df, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            example = json.loads(line)
            assert "text" in example


class TestSelectForLabeling:
    """Tests for select_for_labeling convenience function."""

    def test_function_works(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        selection = select_for_labeling(X, y, n_samples=20, strategy="uncertainty")

        assert isinstance(selection, SampleSelection)
        assert len(selection.indices) == 20


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_dataset(self) -> None:
        X = np.random.randn(5, 3)
        y = np.array([0, 1, 0, 1, 0])

        learner = ActiveLearner(strategy=SamplingStrategy.UNCERTAINTY)
        selection = learner.select_samples(X, y, n_samples=3)

        assert len(selection.indices) == 3

    def test_binary_classification(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1], 50)

        learner = ActiveLearner(strategy=SamplingStrategy.MARGIN)
        selection = learner.select_samples(X, y, n_samples=10)

        assert len(selection.indices) == 10

    def test_diversity_without_labels(self) -> None:
        X = np.random.randn(100, 5)

        learner = ActiveLearner(strategy=SamplingStrategy.DIVERSITY)
        selection = learner.select_samples(X, n_samples=20)

        assert len(selection.indices) == 20

    def test_dataframe_with_non_numeric(self) -> None:
        df = pd.DataFrame({
            "numeric1": np.random.randn(50),
            "numeric2": np.random.randn(50),
            "text": ["text"] * 50,
        })
        y = np.random.choice([0, 1], 50)

        learner = ActiveLearner(strategy=SamplingStrategy.RANDOM)
        selection = learner.select_samples(df, y, n_samples=10)

        assert len(selection.indices) == 10
