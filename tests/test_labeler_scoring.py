"""Tests for labeler_scoring module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestLabelerScoringModule:
    """Tests for automated labeler performance scoring module."""

    def test_imports(self) -> None:
        from clean.labeler_scoring import (
            LabelerEvaluator,
            LabelerMetrics,
            LabelerReport,
            SmartRouter,
            evaluate_labelers,
            get_labeler_report,
        )
        assert LabelerEvaluator is not None
        assert LabelerMetrics is not None
        assert LabelerReport is not None
        assert SmartRouter is not None

    def test_evaluator_init(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator
        
        evaluator = LabelerEvaluator()
        assert evaluator is not None
        
        evaluator_with_params = LabelerEvaluator(window_size=50, min_labels_for_evaluation=5)
        assert evaluator_with_params is not None

    def test_evaluator_fit_basic(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator
        
        np.random.seed(42)
        n_labels = 200
        
        labels = np.random.choice([0, 1], n_labels)
        labeler_ids = np.random.choice(["labeler_1", "labeler_2", "labeler_3"], n_labels)
        
        evaluator = LabelerEvaluator()
        evaluator.fit(list(labels), list(labeler_ids), ground_truth=list(labels))
        
        assert evaluator is not None

    def test_get_labeler_metrics(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator, LabelerMetrics
        
        np.random.seed(42)
        n_labels = 200
        
        labels = list(np.random.choice([0, 1], n_labels))
        labeler_ids = list(np.random.choice(["labeler_1", "labeler_2"], n_labels))
        
        evaluator = LabelerEvaluator(min_labels_for_evaluation=5)
        evaluator.fit(labels, labeler_ids, ground_truth=labels)
        
        metrics = evaluator.get_labeler_metrics("labeler_1")
        
        if metrics is not None:
            assert isinstance(metrics, LabelerMetrics)
            assert metrics.labeler_id == "labeler_1"
            assert metrics.n_labels > 0

    def test_get_all_labeler_metrics(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator
        
        np.random.seed(42)
        n_labels = 200
        
        labels = list(np.random.choice([0, 1], n_labels))
        labeler_ids = list(np.random.choice(["labeler_1", "labeler_2", "labeler_3"], n_labels))
        
        evaluator = LabelerEvaluator(min_labels_for_evaluation=5)
        evaluator.fit(labels, labeler_ids, ground_truth=labels)
        
        all_metrics = evaluator.get_all_labeler_metrics()
        
        assert isinstance(all_metrics, dict)

    def test_get_labeler_ranking(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator
        
        np.random.seed(42)
        n_labels = 300
        
        labels = list(np.random.choice([0, 1], n_labels))
        labeler_ids = list(np.random.choice(["labeler_1", "labeler_2", "labeler_3"], n_labels))
        
        evaluator = LabelerEvaluator(min_labels_for_evaluation=5)
        evaluator.fit(labels, labeler_ids, ground_truth=labels)
        
        ranking = evaluator.get_labeler_ranking(metric="accuracy")
        
        assert isinstance(ranking, list)
        for entry in ranking:
            assert len(entry) == 2

    def test_labeler_metrics_fields(self) -> None:
        from clean.labeler_scoring import LabelerMetrics, ExpertiseLevel
        from datetime import datetime
        
        # Get actual ExpertiseLevel values
        expertise = ExpertiseLevel.EXPERT if hasattr(ExpertiseLevel, 'EXPERT') else list(ExpertiseLevel)[0]
        
        metrics = LabelerMetrics(
            labeler_id="test_labeler",
            n_labels=100,
            n_tasks=50,
            accuracy=0.95,
            error_rate=0.05,
            agreement_rate=0.90,
            consistency_score=0.92,
            self_agreement=0.88,
            avg_labels_per_day=20.0,
            completion_rate=0.95,
            expertise_level=expertise,
            strong_categories=["cat_a"],
            weak_categories=["cat_b"],
            performance_trend="improving",
            recent_accuracy=0.97,
            accuracy_change=0.02,
            first_label_date=datetime.now(),
            last_label_date=datetime.now(),
            active_days=30,
            metadata={},
        )
        
        assert metrics.labeler_id == "test_labeler"
        assert metrics.accuracy == 0.95

    def test_smart_router_init(self) -> None:
        from clean.labeler_scoring import SmartRouter, LabelerEvaluator
        
        evaluator = LabelerEvaluator()
        router = SmartRouter(evaluator)
        
        assert router is not None

    def test_evaluator_with_timestamps(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        n_labels = 100
        
        labels = list(np.random.choice([0, 1], n_labels))
        labeler_ids = list(np.random.choice(["labeler_1", "labeler_2"], n_labels))
        
        base_time = datetime.now()
        timestamps = [base_time + timedelta(hours=i) for i in range(n_labels)]
        
        evaluator = LabelerEvaluator(min_labels_for_evaluation=5)
        evaluator.fit(labels, labeler_ids, ground_truth=labels, timestamps=timestamps)
        
        assert evaluator is not None

    def test_evaluator_with_task_ids(self) -> None:
        from clean.labeler_scoring import LabelerEvaluator
        
        np.random.seed(42)
        n_labels = 100
        
        labels = list(np.random.choice([0, 1], n_labels))
        labeler_ids = list(np.random.choice(["labeler_1", "labeler_2"], n_labels))
        task_ids = [f"task_{i % 50}" for i in range(n_labels)]
        
        evaluator = LabelerEvaluator(min_labels_for_evaluation=5)
        evaluator.fit(labels, labeler_ids, ground_truth=labels, task_ids=task_ids)
        
        assert evaluator is not None
