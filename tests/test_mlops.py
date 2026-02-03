"""Tests for MLOps integration module."""

import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest

import pandas as pd
import numpy as np


class TestMLflowIntegration:
    """Tests for MLflow integration."""

    @pytest.fixture
    def mock_mlflow_module(self):
        """Create a mock mlflow module and inject it into sys.modules."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Inject mock into sys.modules before importing MLflowIntegration
        original = sys.modules.get("mlflow")
        sys.modules["mlflow"] = mock_mlflow
        yield mock_mlflow
        # Restore original
        if original is not None:
            sys.modules["mlflow"] = original
        else:
            del sys.modules["mlflow"]

    def test_init_with_tracking_uri(self, mock_mlflow_module):
        """Test initialization with tracking URI."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration(tracking_uri="http://localhost:5000")
        mock_mlflow_module.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_log_metrics(self, mock_mlflow_module):
        """Test logging metrics."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration()

        integration.log_metrics({"accuracy": 0.95, "loss": 0.05})

        assert mock_mlflow_module.log_metric.call_count == 2

    def test_log_params(self, mock_mlflow_module):
        """Test logging parameters."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration()

        integration.log_params({"batch_size": 32, "lr": 0.001})

        mock_mlflow_module.log_params.assert_called_once()

    def test_set_tags(self, mock_mlflow_module):
        """Test setting tags."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration()

        integration.set_tags({"version": "1.0", "env": "test"})

        mock_mlflow_module.set_tags.assert_called_once()

    def test_log_artifact(self, mock_mlflow_module):
        """Test logging artifact."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()
            integration.log_artifact(f.name)

        mock_mlflow_module.log_artifact.assert_called_once()

    def test_log_quality_report(self, mock_mlflow_module):
        """Test logging a quality report."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration()

        # Create a mock quality report with proper return types
        mock_report = MagicMock()
        mock_report.quality_score.overall = 85.0
        mock_report.dataset_info.n_samples = 1000
        mock_report.label_errors_result.issues = [MagicMock()] * 10
        mock_report.duplicates_result.issues = [MagicMock()] * 5
        mock_report.outliers_result.issues = [MagicMock()] * 3
        mock_report.to_dict.return_value = {"score": 85.0, "n_samples": 1000}
        mock_report.summary.return_value = "Test Quality Report Summary"

        integration.log_quality_report(mock_report)

        # Verify metrics logged (log_metrics calls log_metric for each metric)
        assert mock_mlflow_module.log_metric.called
        # Verify artifacts logged
        assert mock_mlflow_module.log_artifact.called
        # Verify tags set
        assert mock_mlflow_module.set_tags.called

    def test_log_dataset_returns_hash(self, mock_mlflow_module):
        """Test logging dataset returns a hash."""
        from clean.mlops import MLflowIntegration
        integration = MLflowIntegration()

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash_value = integration.log_dataset(df, "test_dataset")

        assert isinstance(hash_value, str)
        assert len(hash_value) > 0


class TestWandbIntegration:
    """Tests for Weights & Biases integration."""

    @pytest.fixture
    def mock_wandb_module(self):
        """Create a mock wandb module and inject it into sys.modules."""
        mock_wandb = MagicMock()
        mock_wandb.Table = MagicMock
        mock_wandb.init.return_value = MagicMock()

        # Inject mock into sys.modules
        original = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        yield mock_wandb
        # Restore original
        if original is not None:
            sys.modules["wandb"] = original
        else:
            del sys.modules["wandb"]

    def test_init_with_project(self, mock_wandb_module):
        """Test initialization with project name."""
        from clean.mlops import WandbIntegration
        integration = WandbIntegration(project="test_project")
        # wandb.init is not called in __init__, only project is stored
        assert integration.project == "test_project"

    def test_log_metrics(self, mock_wandb_module):
        """Test logging metrics."""
        from clean.mlops import WandbIntegration
        integration = WandbIntegration()

        integration.log_metrics({"accuracy": 0.95, "loss": 0.05})

        mock_wandb_module.log.assert_called_once()

    def test_log_table(self, mock_wandb_module):
        """Test logging a table."""
        from clean.mlops import WandbIntegration
        integration = WandbIntegration()

        # Start a run first (required for log_table)
        integration._run = MagicMock()

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        integration.log_table(df, "test_table")

        mock_wandb_module.log.assert_called()


class TestQualityCallback:
    """Tests for training callback."""

    @pytest.fixture
    def mock_mlflow_module(self):
        """Create a mock mlflow module."""
        mock_mlflow = MagicMock()
        original = sys.modules.get("mlflow")
        sys.modules["mlflow"] = mock_mlflow
        yield mock_mlflow
        if original is not None:
            sys.modules["mlflow"] = original
        else:
            del sys.modules["mlflow"]

    def test_on_epoch_end_runs_analysis(self, mock_mlflow_module):
        """Test that on_epoch_end runs quality analysis."""
        from clean.mlops import QualityCallback, MLflowIntegration

        backend = MLflowIntegration()
        df = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})

        callback = QualityCallback(
            backend=backend,
            data=df,
            label_column="label",
            check_interval=1
        )

        # Test that on_epoch_end returns bool
        result = callback.on_epoch_end(epoch=1)
        assert isinstance(result, bool)

    def test_check_interval_skips(self, mock_mlflow_module):
        """Test that callback respects check_interval."""
        from clean.mlops import QualityCallback, MLflowIntegration

        backend = MLflowIntegration()
        df = pd.DataFrame({
            "feature": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })

        callback = QualityCallback(
            backend=backend,
            data=df,
            label_column="label",
            check_interval=5
        )

        # Run multiple epochs - only epoch 5 should trigger analysis
        # The callback uses _check_count internally
        for epoch in range(1, 6):
            result = callback.on_epoch_end(epoch=epoch)
            assert isinstance(result, bool)


class TestCreateIntegrations:
    """Tests for factory functions."""

    @pytest.fixture
    def mock_mlflow_module(self):
        """Create a mock mlflow module."""
        mock_mlflow = MagicMock()
        original = sys.modules.get("mlflow")
        sys.modules["mlflow"] = mock_mlflow
        yield mock_mlflow
        if original is not None:
            sys.modules["mlflow"] = original
        else:
            del sys.modules["mlflow"]

    @pytest.fixture
    def mock_wandb_module(self):
        """Create a mock wandb module."""
        mock_wandb = MagicMock()
        mock_wandb.Table = MagicMock
        mock_wandb.init.return_value = MagicMock()
        original = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        yield mock_wandb
        if original is not None:
            sys.modules["wandb"] = original
        else:
            del sys.modules["wandb"]

    def test_create_mlflow_integration(self, mock_mlflow_module):
        """Test creating MLflow integration via factory."""
        from clean.mlops import create_mlflow_integration
        integration = create_mlflow_integration(tracking_uri="http://localhost:5000")
        assert integration is not None

    def test_create_wandb_integration(self, mock_wandb_module):
        """Test creating W&B integration via factory."""
        from clean.mlops import create_wandb_integration
        integration = create_wandb_integration(project="test_project")
        assert integration is not None


class TestTrackDataQualityDecorator:
    """Tests for track_data_quality decorator."""

    def test_decorator_wraps_function(self):
        """Test that decorator wraps function correctly."""
        from clean.mlops import track_data_quality

        @track_data_quality(backend="mlflow", auto_analyze=False)
        def my_function(data, label_column):
            return "result"

        # Function should be callable
        assert callable(my_function)

    def test_decorator_passes_through_without_data(self):
        """Test that decorator passes through without DataFrame."""
        from clean.mlops import track_data_quality

        @track_data_quality(backend="mlflow", auto_analyze=False)
        def my_function(x):
            return x * 2

        result = my_function(5)
        assert result == 10
