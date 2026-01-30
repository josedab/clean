"""Tests for the web dashboard module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient
    from clean.dashboard import (
        DashboardApp,
        DashboardConfig,
        create_dashboard_app,
    )
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed"
)


class TestDashboardConfig:
    """Tests for DashboardConfig class."""

    def test_config_defaults(self) -> None:
        config = DashboardConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.enable_upload is True

    def test_config_custom(self) -> None:
        config = DashboardConfig(
            host="0.0.0.0",
            port=3000,
            title="Test Dashboard",
        )

        assert config.host == "0.0.0.0"
        assert config.port == 3000
        assert config.title == "Test Dashboard"

    def test_config_to_dict(self) -> None:
        config = DashboardConfig(port=9000)
        d = config.to_dict()

        assert d["port"] == 9000
        assert "host" in d
        assert "title" in d


class TestDashboardApp:
    """Tests for DashboardApp class."""

    @pytest.fixture
    def app(self) -> DashboardApp:
        """Create a dashboard app."""
        return create_dashboard_app(port=8888)

    @pytest.fixture
    def client(self, app: DashboardApp) -> TestClient:
        """Create a test client."""
        return TestClient(app.app)

    def test_index_route(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Data Quality Dashboard" in response.text

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_config_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert data["port"] == 8888

    def test_analyze_endpoint(self, client: TestClient, tmp_path) -> None:
        # Create test CSV
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1, 2], 100),
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("test.csv", f, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 100
        assert data["total_features"] == 3
        assert "quality_score" in data
        assert "features" in data

    def test_analyze_with_label_column(self, client: TestClient, tmp_path) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "target": np.random.choice([0, 1], 50),
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze?label_column=target",
                files={"file": ("test.csv", f, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 50

    def test_analyze_detects_duplicates(self, client: TestClient, tmp_path) -> None:
        # Create CSV with duplicates
        df = pd.DataFrame({
            "x": [1, 2, 3, 3, 3],  # 2 duplicates
            "y": [1, 2, 3, 3, 3],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("test.csv", f, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["issue_counts"]["duplicates"] == 2

    def test_analyze_detects_missing_values(self, client: TestClient, tmp_path) -> None:
        df = pd.DataFrame({
            "x": [1, 2, None, 4, None],
            "y": [1, 2, 3, 4, 5],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("test.csv", f, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["issue_counts"]["missing_values"] == 2

    def test_analyze_invalid_file(self, client: TestClient, tmp_path) -> None:
        # Create invalid file
        invalid_path = tmp_path / "invalid.csv"
        invalid_path.write_text("not,valid,csv\n1,2")

        with open(invalid_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("invalid.csv", f, "text/csv")},
            )

        # Should either succeed with partial data or return 400
        assert response.status_code in [200, 400]


class TestCreateDashboardApp:
    """Tests for create_dashboard_app function."""

    def test_create_with_defaults(self) -> None:
        app = create_dashboard_app()

        assert isinstance(app, DashboardApp)
        assert app.config.port == 8080

    def test_create_with_custom_port(self) -> None:
        app = create_dashboard_app(port=5000)

        assert app.config.port == 5000

    def test_create_with_custom_title(self) -> None:
        app = create_dashboard_app(title="My Dashboard")

        assert app.config.title == "My Dashboard"


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = create_dashboard_app()
        return TestClient(app.app)

    def test_empty_csv(self, client: TestClient, tmp_path) -> None:
        # Create empty CSV
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("x,y,label\n")

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("empty.csv", f, "text/csv")},
            )

        # Empty CSV may return 200 with 0 samples or 400 if analyzer fails
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert data["total_samples"] == 0

    def test_single_row(self, client: TestClient, tmp_path) -> None:
        csv_path = tmp_path / "single.csv"
        csv_path.write_text("x,y,label\n1,2,0\n")

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("single.csv", f, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 1

    def test_feature_info(self, client: TestClient, tmp_path) -> None:
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": ("test.csv", f, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()

        assert len(data["features"]) == 3
        feature_names = [f["name"] for f in data["features"]]
        assert "int_col" in feature_names
        assert "float_col" in feature_names
