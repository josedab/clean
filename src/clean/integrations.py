"""Native dbt and Airflow Integration.

First-class plugins for dbt tests and Airflow operators enabling quality
checks as part of existing data pipelines without code changes.

Example dbt usage:
    # In dbt project, add to schema.yml:
    # tests:
    #   - clean_quality_check:
    #       min_score: 80
    #       detect_label_errors: true

Example Airflow usage:
    >>> from clean.integrations import CleanQualityOperator
    >>>
    >>> quality_check = CleanQualityOperator(
    ...     task_id="check_data_quality",
    ...     dataset_path="/path/to/data.csv",
    ...     label_column="target",
    ...     min_quality_score=80,
    ... )
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Status of integration check."""

    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class IntegrationResult:
    """Result from an integration check."""

    status: IntegrationStatus
    quality_score: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "quality_score": self.quality_score,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# dbt Integration
# =============================================================================


class DbtTestBase(ABC):
    """Base class for dbt data quality tests."""

    @abstractmethod
    def run_test(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Run the test and return (passed, message)."""
        pass


class DbtQualityScoreTest(DbtTestBase):
    """dbt test for minimum quality score."""

    def __init__(self, min_score: float = 80.0):
        self.min_score = min_score

    def run_test(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if data meets minimum quality score."""
        from clean import DatasetCleaner

        cleaner = DatasetCleaner(data=data, label_column=label_column)
        report = cleaner.analyze(show_progress=False)

        score = report.quality_score.overall
        passed = score >= self.min_score

        if passed:
            message = f"Quality score {score:.1f} meets minimum threshold {self.min_score}"
        else:
            message = f"Quality score {score:.1f} below minimum threshold {self.min_score}"

        return passed, message


class DbtDuplicateTest(DbtTestBase):
    """dbt test for duplicate rate threshold."""

    def __init__(self, max_duplicate_rate: float = 0.05):
        self.max_duplicate_rate = max_duplicate_rate

    def run_test(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if duplicate rate is below threshold."""
        dup_rate = data.duplicated().mean()
        passed = dup_rate <= self.max_duplicate_rate

        if passed:
            message = f"Duplicate rate {dup_rate:.2%} within threshold {self.max_duplicate_rate:.2%}"
        else:
            message = f"Duplicate rate {dup_rate:.2%} exceeds threshold {self.max_duplicate_rate:.2%}"

        return passed, message


class DbtMissingValueTest(DbtTestBase):
    """dbt test for missing value rate threshold."""

    def __init__(self, max_missing_rate: float = 0.1, columns: list[str] | None = None):
        self.max_missing_rate = max_missing_rate
        self.columns = columns

    def run_test(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if missing value rate is below threshold."""
        columns = self.columns or data.columns.tolist()
        missing_rate = data[columns].isna().mean().mean()
        passed = missing_rate <= self.max_missing_rate

        if passed:
            message = f"Missing rate {missing_rate:.2%} within threshold {self.max_missing_rate:.2%}"
        else:
            message = f"Missing rate {missing_rate:.2%} exceeds threshold {self.max_missing_rate:.2%}"

        return passed, message


class DbtOutlierTest(DbtTestBase):
    """dbt test for outlier rate threshold."""

    def __init__(self, max_outlier_rate: float = 0.05):
        self.max_outlier_rate = max_outlier_rate

    def run_test(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if outlier rate is below threshold."""
        from clean import DatasetCleaner

        cleaner = DatasetCleaner(data=data)
        report = cleaner.analyze(
            detect_label_errors=False,
            detect_duplicates=False,
            detect_outliers=True,
            detect_imbalance=False,
            detect_bias=False,
            show_progress=False,
        )

        n_outliers = len(report.outliers()) if report.outliers_result else 0
        outlier_rate = n_outliers / len(data) if len(data) > 0 else 0

        passed = outlier_rate <= self.max_outlier_rate

        if passed:
            message = f"Outlier rate {outlier_rate:.2%} within threshold {self.max_outlier_rate:.2%}"
        else:
            message = f"Outlier rate {outlier_rate:.2%} exceeds threshold {self.max_outlier_rate:.2%}"

        return passed, message


class DbtTestRunner:
    """Runner for dbt data quality tests."""

    AVAILABLE_TESTS = {
        "quality_score": DbtQualityScoreTest,
        "duplicate_rate": DbtDuplicateTest,
        "missing_rate": DbtMissingValueTest,
        "outlier_rate": DbtOutlierTest,
    }

    def __init__(self):
        self.results: list[dict[str, Any]] = []

    def run_test(
        self,
        test_name: str,
        data: pd.DataFrame,
        test_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> IntegrationResult:
        """Run a specific dbt test.

        Args:
            test_name: Name of the test to run
            data: DataFrame to test
            test_config: Test configuration parameters
            **kwargs: Additional arguments

        Returns:
            IntegrationResult with test outcome
        """
        test_config = test_config or {}

        if test_name not in self.AVAILABLE_TESTS:
            return IntegrationResult(
                status=IntegrationStatus.ERROR,
                quality_score=0.0,
                message=f"Unknown test: {test_name}",
                details={"available_tests": list(self.AVAILABLE_TESTS.keys())},
            )

        test_class = self.AVAILABLE_TESTS[test_name]
        test_instance = test_class(**test_config)

        try:
            passed, message = test_instance.run_test(data, **kwargs)

            status = IntegrationStatus.SUCCESS if passed else IntegrationStatus.FAILURE
            quality_score = 100.0 if passed else 0.0

            result = IntegrationResult(
                status=status,
                quality_score=quality_score,
                message=message,
                details={
                    "test_name": test_name,
                    "test_config": test_config,
                    "passed": passed,
                },
            )

        except Exception as e:
            result = IntegrationResult(
                status=IntegrationStatus.ERROR,
                quality_score=0.0,
                message=f"Test error: {e!s}",
                details={"test_name": test_name, "error": str(e)},
            )

        self.results.append(result.to_dict())
        return result

    def run_all_tests(
        self,
        data: pd.DataFrame,
        tests: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> list[IntegrationResult]:
        """Run multiple tests.

        Args:
            data: DataFrame to test
            tests: Dict of {test_name: config}
            **kwargs: Additional arguments

        Returns:
            List of IntegrationResults
        """
        results = []
        for test_name, test_config in tests.items():
            result = self.run_test(test_name, data, test_config, **kwargs)
            results.append(result)
        return results


def generate_dbt_schema_yaml(
    model_name: str,
    tests: dict[str, dict[str, Any]],
    label_column: str | None = None,
) -> str:
    """Generate dbt schema.yml content for Clean tests.

    Args:
        model_name: Name of the dbt model
        tests: Dict of {test_name: config}
        label_column: Name of label column

    Returns:
        YAML content as string
    """
    yaml_content = f"""version: 2

models:
  - name: {model_name}
    description: "Model with Clean data quality tests"
    config:
      clean_label_column: {label_column or 'null'}
    tests:
"""

    for test_name, config in tests.items():
        yaml_content += f"      - clean_{test_name}:\n"
        for key, value in config.items():
            yaml_content += f"          {key}: {value}\n"

    return yaml_content


# =============================================================================
# Airflow Integration
# =============================================================================


@dataclass
class AirflowTaskConfig:
    """Configuration for Airflow Clean tasks."""

    task_id: str
    dataset_path: str | None = None
    dataset_connection_id: str | None = None
    sql_query: str | None = None
    label_column: str | None = None
    min_quality_score: float = 80.0
    fail_on_warning: bool = False
    detect_label_errors: bool = True
    detect_duplicates: bool = True
    detect_outliers: bool = True
    detect_imbalance: bool = True
    detect_bias: bool = True
    output_path: str | None = None
    xcom_push: bool = True


class BaseCleanOperator:
    """Base class for Airflow operators (framework-agnostic)."""

    def __init__(self, config: AirflowTaskConfig):
        self.config = config
        self._result: IntegrationResult | None = None

    def load_data(self) -> pd.DataFrame:
        """Load data from configured source."""
        if self.config.dataset_path:
            path = Path(self.config.dataset_path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
            elif path.suffix == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix == ".json":
                return pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        raise ValueError("No data source configured")

    def execute(self, context: dict[str, Any] | None = None) -> IntegrationResult:
        """Execute the quality check.

        Args:
            context: Airflow context (optional)

        Returns:
            IntegrationResult with check outcome
        """
        from clean import DatasetCleaner

        logger.info("Starting Clean quality check: %s", self.config.task_id)

        try:
            # Load data
            data = self.load_data()
            logger.info("Loaded %d rows", len(data))

            # Run analysis
            cleaner = DatasetCleaner(
                data=data,
                label_column=self.config.label_column,
            )

            report = cleaner.analyze(
                detect_label_errors=self.config.detect_label_errors,
                detect_duplicates=self.config.detect_duplicates,
                detect_outliers=self.config.detect_outliers,
                detect_imbalance=self.config.detect_imbalance,
                detect_bias=self.config.detect_bias,
                show_progress=False,
            )

            quality_score = report.quality_score.overall

            # Determine status
            if quality_score >= self.config.min_quality_score:
                status = IntegrationStatus.SUCCESS
                message = f"Quality check passed with score {quality_score:.1f}"
            elif quality_score >= self.config.min_quality_score * 0.9:
                status = IntegrationStatus.WARNING
                message = f"Quality check warning: score {quality_score:.1f} near threshold"
            else:
                status = IntegrationStatus.FAILURE
                message = f"Quality check failed: score {quality_score:.1f} below {self.config.min_quality_score}"

            # Collect details
            details = {
                "dataset_path": self.config.dataset_path,
                "n_samples": len(data),
                "quality_score": quality_score,
                "label_errors": len(report.label_errors()) if report.label_errors_result else 0,
                "duplicates": len(report.duplicates()) if report.duplicates_result else 0,
                "outliers": len(report.outliers()) if report.outliers_result else 0,
            }

            # Save output if configured
            if self.config.output_path:
                output_path = Path(self.config.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                output_data = {
                    "quality_score": quality_score,
                    "summary": report.summary(),
                    "details": details,
                    "timestamp": datetime.now().isoformat(),
                }

                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)

            self._result = IntegrationResult(
                status=status,
                quality_score=quality_score,
                message=message,
                details=details,
            )

        except Exception as e:
            logger.exception("Quality check failed")
            self._result = IntegrationResult(
                status=IntegrationStatus.ERROR,
                quality_score=0.0,
                message=f"Quality check error: {e!s}",
                details={"error": str(e)},
            )

        # Check if we should fail the task
        if self._result.status == IntegrationStatus.FAILURE:
            raise QualityCheckFailed(self._result.message)

        if self._result.status == IntegrationStatus.WARNING and self.config.fail_on_warning:
            raise QualityCheckFailed(self._result.message)

        return self._result


class QualityCheckFailed(Exception):
    """Exception raised when quality check fails."""

    pass


class CleanQualityOperator(BaseCleanOperator):
    """Airflow operator for running Clean quality checks.

    This is a framework-agnostic implementation that can be used
    directly or wrapped for specific Airflow versions.

    Example:
        >>> from clean.integrations import CleanQualityOperator, AirflowTaskConfig
        >>>
        >>> config = AirflowTaskConfig(
        ...     task_id="check_training_data",
        ...     dataset_path="/data/training.csv",
        ...     label_column="target",
        ...     min_quality_score=85,
        ... )
        >>> operator = CleanQualityOperator(config)
        >>> result = operator.execute()
    """

    def __init__(
        self,
        task_id: str,
        dataset_path: str | None = None,
        label_column: str | None = None,
        min_quality_score: float = 80.0,
        fail_on_warning: bool = False,
        output_path: str | None = None,
        **kwargs: Any,
    ):
        config = AirflowTaskConfig(
            task_id=task_id,
            dataset_path=dataset_path,
            label_column=label_column,
            min_quality_score=min_quality_score,
            fail_on_warning=fail_on_warning,
            output_path=output_path,
        )
        super().__init__(config)


class CleanDriftOperator(BaseCleanOperator):
    """Airflow operator for running drift detection.

    Example:
        >>> from clean.integrations import CleanDriftOperator
        >>>
        >>> operator = CleanDriftOperator(
        ...     task_id="check_drift",
        ...     reference_path="/data/reference.csv",
        ...     current_path="/data/current.csv",
        ...     drift_threshold=0.1,
        ... )
    """

    def __init__(
        self,
        task_id: str,
        reference_path: str,
        current_path: str,
        drift_threshold: float = 0.1,
        fail_on_drift: bool = True,
        **kwargs: Any,
    ):
        config = AirflowTaskConfig(task_id=task_id, dataset_path=current_path)
        super().__init__(config)
        self.reference_path = reference_path
        self.drift_threshold = drift_threshold
        self.fail_on_drift = fail_on_drift

    def execute(self, context: dict[str, Any] | None = None) -> IntegrationResult:
        """Execute drift detection."""
        from clean import DriftDetector

        logger.info("Starting drift detection: %s", self.config.task_id)

        try:
            # Load data
            reference_data = pd.read_csv(self.reference_path)
            current_data = self.load_data()

            # Run drift detection
            detector = DriftDetector()
            detector.fit(reference_data)
            report = detector.detect(current_data)

            drift_score = report.overall_drift_score

            if drift_score <= self.drift_threshold:
                status = IntegrationStatus.SUCCESS
                message = f"No significant drift detected (score: {drift_score:.3f})"
            else:
                status = IntegrationStatus.FAILURE
                message = f"Drift detected (score: {drift_score:.3f} > {self.drift_threshold})"

            details = {
                "drift_score": drift_score,
                "drifted_features": report.drifted_features,
                "reference_size": len(reference_data),
                "current_size": len(current_data),
            }

            self._result = IntegrationResult(
                status=status,
                quality_score=100 * (1 - drift_score),
                message=message,
                details=details,
            )

            if status == IntegrationStatus.FAILURE and self.fail_on_drift:
                raise QualityCheckFailed(message)

        except Exception as e:
            if isinstance(e, QualityCheckFailed):
                raise
            logger.exception("Drift detection failed")
            self._result = IntegrationResult(
                status=IntegrationStatus.ERROR,
                quality_score=0.0,
                message=f"Drift detection error: {e!s}",
                details={"error": str(e)},
            )

        return self._result


def generate_airflow_dag(
    dag_id: str,
    dataset_path: str,
    label_column: str | None = None,
    schedule: str = "@daily",
    min_quality_score: float = 80.0,
) -> str:
    """Generate Airflow DAG code for Clean quality checks.

    Args:
        dag_id: ID for the DAG
        dataset_path: Path to the dataset
        label_column: Name of label column
        schedule: Cron schedule
        min_quality_score: Minimum quality score

    Returns:
        Python code for the DAG
    """
    label_param = f'"{label_column}"' if label_column else "None"

    dag_code = f'''"""Auto-generated Clean quality check DAG."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import Clean integration
from clean.integrations import CleanQualityOperator

default_args = {{
    "owner": "clean",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}}

with DAG(
    dag_id="{dag_id}",
    default_args=default_args,
    description="Data quality check with Clean",
    schedule_interval="{schedule}",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data-quality", "clean"],
) as dag:

    def run_quality_check(**context):
        """Run Clean quality check."""
        from clean.integrations import CleanQualityOperator, AirflowTaskConfig

        config = AirflowTaskConfig(
            task_id="quality_check",
            dataset_path="{dataset_path}",
            label_column={label_param},
            min_quality_score={min_quality_score},
        )
        operator = CleanQualityOperator(config)
        result = operator.execute(context)

        # Push results to XCom
        context["ti"].xcom_push(key="quality_score", value=result.quality_score)
        context["ti"].xcom_push(key="status", value=result.status.value)

        return result.to_dict()

    quality_check = PythonOperator(
        task_id="quality_check",
        python_callable=run_quality_check,
        provide_context=True,
    )
'''

    return dag_code


# =============================================================================
# Generic Pipeline Integration
# =============================================================================


class PipelineQualityGate:
    """Generic quality gate for any data pipeline."""

    def __init__(
        self,
        min_quality_score: float = 80.0,
        max_duplicate_rate: float = 0.05,
        max_missing_rate: float = 0.1,
        max_outlier_rate: float = 0.05,
        fail_fast: bool = True,
    ):
        self.min_quality_score = min_quality_score
        self.max_duplicate_rate = max_duplicate_rate
        self.max_missing_rate = max_missing_rate
        self.max_outlier_rate = max_outlier_rate
        self.fail_fast = fail_fast
        self.results: list[IntegrationResult] = []

    def check(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
    ) -> IntegrationResult:
        """Run all quality checks.

        Args:
            data: DataFrame to check
            label_column: Name of label column

        Returns:
            IntegrationResult with overall outcome
        """
        from clean import DatasetCleaner

        all_passed = True
        messages = []
        details: dict[str, Any] = {}

        # Full quality analysis
        cleaner = DatasetCleaner(data=data, label_column=label_column)
        report = cleaner.analyze(show_progress=False)

        quality_score = report.quality_score.overall
        details["quality_score"] = quality_score

        if quality_score < self.min_quality_score:
            all_passed = False
            messages.append(f"Quality score {quality_score:.1f} below {self.min_quality_score}")
        else:
            messages.append(f"Quality score {quality_score:.1f} OK")

        # Check duplicate rate
        dup_rate = data.duplicated().mean()
        details["duplicate_rate"] = dup_rate
        if dup_rate > self.max_duplicate_rate:
            all_passed = False
            messages.append(f"Duplicate rate {dup_rate:.2%} exceeds {self.max_duplicate_rate:.2%}")

        # Check missing rate
        missing_rate = data.isna().mean().mean()
        details["missing_rate"] = missing_rate
        if missing_rate > self.max_missing_rate:
            all_passed = False
            messages.append(f"Missing rate {missing_rate:.2%} exceeds {self.max_missing_rate:.2%}")

        # Check outlier rate
        if report.outliers_result:
            outlier_rate = len(report.outliers()) / len(data) if len(data) > 0 else 0
            details["outlier_rate"] = outlier_rate
            if outlier_rate > self.max_outlier_rate:
                all_passed = False
                messages.append(f"Outlier rate {outlier_rate:.2%} exceeds {self.max_outlier_rate:.2%}")

        if all_passed:
            status = IntegrationStatus.SUCCESS
            message = "All quality gates passed"
        else:
            status = IntegrationStatus.FAILURE
            message = "; ".join(messages)

        result = IntegrationResult(
            status=status,
            quality_score=quality_score,
            message=message,
            details=details,
        )

        self.results.append(result)

        if not all_passed and self.fail_fast:
            raise QualityCheckFailed(message)

        return result


def create_quality_gate(
    min_quality_score: float = 80.0,
    fail_fast: bool = True,
) -> PipelineQualityGate:
    """Create a pipeline quality gate.

    Args:
        min_quality_score: Minimum quality score to pass
        fail_fast: Whether to raise exception on failure

    Returns:
        Configured PipelineQualityGate
    """
    return PipelineQualityGate(
        min_quality_score=min_quality_score,
        fail_fast=fail_fast,
    )


def check_data_quality(
    data: pd.DataFrame,
    label_column: str | None = None,
    min_score: float = 80.0,
    raise_on_failure: bool = False,
) -> IntegrationResult:
    """Convenience function for quick quality checks.

    Args:
        data: DataFrame to check
        label_column: Name of label column
        min_score: Minimum quality score
        raise_on_failure: Whether to raise exception on failure

    Returns:
        IntegrationResult
    """
    gate = PipelineQualityGate(
        min_quality_score=min_score,
        fail_fast=raise_on_failure,
    )
    return gate.check(data, label_column)
