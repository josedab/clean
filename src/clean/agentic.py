"""Agentic Data Curation System.

AI-powered autonomous data cleaning agent that identifies issues, suggests fixes,
executes approved changes, and learns from human feedback.

Example:
    >>> from clean.agentic import DataCurationAgent, AgentConfig
    >>>
    >>> # Create agent
    >>> agent = DataCurationAgent(data=df, label_column="label")
    >>>
    >>> # Run autonomous analysis
    >>> plan = agent.analyze_and_plan()
    >>> print(plan.summary())
    >>>
    >>> # Review and approve fixes
    >>> agent.approve_actions([0, 1, 2])  # Approve first 3 actions
    >>> result = agent.execute_approved()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of curation actions."""

    REMOVE_DUPLICATE = "remove_duplicate"
    REMOVE_OUTLIER = "remove_outlier"
    RELABEL = "relabel"
    FLAG_FOR_REVIEW = "flag_for_review"
    IMPUTE_MISSING = "impute_missing"
    NORMALIZE = "normalize"
    FIX_ENCODING = "fix_encoding"
    MERGE_SIMILAR = "merge_similar"
    SPLIT_COMPOUND = "split_compound"
    AUGMENT_MINORITY = "augment_minority"


class ActionPriority(Enum):
    """Priority levels for actions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionStatus(Enum):
    """Status of a curation action."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    REVERTED = "reverted"


@dataclass
class CurationAction:
    """A single curation action proposed by the agent."""

    action_id: int
    action_type: ActionType
    priority: ActionPriority
    target_indices: list[int]
    description: str
    confidence: float
    estimated_impact: float
    details: dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PROPOSED
    execution_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "priority": self.priority.value,
            "target_indices": self.target_indices,
            "description": self.description,
            "confidence": self.confidence,
            "estimated_impact": self.estimated_impact,
            "details": self.details,
            "status": self.status.value,
        }


@dataclass
class CurationPlan:
    """Complete curation plan with all proposed actions."""

    timestamp: datetime
    dataset_info: dict[str, Any]
    quality_score_before: float
    estimated_score_after: float
    actions: list[CurationAction]
    total_affected_rows: int
    priority_breakdown: dict[ActionPriority, int]
    estimated_time_minutes: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Data Curation Plan",
            "=" * 50,
            f"Generated: {self.timestamp.isoformat()}",
            f"Dataset: {self.dataset_info.get('n_samples', 'unknown'):,} samples",
            "",
            f"Quality Score: {self.quality_score_before:.1f} → {self.estimated_score_after:.1f} (estimated)",
            f"Total Actions: {len(self.actions)}",
            f"Affected Rows: {self.total_affected_rows:,}",
            "",
            "Priority Breakdown:",
        ]

        for priority, count in self.priority_breakdown.items():
            lines.append(f"  • {priority.value}: {count}")

        lines.append("")
        lines.append("Proposed Actions:")

        for action in sorted(self.actions, key=lambda a: (a.priority.value, -a.confidence)):
            status_icon = "○" if action.status == ActionStatus.PROPOSED else "✓"
            lines.append(
                f"  {status_icon} [{action.action_id}] {action.action_type.value} "
                f"(confidence: {action.confidence:.2f}, impact: {action.estimated_impact:.2f})"
            )
            lines.append(f"      {action.description}")

        return "\n".join(lines)

    def get_pending_actions(self) -> list[CurationAction]:
        """Get actions that haven't been executed."""
        return [a for a in self.actions if a.status in (ActionStatus.PROPOSED, ActionStatus.APPROVED)]


@dataclass
class ExecutionResult:
    """Result of executing curation actions."""

    timestamp: datetime
    actions_executed: int
    actions_failed: int
    rows_modified: int
    rows_removed: int
    quality_score_after: float
    errors: list[str]
    rollback_available: bool

    def summary(self) -> str:
        """Generate summary."""
        lines = [
            "Curation Execution Result",
            "=" * 50,
            f"Executed: {self.timestamp.isoformat()}",
            f"Actions Executed: {self.actions_executed}",
            f"Actions Failed: {self.actions_failed}",
            f"Rows Modified: {self.rows_modified:,}",
            f"Rows Removed: {self.rows_removed:,}",
            f"Quality Score: {self.quality_score_after:.1f}",
        ]

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for err in self.errors[:5]:
                lines.append(f"  • {err}")

        if self.rollback_available:
            lines.append("")
            lines.append("Rollback available: call agent.rollback() to revert changes")

        return "\n".join(lines)


@dataclass
class FeedbackRecord:
    """Record of human feedback on agent actions."""

    action_id: int
    action_type: ActionType
    was_approved: bool
    was_modified: bool
    human_correction: str | None
    timestamp: datetime = field(default_factory=datetime.now)


class ActionExecutor(ABC):
    """Base class for action executors."""

    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """The action type this executor handles."""
        pass

    @abstractmethod
    def execute(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute the action on the data.

        Returns:
            Tuple of (modified_data, execution_info)
        """
        pass

    @abstractmethod
    def estimate_impact(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> float:
        """Estimate the impact of executing this action."""
        pass


class DuplicateRemovalExecutor(ActionExecutor):
    """Executor for duplicate removal actions."""

    @property
    def action_type(self) -> ActionType:
        return ActionType.REMOVE_DUPLICATE

    def execute(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        indices_to_remove = set(action.target_indices)
        original_len = len(data)

        # Keep first occurrence, remove duplicates
        mask = ~data.index.isin(indices_to_remove)
        modified_data = data[mask].copy()

        return modified_data, {
            "rows_removed": original_len - len(modified_data),
            "indices_removed": list(indices_to_remove),
        }

    def estimate_impact(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> float:
        # Impact based on proportion of data affected
        removal_rate = len(action.target_indices) / max(len(data), 1)
        # Removing duplicates generally improves quality
        return min(0.5, removal_rate * 10)


class OutlierRemovalExecutor(ActionExecutor):
    """Executor for outlier removal actions."""

    @property
    def action_type(self) -> ActionType:
        return ActionType.REMOVE_OUTLIER

    def execute(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        indices_to_remove = set(action.target_indices)
        original_len = len(data)

        mask = ~data.index.isin(indices_to_remove)
        modified_data = data[mask].copy()

        return modified_data, {
            "rows_removed": original_len - len(modified_data),
            "outlier_indices": list(indices_to_remove),
        }

    def estimate_impact(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> float:
        # Moderate impact for outlier removal
        return min(0.3, len(action.target_indices) / max(len(data), 1) * 5)


class RelabelExecutor(ActionExecutor):
    """Executor for relabeling actions."""

    @property
    def action_type(self) -> ActionType:
        return ActionType.RELABEL

    def execute(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        modified_data = data.copy()
        label_column = action.details.get("label_column", "label")
        new_labels = action.details.get("new_labels", {})

        relabeled_count = 0
        for idx in action.target_indices:
            if idx in new_labels and idx in modified_data.index:
                modified_data.loc[idx, label_column] = new_labels[idx]
                relabeled_count += 1

        return modified_data, {
            "rows_relabeled": relabeled_count,
            "label_changes": new_labels,
        }

    def estimate_impact(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> float:
        # Higher impact for relabeling - directly affects model training
        return min(0.8, len(action.target_indices) / max(len(data), 1) * 20)


class ImputationExecutor(ActionExecutor):
    """Executor for missing value imputation."""

    @property
    def action_type(self) -> ActionType:
        return ActionType.IMPUTE_MISSING

    def execute(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        modified_data = data.copy()
        column = action.details.get("column")
        method = action.details.get("method", "mean")
        impute_value = action.details.get("impute_value")

        if column is None:
            return modified_data, {"error": "No column specified"}

        original_missing = modified_data[column].isna().sum()

        if impute_value is not None:
            modified_data[column] = modified_data[column].fillna(impute_value)
        elif method == "mean":
            modified_data[column] = modified_data[column].fillna(
                modified_data[column].mean()
            )
        elif method == "median":
            modified_data[column] = modified_data[column].fillna(
                modified_data[column].median()
            )
        elif method == "mode":
            mode_val = modified_data[column].mode()
            if len(mode_val) > 0:
                modified_data[column] = modified_data[column].fillna(mode_val.iloc[0])
        elif method == "forward_fill":
            modified_data[column] = modified_data[column].ffill()
        elif method == "backward_fill":
            modified_data[column] = modified_data[column].bfill()

        new_missing = modified_data[column].isna().sum()

        return modified_data, {
            "column": column,
            "method": method,
            "values_imputed": int(original_missing - new_missing),
        }

    def estimate_impact(
        self,
        data: pd.DataFrame,
        action: CurationAction,
    ) -> float:
        column = action.details.get("column")
        if column and column in data.columns:
            missing_rate = data[column].isna().mean()
            return min(0.4, missing_rate * 2)
        return 0.1


class AgentConfig:
    """Configuration for the curation agent."""

    def __init__(
        self,
        auto_approve_threshold: float = 0.95,
        max_actions_per_run: int = 100,
        min_confidence: float = 0.5,
        enable_learning: bool = True,
        require_approval: bool = True,
        priority_weights: dict[ActionPriority, float] | None = None,
    ):
        self.auto_approve_threshold = auto_approve_threshold
        self.max_actions_per_run = max_actions_per_run
        self.min_confidence = min_confidence
        self.enable_learning = enable_learning
        self.require_approval = require_approval
        self.priority_weights = priority_weights or {
            ActionPriority.CRITICAL: 4.0,
            ActionPriority.HIGH: 3.0,
            ActionPriority.MEDIUM: 2.0,
            ActionPriority.LOW: 1.0,
        }


class DataCurationAgent:
    """AI-powered autonomous data curation agent."""

    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize the curation agent.

        Args:
            data: Input DataFrame to curate
            label_column: Name of the label column
            config: Agent configuration
        """
        self.original_data = data.copy()
        self.current_data = data.copy()
        self.label_column = label_column
        self.config = config or AgentConfig()

        self._executors: dict[ActionType, ActionExecutor] = {
            ActionType.REMOVE_DUPLICATE: DuplicateRemovalExecutor(),
            ActionType.REMOVE_OUTLIER: OutlierRemovalExecutor(),
            ActionType.RELABEL: RelabelExecutor(),
            ActionType.IMPUTE_MISSING: ImputationExecutor(),
        }

        self._current_plan: CurationPlan | None = None
        self._action_counter = 0
        self._feedback_history: list[FeedbackRecord] = []
        self._execution_history: list[ExecutionResult] = []
        self._data_snapshots: list[pd.DataFrame] = []

    def analyze_and_plan(self) -> CurationPlan:
        """Analyze data and create a curation plan.

        Returns:
            CurationPlan with proposed actions
        """
        logger.info("Starting data analysis for curation planning")

        actions: list[CurationAction] = []

        # Detect and plan duplicate removal
        dup_actions = self._plan_duplicate_removal()
        actions.extend(dup_actions)

        # Detect and plan outlier handling
        outlier_actions = self._plan_outlier_handling()
        actions.extend(outlier_actions)

        # Detect and plan missing value imputation
        impute_actions = self._plan_imputation()
        actions.extend(impute_actions)

        # Detect and plan label error fixes
        if self.label_column:
            label_actions = self._plan_label_fixes()
            actions.extend(label_actions)

        # Filter by minimum confidence
        actions = [a for a in actions if a.confidence >= self.config.min_confidence]

        # Sort by priority and confidence
        actions.sort(
            key=lambda a: (
                -self.config.priority_weights.get(a.priority, 1.0),
                -a.confidence,
            )
        )

        # Limit actions
        actions = actions[: self.config.max_actions_per_run]

        # Compute metrics
        quality_before = self._compute_quality_score()
        affected_rows = len(set(idx for a in actions for idx in a.target_indices))
        priority_breakdown = {}
        for priority in ActionPriority:
            priority_breakdown[priority] = sum(
                1 for a in actions if a.priority == priority
            )

        # Estimate quality after
        estimated_impact = sum(a.estimated_impact for a in actions)
        quality_after = min(100, quality_before + estimated_impact * 10)

        self._current_plan = CurationPlan(
            timestamp=datetime.now(),
            dataset_info={
                "n_samples": len(self.current_data),
                "n_features": len(self.current_data.columns),
                "label_column": self.label_column,
            },
            quality_score_before=quality_before,
            estimated_score_after=quality_after,
            actions=actions,
            total_affected_rows=affected_rows,
            priority_breakdown=priority_breakdown,
            estimated_time_minutes=len(actions) * 0.1,
        )

        logger.info("Created plan with %d actions", len(actions))
        return self._current_plan

    def _plan_duplicate_removal(self) -> list[CurationAction]:
        """Plan duplicate removal actions."""
        actions = []

        # Find exact duplicates
        duplicated_mask = self.current_data.duplicated(keep="first")
        duplicate_indices = self.current_data[duplicated_mask].index.tolist()

        if duplicate_indices:
            self._action_counter += 1
            actions.append(
                CurationAction(
                    action_id=self._action_counter,
                    action_type=ActionType.REMOVE_DUPLICATE,
                    priority=ActionPriority.HIGH,
                    target_indices=duplicate_indices,
                    description=f"Remove {len(duplicate_indices)} exact duplicate rows",
                    confidence=0.95,
                    estimated_impact=self._executors[ActionType.REMOVE_DUPLICATE].estimate_impact(
                        self.current_data,
                        CurationAction(
                            action_id=0,
                            action_type=ActionType.REMOVE_DUPLICATE,
                            priority=ActionPriority.HIGH,
                            target_indices=duplicate_indices,
                            description="",
                            confidence=0,
                            estimated_impact=0,
                        ),
                    ),
                    details={"method": "exact_match"},
                )
            )

        return actions

    def _plan_outlier_handling(self) -> list[CurationAction]:
        """Plan outlier handling actions."""
        actions = []

        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = self.current_data[col].dropna()
            if len(col_data) < 10:
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 3 * iqr  # Use 3x IQR for extreme outliers
            upper = q3 + 3 * iqr

            outlier_mask = (self.current_data[col] < lower) | (self.current_data[col] > upper)
            outlier_indices = self.current_data[outlier_mask].index.tolist()

            if outlier_indices and len(outlier_indices) < len(self.current_data) * 0.05:
                self._action_counter += 1
                actions.append(
                    CurationAction(
                        action_id=self._action_counter,
                        action_type=ActionType.REMOVE_OUTLIER,
                        priority=ActionPriority.MEDIUM,
                        target_indices=outlier_indices,
                        description=f"Remove {len(outlier_indices)} extreme outliers in '{col}'",
                        confidence=0.75,
                        estimated_impact=0.2,
                        details={
                            "column": col,
                            "method": "iqr",
                            "lower_bound": float(lower),
                            "upper_bound": float(upper),
                        },
                    )
                )

        return actions

    def _plan_imputation(self) -> list[CurationAction]:
        """Plan missing value imputation actions."""
        actions = []

        for col in self.current_data.columns:
            if col == self.label_column:
                continue

            missing_rate = self.current_data[col].isna().mean()

            if 0 < missing_rate < 0.3:  # Only impute if < 30% missing
                missing_indices = self.current_data[self.current_data[col].isna()].index.tolist()

                # Choose imputation method based on data type
                if pd.api.types.is_numeric_dtype(self.current_data[col]):
                    method = "median"
                else:
                    method = "mode"

                self._action_counter += 1
                actions.append(
                    CurationAction(
                        action_id=self._action_counter,
                        action_type=ActionType.IMPUTE_MISSING,
                        priority=ActionPriority.LOW if missing_rate < 0.1 else ActionPriority.MEDIUM,
                        target_indices=missing_indices,
                        description=f"Impute {len(missing_indices)} missing values in '{col}' using {method}",
                        confidence=0.7,
                        estimated_impact=missing_rate * 2,
                        details={
                            "column": col,
                            "method": method,
                            "missing_rate": float(missing_rate),
                        },
                    )
                )

        return actions

    def _plan_label_fixes(self) -> list[CurationAction]:
        """Plan label correction actions."""
        actions = []

        if self.label_column not in self.current_data.columns:
            return actions

        # Check for missing labels
        missing_labels = self.current_data[self.label_column].isna()
        missing_indices = self.current_data[missing_labels].index.tolist()

        if missing_indices:
            self._action_counter += 1
            actions.append(
                CurationAction(
                    action_id=self._action_counter,
                    action_type=ActionType.FLAG_FOR_REVIEW,
                    priority=ActionPriority.HIGH,
                    target_indices=missing_indices,
                    description=f"Flag {len(missing_indices)} samples with missing labels for review",
                    confidence=1.0,
                    estimated_impact=0.3,
                    details={"reason": "missing_label"},
                )
            )

        return actions

    def _compute_quality_score(self) -> float:
        """Compute current data quality score."""
        score = 100.0

        # Penalize duplicates
        dup_rate = self.current_data.duplicated().mean()
        score -= dup_rate * 30

        # Penalize missing values
        missing_rate = self.current_data.isna().mean().mean()
        score -= missing_rate * 20

        # Penalize class imbalance
        if self.label_column and self.label_column in self.current_data.columns:
            label_counts = self.current_data[self.label_column].value_counts()
            if len(label_counts) > 1:
                imbalance = label_counts.max() / max(label_counts.min(), 1)
                if imbalance > 10:
                    score -= 15
                elif imbalance > 5:
                    score -= 10

        return max(0, min(100, score))

    def approve_actions(self, action_ids: list[int]) -> int:
        """Approve specific actions for execution.

        Args:
            action_ids: List of action IDs to approve

        Returns:
            Number of actions approved
        """
        if self._current_plan is None:
            raise ValueError("No plan available. Call analyze_and_plan() first.")

        approved_count = 0
        for action in self._current_plan.actions:
            if action.action_id in action_ids and action.status == ActionStatus.PROPOSED:
                action.status = ActionStatus.APPROVED
                approved_count += 1

        logger.info("Approved %d actions", approved_count)
        return approved_count

    def reject_actions(self, action_ids: list[int], reason: str = "") -> int:
        """Reject specific actions.

        Args:
            action_ids: List of action IDs to reject
            reason: Rejection reason

        Returns:
            Number of actions rejected
        """
        if self._current_plan is None:
            raise ValueError("No plan available.")

        rejected_count = 0
        for action in self._current_plan.actions:
            if action.action_id in action_ids and action.status == ActionStatus.PROPOSED:
                action.status = ActionStatus.REJECTED
                rejected_count += 1

                # Record feedback for learning
                self._feedback_history.append(
                    FeedbackRecord(
                        action_id=action.action_id,
                        action_type=action.action_type,
                        was_approved=False,
                        was_modified=False,
                        human_correction=reason,
                    )
                )

        logger.info("Rejected %d actions", rejected_count)
        return rejected_count

    def approve_all_high_confidence(self) -> int:
        """Auto-approve all actions above the confidence threshold.

        Returns:
            Number of actions approved
        """
        if self._current_plan is None:
            raise ValueError("No plan available.")

        high_confidence_ids = [
            a.action_id
            for a in self._current_plan.actions
            if a.confidence >= self.config.auto_approve_threshold
            and a.status == ActionStatus.PROPOSED
        ]

        return self.approve_actions(high_confidence_ids)

    def execute_approved(self) -> ExecutionResult:
        """Execute all approved actions.

        Returns:
            ExecutionResult with execution details
        """
        if self._current_plan is None:
            raise ValueError("No plan available.")

        # Save snapshot for rollback
        self._data_snapshots.append(self.current_data.copy())

        approved_actions = [
            a for a in self._current_plan.actions if a.status == ActionStatus.APPROVED
        ]

        if not approved_actions:
            return ExecutionResult(
                timestamp=datetime.now(),
                actions_executed=0,
                actions_failed=0,
                rows_modified=0,
                rows_removed=0,
                quality_score_after=self._compute_quality_score(),
                errors=[],
                rollback_available=True,
            )

        logger.info("Executing %d approved actions", len(approved_actions))

        executed = 0
        failed = 0
        rows_modified = 0
        rows_removed = 0
        errors = []
        original_len = len(self.current_data)

        for action in approved_actions:
            try:
                executor = self._executors.get(action.action_type)
                if executor is None:
                    action.status = ActionStatus.FAILED
                    errors.append(f"No executor for action type: {action.action_type.value}")
                    failed += 1
                    continue

                self.current_data, exec_info = executor.execute(self.current_data, action)
                action.status = ActionStatus.EXECUTED
                action.execution_result = exec_info
                executed += 1

                rows_modified += exec_info.get("rows_relabeled", 0)
                rows_modified += exec_info.get("values_imputed", 0)

                # Record successful execution as positive feedback
                self._feedback_history.append(
                    FeedbackRecord(
                        action_id=action.action_id,
                        action_type=action.action_type,
                        was_approved=True,
                        was_modified=False,
                        human_correction=None,
                    )
                )

            except Exception as e:
                action.status = ActionStatus.FAILED
                errors.append(f"Action {action.action_id} failed: {e!s}")
                failed += 1
                logger.warning("Action %d failed: %s", action.action_id, e)

        rows_removed = original_len - len(self.current_data)

        result = ExecutionResult(
            timestamp=datetime.now(),
            actions_executed=executed,
            actions_failed=failed,
            rows_modified=rows_modified,
            rows_removed=rows_removed,
            quality_score_after=self._compute_quality_score(),
            errors=errors,
            rollback_available=True,
        )

        self._execution_history.append(result)
        logger.info("Execution complete: %d executed, %d failed", executed, failed)

        return result

    def rollback(self) -> bool:
        """Rollback to the previous data state.

        Returns:
            True if rollback successful
        """
        if not self._data_snapshots:
            logger.warning("No snapshots available for rollback")
            return False

        self.current_data = self._data_snapshots.pop()
        logger.info("Rolled back to previous state")
        return True

    def reset(self) -> None:
        """Reset to original data."""
        self.current_data = self.original_data.copy()
        self._current_plan = None
        self._data_snapshots.clear()
        logger.info("Reset to original data")

    def get_curated_data(self) -> pd.DataFrame:
        """Get the current curated data.

        Returns:
            Current DataFrame after curation
        """
        return self.current_data.copy()

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get summary of human feedback for learning.

        Returns:
            Dictionary with feedback statistics
        """
        if not self._feedback_history:
            return {"total_feedback": 0}

        total = len(self._feedback_history)
        approved = sum(1 for f in self._feedback_history if f.was_approved)
        rejected = total - approved

        by_action_type: dict[str, dict[str, int]] = {}
        for feedback in self._feedback_history:
            action_type = feedback.action_type.value
            if action_type not in by_action_type:
                by_action_type[action_type] = {"approved": 0, "rejected": 0}
            if feedback.was_approved:
                by_action_type[action_type]["approved"] += 1
            else:
                by_action_type[action_type]["rejected"] += 1

        return {
            "total_feedback": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / max(total, 1),
            "by_action_type": by_action_type,
        }


class AutoCurator:
    """Fully automated data curator that runs without human intervention."""

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        max_iterations: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations

    def curate(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
    ) -> tuple[pd.DataFrame, list[ExecutionResult]]:
        """Automatically curate data without human intervention.

        Args:
            data: Input DataFrame
            label_column: Name of label column

        Returns:
            Tuple of (curated_data, execution_history)
        """
        config = AgentConfig(
            auto_approve_threshold=self.confidence_threshold,
            require_approval=False,
        )

        agent = DataCurationAgent(data, label_column, config)
        results = []

        for iteration in range(self.max_iterations):
            logger.info("Auto-curation iteration %d", iteration + 1)

            plan = agent.analyze_and_plan()

            # Auto-approve high confidence actions
            approved = agent.approve_all_high_confidence()

            if approved == 0:
                logger.info("No high-confidence actions found. Stopping.")
                break

            result = agent.execute_approved()
            results.append(result)

            if result.actions_executed == 0:
                break

        return agent.get_curated_data(), results


def create_curation_agent(
    data: pd.DataFrame,
    label_column: str | None = None,
    auto_approve_threshold: float = 0.95,
    require_approval: bool = True,
) -> DataCurationAgent:
    """Convenience function to create a curation agent.

    Args:
        data: Input DataFrame
        label_column: Name of label column
        auto_approve_threshold: Confidence threshold for auto-approval
        require_approval: Whether to require human approval

    Returns:
        Configured DataCurationAgent
    """
    config = AgentConfig(
        auto_approve_threshold=auto_approve_threshold,
        require_approval=require_approval,
    )
    return DataCurationAgent(data, label_column, config)


def auto_curate(
    data: pd.DataFrame,
    label_column: str | None = None,
    confidence_threshold: float = 0.9,
) -> pd.DataFrame:
    """Automatically curate data without human intervention.

    Args:
        data: Input DataFrame
        label_column: Name of label column
        confidence_threshold: Minimum confidence for auto-execution

    Returns:
        Curated DataFrame
    """
    curator = AutoCurator(confidence_threshold=confidence_threshold)
    curated_data, _ = curator.curate(data, label_column)
    return curated_data
