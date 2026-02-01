"""Collaborative Review Workspace for multi-user annotation review.

This module provides real-time multi-user annotation review capabilities
with conflict resolution and consensus building features.

Example:
    >>> from clean.collaboration import ReviewWorkspace, ReviewSession
    >>>
    >>> workspace = ReviewWorkspace(workspace_id="my-workspace")
    >>> session = workspace.create_session(
    ...     data=df,
    ...     reviewers=["alice@example.com", "bob@example.com"],
    ... )
    >>> await session.start()
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    pass


class ReviewStatus(Enum):
    """Status of a review item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEWED = "reviewed"
    CONFLICT = "conflict"
    RESOLVED = "resolved"
    SKIPPED = "skipped"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving annotation conflicts."""

    MAJORITY_VOTE = "majority_vote"
    EXPERT_OVERRIDE = "expert_override"
    CONSENSUS_REQUIRED = "consensus_required"
    WEIGHTED_VOTE = "weighted_vote"
    DISCUSSION = "discussion"


class ReviewerRole(Enum):
    """Roles in the review workflow."""

    REVIEWER = "reviewer"
    SENIOR_REVIEWER = "senior_reviewer"
    EXPERT = "expert"
    ADMIN = "admin"


@dataclass
class Reviewer:
    """A participant in the review process."""

    user_id: str
    email: str
    name: str
    role: ReviewerRole = ReviewerRole.REVIEWER
    weight: float = 1.0  # For weighted voting
    expertise_areas: list[str] = field(default_factory=list)
    is_active: bool = True
    last_activity: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "weight": self.weight,
            "is_active": self.is_active,
        }


@dataclass
class Annotation:
    """An annotation made by a reviewer."""

    annotation_id: str
    sample_index: int
    reviewer_id: str
    label: Any
    confidence: float  # 0-1
    timestamp: datetime
    comment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "sample_index": self.sample_index,
            "reviewer_id": self.reviewer_id,
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "comment": self.comment,
        }


@dataclass
class Conflict:
    """A conflict between annotations."""

    conflict_id: str
    sample_index: int
    annotations: list[Annotation]
    status: str = "open"
    resolution: Any | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    discussion: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "sample_index": self.sample_index,
            "n_annotations": len(self.annotations),
            "labels": [a.label for a in self.annotations],
            "status": self.status,
            "resolution": self.resolution,
        }


@dataclass
class ReviewItem:
    """An item to be reviewed."""

    index: int
    data: dict[str, Any]
    original_label: Any | None = None
    suggested_label: Any | None = None
    status: ReviewStatus = ReviewStatus.PENDING
    annotations: list[Annotation] = field(default_factory=list)
    assigned_to: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = more urgent
    conflict: Conflict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "data": self.data,
            "original_label": self.original_label,
            "suggested_label": self.suggested_label,
            "status": self.status.value,
            "n_annotations": len(self.annotations),
            "priority": self.priority,
            "has_conflict": self.conflict is not None,
        }


@dataclass
class SessionStats:
    """Statistics for a review session."""

    total_items: int
    pending: int
    in_progress: int
    reviewed: int
    conflicts: int
    resolved: int
    skipped: int
    agreement_rate: float
    avg_time_per_item: float
    active_reviewers: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "pending": self.pending,
            "reviewed": self.reviewed,
            "conflicts": self.conflicts,
            "agreement_rate": self.agreement_rate,
            "active_reviewers": self.active_reviewers,
        }


class ReviewSession:
    """A collaborative review session.

    Manages the review workflow for a set of samples with multiple
    reviewers, conflict detection, and resolution.
    """

    def __init__(
        self,
        session_id: str,
        workspace_id: str,
        items: list[ReviewItem],
        reviewers: list[Reviewer],
        labels: list[Any],
        min_reviews_per_item: int = 2,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MAJORITY_VOTE,
    ):
        """Initialize the review session.

        Args:
            session_id: Unique session identifier
            workspace_id: Parent workspace ID
            items: Items to review
            reviewers: List of reviewers
            labels: Valid label options
            min_reviews_per_item: Minimum reviews needed per item
            conflict_strategy: Strategy for resolving conflicts
        """
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.items = {item.index: item for item in items}
        self.reviewers = {r.user_id: r for r in reviewers}
        self.labels = labels
        self.min_reviews_per_item = min_reviews_per_item
        self.conflict_strategy = conflict_strategy

        self._created_at = datetime.utcnow()
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._event_handlers: list[Callable] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the review session."""
        self._started_at = datetime.utcnow()
        await self._emit_event("session_started", {
            "session_id": self.session_id,
            "n_items": len(self.items),
            "n_reviewers": len(self.reviewers),
        })

    async def submit_annotation(
        self,
        reviewer_id: str,
        sample_index: int,
        label: Any,
        confidence: float = 1.0,
        comment: str | None = None,
    ) -> Annotation:
        """Submit an annotation for a sample.

        Args:
            reviewer_id: ID of the reviewer
            sample_index: Index of the sample
            label: Assigned label
            confidence: Confidence score (0-1)
            comment: Optional comment

        Returns:
            Created Annotation
        """
        if reviewer_id not in self.reviewers:
            raise ValueError(f"Unknown reviewer: {reviewer_id}")

        if sample_index not in self.items:
            raise ValueError(f"Unknown sample index: {sample_index}")

        async with self._lock:
            item = self.items[sample_index]

            # Check if reviewer already annotated this item
            existing = [a for a in item.annotations if a.reviewer_id == reviewer_id]
            if existing:
                # Update existing annotation
                existing[0].label = label
                existing[0].confidence = confidence
                existing[0].comment = comment
                existing[0].timestamp = datetime.utcnow()
                annotation = existing[0]
            else:
                # Create new annotation
                annotation = Annotation(
                    annotation_id=str(uuid.uuid4()),
                    sample_index=sample_index,
                    reviewer_id=reviewer_id,
                    label=label,
                    confidence=confidence,
                    timestamp=datetime.utcnow(),
                    comment=comment,
                )
                item.annotations.append(annotation)

            # Update item status
            if len(item.annotations) >= self.min_reviews_per_item:
                item.status = ReviewStatus.REVIEWED

                # Check for conflicts
                unique_labels = set(a.label for a in item.annotations)
                if len(unique_labels) > 1:
                    item.status = ReviewStatus.CONFLICT
                    item.conflict = Conflict(
                        conflict_id=str(uuid.uuid4()),
                        sample_index=sample_index,
                        annotations=item.annotations.copy(),
                    )
            else:
                item.status = ReviewStatus.IN_PROGRESS

            # Update reviewer activity
            self.reviewers[reviewer_id].last_activity = datetime.utcnow()

        await self._emit_event("annotation_submitted", {
            "annotation": annotation.to_dict(),
            "item_status": item.status.value,
        })

        return annotation

    async def resolve_conflict(
        self,
        sample_index: int,
        resolution: Any,
        resolver_id: str,
        reason: str | None = None,
    ) -> None:
        """Resolve a conflict on a sample.

        Args:
            sample_index: Index of the conflicted sample
            resolution: Resolved label
            resolver_id: ID of the user resolving
            reason: Optional reason for resolution
        """
        if sample_index not in self.items:
            raise ValueError(f"Unknown sample index: {sample_index}")

        item = self.items[sample_index]
        if item.conflict is None:
            raise ValueError(f"No conflict on sample {sample_index}")

        async with self._lock:
            item.conflict.resolution = resolution
            item.conflict.resolved_by = resolver_id
            item.conflict.resolved_at = datetime.utcnow()
            item.conflict.status = "resolved"

            if reason:
                item.conflict.discussion.append({
                    "user_id": resolver_id,
                    "message": f"Resolution: {reason}",
                    "timestamp": datetime.utcnow().isoformat(),
                })

            item.status = ReviewStatus.RESOLVED

        await self._emit_event("conflict_resolved", {
            "sample_index": sample_index,
            "resolution": resolution,
            "resolver_id": resolver_id,
        })

    async def add_discussion_message(
        self,
        sample_index: int,
        user_id: str,
        message: str,
    ) -> None:
        """Add a discussion message to a conflicted item.

        Args:
            sample_index: Index of the sample
            user_id: ID of the user
            message: Discussion message
        """
        if sample_index not in self.items:
            raise ValueError(f"Unknown sample index: {sample_index}")

        item = self.items[sample_index]
        if item.conflict is None:
            # Create conflict for discussion
            item.conflict = Conflict(
                conflict_id=str(uuid.uuid4()),
                sample_index=sample_index,
                annotations=item.annotations.copy(),
            )

        item.conflict.discussion.append({
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        })

        await self._emit_event("discussion_message", {
            "sample_index": sample_index,
            "user_id": user_id,
            "message": message,
        })

    async def auto_resolve_conflicts(self) -> int:
        """Automatically resolve conflicts using the configured strategy.

        Returns:
            Number of conflicts resolved
        """
        resolved = 0

        for item in self.items.values():
            if item.status != ReviewStatus.CONFLICT or item.conflict is None:
                continue

            resolution = await self._apply_resolution_strategy(item)

            if resolution is not None:
                await self.resolve_conflict(
                    sample_index=item.index,
                    resolution=resolution,
                    resolver_id="auto_resolver",
                    reason=f"Auto-resolved using {self.conflict_strategy.value}",
                )
                resolved += 1

        return resolved

    async def _apply_resolution_strategy(
        self,
        item: ReviewItem,
    ) -> Any | None:
        """Apply conflict resolution strategy."""
        annotations = item.annotations

        if self.conflict_strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            # Simple majority vote
            label_counts: dict[Any, int] = {}
            for ann in annotations:
                label_counts[ann.label] = label_counts.get(ann.label, 0) + 1

            max_count = max(label_counts.values())
            winners = [l for l, c in label_counts.items() if c == max_count]

            if len(winners) == 1:
                return winners[0]
            return None  # Tie, can't auto-resolve

        elif self.conflict_strategy == ConflictResolutionStrategy.WEIGHTED_VOTE:
            # Weighted by reviewer weight and confidence
            label_weights: dict[Any, float] = {}
            for ann in annotations:
                reviewer = self.reviewers.get(ann.reviewer_id)
                weight = (reviewer.weight if reviewer else 1.0) * ann.confidence
                label_weights[ann.label] = label_weights.get(ann.label, 0) + weight

            max_weight = max(label_weights.values())
            return max(label_weights.items(), key=lambda x: x[1])[0]

        elif self.conflict_strategy == ConflictResolutionStrategy.EXPERT_OVERRIDE:
            # Expert annotations take precedence
            expert_anns = [
                a for a in annotations
                if self.reviewers.get(a.reviewer_id, Reviewer("", "", "", ReviewerRole.REVIEWER)).role
                in (ReviewerRole.EXPERT, ReviewerRole.ADMIN)
            ]

            if expert_anns:
                return expert_anns[0].label

            return None

        elif self.conflict_strategy == ConflictResolutionStrategy.CONSENSUS_REQUIRED:
            # All must agree
            labels = set(a.label for a in annotations)
            if len(labels) == 1:
                return list(labels)[0]
            return None

        return None

    def get_next_item(
        self,
        reviewer_id: str,
    ) -> ReviewItem | None:
        """Get the next item for a reviewer to review.

        Args:
            reviewer_id: ID of the reviewer

        Returns:
            Next ReviewItem or None if all done
        """
        # Priority: conflicts > pending > assigned
        candidates = []

        for item in self.items.values():
            # Skip already reviewed by this reviewer
            if any(a.reviewer_id == reviewer_id for a in item.annotations):
                continue

            # Skip completed items
            if item.status in (ReviewStatus.RESOLVED, ReviewStatus.SKIPPED):
                continue

            # Prioritize conflicts
            if item.status == ReviewStatus.CONFLICT:
                candidates.append((0, item.priority, item))
            # Then pending
            elif item.status == ReviewStatus.PENDING:
                candidates.append((1, item.priority, item))
            # Then in-progress
            elif item.status == ReviewStatus.IN_PROGRESS:
                candidates.append((2, item.priority, item))

        if not candidates:
            return None

        # Sort by priority (lower is better), then by item priority (higher is better)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0][2]

    def get_stats(self) -> SessionStats:
        """Get session statistics."""
        status_counts = {s: 0 for s in ReviewStatus}
        total_time = 0
        n_reviewed = 0
        agreements = 0
        total_comparisons = 0

        for item in self.items.values():
            status_counts[item.status] += 1

            if item.annotations:
                # Calculate agreement
                labels = [a.label for a in item.annotations]
                if len(labels) >= 2:
                    # Compare all pairs
                    for i in range(len(labels)):
                        for j in range(i + 1, len(labels)):
                            total_comparisons += 1
                            if labels[i] == labels[j]:
                                agreements += 1

        agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 1.0

        active_reviewers = sum(
            1 for r in self.reviewers.values()
            if r.is_active and r.last_activity
        )

        return SessionStats(
            total_items=len(self.items),
            pending=status_counts[ReviewStatus.PENDING],
            in_progress=status_counts[ReviewStatus.IN_PROGRESS],
            reviewed=status_counts[ReviewStatus.REVIEWED],
            conflicts=status_counts[ReviewStatus.CONFLICT],
            resolved=status_counts[ReviewStatus.RESOLVED],
            skipped=status_counts[ReviewStatus.SKIPPED],
            agreement_rate=agreement_rate,
            avg_time_per_item=total_time / max(n_reviewed, 1),
            active_reviewers=active_reviewers,
        )

    def get_final_labels(self) -> dict[int, Any]:
        """Get final labels after review.

        Returns:
            Dictionary mapping sample index to final label
        """
        final_labels = {}

        for item in self.items.values():
            if item.status == ReviewStatus.RESOLVED and item.conflict:
                final_labels[item.index] = item.conflict.resolution
            elif item.annotations:
                # Take most common or most confident
                label_scores: dict[Any, float] = {}
                for ann in item.annotations:
                    label_scores[ann.label] = label_scores.get(ann.label, 0) + ann.confidence
                final_labels[item.index] = max(label_scores.items(), key=lambda x: x[1])[0]
            elif item.original_label is not None:
                final_labels[item.index] = item.original_label

        return final_labels

    def add_event_handler(self, handler: Callable) -> None:
        """Add event handler for session events."""
        self._event_handlers.append(handler)

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to all handlers."""
        event = {
            "type": event_type,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                pass  # Don't let handler errors break the session


class ReviewWorkspace:
    """Workspace for managing multiple review sessions.

    Provides team management, session creation, and coordination
    across multiple review efforts.
    """

    def __init__(
        self,
        workspace_id: str,
        name: str = "Review Workspace",
    ):
        """Initialize the workspace.

        Args:
            workspace_id: Unique workspace identifier
            name: Workspace name
        """
        self.workspace_id = workspace_id
        self.name = name

        self._sessions: dict[str, ReviewSession] = {}
        self._reviewers: dict[str, Reviewer] = {}
        self._created_at = datetime.utcnow()

    def add_reviewer(
        self,
        email: str,
        name: str,
        role: ReviewerRole = ReviewerRole.REVIEWER,
        **kwargs: Any,
    ) -> Reviewer:
        """Add a reviewer to the workspace.

        Args:
            email: Reviewer email
            name: Reviewer name
            role: Reviewer role
            **kwargs: Additional reviewer attributes

        Returns:
            Created Reviewer
        """
        user_id = str(uuid.uuid4())
        reviewer = Reviewer(
            user_id=user_id,
            email=email,
            name=name,
            role=role,
            **kwargs,
        )
        self._reviewers[user_id] = reviewer
        return reviewer

    def create_session(
        self,
        data: pd.DataFrame,
        reviewer_ids: list[str] | None = None,
        label_column: str | None = None,
        suggested_label_column: str | None = None,
        labels: list[Any] | None = None,
        **kwargs: Any,
    ) -> ReviewSession:
        """Create a new review session.

        Args:
            data: DataFrame with items to review
            reviewer_ids: IDs of reviewers for this session
            label_column: Column with original labels
            suggested_label_column: Column with suggested labels
            labels: Valid label options
            **kwargs: Additional session configuration

        Returns:
            Created ReviewSession
        """
        session_id = str(uuid.uuid4())

        # Create review items
        items = []
        for idx, row in data.iterrows():
            item = ReviewItem(
                index=int(idx),
                data=row.to_dict(),
                original_label=row.get(label_column) if label_column else None,
                suggested_label=row.get(suggested_label_column) if suggested_label_column else None,
            )
            items.append(item)

        # Get reviewers
        if reviewer_ids:
            reviewers = [
                self._reviewers[rid] for rid in reviewer_ids
                if rid in self._reviewers
            ]
        else:
            reviewers = list(self._reviewers.values())

        # Infer labels if not provided
        if labels is None and label_column and label_column in data.columns:
            labels = data[label_column].dropna().unique().tolist()

        session = ReviewSession(
            session_id=session_id,
            workspace_id=self.workspace_id,
            items=items,
            reviewers=reviewers,
            labels=labels or [],
            **kwargs,
        )

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ReviewSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions in the workspace."""
        return [
            {
                "session_id": s.session_id,
                "n_items": len(s.items),
                "n_reviewers": len(s.reviewers),
                "stats": s.get_stats().to_dict(),
            }
            for s in self._sessions.values()
        ]

    def get_reviewer_performance(
        self,
        reviewer_id: str,
    ) -> dict[str, Any]:
        """Get performance metrics for a reviewer.

        Args:
            reviewer_id: ID of the reviewer

        Returns:
            Performance metrics
        """
        total_annotations = 0
        total_agreements = 0
        total_comparisons = 0
        sessions_participated = 0

        for session in self._sessions.values():
            reviewer_anns = []

            for item in session.items.values():
                rev_ann = [a for a in item.annotations if a.reviewer_id == reviewer_id]
                if rev_ann:
                    total_annotations += 1
                    reviewer_anns.append(rev_ann[0])

                    # Calculate agreement with others
                    other_anns = [a for a in item.annotations if a.reviewer_id != reviewer_id]
                    for other in other_anns:
                        total_comparisons += 1
                        if rev_ann[0].label == other.label:
                            total_agreements += 1

            if reviewer_anns:
                sessions_participated += 1

        agreement_rate = total_agreements / total_comparisons if total_comparisons > 0 else 0

        return {
            "reviewer_id": reviewer_id,
            "total_annotations": total_annotations,
            "sessions_participated": sessions_participated,
            "agreement_rate": agreement_rate,
            "avg_confidence": 0.0,  # Would need to track
        }


def create_review_session(
    data: pd.DataFrame,
    reviewers: list[dict[str, Any]],
    **kwargs: Any,
) -> ReviewSession:
    """Convenience function to create a review session.

    Args:
        data: DataFrame with items to review
        reviewers: List of reviewer dictionaries with email and name
        **kwargs: Additional session configuration

    Returns:
        Created ReviewSession
    """
    workspace = ReviewWorkspace(workspace_id=str(uuid.uuid4()))

    for reviewer_info in reviewers:
        workspace.add_reviewer(
            email=reviewer_info.get("email", ""),
            name=reviewer_info.get("name", ""),
            role=ReviewerRole(reviewer_info.get("role", "reviewer")),
        )

    reviewer_ids = list(workspace._reviewers.keys())
    return workspace.create_session(data, reviewer_ids=reviewer_ids, **kwargs)
