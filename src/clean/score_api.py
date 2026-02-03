"""Data Quality Score API - Public API for Dataset Scoring.

This module provides a public API endpoint for scoring datasets,
with free tier support and rate limiting.

Example:
    >>> from clean.score_api import QualityScoreAPI, create_api_app
    >>>
    >>> # Create and run API
    >>> app = create_api_app()
    >>> # Run with: uvicorn clean.score_api:app --port 8080
    >>>
    >>> # Or use programmatically
    >>> api = QualityScoreAPI()
    >>> score = api.score_dataset(df, label_column="label")
    >>> print(f"Quality Score: {score.overall_score}/100")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.core.cleaner import DatasetCleaner
from clean.exceptions import CleanError, ConfigurationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TierLevel(Enum):
    """API tier levels."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Rate limits and constraints for each tier."""

    max_samples_per_request: int
    max_requests_per_day: int
    max_requests_per_minute: int
    features_enabled: list[str]
    priority: int  # Higher = faster processing

    @classmethod
    def for_tier(cls, tier: TierLevel) -> "TierLimits":
        """Get limits for a tier."""
        limits = {
            TierLevel.FREE: cls(
                max_samples_per_request=1000,
                max_requests_per_day=10,
                max_requests_per_minute=1,
                features_enabled=["basic_score", "label_errors", "duplicates"],
                priority=1,
            ),
            TierLevel.BASIC: cls(
                max_samples_per_request=10000,
                max_requests_per_day=100,
                max_requests_per_minute=10,
                features_enabled=["basic_score", "label_errors", "duplicates", "outliers"],
                priority=2,
            ),
            TierLevel.PRO: cls(
                max_samples_per_request=100000,
                max_requests_per_day=1000,
                max_requests_per_minute=60,
                features_enabled=["basic_score", "label_errors", "duplicates", "outliers", "bias", "drift"],
                priority=3,
            ),
            TierLevel.ENTERPRISE: cls(
                max_samples_per_request=1000000,
                max_requests_per_day=10000,
                max_requests_per_minute=300,
                features_enabled=["all"],
                priority=4,
            ),
        }
        return limits[tier]


@dataclass
class RateLimitStatus:
    """Current rate limit status."""

    requests_remaining_day: int
    requests_remaining_minute: int
    reset_time_day: datetime
    reset_time_minute: datetime
    is_limited: bool


@dataclass
class QuickScore:
    """Quick quality score result."""

    overall_score: float  # 0-100
    grade: str  # A, B, C, D, F
    n_samples: int
    n_issues: int
    issue_breakdown: dict[str, int]
    processing_time: float
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "n_samples": self.n_samples,
            "n_issues": self.n_issues,
            "issue_breakdown": self.issue_breakdown,
            "processing_time": self.processing_time,
            "recommendations": self.recommendations,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def score_to_grade(score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self):
        """Initialize rate limiter."""
        self._requests_day: dict[str, list[datetime]] = defaultdict(list)
        self._requests_minute: dict[str, list[datetime]] = defaultdict(list)

    def check_limit(self, api_key: str, tier: TierLevel) -> RateLimitStatus:
        """Check if request is within rate limits.

        Args:
            api_key: API key or identifier
            tier: User's tier level

        Returns:
            RateLimitStatus with current limits
        """
        limits = TierLimits.for_tier(tier)
        now = datetime.now()

        # Clean old requests
        day_cutoff = now - timedelta(days=1)
        minute_cutoff = now - timedelta(minutes=1)

        self._requests_day[api_key] = [
            t for t in self._requests_day[api_key] if t > day_cutoff
        ]
        self._requests_minute[api_key] = [
            t for t in self._requests_minute[api_key] if t > minute_cutoff
        ]

        # Count remaining
        day_used = len(self._requests_day[api_key])
        minute_used = len(self._requests_minute[api_key])

        day_remaining = limits.max_requests_per_day - day_used
        minute_remaining = limits.max_requests_per_minute - minute_used

        is_limited = day_remaining <= 0 or minute_remaining <= 0

        return RateLimitStatus(
            requests_remaining_day=max(0, day_remaining),
            requests_remaining_minute=max(0, minute_remaining),
            reset_time_day=now + timedelta(days=1),
            reset_time_minute=now + timedelta(minutes=1),
            is_limited=is_limited,
        )

    def record_request(self, api_key: str) -> None:
        """Record a request."""
        now = datetime.now()
        self._requests_day[api_key].append(now)
        self._requests_minute[api_key].append(now)


class QualityScoreAPI:
    """API for dataset quality scoring.

    Provides quick quality scores for datasets with configurable
    depth of analysis based on tier level.

    Example:
        >>> api = QualityScoreAPI()
        >>> score = api.score_dataset(df, label_column="label")
        >>> print(f"Score: {score.overall_score}/100 ({score.grade})")
    """

    def __init__(
        self,
        default_tier: TierLevel = TierLevel.FREE,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        """Initialize the API.

        Args:
            default_tier: Default tier for unauthenticated requests
            enable_caching: Enable result caching
            cache_ttl_seconds: Cache time-to-live
        """
        self.default_tier = default_tier
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl_seconds

        self._rate_limiter = RateLimiter()
        self._cache: dict[str, tuple[QuickScore, datetime]] = {}
        self._api_keys: dict[str, TierLevel] = {}

    def register_api_key(self, api_key: str, tier: TierLevel) -> None:
        """Register an API key with a tier level.

        Args:
            api_key: API key to register
            tier: Tier level for the key
        """
        self._api_keys[api_key] = tier

    def score_dataset(
        self,
        data: pd.DataFrame,
        label_column: str | None = None,
        api_key: str | None = None,
        features: list[str] | None = None,
    ) -> QuickScore:
        """Score a dataset for quality.

        Args:
            data: DataFrame to score
            label_column: Column containing labels (optional)
            api_key: API key for authentication
            features: Specific features to analyze

        Returns:
            QuickScore with results
        """
        start_time = time.time()

        # Determine tier
        tier = self._api_keys.get(api_key, self.default_tier) if api_key else self.default_tier
        limits = TierLimits.for_tier(tier)

        # Check rate limits
        key = api_key or "anonymous"
        status = self._rate_limiter.check_limit(key, tier)
        if status.is_limited:
            raise CleanError(
                f"Rate limit exceeded. Resets at {status.reset_time_minute.isoformat()}"
            )

        # Check sample limit
        if len(data) > limits.max_samples_per_request:
            raise CleanError(
                f"Dataset too large ({len(data)} samples). "
                f"Max for {tier.value} tier: {limits.max_samples_per_request}"
            )

        # Check cache
        cache_key = self._compute_cache_key(data, label_column)
        if self.enable_caching and cache_key in self._cache:
            cached_score, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                self._rate_limiter.record_request(key)
                return cached_score

        # Determine which features to analyze
        if features is None:
            features = limits.features_enabled

        # Run analysis
        score = self._analyze_dataset(data, label_column, features, limits)

        # Update timing
        score.processing_time = time.time() - start_time

        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = (score, datetime.now())

        # Record request
        self._rate_limiter.record_request(key)

        return score

    def get_rate_limit_status(self, api_key: str | None = None) -> RateLimitStatus:
        """Get current rate limit status.

        Args:
            api_key: API key to check

        Returns:
            RateLimitStatus
        """
        key = api_key or "anonymous"
        tier = self._api_keys.get(key, self.default_tier)
        return self._rate_limiter.check_limit(key, tier)

    def _compute_cache_key(
        self,
        data: pd.DataFrame,
        label_column: str | None,
    ) -> str:
        """Compute cache key for dataset."""
        # Hash based on shape, columns, and sample of data
        shape_str = f"{data.shape}"
        cols_str = ",".join(sorted(data.columns))
        sample_str = data.head(10).to_json()
        label_str = label_column or "none"

        content = f"{shape_str}|{cols_str}|{sample_str}|{label_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def _analyze_dataset(
        self,
        data: pd.DataFrame,
        label_column: str | None,
        features: list[str],
        limits: TierLimits,
    ) -> QuickScore:
        """Run quality analysis on dataset."""
        # Use DatasetCleaner for analysis
        cleaner = DatasetCleaner(
            data=data,
            label_column=label_column,
        )

        report = cleaner.analyze()

        # Collect issues
        issue_breakdown: dict[str, int] = {}
        n_issues = 0

        if "label_errors" in features or "all" in features:
            if hasattr(report, "label_errors"):
                err_df = report.label_errors()
                issue_breakdown["label_errors"] = len(err_df)
                n_issues += len(err_df)

        if "duplicates" in features or "all" in features:
            if hasattr(report, "duplicates"):
                dup_df = report.duplicates()
                issue_breakdown["duplicates"] = len(dup_df)
                n_issues += len(dup_df)

        if "outliers" in features or "all" in features:
            if hasattr(report, "outliers"):
                out_df = report.outliers()
                issue_breakdown["outliers"] = len(out_df)
                n_issues += len(out_df)

        # Calculate overall score (0-100)
        overall_score = report.score.overall * 100

        # Generate recommendations
        recommendations = self._generate_recommendations(
            issue_breakdown, len(data), overall_score
        )

        return QuickScore(
            overall_score=overall_score,
            grade=QuickScore.score_to_grade(overall_score),
            n_samples=len(data),
            n_issues=n_issues,
            issue_breakdown=issue_breakdown,
            processing_time=0.0,  # Will be updated
            recommendations=recommendations,
            metadata={
                "tier": limits.priority,
                "features_analyzed": features,
            },
        )

    def _generate_recommendations(
        self,
        issues: dict[str, int],
        n_samples: int,
        score: float,
    ) -> list[str]:
        """Generate recommendations based on issues."""
        recommendations = []

        if issues.get("label_errors", 0) > n_samples * 0.05:
            recommendations.append(
                "High label error rate detected. Consider reviewing flagged samples."
            )

        if issues.get("duplicates", 0) > n_samples * 0.1:
            recommendations.append(
                "Significant duplicate content found. Consider deduplication."
            )

        if issues.get("outliers", 0) > n_samples * 0.1:
            recommendations.append(
                "Many outliers detected. Review data collection process."
            )

        if score < 70:
            recommendations.append(
                "Overall quality is below acceptable threshold. Data cleaning recommended."
            )
        elif score >= 90:
            recommendations.append(
                "Excellent data quality! Continue monitoring for drift."
            )

        if not recommendations:
            recommendations.append("Data quality is acceptable.")

        return recommendations


def create_api_app(
    default_tier: TierLevel = TierLevel.FREE,
) -> Any:
    """Create FastAPI application for quality scoring.

    Args:
        default_tier: Default tier for unauthenticated requests

    Returns:
        FastAPI application

    Example:
        >>> app = create_api_app()
        >>> # Run with: uvicorn module:app --port 8080
    """
    try:
        from fastapi import FastAPI, HTTPException, Header, UploadFile, File
        from fastapi.responses import JSONResponse
        import io
    except ImportError as e:
        raise CleanError(
            "FastAPI required for API app. Install with: pip install fastapi uvicorn"
        ) from e

    app = FastAPI(
        title="Clean Data Quality API",
        description="Public API for dataset quality scoring",
        version="1.0.0",
    )

    api = QualityScoreAPI(default_tier=default_tier)

    @app.post("/score")
    async def score_dataset(
        file: UploadFile = File(...),
        label_column: str | None = None,
        api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> dict:
        """Score a dataset for quality."""
        try:
            # Read file
            content = await file.read()
            if file.filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            elif file.filename.endswith(".json"):
                df = pd.read_json(io.BytesIO(content))
            elif file.filename.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content))
            else:
                raise HTTPException(400, "Unsupported file format")

            # Score
            score = api.score_dataset(df, label_column, api_key)
            return score.to_dict()

        except CleanError as e:
            raise HTTPException(429 if "rate limit" in str(e).lower() else 400, str(e))

    @app.get("/limits")
    async def get_limits(
        api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> dict:
        """Get current rate limit status."""
        status = api.get_rate_limit_status(api_key)
        return {
            "requests_remaining_day": status.requests_remaining_day,
            "requests_remaining_minute": status.requests_remaining_minute,
            "is_limited": status.is_limited,
        }

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


# Create default app for uvicorn
app = create_api_app()


__all__ = [
    # Core classes
    "QualityScoreAPI",
    "QuickScore",
    "RateLimiter",
    "RateLimitStatus",
    # Tier system
    "TierLevel",
    "TierLimits",
    # Functions
    "create_api_app",
    # Default app
    "app",
]
