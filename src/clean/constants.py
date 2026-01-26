"""Constants and configurable defaults for Clean.

This module centralizes magic numbers and default values used throughout
the codebase, making them easier to configure and maintain.
"""

from __future__ import annotations

# ==============================================================================
# Outlier Detection Constants
# ==============================================================================

#: Default IQR multiplier for outlier detection (standard is 1.5)
DEFAULT_IQR_MULTIPLIER: float = 1.5

#: Default z-score threshold for statistical outlier detection
DEFAULT_ZSCORE_THRESHOLD: float = 3.0

#: Default contamination rate for Isolation Forest
DEFAULT_CONTAMINATION: float = 0.1

#: Default percentile threshold for conservative outlier removal
DEFAULT_OUTLIER_PERCENTILE: float = 75.0


# ==============================================================================
# Duplicate Detection Constants
# ==============================================================================

#: Default similarity threshold for near-duplicate detection
DEFAULT_SIMILARITY_THRESHOLD: float = 0.9

#: High similarity threshold for exact-like matches
HIGH_SIMILARITY_THRESHOLD: float = 0.95

#: Maximum dataset size for pairwise similarity computation
MAX_PAIRWISE_SAMPLES: int = 10000


# ==============================================================================
# Label Error Detection Constants
# ==============================================================================

#: Default confidence threshold for flagging label errors
DEFAULT_LABEL_CONFIDENCE_THRESHOLD: float = 0.5

#: Default number of cross-validation folds
DEFAULT_CV_FOLDS: int = 5

#: High confidence threshold for auto-relabeling
HIGH_CONFIDENCE_RELABEL_THRESHOLD: float = 0.9


# ==============================================================================
# Text Analysis Constants
# ==============================================================================

#: Minimum average character length to consider a column as text data
TEXT_COLUMN_MIN_AVG_LENGTH: int = 50

#: Minimum length for text/response content in LLM analysis
DEFAULT_MIN_TEXT_LENGTH: int = 10

#: Maximum length for text/response content in LLM analysis
DEFAULT_MAX_TEXT_LENGTH: int = 10000

#: Minimum word length to consider content as "short"
SHORT_CONTENT_THRESHOLD: int = 5


# ==============================================================================
# Imbalance Detection Constants
# ==============================================================================

#: Default ratio threshold for class imbalance warning
DEFAULT_IMBALANCE_THRESHOLD: float = 5.0

#: Critical imbalance ratio requiring attention
CRITICAL_IMBALANCE_RATIO: float = 10.0


# ==============================================================================
# Quality Scoring Constants
# ==============================================================================

#: Threshold for "excellent" quality score
QUALITY_EXCELLENT_THRESHOLD: float = 90.0

#: Threshold for "good" quality score
QUALITY_GOOD_THRESHOLD: float = 75.0

#: Threshold for "moderate" quality score
QUALITY_MODERATE_THRESHOLD: float = 60.0

#: Threshold for "poor" quality score
QUALITY_POOR_THRESHOLD: float = 40.0


# ==============================================================================
# Fix Engine Constants
# ==============================================================================

#: Default confidence threshold for suggesting label fixes
DEFAULT_FIX_LABEL_THRESHOLD: float = 0.9

#: Default similarity threshold for suggesting duplicate removal
DEFAULT_FIX_DUPLICATE_THRESHOLD: float = 0.98

#: Default score threshold for suggesting outlier removal
DEFAULT_FIX_OUTLIER_THRESHOLD: float = 0.9


# ==============================================================================
# Streaming Constants
# ==============================================================================

#: Default chunk size for streaming analysis
DEFAULT_CHUNK_SIZE: int = 10000

#: Default window size for real-time metrics
DEFAULT_WINDOW_SIZE: int = 1000


# ==============================================================================
# API and Performance Constants
# ==============================================================================

#: Maximum results per page for pagination
DEFAULT_PAGE_SIZE: int = 100

#: Maximum samples for expensive operations
MAX_SAMPLES_FOR_EXPENSIVE_OPS: int = 100000

#: Batch size for embedding generation
DEFAULT_EMBEDDING_BATCH_SIZE: int = 32


# ==============================================================================
# Data Type Detection Constants
# ==============================================================================

#: Number of samples to check for data type inference
DATA_TYPE_SAMPLE_SIZE: int = 100

#: Number of samples for text column detection
TEXT_DETECTION_SAMPLE_SIZE: int = 10

#: Threshold for unique values to consider regression vs classification
REGRESSION_UNIQUE_THRESHOLD: int = 20


# ==============================================================================
# Privacy Constants
# ==============================================================================

#: Default minimum confidence for PII detection
DEFAULT_PII_CONFIDENCE: float = 0.5

#: High-risk PII confidence threshold
HIGH_RISK_PII_CONFIDENCE: float = 0.9


__all__ = [
    # Outlier detection
    "DEFAULT_IQR_MULTIPLIER",
    "DEFAULT_ZSCORE_THRESHOLD",
    "DEFAULT_CONTAMINATION",
    "DEFAULT_OUTLIER_PERCENTILE",
    # Duplicate detection
    "DEFAULT_SIMILARITY_THRESHOLD",
    "HIGH_SIMILARITY_THRESHOLD",
    "MAX_PAIRWISE_SAMPLES",
    # Label error detection
    "DEFAULT_LABEL_CONFIDENCE_THRESHOLD",
    "DEFAULT_CV_FOLDS",
    "HIGH_CONFIDENCE_RELABEL_THRESHOLD",
    # Text analysis
    "TEXT_COLUMN_MIN_AVG_LENGTH",
    "DEFAULT_MIN_TEXT_LENGTH",
    "DEFAULT_MAX_TEXT_LENGTH",
    "SHORT_CONTENT_THRESHOLD",
    # Imbalance detection
    "DEFAULT_IMBALANCE_THRESHOLD",
    "CRITICAL_IMBALANCE_RATIO",
    # Quality scoring
    "QUALITY_EXCELLENT_THRESHOLD",
    "QUALITY_GOOD_THRESHOLD",
    "QUALITY_MODERATE_THRESHOLD",
    "QUALITY_POOR_THRESHOLD",
    # Fix engine
    "DEFAULT_FIX_LABEL_THRESHOLD",
    "DEFAULT_FIX_DUPLICATE_THRESHOLD",
    "DEFAULT_FIX_OUTLIER_THRESHOLD",
    # Streaming
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_WINDOW_SIZE",
    # API and performance
    "DEFAULT_PAGE_SIZE",
    "MAX_SAMPLES_FOR_EXPENSIVE_OPS",
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    # Data type detection
    "DATA_TYPE_SAMPLE_SIZE",
    "TEXT_DETECTION_SAMPLE_SIZE",
    "REGRESSION_UNIQUE_THRESHOLD",
    # Privacy
    "DEFAULT_PII_CONFIDENCE",
    "HIGH_RISK_PII_CONFIDENCE",
]
