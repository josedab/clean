"""Clean: AI-powered data quality platform for ML datasets.

Clean automatically detects and helps fix the issues that make ML models fail:
label errors, duplicates, outliers, and biases.

Basic usage:
    >>> from clean import DatasetCleaner
    >>> cleaner = DatasetCleaner(data=df, label_column='label')
    >>> report = cleaner.analyze()
    >>> print(report.summary())
"""

from clean.__version__ import __version__
from clean.core.cleaner import DatasetCleaner
from clean.core.report import QualityReport
from clean.core.types import (
    DatasetInfo,
    DataType,
    DetectionResults,
    IssueType,
    QualityScore,
    TaskType,
)
from clean.detection import (
    BiasDetector,
    DuplicateDetector,
    ImbalanceDetector,
    LabelErrorDetector,
    OutlierDetector,
    analyze_bias,
    analyze_imbalance,
    find_duplicates,
    find_label_errors,
    find_outliers,
)
from clean.exceptions import (
    CleanError,
    ConfigurationError,
    DependencyError,
    DetectionError,
    ExportError,
    FixError,
    LoaderError,
    PluginError,
    StreamingError,
    ValidationError,
    require_package,
)
from clean.fixes import (
    FixConfig,
    FixEngine,
    FixResult,
    FixStrategy,
    apply_fixes,
    suggest_fixes,
)
from clean.loaders import (
    load_arrays,
    load_csv,
    load_dataframe,
)
from clean.plugins import (
    DetectorPlugin,
    ExporterPlugin,
    FixerPlugin,
    LoaderPlugin,
    PluginRegistry,
    SuggestedFix,
    registry,
)

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "CleanError",
    "ConfigurationError",
    "DependencyError",
    "DetectionError",
    "ExportError",
    "FixError",
    "LoaderError",
    "PluginError",
    "StreamingError",
    "ValidationError",
    "require_package",
    # Core classes
    "DatasetCleaner",
    "DatasetInfo",
    "DataType",
    "DetectionResults",
    "IssueType",
    "QualityReport",
    "QualityScore",
    "TaskType",
    # Detector classes
    "BiasDetector",
    "DuplicateDetector",
    "ImbalanceDetector",
    "LabelErrorDetector",
    "OutlierDetector",
    # Detection functions
    "analyze_bias",
    "analyze_imbalance",
    "find_duplicates",
    "find_label_errors",
    "find_outliers",
    # Loaders
    "load_arrays",
    "load_csv",
    "load_dataframe",
    # Fix system
    "FixConfig",
    "FixEngine",
    "FixResult",
    "FixStrategy",
    "apply_fixes",
    "suggest_fixes",
    # Plugin system
    "DetectorPlugin",
    "ExporterPlugin",
    "FixerPlugin",
    "LoaderPlugin",
    "PluginRegistry",
    "SuggestedFix",
    "registry",
]

# Lineage tracking
from clean.lineage import (
    AnalysisRun,
    LineageTracker,
    ReviewRecord,
    SampleHistory,
)

__all__.extend([
    "AnalysisRun",
    "LineageTracker",
    "ReviewRecord",
    "SampleHistory",
])

# LLM data quality
from clean.llm import (
    InstructionIssue,
    LLMDataCleaner,
    LLMQualityReport,
)

__all__.extend([
    "InstructionIssue",
    "LLMDataCleaner",
    "LLMQualityReport",
])

# Optional HuggingFace loader
try:
    from clean.loaders import load_huggingface

    __all__.append("load_huggingface")
except ImportError:
    pass

# Optional image loader
try:
    from clean.loaders import load_image_folder

    __all__.append("load_image_folder")
except ImportError:
    pass

# Streaming support
from clean.streaming import (
    ChunkResult,
    StreamingCleaner,
    StreamingSummary,
    stream_analyze,
)

__all__.extend([
    "ChunkResult",
    "StreamingCleaner",
    "StreamingSummary",
    "stream_analyze",
])

# Annotation quality analysis
from clean.annotation import (
    AgreementMetrics,
    AnnotationAnalyzer,
    AnnotationQualityReport,
    AnnotatorMetrics,
    analyze_annotations,
)

__all__.extend([
    "AnnotationAnalyzer",
    "AnnotationQualityReport",
    "AnnotatorMetrics",
    "AgreementMetrics",
    "analyze_annotations",
])

# Data drift detection
from clean.drift import (
    DriftDetector,
    DriftMonitor,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDrift,
    detect_drift,
)

__all__.extend([
    "DriftDetector",
    "DriftMonitor",
    "DriftReport",
    "DriftSeverity",
    "DriftType",
    "FeatureDrift",
    "detect_drift",
])

# LLM Evaluation Suite
from clean.llm_eval import (
    LLMEvalReport,
    LLMEvaluator,
    PIIDetector,
    SafetyCategory,
    SampleEvaluation,
    ToxicityDetector,
    evaluate_llm_data,
)

__all__.extend([
    "LLMEvaluator",
    "LLMEvalReport",
    "SampleEvaluation",
    "SafetyCategory",
    "ToxicityDetector",
    "PIIDetector",
    "evaluate_llm_data",
])

# Synthetic Data Validation
from clean.synthetic import (
    SyntheticDataValidator,
    SyntheticIssue,
    SyntheticIssueType,
    SyntheticValidationReport,
    validate_synthetic_data,
)

__all__.extend([
    "SyntheticDataValidator",
    "SyntheticValidationReport",
    "SyntheticIssue",
    "SyntheticIssueType",
    "validate_synthetic_data",
])

# Compliance Report Generation
from clean.compliance import (
    ComplianceFramework,
    ComplianceReport,
    ComplianceReportGenerator,
    ComplianceStatus,
    RiskLevel,
    generate_compliance_report,
)

__all__.extend([
    "ComplianceReportGenerator",
    "ComplianceReport",
    "ComplianceFramework",
    "ComplianceStatus",
    "RiskLevel",
    "generate_compliance_report",
])

# Active Learning Integration
from clean.active_learning import (
    ActiveLearner,
    CVATExporter,
    LabelStudioExporter,
    ProdigyExporter,
    SampleSelection,
    SamplingStrategy,
    select_for_labeling,
)

__all__.extend([
    "ActiveLearner",
    "SamplingStrategy",
    "SampleSelection",
    "LabelStudioExporter",
    "CVATExporter",
    "ProdigyExporter",
    "select_for_labeling",
])

# Multi-Modal Analysis
from clean.multimodal import (
    AlignmentIssue,
    ModalityType,
    MultiModalAnalyzer,
    MultiModalReport,
    analyze_multimodal,
)

__all__.extend([
    "MultiModalAnalyzer",
    "MultiModalReport",
    "ModalityType",
    "AlignmentIssue",
    "analyze_multimodal",
])

# Distributed Processing
from clean.distributed import (
    ChunkedAnalyzer,
    DaskCleaner,
    DistributedConfig,
    DistributedReport,
    analyze_distributed,
)

__all__.extend([
    "DaskCleaner",
    "ChunkedAnalyzer",
    "DistributedReport",
    "DistributedConfig",
    "analyze_distributed",
])

# Web Dashboard (requires FastAPI)
try:
    from clean.dashboard import (
        DashboardApp,
        DashboardConfig,
        create_dashboard_app,
        run_dashboard,
    )

    __all__.extend([
        "DashboardApp",
        "DashboardConfig",
        "create_dashboard_app",
        "run_dashboard",
    ])
except ImportError:
    pass  # FastAPI not installed

# Real-time Streaming Pipeline
from clean.realtime import (
    AlertSeverity,
    KafkaSource,
    MemorySource,
    PipelineConfig,
    PulsarSource,
    QualityAlert,
    RealtimeMetrics,
    RealtimePipeline,
    RedisSource,
    StreamBackend,
    create_pipeline,
)

__all__.extend([
    "RealtimePipeline",
    "RealtimeMetrics",
    "PipelineConfig",
    "QualityAlert",
    "AlertSeverity",
    "StreamBackend",
    "KafkaSource",
    "PulsarSource",
    "RedisSource",
    "MemorySource",
    "create_pipeline",
])

# AutoML Quality Tuning
from clean.automl import (
    AdaptiveThresholdManager,
    OptimizationMethod,
    QualityTuner,
    ThresholdParams,
    TuningConfig,
    TuningMetric,
    TuningResult,
    tune_quality_thresholds,
)

__all__.extend([
    "QualityTuner",
    "TuningConfig",
    "TuningResult",
    "ThresholdParams",
    "OptimizationMethod",
    "TuningMetric",
    "AdaptiveThresholdManager",
    "tune_quality_thresholds",
])

# Root Cause Analysis
from clean.root_cause import (
    RootCause,
    RootCauseAnalyzer,
    RootCauseReport,
    RootCauseType,
    analyze_root_causes,
)

__all__.extend([
    "RootCauseAnalyzer",
    "RootCauseReport",
    "RootCause",
    "RootCauseType",
    "analyze_root_causes",
])

# Vector DB Integration
from clean.vectordb import (
    MemoryVectorStore,
    VectorDBBackend,
    VectorQualityAnalyzer,
    VectorStore,
    create_vector_store,
)

__all__.extend([
    "VectorStore",
    "MemoryVectorStore",
    "VectorDBBackend",
    "VectorQualityAnalyzer",
    "create_vector_store",
])

# Model-Aware Quality Scoring
from clean.model_aware import (
    ClassMetrics,
    ImpactLevel,
    ModelAwareReport,
    ModelAwareScorer,
    SampleQuality,
    score_with_model,
)

__all__.extend([
    "ModelAwareScorer",
    "ModelAwareReport",
    "SampleQuality",
    "ClassMetrics",
    "ImpactLevel",
    "score_with_model",
])

# Enhanced Active Learning (Intelligent Sampling)
from clean.active_learning import (
    CorrectionFeedback,
    ExpectedModelChange,
    IntelligentSampler,
    LearningSession,
    QueryByCommittee,
)

__all__.extend([
    "IntelligentSampler",
    "CorrectionFeedback",
    "LearningSession",
    "QueryByCommittee",
    "ExpectedModelChange",
])

# Data Slice Discovery
from clean.slice_discovery import (
    DataSlice,
    SliceCondition,
    SliceDiscoverer,
    SliceDiscoveryReport,
    discover_slices,
)

__all__.extend([
    "SliceDiscoverer",
    "SliceDiscoveryReport",
    "DataSlice",
    "SliceCondition",
    "discover_slices",
])

# Privacy Vault
from clean.privacy import (
    AnonymizationMethod,
    AnonymizationResult,
    PIIScanner,
    PIIScanReport,
    PIIType,
    PrivacyVault,
    anonymize_data,
    scan_pii,
)

__all__.extend([
    "PrivacyVault",
    "PIIScanner",
    "PIIScanReport",
    "PIIType",
    "AnonymizationMethod",
    "AnonymizationResult",
    "scan_pii",
    "anonymize_data",
])

# Collaborative Review Workspace
from clean.collaboration import (
    Annotation,
    Conflict,
    ConflictResolutionStrategy,
    Reviewer,
    ReviewItem,
    ReviewSession,
    ReviewStatus,
    ReviewWorkspace,
    create_review_session,
)

__all__.extend([
    "ReviewWorkspace",
    "ReviewSession",
    "ReviewItem",
    "Reviewer",
    "Annotation",
    "Conflict",
    "ReviewStatus",
    "ConflictResolutionStrategy",
    "create_review_session",
])

# Cloud Service (Multi-Tenant)
from clean.cloud import (
    APIKey,
    CloudService,
    Permission,
    RateLimiter,
    Role,
    SubscriptionTier,
    User,
    Workspace,
)

__all__.extend([
    "CloudService",
    "Workspace",
    "User",
    "APIKey",
    "Role",
    "Permission",
    "SubscriptionTier",
    "RateLimiter",
])
