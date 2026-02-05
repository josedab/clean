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

# LLM-as-Judge Integration
from clean.llm_judge import (
    AnthropicProvider,
    CustomProvider,
    EvaluationDimension,
    JudgeConfig,
    JudgeProvider,
    JudgeReport,
    JudgeResult,
    LLMJudge,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
    evaluate_with_llm,
)

__all__.extend([
    "LLMJudge",
    "JudgeConfig",
    "JudgeResult",
    "JudgeReport",
    "EvaluationDimension",
    "JudgeProvider",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "CustomProvider",
    "create_provider",
    "evaluate_with_llm",
])

# Data Quality Copilot
from clean.copilot import (
    DataQualityCopilot,
    FixScript,
    IssueCategory,
    QueryIntent,
    QueryResult,
    create_copilot,
)

__all__.extend([
    "DataQualityCopilot",
    "QueryResult",
    "FixScript",
    "QueryIntent",
    "IssueCategory",
    "create_copilot",
])

# Continuous Learning Feedback Loop
from clean.feedback_loop import (
    CorrelationResult,
    DataPrescription,
    FeedbackLoop,
    InMemoryConnector,
    MetricType,
    MLflowConnector,
    MetricsConnector,
    ActionType,
    create_feedback_loop,
)

__all__.extend([
    "FeedbackLoop",
    "CorrelationResult",
    "DataPrescription",
    "ActionType",
    "MetricType",
    "MetricsConnector",
    "MLflowConnector",
    "InMemoryConnector",
    "create_feedback_loop",
])

# Data Quality Benchmark Suite
from clean.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    DetectorType,
    SyntheticDataGenerator,
    compare_detectors,
    run_benchmark,
)

__all__.extend([
    "BenchmarkSuite",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkSuiteResult",
    "SyntheticDataGenerator",
    "DetectorType",
    "run_benchmark",
    "compare_detectors",
])

# Feature Store Integration
from clean.feature_store import (
    FeastConnector,
    FeatureMetadata,
    FeatureQualityAnalyzer,
    FeatureQualityReport,
    FeatureStoreConnector,
    DataFrameConnector,
    TectonConnector,
    analyze_feature_store,
)

__all__.extend([
    "FeatureQualityAnalyzer",
    "FeatureQualityReport",
    "FeatureMetadata",
    "FeatureStoreConnector",
    "FeastConnector",
    "TectonConnector",
    "DataFrameConnector",
    "analyze_feature_store",
])

# Automated Data Augmentation
from clean.augmentation import (
    AugmentationConfig,
    AugmentationRecommendation,
    AugmentationResult,
    AugmentationStrategy,
    DataAugmenter,
    InterpolationOperation,
    NoiseAugmentationOperation,
    OversamplingOperation,
    SMOTEOperation,
    augment_for_quality,
)

__all__.extend([
    "DataAugmenter",
    "AugmentationConfig",
    "AugmentationResult",
    "AugmentationRecommendation",
    "AugmentationStrategy",
    "SMOTEOperation",
    "InterpolationOperation",
    "NoiseAugmentationOperation",
    "OversamplingOperation",
    "augment_for_quality",
])

# Multi-Language Label Error Detection
from clean.multilingual import (
    LanguageDetector,
    LanguageDetectionResult,
    MultilingualDetector,
    MultilingualEmbedder,
    MultilingualReport,
    MultilingualLabelError,
    detect_multilingual_errors,
)

__all__.extend([
    "MultilingualDetector",
    "MultilingualLabelError",
    "MultilingualReport",
    "MultilingualEmbedder",
    "LanguageDetector",
    "LanguageDetectionResult",
    "detect_multilingual_errors",
])

# Data Quality Score API
from clean.score_api import (
    QualityScoreAPI,
    QuickScore,
    RateLimiter as ScoreAPIRateLimiter,
    RateLimitStatus,
    TierLevel,
    TierLimits,
    create_api_app,
)

__all__.extend([
    "QualityScoreAPI",
    "QuickScore",
    "ScoreAPIRateLimiter",
    "RateLimitStatus",
    "TierLevel",
    "TierLimits",
    "create_api_app",
])

# Version Control for Data Quality
from clean.version_control import (
    BranchInfo,
    ChangeType,
    MetricChange,
    QualityDiff,
    QualitySnapshot,
    QualityVersionControl,
    create_version_control,
)

__all__.extend([
    "QualityVersionControl",
    "QualitySnapshot",
    "QualityDiff",
    "MetricChange",
    "BranchInfo",
    "ChangeType",
    "create_version_control",
])

# Custom Model Distillation
from clean.distillation import (
    CompressionLevel,
    DistillationConfig,
    DistillationPipeline,
    DistillationResult,
    ExportResult,
    LightweightDetector,
    ModelDistiller,
    ModelFormat,
    create_distillation_pipeline,
    create_distiller,
)

__all__.extend([
    "ModelDistiller",
    "DistillationPipeline",
    "DistillationConfig",
    "DistillationResult",
    "ExportResult",
    "LightweightDetector",
    "ModelFormat",
    "CompressionLevel",
    "create_distiller",
    "create_distillation_pipeline",
])

# Federated Data Quality (Privacy-Preserving Analysis)
from clean.federated import (
    FederatedAnalyzer,
    FederatedQualityReport,
    LocalNode,
    PrivacyConfig,
    PrivacyLevel as FederatedPrivacyLevel,
    SecureAggregator,
    federated_analyze,
)

__all__.extend([
    "FederatedAnalyzer",
    "FederatedQualityReport",
    "LocalNode",
    "PrivacyConfig",
    "FederatedPrivacyLevel",
    "SecureAggregator",
    "federated_analyze",
])

# Foundation Model Quality Benchmark
from clean.fm_benchmark import (
    BenchmarkLeaderboard,
    BenchmarkMetric,
    BenchmarkResult as FMBenchmarkResult,
    FMBenchmark,
    benchmark_dataset,
)

__all__.extend([
    "FMBenchmark",
    "FMBenchmarkResult",
    "BenchmarkMetric",
    "BenchmarkLeaderboard",
    "benchmark_dataset",
])

# Agentic Data Curation
from clean.agentic import (
    ActionStatus,
    CurationAction,
    CurationPlan,
    DataCurationAgent,
    create_curation_agent,
)

__all__.extend([
    "DataCurationAgent",
    "CurationPlan",
    "CurationAction",
    "ActionStatus",
    "create_curation_agent",
])

# Data Quality Knowledge Graph
from clean.knowledge_graph import (
    GraphEdge,
    GraphNode,
    ImpactAnalyzer,
    ImpactPrediction,
    NodeType,
    QualityKnowledgeGraph,
    create_knowledge_graph,
)

__all__.extend([
    "QualityKnowledgeGraph",
    "ImpactAnalyzer",
    "ImpactPrediction",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "create_knowledge_graph",
])

# dbt/Airflow Integration
from clean.integrations import (
    CleanQualityOperator,
    DbtTestRunner,
    IntegrationResult,
    PipelineQualityGate,
    generate_airflow_dag,
    generate_dbt_schema_yaml,
)

__all__.extend([
    "DbtTestRunner",
    "IntegrationResult",
    "CleanQualityOperator",
    "PipelineQualityGate",
    "generate_dbt_schema_yaml",
    "generate_airflow_dag",
])

# Visual Data Quality Studio
try:
    from clean.studio import (
        DataStudio,
        StudioAPI,
        StudioSession,
        create_studio,
        run_studio,
    )

    __all__.extend([
        "DataStudio",
        "StudioAPI",
        "StudioSession",
        "create_studio",
        "run_studio",
    ])
except ImportError:
    pass  # FastAPI not installed

# Multi-Language SDK Generation
from clean.sdk import (
    JavaScriptSDKGenerator,
    JuliaSDKGenerator,
    RSDKGenerator,
    SDKConfig,
    SDKGenerator,
    TargetLanguage,
    generate_all_sdks,
    generate_js_client,
    generate_julia_package,
    generate_r_package,
)

__all__.extend([
    "SDKGenerator",
    "SDKConfig",
    "TargetLanguage",
    "RSDKGenerator",
    "JavaScriptSDKGenerator",
    "JuliaSDKGenerator",
    "generate_r_package",
    "generate_js_client",
    "generate_julia_package",
    "generate_all_sdks",
])

# Certified Synthetic Data Generation (extends synthetic module)
from clean.synthetic import (
    CertificationConfig,
    CertificationStatus,
    CertifiedDataGenerator,
    GenerationResult,
    QualityCertificate,
    QualityMetric as SyntheticQualityMetric,
    generate_certified_data,
)

__all__.extend([
    "CertifiedDataGenerator",
    "CertificationConfig",
    "CertificationStatus",
    "GenerationResult",
    "QualityCertificate",
    "SyntheticQualityMetric",
    "generate_certified_data",
])

# Foundation Model Embeddings
from clean.embeddings.foundation import (
    CachedEmbedder,
    CohereEmbedder,
    EmbeddingProvider,
    EmbeddingStats,
    FoundationEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    create_embedder,
)

__all__.extend([
    "EmbeddingProvider",
    "EmbeddingStats",
    "FoundationEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    "CachedEmbedder",
    "create_embedder",
])

# Zero-Config Auto-Detection
from clean.auto_detect import (
    AutoConfig,
    AutoDetector,
    ColumnProfile,
    DataModality,
    auto_analyze,
    detect_config,
)

__all__.extend([
    "DataModality",
    "ColumnProfile",
    "AutoConfig",
    "AutoDetector",
    "auto_analyze",
    "detect_config",
])

# Cost-Impact Estimator
from clean.cost_impact import (
    ActionCost,
    CleaningAction,
    CostConfig,
    CostImpactEstimator,
    ImpactReport,
    estimate_impact,
)

__all__.extend([
    "CleaningAction",
    "ActionCost",
    "ImpactReport",
    "CostConfig",
    "CostImpactEstimator",
    "estimate_impact",
])

# MLflow/W&B Integration
from clean.mlops import (
    MLflowIntegration,
    MLOpsBackend,
    QualityCallback,
    WandbIntegration,
    create_mlflow_integration,
    create_wandb_integration,
    track_data_quality,
)

__all__.extend([
    "MLOpsBackend",
    "MLflowIntegration",
    "WandbIntegration",
    "QualityCallback",
    "track_data_quality",
    "create_mlflow_integration",
    "create_wandb_integration",
])

# Notebook Widgets (optional - requires ipywidgets)
try:
    from clean.widgets import (
        FixWidget,
        QualityExplorer,
        show_report,
    )

    __all__.extend([
        "QualityExplorer",
        "FixWidget",
        "show_report",
    ])
except ImportError:
    pass  # ipywidgets not installed

# Automated Data Documentation
from clean.auto_docs import (
    ColumnDocumentation,
    DataDocumenter,
    DatasetDocumentation,
    DocumentationLevel,
    generate_docs,
)

__all__.extend([
    "DataDocumenter",
    "DatasetDocumentation",
    "ColumnDocumentation",
    "DocumentationLevel",
    "generate_docs",
])

# Version-Aware Quality Tracking
from clean.quality_tracker import (
    QualityDiff,
    QualitySnapshot,
    QualityTracker,
    QualityTrend,
    track_quality,
)

__all__.extend([
    "QualityTracker",
    "QualitySnapshot",
    "QualityDiff",
    "QualityTrend",
    "track_quality",
])

# Proactive Quality Advisor
from clean.advisor import (
    ActionCategory,
    AdvisorReport,
    Priority,
    QualityAdvisor,
    Recommendation,
    get_recommendations,
)

__all__.extend([
    "QualityAdvisor",
    "Recommendation",
    "AdvisorReport",
    "Priority",
    "ActionCategory",
    "get_recommendations",
])

# Federated Quality Analysis
from clean.federated import (
    FederatedAnalyzer,
    FederatedQualityReport,
    LocalNode,
    LocalStatistics,
    PrivacyConfig,
    PrivacyLevel,
    SecureAggregator,
    create_federated_analyzer,
    federated_analyze,
)

__all__.extend([
    "FederatedAnalyzer",
    "FederatedQualityReport",
    "LocalNode",
    "LocalStatistics",
    "PrivacyConfig",
    "PrivacyLevel",
    "SecureAggregator",
    "create_federated_analyzer",
    "federated_analyze",
])

# Benchmark Suite
from clean.benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    DetectionMetrics,
    DetectorType,
    SyntheticDataGenerator,
    compare_detectors,
    run_benchmark,
)

__all__.extend([
    "BenchmarkConfig",
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkSuiteResult",
    "DetectionMetrics",
    "DetectorType",
    "SyntheticDataGenerator",
    "compare_detectors",
    "run_benchmark",
])

# =============================================================================
# Next-Gen Features (2025)
# =============================================================================

# Quality Prediction Model
from clean.quality_predictor import (
    FeatureExtractor,
    PredictorConfig,
    QualityGate,
    QualityPrediction,
    QualityPredictor,
    predict_quality,
)

__all__.extend([
    "QualityPredictor",
    "QualityPrediction",
    "FeatureExtractor",
    "PredictorConfig",
    "QualityGate",
    "predict_quality",
])

# Natural Language Query Interface
from clean.nl_query import (
    NLQueryEngine,
    QueryIntent as NLQueryIntent,
    QueryResult as NLQueryResult,
    create_query_engine,
    query_report,
)

__all__.extend([
    "NLQueryEngine",
    "NLQueryIntent",
    "NLQueryResult",
    "create_query_engine",
    "query_report",
])

# Quality-Aware Data Augmentation
from clean.quality_augmentation import (
    AugmentationConfig,
    AugmentationMethod,
    AugmentationResult,
    GapType,
    QualityAwareAugmenter,
    QualityGap,
    augment_for_quality,
)

__all__.extend([
    "QualityAwareAugmenter",
    "AugmentationConfig",
    "AugmentationResult",
    "AugmentationMethod",
    "GapType",
    "QualityGap",
    "augment_for_quality",
])

# Cross-Dataset Contamination Detector
from clean.contamination import (
    ContaminationConfig,
    ContaminationDetector,
    ContaminationReport,
    ContaminationType,
    SeverityLevel as ContaminationSeverity,
    detect_contamination,
)

__all__.extend([
    "ContaminationDetector",
    "ContaminationReport",
    "ContaminationType",
    "ContaminationConfig",
    "ContaminationSeverity",
    "detect_contamination",
])

# Curriculum Learning Optimizer
from clean.curriculum import (
    CurriculumConfig,
    CurriculumDataLoader,
    CurriculumOptimizer,
    CurriculumSchedule,
    CurriculumStrategy,
    create_curriculum,
)

__all__.extend([
    "CurriculumOptimizer",
    "CurriculumSchedule",
    "CurriculumConfig",
    "CurriculumStrategy",
    "CurriculumDataLoader",
    "create_curriculum",
])

# Quality Regression Testing
from clean.quality_regression import (
    QualityHistoryStore,
    QualityRegressionTester,
    QualitySnapshot,
    QualityTestResult,
    RegressionSeverity,
    RegressionTestConfig,
    run_quality_test,
    QualityGate as RegressionQualityGate,
)

__all__.extend([
    "QualityRegressionTester",
    "QualityTestResult",
    "QualitySnapshot",
    "RegressionSeverity",
    "RegressionTestConfig",
    "QualityHistoryStore",
    "RegressionQualityGate",
    "run_quality_test",
])

# Embedding Space Visualizer
from clean.embedding_viz import (
    EmbeddingVisualizer,
    ReductionMethod,
    VisualizationConfig,
    VisualizationResult,
    visualize_embeddings,
)

__all__.extend([
    "EmbeddingVisualizer",
    "VisualizationConfig",
    "VisualizationResult",
    "ReductionMethod",
    "visualize_embeddings",
])

# Synthetic Data Quality Certification
from clean.synthetic_certification import (
    CertificationConfig,
    CertificationStatus,
    DimensionScore,
    QualityCertificate,
    QualityDimension,
    SyntheticCertifier,
    certify_synthetic_data,
)

__all__.extend([
    "SyntheticCertifier",
    "QualityCertificate",
    "DimensionScore",
    "CertificationConfig",
    "CertificationStatus",
    "QualityDimension",
    "certify_synthetic_data",
])

# Multi-Organization Data Marketplace
from clean.marketplace import (
    AnonymizedBenchmark,
    DataType as MarketplaceDataType,
    Domain,
    IndustryBenchmark,
    PercentileResult,
    PrivacyLevel as MarketplacePrivacyLevel,
    QualityMarketplace,
    create_marketplace,
    get_industry_percentile,
)

__all__.extend([
    "QualityMarketplace",
    "IndustryBenchmark",
    "AnonymizedBenchmark",
    "Domain",
    "MarketplaceDataType",
    "MarketplacePrivacyLevel",
    "PercentileResult",
    "create_marketplace",
    "get_industry_percentile",
])

# Automated Labeler Performance Scoring
from clean.labeler_scoring import (
    ExpertiseLevel,
    LabelerEvaluator,
    LabelerMetrics,
    LabelerRecommendation,
    LabelerReport,
    PerformanceStatus,
    SmartRouter,
    TaskAssignment,
    evaluate_labelers,
    get_labeler_report,
)

__all__.extend([
    "LabelerEvaluator",
    "LabelerMetrics",
    "LabelerReport",
    "LabelerRecommendation",
    "ExpertiseLevel",
    "PerformanceStatus",
    "SmartRouter",
    "TaskAssignment",
    "evaluate_labelers",
    "get_labeler_report",
])
