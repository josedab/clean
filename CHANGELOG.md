# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Real-Time Streaming Pipeline
- `RealtimePipeline` for continuous quality monitoring on streaming data
- `KafkaSource`, `PulsarSource`, `RedisSource` connectors
- `PipelineConfig` for window size, thresholds, and alerting
- `QualityAlert` with Slack and webhook handlers
- `create_realtime_pipeline()` convenience function

#### AutoML Quality Tuning
- `QualityTuner` for automated threshold optimization
- Grid search, random search, Bayesian, and evolutionary methods
- `TuningConfig` for method selection and hyperparameters
- `ThresholdParams` dataclass for tunable parameters
- `tune_quality_thresholds()` convenience function

#### Multi-Tenant Cloud Service
- `CloudService` for SaaS deployment
- `CloudService` with user management and JWT authentication
- Workspace isolation via `create_workspace()` method
- API key management via `create_api_key()` method
- RBAC with owner/admin/analyst/viewer roles
- Usage tracking and rate limiting via `RateLimiter`

#### Automated Root Cause Analysis
- `RootCauseAnalyzer` for identifying issue causes
- Statistical correlation analysis
- Feature importance with ML models
- Cluster analysis for pattern detection
- `RootCause` with descriptions and suggested fixes
- `analyze_root_causes()` convenience function

#### Vector Database Integration
- `VectorStore` abstract base class
- `PineconeStore` for Pinecone cloud
- `WeaviateStore` for Weaviate
- `MilvusStore` for Milvus
- `QdrantStore` for Qdrant
- `create_vector_store()` factory function

#### Model-Aware Quality Scoring
- `ModelAwareScorer` for model-specific quality assessment
- Support for neural networks, tree-based, SVM, linear, KNN
- Issue weighting by model sensitivity
- Impact simulation for cleanup ROI estimation
- `score_for_model()` convenience function

#### Intelligent Sampling for Labeling
- `IntelligentSampler` with advanced query strategies
- `QueryByCommittee` for disagreement-based sampling
- `ExpectedModelChange` for gradient-based selection
- Hybrid strategies combining uncertainty and diversity
- `LabelingSession` for iterative human-in-the-loop learning

#### Data Slice Discovery
- `SliceDiscovery` for finding underperforming subgroups
- Decision tree, clustering, rule mining, and Domino methods
- `Slice` with predicates and metric gaps
- `SliceAwareScorer` for adjusted quality scores
- `discover_slices()` convenience function

#### Data Privacy Vault
- `PrivacyVault` for PII management
- `PIIDetector` for email, phone, SSN, credit card, name, address
- Anonymization: pseudonymize, mask, generalize, differential privacy
- Column-level and format-preserving encryption
- Audit logging for compliance
- `scan_for_pii()` and `anonymize_dataframe()` convenience functions

#### Collaborative Review Workspace
- `ReviewWorkspace` for multi-user annotation review
- `ReviewSession` with item assignment and tracking
- Voting strategies: majority, unanimous, weighted, Dawid-Skene
- Conflict detection and resolution workflow
- `ReviewerAnalytics` for annotator quality metrics
- Label Studio, CVAT, Prodigy export integration

#### Data Drift Monitoring
- `DriftDetector` class for one-time distribution shift detection
- `DriftMonitor` class for continuous monitoring with alerting
- Support for KS test, PSI, Chi-squared, and Wasserstein distance methods
- `detect_drift()` convenience function

#### Annotation Quality Analysis
- `AnnotationAnalyzer` for inter-annotator agreement measurement
- Krippendorff's alpha, Fleiss' kappa, and Cohen's kappa metrics
- Per-annotator quality metrics and disagreement analysis
- `analyze_annotations()` convenience function

#### LLM Evaluation Suite
- `LLMEvaluator` for safety and quality assessment
- `ToxicityDetector` with pattern-based profanity/hate speech detection
- `PIIDetector` for emails, phones, SSNs, credit cards
- Prompt injection detection
- `evaluate_llm_data()` convenience function

#### Synthetic Data Validation
- `SyntheticDataValidator` for generated data quality assessment
- Mode collapse detection
- Memorization risk analysis via nearest-neighbor distances
- Distribution gap measurement
- `validate_synthetic_data()` convenience function

#### Compliance Report Generator
- `ComplianceReportGenerator` for regulatory documentation
- EU AI Act compliance checking
- NIST AI RMF framework support
- Customizable compliance frameworks
- `generate_compliance_report()` convenience function

#### Active Learning Integration
- `ActiveLearner` with uncertainty, margin, entropy, and diversity sampling
- `LabelStudioExporter` for Label Studio JSON export
- `CVATExporter` for CVAT XML export
- `ProdigyExporter` for Prodigy JSONL export
- `select_for_labeling()` convenience function

#### Multi-Modal Analysis
- `MultiModalAnalyzer` for image-text consistency checking
- Cross-modal alignment scoring
- Per-modality quality assessment
- `analyze_multimodal()` convenience function

#### Distributed Processing
- `DaskCleaner` for parallel processing with ThreadPoolExecutor
- `ChunkedAnalyzer` for memory-efficient file processing
- `SparkCleaner` for PySpark integration (optional)
- `analyze_distributed()` convenience function

#### Web Dashboard
- `DashboardApp` FastAPI-based interactive dashboard
- Real-time file upload and analysis
- Quality score visualization with Chart.js
- Issue breakdown and feature-level analysis
- `run_dashboard()` convenience function

#### CI/CD Integration
- GitHub Action (`action/action.yml`) for automated quality gates
- `clean check` CLI command with pass/fail exit codes
- GitHub Actions output format support
- Configurable quality thresholds

### Changed
- CLI now includes `dashboard` and `check` commands
- Module exports expanded in `__init__.py`
- `ActiveLearner` enhanced with `IntelligentSampler`, `QueryByCommittee`, `ExpectedModelChange`

### Dependencies
- Added optional `streaming` extras: `aiokafka`, `pulsar-client`, `redis`
- Added optional `vectordb` extras: `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`
- Added optional `cloud` extras: `aiohttp`

## [1.0.0] - 2026-01-29

### Added
- Initial public release of Clean data quality platform
- Core detection: label errors, duplicates, outliers, bias, class imbalance
- LLM data quality analysis for instruction-tuning datasets
- Auto-fix engine with configurable strategies
- Plugin system for custom detectors and fixers
- Streaming analysis for large datasets
- REST API with FastAPI
- Command-line interface
- Data lineage tracking
