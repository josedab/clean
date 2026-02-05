# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-02-05)


### Features

* **action:** add GitHub Action for CI/CD quality gates ([db46080](https://github.com/josedab/clean/commit/db46080d602740ea0822100bb4fc75ec9108a26b))
* add drift detection and annotation analysis ([f66c30c](https://github.com/josedab/clean/commit/f66c30c32d364736bc8b134e398bf6018addbffe))
* **advisor:** add proactive quality advisor with AI recommendations ([6dd4081](https://github.com/josedab/clean/commit/6dd4081e828476cc1b477d85364569ccd723117b))
* **agentic:** add AI-powered autonomous data curation agent ([46d577d](https://github.com/josedab/clean/commit/46d577d918cd4db5078dbaa7d78449d9dd3ff59c))
* **augmentation:** add quality-aware data augmentation ([1f61725](https://github.com/josedab/clean/commit/1f61725e47f318eea4296991362688c5887e97a1))
* **auto-detect:** add zero-config auto-detection ([bb28f46](https://github.com/josedab/clean/commit/bb28f46ba31c697198a348611cc40325e31ce250))
* **auto-docs:** add automated data documentation generator ([d81becf](https://github.com/josedab/clean/commit/d81becfce3cc1780e8e1d200f729d2aa7889889b))
* **benchmark:** add data quality benchmark suite ([ccd7b22](https://github.com/josedab/clean/commit/ccd7b22959572cfde593bf3abe7ea8eb11464cdf))
* **contamination:** add cross-dataset contamination detector ([19b6a48](https://github.com/josedab/clean/commit/19b6a4856660c3319ee68084b35df2f8b818ef00))
* **copilot:** add natural language data quality copilot ([4c6a928](https://github.com/josedab/clean/commit/4c6a9289c0b86d97d7f27fd7a978a6ee2d93287d))
* **core:** add constants and configuration defaults ([acba5fb](https://github.com/josedab/clean/commit/acba5fb6878ed05867d2a353ead1c5885cf94f81))
* **core:** add core types and data structures ([c6f3de2](https://github.com/josedab/clean/commit/c6f3de26b8e760a34247decaa254726f69ece210))
* **core:** add QualityReport and scoring system ([eab97cb](https://github.com/josedab/clean/commit/eab97cb0adf15a940c0e8fe75bb79545964847a5))
* **core:** implement base exception hierarchy ([866964c](https://github.com/josedab/clean/commit/866964c1b8ed41e87c72bdb392a66d8ce27986a4))
* **core:** implement DatasetCleaner main interface ([f6a2e61](https://github.com/josedab/clean/commit/f6a2e614dd1c24f6f60f7a2e56932a7d452f7a20))
* **cost-impact:** add cost-impact estimator for data cleaning ([2a8652b](https://github.com/josedab/clean/commit/2a8652b97171d7fef63da941f18a58af121e7363))
* **curriculum:** add curriculum learning optimizer ([46e4816](https://github.com/josedab/clean/commit/46e4816f99906cdaaa914f177a92da9758c63ee9))
* **detection:** add base detector abstraction and factory ([c1ae136](https://github.com/josedab/clean/commit/c1ae1367b8bd1d08e3dc2fff57123f149b1a5ce7))
* **detection:** add duplicate detection with strategies ([b20f0db](https://github.com/josedab/clean/commit/b20f0dbdf9b84ecbb0f1bbe9a269b6cf07c69e6e))
* **detection:** add imbalance and bias detectors ([8b1f899](https://github.com/josedab/clean/commit/8b1f8995937eedd6fb67b109890870f4c8b9ca6f))
* **detection:** implement label error detector ([2b3de0f](https://github.com/josedab/clean/commit/2b3de0f710ba31b7b5bee35b38eb947f1f1a2f39))
* **detection:** implement outlier detector with multiple algorithms ([bf97b35](https://github.com/josedab/clean/commit/bf97b35ee356858a8bb23174da3c911efa7be4a0))
* **distillation:** add lightweight detector model distillation ([0956660](https://github.com/josedab/clean/commit/09566607eacfed2755bf88430f336939c922bf88))
* **embedding-viz:** add embedding space visualizer ([0842660](https://github.com/josedab/clean/commit/08426604ac33a38f80b62abd662a632625059398))
* **embeddings:** add foundation model embeddings support ([41a3f86](https://github.com/josedab/clean/commit/41a3f867506d6b804d44f1782f6e3912c0355f0c))
* **embeddings:** add text and image embedding providers ([8494200](https://github.com/josedab/clean/commit/849420050f2a515dcf321bd05110b0ae9999abb4))
* **enterprise:** add privacy, compliance and collaboration ([b5b8da4](https://github.com/josedab/clean/commit/b5b8da4ae425501c9f9d8dd7c5a9ee53118a9c04))
* **enterprise:** add realtime, cloud and AutoML features ([cccd42d](https://github.com/josedab/clean/commit/cccd42d48a06afb58b1f95aac033126e9ecb0304))
* **enterprise:** implement vector DB and active learning ([fc78ae7](https://github.com/josedab/clean/commit/fc78ae7588657113829553957e5325404b6d3ebe))
* **feature-store:** add feature store integration ([8cd9238](https://github.com/josedab/clean/commit/8cd92381f21892dd67d7752c1eb84f5509761d6e))
* **federated:** add privacy-preserving federated quality analysis ([93a5363](https://github.com/josedab/clean/commit/93a53638225a221b7e65123f36190f0a85b27b9f))
* **feedback-loop:** add continuous learning feedback loop ([58e25b3](https://github.com/josedab/clean/commit/58e25b3d9d811462c15e8d5421d4d19b3367c4d0))
* **fixes:** implement auto-fix engine with strategies ([71d4065](https://github.com/josedab/clean/commit/71d406540729c57581aad3ba7dd681152310011b))
* **fm-benchmark:** add foundation model quality benchmark suite ([c53081b](https://github.com/josedab/clean/commit/c53081b31cc06df39c4bde497952811eb666fe1c))
* implement dashboard, CLI and REST API ([54d4e87](https://github.com/josedab/clean/commit/54d4e87af3fb817816e8472058b78154b590c0bc))
* **integrations:** add native dbt and Airflow integration ([c159fa6](https://github.com/josedab/clean/commit/c159fa6f63576ae9ab2e3d87faab5a402d7c4ae8))
* **knowledge-graph:** add data quality knowledge graph for impact analysis ([76bf32e](https://github.com/josedab/clean/commit/76bf32e1d30592f84ffecdd426820f2a2d0e6b58))
* **labeler-scoring:** add automated labeler performance scoring ([7103af3](https://github.com/josedab/clean/commit/7103af3b80e25e25e377e2cda9fc5213c246d7ed))
* **llm-judge:** add LLM-as-judge integration for data quality evaluation ([a1892fb](https://github.com/josedab/clean/commit/a1892fb913e16969fb590b2f43e5f46b74ba1ebb))
* **llm:** implement LLM data quality analyzer ([c06d7dc](https://github.com/josedab/clean/commit/c06d7dc4d0b8d5234f89200968a5750227d5c15e))
* **loaders:** add CSV and HuggingFace loaders ([1299788](https://github.com/josedab/clean/commit/1299788aea00ffaa113e79de2c4bdedd4e7cd9a2))
* **loaders:** add pandas and numpy data loaders ([d1d68bd](https://github.com/josedab/clean/commit/d1d68bdce9b06e653f9ea4609f8e329dc416d482))
* **loaders:** implement base loader abstraction ([c9650bf](https://github.com/josedab/clean/commit/c9650bfba6131f53b93bf2b1c0bb5e713f15fef5))
* **loaders:** implement image folder loader ([6982a67](https://github.com/josedab/clean/commit/6982a6757eeb8d392cef47cc603bb204189eaeab))
* **marketplace:** add multi-organization data marketplace ([3b0342e](https://github.com/josedab/clean/commit/3b0342e1a498d1fcf8558ba6eecd1bdc56cbc4ff))
* **mlops:** add MLflow and Weights & Biases integration ([7f47592](https://github.com/josedab/clean/commit/7f47592a4a4a46061783c7a1830a55327e682004))
* **multilingual:** add multi-language label error detection ([74e6d96](https://github.com/josedab/clean/commit/74e6d96d3be1146d998ad520efefa0a2eb5cabd5))
* **nl-query:** add natural language query interface ([8cdc38d](https://github.com/josedab/clean/commit/8cdc38d7c1297280376156fc0fb0f1049bdaace2))
* **quality-augmentation:** add quality-aware data augmentation ([1e1771b](https://github.com/josedab/clean/commit/1e1771b948f8d87957736447c7743db4fe105a09))
* **quality-predictor:** add ML-based quality score prediction ([6a69fa1](https://github.com/josedab/clean/commit/6a69fa1b5561219adc93b82a3e894249da0d956f))
* **quality-regression:** add quality regression testing ([e2ff04e](https://github.com/josedab/clean/commit/e2ff04e8900b4e4fb7d0359067f70b041514aeed))
* **quality-tracker:** add version-aware quality tracking ([d92e0cd](https://github.com/josedab/clean/commit/d92e0cd55e7c802fdc19fb356f62144edf44d7d7))
* **score-api:** add public data quality score API ([5b19e8c](https://github.com/josedab/clean/commit/5b19e8c39e152856486f02736f83d66046b555b1))
* **sdk:** add multi-language SDK generation ([ab8363f](https://github.com/josedab/clean/commit/ab8363fd4bbdaeffe8a332f7bddef97c59b6e11a))
* **streaming:** add streaming cleaner for large datasets ([dc3ab66](https://github.com/josedab/clean/commit/dc3ab661bdf16959288d94a54e681098b7f6fea2))
* **studio:** add visual data quality studio ([d4043e0](https://github.com/josedab/clean/commit/d4043e09dd382823bb86a521a24a9790393056ff))
* **synthetic-certification:** add synthetic data quality certification ([4a4b057](https://github.com/josedab/clean/commit/4a4b057eb2f6ea9be07824769ead3f3145c1a79f))
* **synthetic:** add certified synthetic data generation ([011e32e](https://github.com/josedab/clean/commit/011e32e49fcfd3220b36c6d725b0991da6ce99cc))
* **utils:** add validation, preprocessing and export utilities ([4ca05d0](https://github.com/josedab/clean/commit/4ca05d0712b1291cab2391013b0e59919c5771e9))
* **version-control:** add git-like version control for quality reports ([34d294e](https://github.com/josedab/clean/commit/34d294eee89020bacbe7b0c42adac7546a62f245))
* **viz:** implement visualization plots and browser export ([b40c9d5](https://github.com/josedab/clean/commit/b40c9d51e2505bab69c161d27a01326d42773023))
* **widgets:** add Jupyter notebook widgets ([e222435](https://github.com/josedab/clean/commit/e22243515778fca828da9f37b827e43c89a18ce9))


### Bug Fixes

* add proper exception chaining with 'from e' ([643ad36](https://github.com/josedab/clean/commit/643ad364d48f973372e4547505991f685e215a0c))


### Documentation

* add documentation, examples and benchmarks ([46ab1c3](https://github.com/josedab/clean/commit/46ab1c3ef805e69e78fb8eb1716bb9eb83bbe100))
* add README, contributing guide and changelog ([b7e0818](https://github.com/josedab/clean/commit/b7e081827f24565cb99b43eb43b35b3ac53b6419))
* **guides:** add next-gen features overview guide ([ee14c65](https://github.com/josedab/clean/commit/ee14c65f7095544fee6610cb4d3209fd5795c26d))

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
