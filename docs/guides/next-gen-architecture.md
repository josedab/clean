# Next-Gen Features Architecture

This document provides an architectural overview of Clean's next-generation features, their interactions, and data flows.

## System Overview

```mermaid
graph TB
    subgraph "Data Ingestion"
        A[Batch Data] --> D[DatasetCleaner]
        B[Streaming Data] --> E[RealtimePipeline]
        C[Vector Store] --> F[VectorDBConnector]
    end

    subgraph "Core Analysis"
        D --> G[Quality Analysis]
        E --> G
        G --> H[Quality Report]
    end

    subgraph "Advanced Analysis"
        H --> I[RootCauseAnalyzer]
        H --> J[SliceDiscovery]
        H --> K[ModelAwareScorer]
    end

    subgraph "Optimization"
        H --> L[QualityTuner/AutoML]
        H --> M[IntelligentSampler]
    end

    subgraph "Privacy & Compliance"
        A --> N[PrivacyVault]
        N --> D
    end

    subgraph "Collaboration"
        H --> O[ReviewWorkspace]
        O --> P[Consensus Engine]
        P --> Q[Final Labels]
    end

    subgraph "Deployment"
        R[CloudService] --> S[Multi-Tenant API]
        S --> T[Workspaces]
        T --> D
    end
```

## Feature Interactions

### 1. Real-Time Pipeline

```mermaid
sequenceDiagram
    participant Source as Kafka/Pulsar/Redis
    participant Pipeline as RealtimePipeline
    participant Buffer as Message Buffer
    participant Analyzer as Quality Analyzer
    participant Alert as Alert Handlers

    Source->>Pipeline: Stream messages
    Pipeline->>Buffer: Buffer messages
    
    loop Window Processing
        Buffer->>Analyzer: Window ready (N samples)
        Analyzer->>Analyzer: Run quality checks
        alt Quality degraded
            Analyzer->>Alert: Trigger alert
            Alert->>Alert: Slack/Webhook/Custom
        end
        Analyzer->>Pipeline: Update metrics
    end
```

### 2. AutoML Tuning Flow

```mermaid
flowchart TD
    A[Training Data] --> B[QualityTuner]
    C[Validation Labels] --> B
    
    B --> D{Optimization Method}
    
    D -->|Grid| E[Exhaustive Search]
    D -->|Random| F[Random Sampling]
    D -->|Bayesian| G[Gaussian Process]
    D -->|Evolutionary| H[Genetic Algorithm]
    
    E --> I[Evaluate Params]
    F --> I
    G --> I
    H --> I
    
    I --> J[Cross-Validation]
    J --> K{Converged?}
    
    K -->|No| L[Update Search]
    L --> D
    
    K -->|Yes| M[Best Parameters]
    M --> N[Optimized Cleaner]
```

### 3. Cloud Service Architecture

```mermaid
graph TD
    subgraph "API Layer"
        A[REST API] --> B[Authentication]
        B --> C[Rate Limiter]
        C --> D[Request Router]
    end

    subgraph "Service Layer"
        D --> E[UserManager]
        D --> F[WorkspaceManager]
        D --> G[AnalysisService]
        D --> H[BillingService]
    end

    subgraph "Data Layer"
        E --> I[(PostgreSQL)]
        F --> I
        G --> J[(Redis Cache)]
        H --> K[Stripe API]
    end

    subgraph "Workspace Isolation"
        F --> L[Workspace A]
        F --> M[Workspace B]
        F --> N[Workspace C]
    end
```

### 4. Privacy Pipeline

```mermaid
flowchart LR
    A[Raw Data] --> B{PII Scan}
    
    B -->|Found| C[PII Report]
    C --> D{Action}
    
    D -->|Anonymize| E[Anonymizer]
    D -->|Encrypt| F[Encryptor]
    D -->|Report| G[Compliance Report]
    
    E --> H{Method}
    H -->|Pseudonymize| I[Fake Values]
    H -->|Mask| J[Character Masking]
    H -->|Generalize| K[Reduced Precision]
    H -->|DP| L[Differential Privacy]
    
    I --> M[Safe Data]
    J --> M
    K --> M
    L --> M
    F --> M
    
    B -->|Clean| M
    M --> N[Analysis Pipeline]
```

### 5. Collaborative Review Flow

```mermaid
stateDiagram-v2
    [*] --> SessionCreated
    
    SessionCreated --> ItemsAdded: Add review items
    ItemsAdded --> InReview: Assign to reviewers
    
    InReview --> ReviewSubmitted: Reviewer votes
    ReviewSubmitted --> InReview: More reviewers
    ReviewSubmitted --> ConsensusCheck: Min votes reached
    
    ConsensusCheck --> Agreed: Consensus reached
    ConsensusCheck --> Conflicted: No agreement
    
    Conflicted --> ExpertReview: Escalate
    ExpertReview --> Resolved: Expert decides
    Resolved --> Agreed
    
    Agreed --> AllComplete: All items done
    AllComplete --> Finalized: Admin approves
    
    Finalized --> [*]
```

## Data Flow Patterns

### End-to-End Quality Pipeline

```mermaid
graph LR
    subgraph "Input"
        A[Raw Dataset]
    end
    
    subgraph "Pre-Processing"
        A --> B[Privacy Scan]
        B --> C[Anonymization]
        C --> D[Vectorization]
    end
    
    subgraph "Analysis"
        D --> E[Quality Analysis]
        E --> F[Root Cause]
        E --> G[Slice Discovery]
        E --> H[Model-Aware Score]
    end
    
    subgraph "Optimization"
        E --> I[AutoML Tuning]
        E --> J[Active Sampling]
    end
    
    subgraph "Review"
        E --> K[Collaborative Review]
        K --> L[Consensus]
    end
    
    subgraph "Output"
        L --> M[Clean Dataset]
        H --> N[Quality Report]
        F --> O[Root Cause Report]
    end
```

### Streaming Quality Monitoring

```mermaid
graph TB
    subgraph "Sources"
        A[Kafka] --> D[Source Connector]
        B[Pulsar] --> D
        C[Redis] --> D
    end
    
    subgraph "Processing"
        D --> E[Message Parser]
        E --> F[Window Buffer]
        F --> G[Quality Checks]
    end
    
    subgraph "Storage"
        G --> H[Metrics Store]
        H --> I[Time Series DB]
    end
    
    subgraph "Alerting"
        G --> J{Threshold}
        J -->|Exceeded| K[Alert Manager]
        K --> L[Slack]
        K --> M[Webhook]
        K --> N[PagerDuty]
    end
    
    subgraph "Visualization"
        I --> O[Dashboard]
        O --> P[Charts]
        O --> Q[Reports]
    end
```

## Module Dependencies

```mermaid
graph TD
    subgraph "Core"
        A[clean.core] --> B[clean.detection]
        B --> C[clean.fixes]
    end
    
    subgraph "Next-Gen"
        D[clean.realtime] --> A
        E[clean.automl] --> A
        F[clean.cloud] --> A
        G[clean.root_cause] --> B
        H[clean.vectordb] --> B
        I[clean.model_aware] --> A
        J[clean.active_learning] --> A
        K[clean.slice_discovery] --> B
        L[clean.privacy] --> A
        M[clean.collaboration] --> A
    end
    
    subgraph "External"
        N[aiokafka] -.-> D
        O[sklearn] -.-> E
        P[pinecone] -.-> H
        Q[fastapi] -.-> F
    end
```

## Scalability Patterns

### Horizontal Scaling

```mermaid
graph TD
    subgraph "Load Balancer"
        A[HAProxy/Nginx]
    end
    
    subgraph "API Instances"
        A --> B[API 1]
        A --> C[API 2]
        A --> D[API N]
    end
    
    subgraph "Workers"
        B --> E[Worker Pool]
        C --> E
        D --> E
    end
    
    subgraph "Shared State"
        E --> F[(PostgreSQL)]
        E --> G[(Redis)]
        E --> H[(S3/MinIO)]
    end
```

### Batch Processing

```mermaid
graph LR
    A[Large Dataset] --> B[Chunker]
    B --> C[Chunk 1]
    B --> D[Chunk 2]
    B --> E[Chunk N]
    
    C --> F[Worker 1]
    D --> G[Worker 2]
    E --> H[Worker N]
    
    F --> I[Aggregator]
    G --> I
    H --> I
    
    I --> J[Final Report]
```

## Integration Points

| Feature | REST API | WebSocket | CLI | Python |
|---------|----------|-----------|-----|--------|
| Real-Time Pipeline | ✅ | ✅ | ✅ | ✅ |
| AutoML Tuning | ✅ | ❌ | ✅ | ✅ |
| Cloud Service | ✅ | ✅ | ✅ | ✅ |
| Root Cause | ✅ | ❌ | ❌ | ✅ |
| Vector DB | ✅ | ❌ | ❌ | ✅ |
| Model-Aware | ✅ | ❌ | ❌ | ✅ |
| Intelligent Sampling | ✅ | ❌ | ✅ | ✅ |
| Slice Discovery | ✅ | ❌ | ❌ | ✅ |
| Privacy Vault | ✅ | ❌ | ✅ | ✅ |
| Collaboration | ✅ | ✅ | ❌ | ✅ |

## Performance Considerations

### Memory Usage

| Feature | Memory Profile | Recommendation |
|---------|---------------|----------------|
| Real-Time Pipeline | Low (streaming) | Use for large datasets |
| AutoML Tuning | Medium (CV folds) | Reduce CV folds for large data |
| Vector DB | Low (external storage) | Use for >100K samples |
| Slice Discovery | High (clustering) | Sample for >1M samples |
| Collaboration | Low (external DB) | Use for any size |

### Compute Requirements

| Feature | CPU | GPU | Parallel |
|---------|-----|-----|----------|
| Real-Time Pipeline | Low | ❌ | ✅ |
| AutoML Tuning | High | Optional | ✅ |
| Root Cause | Medium | ❌ | ✅ |
| Vector DB | Low | ❌ | N/A |
| Slice Discovery | High | Optional | ✅ |
| Privacy (NER) | Medium | Optional | ✅ |
