import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className={styles.heroTitle}>
            Find and fix the data issues that break ML models
          </Heading>
          <p className={styles.heroSubtitle}>
            Clean automatically detects label errors, duplicates, outliers, and biases 
            in your training data. Stop debugging models—start debugging data.
          </p>
          
          <div className={styles.installBox}>
            <code>pip install clean-data-quality</code>
          </div>
          
          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/getting-started">
              Get Started →
            </Link>
            <Link
              className="button button--outline button--lg"
              to="https://github.com/clean-data/clean">
              GitHub ⭐
            </Link>
          </div>
          
          <div className={styles.badges}>
            <img src="https://img.shields.io/pypi/v/clean-data-quality?color=teal" alt="PyPI" />
            <img src="https://img.shields.io/badge/python-3.9+-blue" alt="Python" />
            <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
          </div>
        </div>
      </div>
    </header>
  );
}

function CodeExample() {
  const code = `from clean import DatasetCleaner

# Initialize with your data
cleaner = DatasetCleaner(data=df, label_column='label')

# Run comprehensive analysis
report = cleaner.analyze()

# View results
print(report.summary())
# Quality Score: 82.5/100
# Label errors: 347 (3.5%) - HIGH PRIORITY
# Duplicates: 234 pairs (4.7%)
# Outliers: 156 (1.6%)`;

  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className={styles.codeWrapper}>
          <div className={styles.codeDescription}>
            <Heading as="h2">Three lines to clean data</Heading>
            <p>
              No complex configuration. No learning curve. 
              Just pass your DataFrame and get actionable insights.
            </p>
            <ul className={styles.codeFeatures}>
              <li>✅ Detects mislabeled samples with confidence scores</li>
              <li>✅ Finds exact and near-duplicates</li>
              <li>✅ Identifies statistical outliers</li>
              <li>✅ Analyzes class imbalance and bias</li>
            </ul>
          </div>
          <div className={styles.codeBlock}>
            <CodeBlock language="python">{code}</CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

type FeatureItem = {
  title: string;
  icon: string;
  description: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Label Error Detection',
    icon: '🏷️',
    description: 'Find mislabeled samples using confident learning. Know exactly which labels to fix and what the correct label should be.',
  },
  {
    title: 'Duplicate Detection',
    icon: '🔍',
    description: 'Catch exact matches and near-duplicates that inflate your metrics. Supports text embeddings and image similarity.',
  },
  {
    title: 'Outlier Detection',
    icon: '📊',
    description: 'Multiple methods: Isolation Forest, LOF, IQR, z-score. Ensemble voting for robust detection.',
  },
  {
    title: 'LLM Data Quality',
    icon: '🤖',
    description: 'Specialized analysis for instruction-tuning and RAG datasets. Detect refusals, short responses, and incoherent pairs.',
  },
  {
    title: 'Auto-Fix Engine',
    icon: '🔧',
    description: 'Get fix suggestions with confidence scores. Apply corrections automatically or review manually.',
  },
  {
    title: 'Multiple Interfaces',
    icon: '💻',
    description: 'Python API, CLI tool, REST API for dashboards. Stream large datasets that don\'t fit in memory.',
  },
];

const EnterpriseFeatureList: FeatureItem[] = [
  {
    title: 'Real-Time Streaming',
    icon: '📡',
    description: 'Monitor data quality on Kafka, Pulsar, or Redis streams. Get alerts when quality degrades.',
  },
  {
    title: 'AutoML Tuning',
    icon: '🎯',
    description: 'Automatically optimize quality thresholds using Bayesian or evolutionary optimization.',
  },
  {
    title: 'Root Cause Analysis',
    icon: '🔬',
    description: 'Understand why issues occur. Statistical drill-down identifies the source of problems.',
  },
  {
    title: 'Slice Discovery',
    icon: '🍕',
    description: 'Find underperforming data subgroups automatically. Know where your model fails.',
  },
  {
    title: 'Privacy Vault',
    icon: '🔐',
    description: 'Detect and anonymize PII. Encryption, audit logging, and compliance reporting built-in.',
  },
  {
    title: 'Collaborative Review',
    icon: '👥',
    description: 'Multi-user annotation review with voting, conflict resolution, and consensus tracking.',
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={styles.feature}>
      <div className={styles.featureIcon}>{icon}</div>
      <Heading as="h3">{title}</Heading>
      <p>{description}</p>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <Heading as="h2" className={styles.featuresTitle}>
          Everything you need to debug your data
        </Heading>
        <div className={styles.featureGrid}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function EnterpriseFeatures() {
  return (
    <section className={styles.enterpriseFeatures}>
      <div className="container">
        <Heading as="h2" className={styles.featuresTitle}>
          Enterprise-ready capabilities
        </Heading>
        <p className={styles.enterpriseSubtitle}>
          Scale from prototype to production with features built for teams
        </p>
        <div className={styles.featureGrid}>
          {EnterpriseFeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
        <div className={styles.enterpriseCta}>
          <Link
            className="button button--primary button--lg"
            to="/docs/guides/realtime">
            Explore Enterprise Features →
          </Link>
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className={styles.architecture}>
      <div className="container">
        <Heading as="h2">How Clean works</Heading>
        <p className={styles.architectureSubtitle}>
          A modular pipeline that analyzes, detects, and fixes data quality issues
        </p>
        <div className={styles.architectureDiagram}>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>📥</div>
            <div className={styles.archLabel}>Load Data</div>
            <div className={styles.archDesc}>DataFrame, CSV, HuggingFace, Images</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>🔍</div>
            <div className={styles.archLabel}>Detect Issues</div>
            <div className={styles.archDesc}>Labels, Duplicates, Outliers, Bias</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>📊</div>
            <div className={styles.archLabel}>Quality Report</div>
            <div className={styles.archDesc}>Scores, Rankings, Visualizations</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>🔧</div>
            <div className={styles.archLabel}>Fix & Export</div>
            <div className={styles.archDesc}>Auto-fix, Review, Clean Data</div>
          </div>
        </div>
      </div>
    </section>
  );
}

function WorksWith() {
  return (
    <section className={styles.worksWith}>
      <div className="container">
        <Heading as="h2">Works with your stack</Heading>
        <div className={styles.worksWithGrid}>
          <div className={styles.worksWithItem}>
            <span className={styles.worksWithIcon}>🐼</span>
            <span>Pandas</span>
          </div>
          <div className={styles.worksWithItem}>
            <span className={styles.worksWithIcon}>🔢</span>
            <span>NumPy</span>
          </div>
          <div className={styles.worksWithItem}>
            <span className={styles.worksWithIcon}>🤗</span>
            <span>HuggingFace</span>
          </div>
          <div className={styles.worksWithItem}>
            <span className={styles.worksWithIcon}>🧠</span>
            <span>scikit-learn</span>
          </div>
          <div className={styles.worksWithItem}>
            <span className={styles.worksWithIcon}>🖼️</span>
            <span>Image Folders</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function UseCases() {
  return (
    <section className={styles.useCases}>
      <div className="container">
        <Heading as="h2">Built for ML teams</Heading>
        <p className={styles.useCasesSubtitle}>
          Whether you're training your first model or deploying at scale
        </p>
        <div className={styles.useCasesGrid}>
          <div className={styles.useCase}>
            <div className={styles.useCaseIcon}>🔬</div>
            <Heading as="h3">Research Teams</Heading>
            <p>Ensure reproducible results with clean training data. Catch label errors before they skew your experiments.</p>
          </div>
          <div className={styles.useCase}>
            <div className={styles.useCaseIcon}>🚀</div>
            <Heading as="h3">ML Engineers</Heading>
            <p>Debug model failures faster. When accuracy plateaus, Clean shows you exactly which samples to fix.</p>
          </div>
          <div className={styles.useCase}>
            <div className={styles.useCaseIcon}>⚙️</div>
            <Heading as="h3">MLOps Teams</Heading>
            <p>Add quality gates to your pipelines. Catch data drift and quality regressions before they hit production.</p>
          </div>
          <div className={styles.useCase}>
            <div className={styles.useCaseIcon}>🏷️</div>
            <Heading as="h3">Annotation Teams</Heading>
            <p>Prioritize review efforts. Clean ranks samples by confidence so you review the most impactful labels first.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTA() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <Heading as="h2">Ready to clean your data?</Heading>
        <p>Get started in under 5 minutes. No account required.</p>
        <div className={styles.ctaButtons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started">
            Read the Docs
          </Link>
          <Link
            className="button button--outline button--lg"
            to="https://pypi.org/project/clean-data-quality/">
            View on PyPI
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="AI-powered data quality for ML"
      description="Clean automatically detects and helps fix label errors, duplicates, outliers, and biases in ML datasets.">
      <HomepageHeader />
      <main>
        <CodeExample />
        <HomepageFeatures />
        <Architecture />
        <EnterpriseFeatures />
        <UseCases />
        <WorksWith />
        <CTA />
      </main>
    </Layout>
  );
}
