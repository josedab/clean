import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    {
      type: 'category',
      label: 'Core Concepts',
      collapsed: false,
      items: [
        'concepts/overview',
        'concepts/label-errors',
        'concepts/duplicates',
        'concepts/outliers',
        'concepts/imbalance',
        'concepts/bias',
      ],
    },
    {
      type: 'category',
      label: 'How-To Guides',
      collapsed: false,
      items: [
        'guides/llm-data',
        'guides/streaming',
        'guides/auto-fix',
        'guides/plugins',
        'guides/cli',
        'guides/rest-api',
      ],
    },
    {
      type: 'category',
      label: 'Enterprise Features',
      collapsed: false,
      items: [
        'guides/realtime',
        'guides/automl',
        'guides/root-cause',
        'guides/slice-discovery',
        'guides/model-aware',
        'guides/privacy',
        'guides/collaboration',
        'guides/vectordb',
        'guides/intelligent-sampling',
        'guides/cloud',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: true,
      items: [
        'api/cleaner',
        'api/report',
        'api/detectors',
        'api/fix-engine',
        'api/loaders',
        'api/streaming',
        'api/llm',
        'api/lineage',
      ],
    },
    'architecture',
    'comparison',
    'changelog',
    'contributing',
    'faq',
  ],
};

export default sidebars;
