import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Clean',
  tagline: 'AI-powered data quality for ML datasets',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://clean-data.github.io',
  baseUrl: '/clean/',

  organizationName: 'clean-data',
  projectName: 'clean',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/clean-data/clean/tree/main/website/',
          showLastUpdateTime: false,
          showLastUpdateAuthor: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'star_us',
      content:
        '⭐ If Clean helps you, give us a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/clean-data/clean">GitHub</a>!',
      backgroundColor: '#0d9488',
      textColor: '#ffffff',
      isCloseable: true,
    },
    navbar: {
      title: 'Clean',
      logo: {
        alt: 'Clean Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api/cleaner',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/clean-data/clean',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
            {
              label: 'Core Concepts',
              to: '/docs/concepts/overview',
            },
            {
              label: 'API Reference',
              to: '/docs/api/cleaner',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/clean-data/clean/discussions',
            },
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/clean-data-quality',
            },
            {
              label: 'Contributing',
              to: '/docs/contributing',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Changelog',
              to: '/docs/changelog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/clean-data/clean',
            },
            {
              label: 'PyPI',
              href: 'https://pypi.org/project/clean-data-quality/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Clean Project. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
