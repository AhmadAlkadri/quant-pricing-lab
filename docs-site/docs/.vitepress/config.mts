import { defineConfig } from 'vitepress'

function normalizeBase(base: string): string {
  if (!base || base === '/') {
    return '/'
  }
  let out = base.trim()
  if (!out.startsWith('/')) {
    out = `/${out}`
  }
  if (!out.endsWith('/')) {
    out = `${out}/`
  }
  return out
}

const explicitBase = process.env.BASE
const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const computedBase = explicitBase ?? (repoName ? `/${repoName}/` : '/')

export default defineConfig({
  base: normalizeBase(computedBase),
  title: 'qpl',
  description: 'Quant Pricing Lab: public learning surface for v0.2.0',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Reference', link: '/reference/' }
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Design Philosophy', link: '/guide/design-philosophy' }
          ]
        }
      ],
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Overview', link: '/reference/' },
            { text: 'Pricing', link: '/reference/pricing' },
            { text: 'Numerics', link: '/reference/numerics' },
            { text: 'Transforms', link: '/reference/transforms' },
            { text: 'Dependence', link: '/reference/dependence' },
            { text: 'Dynamic Programming', link: '/reference/dp' }
          ]
        }
      ]
    }
  }
})
