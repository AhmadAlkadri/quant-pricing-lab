import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type DefaultTheme } from 'vitepress'

// docs/curriculum/ is generated from the repository's docs/ by
// scripts/sync-docs.mjs (run by the docs:* npm scripts). The notes sidebar is
// built from that directory, so a new note appears without editing this file.
const curriculumDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'curriculum')

function firstHeading(file: string): string {
  const match = readFileSync(file, 'utf8').match(/^# (.+)$/m)
  return match ? match[1].trim() : file
}

function noteItems(): DefaultTheme.SidebarItem[] {
  const notesDir = join(curriculumDir, 'notes')
  if (!existsSync(notesDir)) return []
  // Order notes by where the curriculum's Delivered section first cites them,
  // which is slice order; anything it does not cite goes last, by name.
  const curriculum = readFileSync(join(curriculumDir, 'index.md'), 'utf8')
  const delivered = curriculum.slice(Math.max(0, curriculum.indexOf('## Delivered')))
  const rank = (name: string) => {
    const at = delivered.indexOf(name)
    return at < 0 ? Number.MAX_SAFE_INTEGER : at
  }
  return readdirSync(notesDir)
    .filter((name) => name.endsWith('.md'))
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((name) => ({
      // The text before a colon is a short label; the page keeps the full title.
      text: firstHeading(join(notesDir, name)).split(':')[0],
      link: `/curriculum/notes/${name.replace(/\.md$/, '')}`
    }))
}

const notes = noteItems()
const curriculumSidebar: DefaultTheme.SidebarItem[] = [
  { text: 'Curriculum', items: [{ text: 'Phase plan and delivered slices', link: '/curriculum/' }] },
  { text: 'Derivation notes', items: notes }
]

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
  description:
    'Quant Pricing Lab: a numerical-methods lab for option pricing with independently checked cases and convergence evidence',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Curriculum', link: '/curriculum/' },
      { text: 'Notes', link: notes[0]?.link ?? '/curriculum/' },
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
      '/curriculum/': curriculumSidebar,
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Overview', link: '/reference/' },
            { text: 'Pricing', link: '/reference/pricing' },
            { text: 'Cases', link: '/reference/cases' },
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
