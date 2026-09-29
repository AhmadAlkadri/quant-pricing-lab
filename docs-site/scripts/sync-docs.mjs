// Copy the repository's docs/ (the single source of truth) into the VitePress
// source tree before dev/build. The output folder is generated and gitignored;
// never edit it by hand.
//
//   docs/CURRICULUM.md   -> docs-site/docs/curriculum/index.md
//   docs/notes/*.md      -> docs-site/docs/curriculum/notes/*.md
//
// Relative Markdown links are kept when they point at another copied file (a
// note linking a sibling note) and rewritten to the file on GitHub otherwise
// (e.g. a link into src/ or tests/), so the VitePress dead-link check stays on.
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, posix, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_BLOB = 'https://github.com/AhmadAlkadri/quant-pricing-lab/blob/main/'

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = resolve(siteRoot, '..')
const outRoot = join(siteRoot, 'docs', 'curriculum')

// source path (repo-relative, posix) -> output path (relative to outRoot)
const mapping = new Map([['docs/CURRICULUM.md', 'index.md']])
for (const name of readdirSync(join(repoRoot, 'docs', 'notes')).sort()) {
  if (name.endsWith('.md')) mapping.set(`docs/notes/${name}`, `notes/${name}`)
}

function rewriteLinks(text, sourcePath, outPath) {
  // Leave fenced code blocks alone.
  return text
    .split(/(^```[\s\S]*?^```)/m)
    .map((chunk, i) => {
      if (i % 2 === 1) return chunk
      return chunk.replace(/\]\(([^)\s]+)\)/g, (match, target) => {
        if (/^([a-z]+:|#|\/)/i.test(target)) return match
        const [path, hash = ''] = target.split('#')
        const repoPath = posix.normalize(posix.join(posix.dirname(sourcePath), path))
        const suffix = hash ? `#${hash}` : ''
        if (mapping.has(repoPath)) {
          let rel = posix.relative(posix.dirname(outPath), mapping.get(repoPath))
          if (!rel.startsWith('.')) rel = `./${rel}`
          return `](${rel}${suffix})`
        }
        return `](${REPO_BLOB}${repoPath}${suffix})`
      })
    })
    .join('')
}

rmSync(outRoot, { recursive: true, force: true })
for (const [source, out] of mapping) {
  const target = join(outRoot, out)
  mkdirSync(dirname(target), { recursive: true })
  const text = readFileSync(join(repoRoot, source), 'utf8')
  const banner =
    `<!-- Generated from ${source} by docs-site/scripts/sync-docs.mjs. Edit the source, not this file. -->\n\n`
  writeFileSync(target, banner + rewriteLinks(text, source, out))
}

if (!existsSync(join(outRoot, 'index.md'))) throw new Error('sync-docs: curriculum not written')
console.log(`sync-docs: ${mapping.size} files -> ${relative(siteRoot, outRoot)}/`)
