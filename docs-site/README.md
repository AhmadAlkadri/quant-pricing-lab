# docs-site

Minimal VitePress site for qpl public documentation.

## Where the content comes from

- `docs/guide/` and `docs/reference/` are hand-written site pages.
- `docs/curriculum/` is **generated** (and gitignored) from the repository's `docs/`
  folder by `scripts/sync-docs.mjs`: `docs/CURRICULUM.md` becomes `curriculum/index.md`
  and every `docs/notes/*.md` becomes `curriculum/notes/*.md`. Edit the files under the
  repository's `docs/`, never the copies.

A copy step was chosen over VitePress `<!--@include: -->` wrapper pages because it needs
no hand-written page per note: a new note under `docs/notes/` gets a page and a sidebar
entry (titled from its first `# ` heading, ordered by where `docs/CURRICULUM.md`'s
Delivered section first cites it) with no change here.

Relative links inside the copied files are kept when they point at another copied file
and rewritten to the file on GitHub (`blob/main`) otherwise, so links into `src/` or
`tests/` never become dead links. VitePress's dead-link check stays on.

The `docs:dev` and `docs:build` scripts run the sync first; `npm run docs:sync` runs it
alone. In dev mode, re-run the sync (or restart) after editing a file under `docs/`.

## Local development

```bash
cd docs-site
npm ci
npm run docs:dev
```

## Build with project-pages base path

This is what `.github/workflows/deploy-docs.yml` runs on pushes to `main`:

```bash
cd docs-site
npm ci
BASE=/quant-pricing-lab/ npm run docs:build
BASE=/quant-pricing-lab/ npm run docs:preview   # serves http://localhost:4173/quant-pricing-lab/
```

Output goes to `docs/.vitepress/dist/` (gitignored).
