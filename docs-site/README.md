# docs-site

Minimal VitePress site for qpl public documentation.

## Local development

```bash
cd docs-site
npm install
npm run docs:dev
```

## Local build + preview

```bash
cd docs-site
npm install
npm run docs:build
npm run docs:preview
```

## Build with project-pages base path

Use this when you want local output matching GitHub Pages project-site paths.

```bash
cd docs-site
BASE=/quant-pricing-lab/ npm run docs:build
```

## One-command sanity check

```bash
cd docs-site
npm run docs:build && npm run docs:preview
```
