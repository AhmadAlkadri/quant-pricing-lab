---
name: notebook-authoring
description: Use when creating or editing notebooks in this repository. Enforces reproducible, smoke-testable notebooks with controlled output size and deterministic seeds.
---

# Notebook Authoring

## Inputs and assumptions
- Work from the repository root.
- Use project dependencies from `pyproject.toml`; do not add notebook-only packages unless justified.
- Keep notebooks runnable headlessly in CI-like mode.

## Rules
1. Reproducibility
- Every stochastic cell must use explicit, deterministic seeds.
- Support a smoke mode via environment variable when runtime could be large.
- Avoid hidden state assumptions across out-of-order execution.

2. Output size
- Keep outputs concise; avoid huge tables, large binary displays, or excessive print logs.
- Favor small sample sizes in smoke mode and larger sizes only in full mode.
- Ensure notebook outputs are strip-friendly with `nbstripout`.

3. Structure
- Use short narrative markdown blocks with clear section headers.
- Place configuration and seed controls near the top.
- End teaching notebooks with concrete exercise prompts.

4. Testing contract
- Add a deterministic smoke test that executes the notebook headlessly with:
```bash
python -m jupyter nbconvert --to notebook --execute <notebook> --output <file> --output-dir <tmp>
```
- Set `PYTHONPATH=src`, `MPLBACKEND=Agg`, and any notebook smoke env var in the test.
- Assert successful return code and generated output notebook file.

## Guardrails
- Do not rely on network calls in notebooks unless the task explicitly requires them.
- Do not add mutable global state that makes reruns non-deterministic.
- Do not ship notebooks that fail when executed from a clean kernel.
