# AGENTS.md

This is the canonical operator guide for agent-driven work in this repository.

## Development Phase Policy (Pre-1.0)
- qpl is in exploratory pre-1.0 mode.
- API churn is expected and acceptable.
- Labs are the authoritative design driver.
- Refactors are encouraged when clarity improves.
- Backward compatibility is not guaranteed.
- The only invariant is always-green delivery with deterministic labs and a clean git tree.

## Operating Policy
- Lab-first development: labs drive package changes.
- Thin vertical slices only: each slice ships runnable behavior + tests.
- Always-green workflow: after every commit, run `ruff check .` and `pytest -q`.
- Deterministic labs: every stochastic notebook must be seedable and CI-friendly with `QPL_LAB_SMOKE=1`.
- End-of-task hygiene: do not report completion unless `git status --short` is empty.

## Source of Detailed Guidance
- Project brain and architecture: `.agents/brain/`
- ADRs: `.agents/brain/adr/`
- Skills and workflows: `.agents/skills/`

## Golden Path Commands
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
ruff check .
pytest -q
```

```bash
QPL_LAB_SMOKE=1 MPLBACKEND=Agg pytest -q tests/test_labs_smoke.py
```

Headless notebook example:
```bash
PYTHONPATH=src MPLBACKEND=Agg QPL_LAB_SMOKE=1 \
python -m jupyter nbconvert --to notebook --execute labs/01_static_monte_carlo.ipynb \
  --output 01_static_monte_carlo.executed.ipynb --output-dir /tmp
```
