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

Quick inner loop (deselects the `slow` marker):
```bash
ruff check .
pytest -q -m "not slow"
```
`pytest -q` is unchanged and still runs everything -- it is the CI contract and
nothing is deselected there. `-m "not slow"` drops 50 cases and takes **120.5 s**
against the full suite's **214.0 s** (measured locally, 2434 tests). The marker
is applied by a measured rule, not by feel: a test function is `slow` when its
worst individual case (setup + call + teardown, read off `pytest --durations=0`)
exceeds about 3 s. Parametrised functions carry the mark as a whole. Everything
it selects is a subprocess or notebook harness (`tests/test_examples_smoke.py`,
`tests/test_labs_smoke.py`, `tests/test_notebooks_smoke.py`) plus two
simulation-heavy American tests, so the quick loop still runs every engine,
case and oracle assertion.

```bash
QPL_LAB_SMOKE=1 MPLBACKEND=Agg pytest -q tests/test_labs_smoke.py
```

Headless notebook example:
```bash
PYTHONPATH=src MPLBACKEND=Agg QPL_LAB_SMOKE=1 \
python -m jupyter nbconvert --to notebook --execute labs/01_static_monte_carlo.ipynb \
  --output 01_static_monte_carlo.executed.ipynb --output-dir /tmp
```
