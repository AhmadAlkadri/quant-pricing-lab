# Getting Started

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run a public example

```bash
PYTHONPATH=src python examples/bs_analytic_greeks.py
```

## Execute public notebooks in smoke mode

```bash
PYTHONPATH=src MPLBACKEND=Agg QPL_LAB_SMOKE=1 pytest -q tests/test_notebooks_smoke.py
```

## Validate the project

```bash
ruff check .
pytest -q
```
