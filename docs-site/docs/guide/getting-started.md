# Getting Started

## Install

`qpl` is installed from a clone of the repository (it is not published on PyPI):

```bash
git clone https://github.com/AhmadAlkadri/quant-pricing-lab.git
cd quant-pricing-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras: `.[oracle]` (QuantLib-Python, for the oracle tests) and `.[data]`
(market-data scripts outside the curriculum).

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
