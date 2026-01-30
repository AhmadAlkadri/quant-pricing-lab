# Roadmap

## DONE
- [x] Milestone A: Analytic Black–Scholes pricer + Greeks + dispatcher + tests
- [x] Milestone B: Monte Carlo European pricer (GBM) + stderr + dispatcher + tests + example
- [x] Milestone C: PDE FD pricer for European under BS (theta-scheme CN/implicit, BCs, deterministic tests vs analytic)
- [x] Patch 1: MC test robustness + edge cases + roadmap update

## NEXT
1) MC Greeks (finite-difference first; then pathwise/LR as stretch)
2) Benchmark harness (pytest-benchmark or ASV)
3) Accuracy/regression baselines (analytic vs PDE vs MC; tolerances; edge cases)
4) CI/polish (ruff/pyright/pytest wiring, docs/examples)
