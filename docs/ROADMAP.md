
### `docs/ROADMAP.md`
```markdown
# Roadmap

## Milestone A — Monte Carlo Engine
- [ ] GBM path generator (exact discretization)
- [ ] European option pricing
- [ ] Greeks: pathwise + likelihood ratio (at least Delta)
- [ ] Variance reduction: antithetic + control variates
- [ ] Convergence plots and runtime benchmarks

## Milestone B — PDE Engine (Finite Differences)
- [ ] Black–Scholes PDE setup
- [ ] Theta scheme (explicit/implicit/Crank–Nicolson)
- [ ] Boundary conditions + stability notes
- [ ] Greeks from grid
- [ ] Compare vs analytic BS and Monte Carlo
- [ ] (Stretch) American put with PSOR

## Milestone C — Case Study
- [ ] Barrier option: MC vs PDE tradeoffs
