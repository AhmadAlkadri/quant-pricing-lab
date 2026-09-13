# Curriculum

`qpl` is built by Textbook-Driven Development (TDD): a textbook or literature
result is re-derived independently, turned into an executable pricing case,
cross-checked by an independent method, backed by convergence or statistical
evidence, and only then becomes a package capability. This page is the
truthful map of that process: the loop, the spine, the dependency order, the
evidence rules, the phase plan, and what each slice has actually delivered.

See also `docs/notes/` for short public derivation notes and
`docs/curriculum_provenance.md` for the per-chapter lab -> `src/` -> tests
mapping.

## The TDD loop

1. **Derive**: re-derive the result independently in a private lab notebook
   under `labs/` (gitignored, never published), citing source/page/equation
   for load-bearing formulas.
2. **Case**: turn it into an executable pricing case in `qpl.cases`, paraphrased
   in this repo's own words and data.
3. **Cross-check**: validate against an independent method (closed form,
   another engine, a published table, or an identity that must hold exactly).
4. **Evidence**: back the claim with convergence-order or statistical evidence
   using `qpl.validation`, tagged with an `EvidenceClass`.
5. **Capability**: only then does it become public `src/` surface with tests
   and, where a plot teaches better than an assertion, a notebook.

Labs are the private *drivers*; `src/`, `tests/`, `qpl.cases`, and
`docs/notes/` are the public *cargo*. No human gating on derivations (D6):
the loop above is the check, not a review queue.

## Identity decision

The project's tie-breaker (D1, 2026-09-13): `qpl` is a numerical-methods lab
with textbook provenance. Correctness and measured convergence beat
instrument breadth, API polish, and performance. Recruiting/portfolio
considerations do not reorder the curriculum (D2).

## Literature spine

| Role | Source |
|---|---|
| Heavy spine (cases) | Fusai & Roncoroni (2008), *Quantitative Finance*, Part II |
| Heavy spine (Monte Carlo) | Glasserman (2003), *Monte Carlo Methods in Financial Engineering* |
| Derivation authority | Shreve, *Stochastic Calculus for Finance II* |
| PDE rigor | Pooley, Forsyth & Vetzal papers; Tavella & Randall / Duffy |
| Instrument/concept map only | Hull, *Options, Futures, and Other Derivatives* |
| Volatility and transforms | Gatheral; Heston (1993); Lewis; Lord & Kahl; Carr & Madan |
| Rates (only if rates become serious) | Andersen & Piterbarg |
| Independent oracle (optional, `[oracle]` extra, never sole evidence) | QuantLib-Python (BSD-3) |

## Method dependency graph

```mermaid
graph LR
    F[Foundations] --> BS[BS analytic]
    BS --> Trees[Trees]
    BS --> PDE[PDE]
    BS --> MC[Monte Carlo]
    Trees --> American[American exercise:\ntree / PDE / LSM]
    PDE --> American
    MC --> American
    BS --> Transforms[Transforms]
    Transforms --> Heston[Heston]
    Heston --> Calibration[Calibration]
    American --> Exotics[Exotics as method drivers]
    Calibration --> Exotics
    Exotics --> MultiAsset[Multi-asset, rates, performance]
```

Compact list form: foundations -> BS analytic -> {trees, PDE, MC} -> American
exercise (via tree / PDE / LSM) -> {transforms -> Heston -> calibration} ->
exotics as method drivers -> {multi-asset, rates, performance}.

## Evidence classes

Defined in `qpl.validation.EvidenceClass`:

- **exact identity** — a relation that holds exactly in the model (e.g. put-call
  parity); tolerance is floating-point round-off, not model error.
- **closed form** — compared against a closed-form formula evaluated in this
  repository.
- **published benchmark** — compared against a number published in a cited
  source.
- **independent engine** — compared against a different engine or library
  computing the same quantity by a different route.
- **convergence order** — the claim is about the measured rate at which error
  decays under refinement, not about a single price.
- **statistical** — the estimate carries sampling error; the tolerance is a
  stated multiple of the reported standard error.
- **negative finding** — a documented failure mode, pinned by a test so it
  cannot silently change or be mistaken for a bug elsewhere.

A fitted slope is never a theorem; Monte Carlo agreement within noise is
never exact; engine agreement is never evidence that shared assumptions are
realistic.

## Provenance rules

- Cite source/page/equation for load-bearing formulas.
- Derive independently; do not copy book code, prose, tables, or a book's
  example sequence.
- Paraphrase problem statements in this repository's own words.
- Published numbers may be used as fixtures, but only with a citation.

## Phase plan

Candidate slices, re-planned as cases expose problems. Each is phrased as "a
user can X by running Y, correct because Z" where that is already known;
unknowns stay unknown.

### Phase 1 — Discrete time (trees)
- A user will be able to price a European call/put via a CRR binomial tree,
  correct because its measured convergence order to the Black-Scholes closed
  form is order 1, with the odd/even node-count oscillation documented rather
  than hidden.
- American exercise becomes an instrument property; CRR American call/put
  with dividends is checked by early-exercise premium >= 0 and American >=
  European.
- Leisen-Reimer smoothing is checked by measured order 2.
- Tree Greeks are read from the lattice.
- Forces ADR-0005: an engine registry keyed by `(instrument, model, method)`
  replacing the `isinstance` ladder in `qpl.pricing`.

### Phase 2 — PDE rigor
- Rannacher start-up, with Greeks read from the grid at measured order.
- A non-uniform grid concentrated at the strike, as the second standard
  remedy for the strike-kink pathology documented in
  `docs/notes/pde_strike_alignment.md`.
- PSOR for American exercise, cross-checked against the Phase 1 trees.
- A digital (discontinuous-payoff) option, showing the pathology and the
  remedies rather than only the smooth-payoff case.
- `qpl.numerics.linear_systems` used where it earns its place, or a recorded
  reason why not.

### Phase 3 — Monte Carlo (Glasserman)
- Explicit stderr/CI discipline throughout.
- Variance reduction — antithetic variates, a control variate (Kemna-Vorst
  geometric Asian as control for arithmetic Asian), and stratification — each
  checked by a measured variance ratio.
- Euler vs Milstein strong/weak order measured against exact GBM.
- Pathwise and likelihood-ratio Greeks checked against the analytic engine.
- Longstaff-Schwartz American put validated against Longstaff & Schwartz
  (2001) Table 1 and against the Phase 1/2 engines — the first three-way
  cross-method case.
- Barrier monitoring bias checked against the Reiner-Rubinstein closed form.
- QMC only if a case motivates it.

### Phase 4 — Transforms and volatility
- Fourier pricing of Black-Scholes as a sanity check.
- The Heston characteristic function in the branch-cut-safe (Lewis / Lord-Kahl)
  form.
- Lewis and Carr-Madan pricing validated against published reference values
  and, optionally, QuantLib.
- Heston Monte Carlo with the QE scheme and a bias study.
- Implied-vol surface utilities.
- Synthetic-recovery calibration with identifiability diagnostics.
- Fusai & Roncoroni Ch. 15 Laplace approach to arithmetic Asians, reusing
  `qpl.transforms`.

### Phase 5+ — only when forced by a case
- Multi-asset (correlated Brownian motion, baskets via existing copulas).
- Rates (Vasicek/CIR/Hull-White).
- Performance: profiling, vectorized kernels, and a benchmark harness with
  reproducibility metadata (instrument, model, engine, grid/paths, tolerance,
  hardware, seed, wall time, error vs reference) — only after reference paths
  exist to benchmark against.

## Delivered

### Slice 0
- Public branch without `labs/`; extras split into `dev`/`data`/`oracle`
  (`pyproject.toml`).
- `qpl.validation`: `fit_convergence_order` (least-squares slope of
  `log(err)` on `log(h)`), `refinement_errors`, and the `EvidenceClass` /
  `BenchmarkRow` taxonomy above.
- `PDEConfig(strike_alignment="midpoint")`, resolving the strike-on-a-node
  pathology. Measured on `S=K=100, r=5%, q=0, sigma=20%, T=1`, European call,
  Crank-Nicolson, `n_s = n_t = n` for `n` in `{50, 100, 200, 400, 800}`:
  - Aligned: fitted order **1.997**, log-space RMS residual **0.022**.
  - Unaligned: fitted order **1.126**, log-space RMS residual **0.860** (the
    high residual reflects a non-monotone error sequence, not just noise —
    refining `n=50 -> 100` makes the unaligned error five times worse because
    the strike lands exactly on a node at `n=100`).
  - Implicit-Euler temporal order, isolated on a fixed aligned spatial grid
    (`n_s=800`): fitted order **0.997**, residual **3e-04**.
  - Full tables and the derivation: `docs/notes/pde_strike_alignment.md`.
- `qpl.cases.european_black_scholes`: parity, degenerate-limit, closed-form
  reference-value, and comparative-statics cases, each carrying a
  `BenchmarkRow` with an explicit `EvidenceClass`; exercised by a cross-engine
  test (`tests/cases/test_european_black_scholes_cases.py`).

## Reconciled old roadmap

`docs/ROADMAP.md` is superseded by this page. Mapping of the old roadmap
items to their actual status:

| Old roadmap item | Status |
|---|---|
| Implied-volatility solver | Done: `qpl.engines.analytic.black_scholes.implied_volatility`. |
| Implied-vol teaching notebook | Exists as a legacy notebook (`notebooks/02_implied_vol_teaching.ipynb`). |
| Historical vol / market data / implied-vs-realized / return-model fit | Done, but frozen behind the `[data]` extra (D9) — outside this curriculum. |
| `BinaryOption` | Deferred to the Phase 2 discontinuous-payoff PDE/MC slice, where it drives a method rather than being added for its own sake. |
| Antithetic variates | Phase 3, first variance-reduction slice, measured by variance ratio. |
| Benchmark harness | Phase 5, only after reference paths exist to benchmark against. |

## Out of scope for this campaign

- Market-data expansion beyond the frozen `[data]` extra.
- A production trading framework.
- A universal pricing ontology built ahead of the cases that would justify it.
- Performance work before reference paths exist (Phase 5+ only).
- Reordering the curriculum for recruiting/portfolio purposes (D2).
