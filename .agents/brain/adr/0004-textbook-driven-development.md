# ADR-0004: Textbook-Driven Development

Status: accepted
Date: 2026-09-13

Context
- Labs drove Part I (Chapters 1-8) of the build, but the resulting toolkit
  (sampling, DP, linear solvers, quadrature, Laplace inversion, copulas)
  never met the pricing core: it accumulated numerical building blocks
  without cases that price an instrument and check the answer.
- Lab notebooks work private textbook problems and, in places, adapt
  textbook example sequences; publishing them as-is raises a copyright/consent
  concern the project owner is not willing to carry.
- Existing tests were sanity checks (shapes, determinism, "does not raise"),
  not validation: they did not, in general, state what evidence justified an
  expected number.
- The project's identity tie-breaker (D1, 2026-09-13) is now settled: `qpl`
  is a numerical-methods lab with textbook provenance, where correctness and
  measured convergence beat instrument breadth, API polish, and performance.

Decision
- Labs (`labs/`) stay private, gitignored, and remain the authoritative
  design driver (unchanged from ADR-0003); they are never published.
- Public cargo is: `src/`, `tests/`, the `qpl.cases` layer, and short
  derivation notes under `docs/notes/`. Notebooks are published only where a
  plot teaches better than an assertion.
- Provenance rules apply to all public cargo: cite source/page/equation for
  load-bearing formulas; derive independently; paraphrase problem statements;
  published numbers may be used as fixtures but only with a citation; never
  copy book code, prose, tables, or a book's example sequence.
- Every numerical or statistical claim in tests or notes must carry an
  `EvidenceClass` (`qpl.validation.EvidenceClass`): exact identity, closed
  form, published benchmark, independent engine, convergence order,
  statistical, or negative finding. A fitted slope is never a theorem; MC
  agreement within noise is never exact; engine agreement is never evidence
  that shared assumptions are realistic.
- Numerical engines require empirical convergence-order evidence
  (`qpl.validation.fit_convergence_order`) before being treated as a
  package capability, not just a passing test.
- Branch/push policy: work happens on `dev/curriculum`; push only at slice
  boundaries; `main` is fast-forwarded by the project owner (D7/D8).
- QuantLib-Python is an optional independent oracle behind the `[oracle]`
  extra; it is never the sole evidence for a claim (D3).
- The market-data strand (`[data]` extra) is frozen and stays outside the
  curriculum (D9).

Alternatives considered
- Publish the labs as-is: rejected on the copyright/consent concern above,
  and because raw lab notebooks mix scratch work with the derivation that
  actually matters.
- A separate private repository for labs: rejected as unnecessary process
  overhead; a gitignored `labs/` directory in the same repository already
  gets the privacy guarantee without splitting history or tooling.
- A single-textbook spine: rejected because no one source covers trees, PDE
  rigor, Monte Carlo variance reduction, and transforms/volatility at the
  needed depth; the literature spine is deliberately a dependency graph of
  several sources (Fusai & Roncoroni; Glasserman; Shreve; Pooley/Forsyth/
  Vetzal and Tavella & Randall for PDE; Hull as a concept map only;
  Gatheral/Heston/Lewis/Lord & Kahl/Carr & Madan for transforms).
- Notebooks as the curriculum: rejected because notebooks are not tested as
  rigorously as `src/` + `tests/`, and reviewing "does the notebook still
  run" is weaker evidence than a pinned convergence-order assertion.

Consequences
- Every new capability must trace to a case, a cross-check, and evidence
  before it counts as delivered; this slows headline feature count but
  removes untested numerical surface.
- Documentation (`docs/CURRICULUM.md`, `docs/curriculum_provenance.md`,
  `docs/notes/`) becomes the public record of what evidence backs each
  capability, replacing the old feature-list roadmap.
- Contributors must budget time for convergence studies alongside
  implementation; a numerical engine without a measured order is not done.
- ADR-0003 (pre-1.0 lab authority and API churn) remains active and
  compatible: labs still drive design and API churn is still acceptable
  pre-1.0; this ADR adds the public-cargo and evidence discipline on top.

Supersedes (optional)
- None.
