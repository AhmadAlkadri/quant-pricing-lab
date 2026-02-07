---
name: qpl-thin-slice
description: Use when implementing or modifying code in Quant Pricing Lab, especially pricing engines, public API dispatch, tests, docs, or notebooks. Apply the repo's thin-vertical-slice workflow: read `.agents/brain` first, preserve numerical and determinism invariants, update tests and docs together, and finish with a clean git tree and small commits.
---

# QPL Thin Slice Delivery

## Inputs and assumptions
- Work from the repository root.
- Treat `.agents/brain/brain.md` as the architectural and invariants source of truth.
- Treat `.agents/brain/adr/0001-public-api-truth-source.md` as the public API contract.
- Use Python >= 3.10 and project tooling from `pyproject.toml`.

## Procedure

1. Frame the change before editing.
- Read `README.md`, `AGENT.md`, and `.agents/brain/brain.md`.
- If the change can affect API or architecture, read relevant ADRs in `.agents/brain/adr/`.
- Define one thin vertical slice with clear non-goals.

2. Confirm constraints and invariants.
- Keep scope within European options + Black-Scholes unless the task explicitly expands scope.
- Preserve deterministic behavior for Monte Carlo (seeded) and PDE outputs.
- Preserve validation and error style (`InvalidInputError`, `NotSupportedError`) at boundaries.

3. Implement minimally.
- Change only the files needed for the requested slice.
- Prefer the stable entry points (`qpl.pricing.price`, `qpl.pricing.greeks`) when wiring behavior.
- Avoid broad refactors unless they are required to ship the slice.

4. Update tests with the behavior.
- Add or adjust focused tests in `tests/` for the new behavior.
- Cover edge cases affected by the change (for example: `T=0`, `sigma=0`, invalid inputs).
- Keep tests deterministic.

5. Validate locally.
- Run:
```bash
pytest -q
```
- If relevant, run example smoke checks:
```bash
python examples/bs_analytic.py
python examples/bs_mc_vs_analytic.py
```
- If notebooks changed, ensure outputs are stripped and naming remains `NN_description.ipynb`.

6. Update project brain and docs when required.
- Update `.agents/brain/steering-brief.md` for meaningful code changes.
- Update `.agents/brain/brain.md` only when architecture/invariants or key workflows changed.
- Add or update an ADR if public API or architecture changed.

7. Prepare delivery.
- Summarize what changed, why, and how it was validated.
- Keep commits small and logically grouped.
- Do not report completion unless `git status` is clean.

## Guardrails
- Do not invent requirements not supported by repo artifacts.
- Do not weaken numerical checks or determinism without documenting the reason.
- Do not leave TODOs in the critical path.
- Do not merge unrelated cleanup into the same slice.
- Do not finish with uncommitted or unstaged leftovers.

## Output expectations
- Provide a concise change summary with file paths.
- Report exact validation commands run and outcomes.
- State any residual risks or follow-up work explicitly.
