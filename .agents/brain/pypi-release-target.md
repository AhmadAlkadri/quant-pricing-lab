# PyPI release target (deferred)

Written 2026-09-29 during the v0.3.0 wrap-up. Status: **specification only.**

- The v0.3.0 consolidation and merge to `main` are **independent of PyPI
  readiness**. v0.3.0 is a source/repository milestone (`CHANGELOG.md`).
- PyPI publication is **deferred** to a future campaign that the owner must
  authorize explicitly. This file is the target for that campaign; it is
  **not** authority to begin it.
- The numerical milestone needs **no additional curriculum features** before
  packaging work can proceed. Everything below is packaging, evidence and
  account work.

Status labels: **gap** (observed, needs work), **done** (observed satisfied),
**unverified** (an open question to answer when the campaign begins).
Observations are from `0ffef19` on 2026-09-29 unless stated. "Probe" means a
throwaway wheel + sdist built in a scratch directory from `git archive 0ffef19`
with isolated build (setuptools 84.0.0 resolved); nothing was committed or
uploaded.

## A. Distribution identity and metadata

| # | Item | Status | Files | Smallest implementation | Acceptance check |
|---|---|---|---|---|---|
| A1 | Index name and ownership | **unverified**: availability of `qpl` and `quant-pricing-lab` on PyPI and TestPyPI was not checked, by design | `pyproject.toml` | At campaign start, look up both names on both indexes; record result and date here. Preferred candidate distribution name `quant-pricing-lab`, **import name stays `qpl`**. Not assumed available; do not rename now | Name decision recorded with the lookup date; `[project].name` matches it; `import qpl` unchanged |
| A2 | Description / readme | **done** (description and `readme = "README.md"` present) | `pyproject.toml`, `README.md` | Re-read README for install text once the index name exists ("not published on PyPI" must change on publication, not before) | `twine check dist/*` passes; README install section matches reality |
| A3 | License metadata | **gap**: no `license` field; `LICENSE` (MIT) is auto-picked up as `License-File` only | `pyproject.toml`, `LICENSE` | `license = "MIT"` (SPDX) + `license-files = ["LICENSE"]`; raise `build-system.requires` to a setuptools that supports PEP 639 (>= 77) | Built METADATA has `License-Expression: MIT` and `License-File: LICENSE`; `twine check` passes |
| A4 | Project URLs | **gap**: none | `pyproject.toml` | `[project.urls]` Homepage/Source/Changelog (+ Documentation only once Pages is live) | METADATA lists `Project-URL` lines; each resolves |
| A5 | Authors, classifiers | **gap**: no authors, no classifiers | `pyproject.toml` | `authors`; classifiers for `Development Status :: 3 - Alpha` (pre-1.0, ADR-0003), supported Python versions, OS, topic | METADATA carries them; classifier Python versions equal the C matrix |
| A6 | Python requirement | **done** as declared (`>=3.10`); evidence is C1 | `pyproject.toml` | None unless C changes it | Same value in `requires-python`, classifiers, ruff `target-version` |
| A7 | Version reconciliation | **done for 0.3.0** (`pyproject` = `qpl.__version__` = 0.3.0) but **gap** in enforcement: nothing keeps them aligned (brain.md, "Version sync risk") | `pyproject.toml`, `src/qpl/__init__.py`, tags | Either single-source the version (e.g. `importlib.metadata`) or a test asserting equality. If any packaging change lands after `v0.3.0`, publish as a **new** version and tag (0.3.1 or 0.4.0); **never move `v0.3.0`** | Installed `importlib.metadata.version(<dist>) == qpl.__version__ ==` tag name without `v` |

## B. Explicit, inspected distribution contents

| # | Item | Status | Files | Smallest implementation | Acceptance check |
|---|---|---|---|---|---|
| B1 | Build from a clean candidate | **gap** (no procedure) | none | Build wheel + sdist once, from a fresh clone at the candidate SHA, with `python -m build` | `dist/` holds exactly one `.whl` and one `.tar.gz` named for the version |
| B2 | All modules, including subpackages | **done in probe, fragile**: wheel held all 82 tracked `src/` files, but `src/qpl/engines/mc/` and `src/qpl/engines/pde/` have **no `__init__.py`** and ship only through setuptools' namespace auto-discovery | `src/qpl/engines/{mc,pde}/`, `pyproject.toml` | Add the two `__init__.py` files (or pin discovery in `[tool.setuptools.packages.find]`) | Wheel file list compared against `git ls-files src` shows no missing module; `import qpl.engines.mc.heston, qpl.engines.pde.barrier` from the installed wheel |
| B3 | Runtime resources | **done**: no non-`.py` files under `src/`; no `importlib.resources` / file reads in `src/` | `src/qpl/` | None now; add package-data explicitly if a resource ever appears | `git ls-files src \| grep -v '\.py$'` empty, or each hit is in the wheel |
| B4 | Private and local material excluded | **done in probe** (sdist top level: `src/`, partial `tests/`, `LICENSE`, `README.md`, `pyproject.toml`, `PKG-INFO`, `setup.cfg`; no `.agents/`, `labs/`, `textbooks/`, `docs-site/`, `.venv`, caches). **Unverified** for a build run inside a working checkout (untracked files, stale `*.egg-info`) | `MANIFEST.in` (absent) | Build only from a fresh clone (B1); add a `MANIFEST.in` with explicit `prune` lines if the tree ever gains a tool that includes all tracked files (e.g. setuptools-scm) | `tar -tzf` and `unzip -l` listings contain none of `labs/ textbooks/ .agents/ docs-site/ node_modules .venv .market_cache __pycache__ .pypirc` |
| B5 | Tests/examples in the sdist | **gap**: the probe sdist carries top-level `tests/test_*.py` only (65 of 92 tracked test files) -- `tests/cases/`, `tests/oracle/` and helper modules (`tests/fourier_points.py`, `tests/heston_points.py`) are missing, so the shipped tests are a broken subset. `examples/` is not included | `MANIFEST.in`, `tests/` | Decide: either ship the whole `tests/` tree (`graft tests`) or none (`prune tests`). Record the decision here | Decision recorded; if shipped, `pytest -q -m "not slow"` passes from the unpacked sdist |
| B6 | No dependence on private workbooks or checkout paths | **done** (observed): `src/` does not reference `labs/`, `textbooks/`, `__file__`-relative repo paths or `parents[...]`; the only relative path is `qpl.market.data`'s `cache_dir=".market_cache"` default, CWD-relative and `[data]`-only | `src/qpl/market/data.py` | None required; optionally document the cache location | `git grep -nE "__file__\|parents\[\|[\"'/]labs/?[\"'/]\|textbooks" -- src/` returns nothing (true at `0ffef19`; `qpl.utils.labs` is only the `QPL_LAB_SMOKE` helper), and the market cache default is reviewed and accepted |

## C. Supported-environment evidence

| # | Item | Status | Files | Smallest implementation | Acceptance check |
|---|---|---|---|---|---|
| C1 | Python matrix | **gap**: CI runs Python 3.11 only; 3.10 (the declared minimum) is never tested | `.github/workflows/ci.yml` | Matrix over a small explicit set decided at campaign start (e.g. 3.10/3.11/3.12/3.13) for `tests-core` | Green CI run on every listed version on the candidate SHA |
| C2 | Operating systems | **gap**: CI is `ubuntu-latest` only; macOS evidence exists only as local runs | `ci.yml` | Add one `macos-latest` core job, or record a local macOS run on the candidate | Linux and macOS both green on the candidate SHA, recorded with versions |
| C3 | Extras stay optional | **done** in CI (`tests-core` without `data`/`oracle`) | `ci.yml`, `tests/oracle/conftest.py` | Keep; smoke D must also run on core install only | Core install imports `qpl` and passes the D smoke with neither extra installed |
| C4 | Support-declaration changes | **unverified** (no change proposed) | `pyproject.toml`, `CHANGELOG.md` | If the minimum Python or OS claims change, justify it in the CHANGELOG and README | Declaration, classifiers and CI matrix agree |

## D. Installed-artifact checks

| # | Item | Status | Files | Smallest implementation | Acceptance check |
|---|---|---|---|---|---|
| D1 | Fresh-venv installs outside the checkout | **gap** (never done; CI uses editable installs) | none (or a short `scripts/` smoke) | Install the wheel, then separately the sdist, into fresh venvs from a directory outside the repo, **no `PYTHONPATH=src`** | `python -c "import qpl; print(qpl.__file__)"` points into site-packages |
| D2 | Metadata and dependency health | **gap** | none | `importlib.metadata.version` check; `pip check` | Version equals A7; `pip check` reports no broken requirements |
| D3 | Bounded numerical smoke | **gap** | a small smoke script, kept out of the package | (i) analytic BS European call vs a known value (e.g. S=K=100, r=5%, sigma=20%, T=1: 10.4506 to 1e-4); (ii) one cross-method comparison within a stated tolerance (e.g. `method="pde"` or `"tree"` vs `"analytic"`); (iii) import and evaluate one `qpl.cases` row | Script exits 0 in under a minute on both artifacts; tolerances written in the script. No CLI is required |

## E. One authorized publication path

| # | Item | Status | Files | Smallest implementation | Acceptance check |
|---|---|---|---|---|---|
| E1 | Choose the path | **unverified** (not chosen) | none | Choose **one**: manual Twine upload, or a Trusted Publisher. Automated publishing is optional. A workflow file is not evidence that the account side is configured | Choice recorded here with the reason |
| E2 | Owner-side configuration | **unverified**, owner-only | index accounts (not in repo) | Owner creates/claims the project, 2FA/API token or Trusted Publisher binding on TestPyPI (optional) and PyPI. Agents do not touch accounts or credentials | Owner confirms in writing; agent verifies only what is visible publicly |
| E3 | Build once, freeze, promote identical files | **gap** (no procedure) | none | Build per B1; record SHA-256 of both files; upload those exact files to TestPyPI (optional rehearsal) then PyPI, then optionally attach to a GitHub release. Never rebuild between stages; never re-upload a version (a fix is a new version) | Checksums of downloaded index files equal the frozen ones at every stage |
| E4 | Verify from the index | **gap** | none | Repeat D1-D3 installing `<dist>==<version>` from the index | Same results as D against the local files |
| E5 | Record | **gap** | a dated record under `.agents/brain/` | Record version, commit SHA, tag, file names, SHA-256s, index URLs, validation results | Record committed after the tag, not moving it |

Implementation work (A-D, E3-E5 procedure) is agent work in normal slices.
Owner-side work (A1 final name choice, E1 choice, E2 accounts, and the
**authorization to publish**) is not, and no agent step may substitute for it.

## Proposed slice order for the future campaign

1. Identity: A1 name lookup and decision, A3-A5 metadata, A7 version single
   source. One commit plus tests.
2. Contents: B2 `__init__.py` files, B5 tests-in-sdist decision (`MANIFEST.in`),
   B1/B4 build-and-inspect from a clean clone.
3. Environments: C1/C2 CI matrix and macOS evidence.
4. Artifacts: D1-D3 smoke on wheel and sdist.
5. Owner gate: E1 and E2; then E3-E5 publication and record under a new
   version (0.3.1 or 0.4.0) and tag.

## Completion condition

The target is met when one version of the distribution is on PyPI whose files
were built once from a tagged commit on `main`, whose SHA-256s match the
recorded ones, whose metadata satisfies A, whose contents were inspected per B,
which passed C on the declared matrix and D from the index in a fresh venv, and
whose record (E5) is committed. Until the owner authorizes publication, the most
an agent campaign can reach is "ready to publish": A-D satisfied and E3 rehearsed
locally.

## Out of scope for this target

- QMC, multi-asset, rates, further exotics, additional Heston/calibration research.
- Performance optimization and benchmarks.
- API redesign or any 1.0 stability promise.
- CLI or service development.
- Broad provenance/legal investigation absent a concrete new finding.
- Copying Chemical-Thermodynamics' full release machinery (release packets,
  history-rewrite reports, ADR sets); take only the build-once/checksum habit.
