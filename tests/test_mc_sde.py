"""Scheme identities, determinism, and the CIR boundary, for `qpl.engines.mc.sde`.

Evidence classes (`qpl.validation.EvidenceClass`) used here:

- ``EXACT_IDENTITY`` for the algebraic relations between the schemes on GBM.
  These are statements about floating-point arithmetic on one sample, not
  about a law, so the tolerances are round-off tolerances -- stated as
  relative tolerances rather than exact equality, because the two sides
  evaluate the same quantity through different expression trees and the last
  bit is not portable.
- ``STATISTICAL`` for the CIR exact sampler's moments, whose tolerances are
  multiples of the sample's own standard error.
- ``NEGATIVE_FINDING`` for what plain Euler does to a square-root diffusion,
  and for what full truncation does and does not repair.

The convergence orders live in `tests/test_sde_convergence.py`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.cases import (
    CIR_NEGATIVITY_PATHS,
    CIR_NEGATIVITY_SEED,
    CIR_NEGATIVITY_STEPS,
    GBM_STUDY_SPEC,
    SDE_NEGATIVITY_CASES,
    SDECase,
)
from qpl.cases.sde_discretization import CIR_FELLER_OK, CIR_FELLER_VIOLATED
from qpl.engines.mc.processes import simulate_gbm_exact
from qpl.engines.mc.sde import (
    SDEModel,
    cir_expected_excess,
    cir_moments,
    cir_sde,
    cir_transition_parameters,
    coarsen_normals,
    gbm_sde,
    simulate,
    uniform_time_grid,
)
from qpl.exceptions import InvalidInputError

GBM_S0 = GBM_STUDY_SPEC.s0
GBM_MU = GBM_STUDY_SPEC.mu
GBM_SIGMA = GBM_STUDY_SPEC.sigma
GBM_T = GBM_STUDY_SPEC.expiry

# `CIR_FELLER_OK` has Feller number 4 kappa theta / xi^2 = 8.0 and
# `CIR_FELLER_VIOLATED` has 0.08; the Feller condition 2 kappa theta >= xi^2 is
# exactly "Feller number >= 2". Both specs, and the negativity settings and
# measured frequencies below, live in `qpl.cases.sde_discretization`.
CIR_V0 = CIR_FELLER_OK.v0
CIR_PARAMS = {
    "feller_ok": CIR_FELLER_OK.params,
    "feller_violated": CIR_FELLER_VIOLATED.params,
}


def _log_gbm_model() -> SDEModel:
    """``d log S = (mu - sigma^2/2) dt + sigma dW``: constant coefficients.

    Deliberately built without a ``state_floor``: the log price is
    unconstrained, and one of the validation tests below needs a model for
    which ``truncation='full'`` is meaningless.
    """
    drift = GBM_MU - 0.5 * GBM_SIGMA * GBM_SIGMA

    def a(x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), drift)

    def b(x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), GBM_SIGMA)

    def db(x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))

    return SDEModel(name="log_gbm", drift=a, diffusion=b, diffusion_derivative=db)


# ---------------------------------------------------------------------------
# (e) determinism and the untouched exact GBM sampler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["euler", "milstein", "exact"])
def test_gbm_schemes_are_deterministic_per_seed(scheme: str) -> None:
    """EXACT_IDENTITY: same seed, same bytes, for every scheme."""
    model = gbm_sde(GBM_MU, GBM_SIGMA)
    grid = uniform_time_grid(GBM_T, 8)
    a = simulate(model, GBM_S0, grid, 500, scheme=scheme, seed=4242)
    b = simulate(model, GBM_S0, grid, 500, scheme=scheme, seed=4242)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.normals, b.normals)
    assert a.values.shape == (500, 9)


def test_cir_exact_sampler_is_deterministic_per_seed() -> None:
    """EXACT_IDENTITY: the noncentral chi-square draws are seeded too."""
    model = cir_sde(**CIR_PARAMS["feller_violated"])
    grid = uniform_time_grid(1.0, 4)
    a = simulate(model, CIR_V0, grid, 2_000, scheme="exact", seed=11)
    b = simulate(model, CIR_V0, grid, 2_000, scheme="exact", seed=11)
    assert np.array_equal(a.values, b.values)
    assert a.normals.shape == (2_000, 0)  # the transition consumes no normals


def test_new_exact_scheme_reproduces_the_old_uniform_grid_sampler() -> None:
    """EXACT_IDENTITY: `simulate_gbm_exact` is unchanged, to round-off.

    The Slice 8 sampler draws ``n_paths`` normals per step; feeding the same
    draws to the new module's ``scheme='exact'`` has to reproduce it. Not
    bit-for-bit: the old path forms ``dt = t / n_steps`` once while the new one
    reads ``t_{i+1} - t_i`` off the grid, and those differ in the last bit. A
    relative tolerance is what is portable across platforms here.
    """
    n_paths, n_steps = 400, 12
    rng = np.random.default_rng(7)
    normals = np.column_stack([rng.normal(size=n_paths) for _ in range(n_steps)])

    old = simulate_gbm_exact(
        s0=GBM_S0, mu=0.04, sigma=GBM_SIGMA, t=GBM_T, n_steps=n_steps, n_paths=n_paths, seed=7
    )
    new = simulate(
        gbm_sde(0.04, GBM_SIGMA),
        GBM_S0,
        uniform_time_grid(GBM_T, n_steps),
        n_paths,
        scheme="exact",
        normals=normals,
    )
    assert np.allclose(new.values, old, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# (b) what Milstein on GBM actually is
# ---------------------------------------------------------------------------


def test_euler_on_log_gbm_is_the_exact_sampler() -> None:
    """EXACT_IDENTITY, and the slice statement's expectation corrected.

    The slice said "Milstein on GBM matches exp-of-Euler-on-log". It does not,
    and the reason is that exp-of-Euler-on-log is not an approximation at all:
    ``d log S = (mu - sigma^2/2) dt + sigma dW`` has **state-independent**
    coefficients, so the Euler step reproduces the increment of the log exactly
    and its exponential is the exact lognormal transition. Euler on the log has
    infinite order, not order one.
    """
    n_steps, n_paths = 16, 2_000
    grid = uniform_time_grid(GBM_T, n_steps)
    rng = np.random.default_rng(3)
    z = rng.normal(size=(n_paths, n_steps))

    log_euler = simulate(
        _log_gbm_model(), math.log(GBM_S0), grid, n_paths, scheme="euler", normals=z
    )
    exact = simulate(
        gbm_sde(GBM_MU, GBM_SIGMA), GBM_S0, grid, n_paths, scheme="exact", normals=z
    )
    assert np.allclose(np.exp(log_euler.values), exact.values, rtol=1e-12, atol=0.0)


def test_milstein_on_log_gbm_is_bit_for_bit_euler_on_log_gbm() -> None:
    """EXACT_IDENTITY: ``db/dx = 0``, so the correction term is exactly ``0``.

    This is why "add a Milstein correction" is not a universal improvement:
    on the log scale there is nothing to correct, and the first-order scheme
    is already exact.
    """
    n_steps, n_paths = 16, 500
    grid = uniform_time_grid(GBM_T, n_steps)
    rng = np.random.default_rng(3)
    z = rng.normal(size=(n_paths, n_steps))
    model = _log_gbm_model()
    euler = simulate(model, math.log(GBM_S0), grid, n_paths, scheme="euler", normals=z)
    milstein = simulate(model, math.log(GBM_S0), grid, n_paths, scheme="milstein", normals=z)
    assert np.array_equal(euler.values, milstein.values)


def test_milstein_gbm_is_the_quadratic_truncation_of_the_exact_factor() -> None:
    """EXACT_IDENTITY: the relation that *does* hold, to round-off.

    Write the exact log increment as ``u = (mu - sigma^2/2) h + sigma dW``, so
    the exact transition multiplies by ``exp(u)``. The Milstein step is

        X + mu X h + sigma X dW + (1/2) sigma^2 X (dW^2 - h)
          = X (1 + (mu - sigma^2/2) h + sigma dW + (1/2) sigma^2 dW^2)
          = X (1 + u + (1/2) sigma^2 dW^2),

    which is ``exp(u)`` truncated after the quadratic term *in dW only*: the
    genuine second-order Taylor polynomial ``1 + u + u^2/2`` also carries the
    ``h dW`` and ``h^2`` pieces of ``u^2``, and Milstein drops them because
    they are ``O(h^{3/2})`` and ``O(h^2)``. The identity asserted here is the
    multiplicative one, and it holds for every path and every step.
    """
    n_steps, n_paths = 16, 2_000
    h = GBM_T / n_steps
    grid = uniform_time_grid(GBM_T, n_steps)
    rng = np.random.default_rng(3)
    z = rng.normal(size=(n_paths, n_steps))
    milstein = simulate(
        gbm_sde(GBM_MU, GBM_SIGMA), GBM_S0, grid, n_paths, scheme="milstein", normals=z
    ).values

    dw = math.sqrt(h) * z
    u = (GBM_MU - 0.5 * GBM_SIGMA**2) * h + GBM_SIGMA * dw
    predicted = milstein[:, :-1] * (1.0 + u + 0.5 * GBM_SIGMA**2 * dw * dw)
    assert np.allclose(milstein[:, 1:], predicted, rtol=1e-13, atol=0.0)


def test_milstein_gbm_differs_from_the_exact_factor_at_third_order() -> None:
    """CONVERGENCE_ORDER (local): the residual of the truncation above.

    ``exp(u) - (1 + u + sigma^2 dW^2 / 2)`` is ``O(h^{3/2})`` per step, so the
    largest single-step relative gap should shrink by about ``2^{-3/2} = 0.354``
    when the step is halved. Measured over the same normals at three step
    counts; the assertion is on the ratio being in a band around that, not on
    a fitted slope, because a maximum over paths is not a smooth statistic.
    """
    n_paths = 4_000
    rng = np.random.default_rng(17)
    gaps = []
    for n_steps in (8, 16, 32):
        grid = uniform_time_grid(GBM_T, n_steps)
        z = rng.normal(size=(n_paths, n_steps))
        model = gbm_sde(GBM_MU, GBM_SIGMA)
        mil = simulate(model, GBM_S0, grid, n_paths, scheme="milstein", normals=z).values
        exact = simulate(model, GBM_S0, grid, n_paths, scheme="exact", normals=z).values
        ratio = mil[:, 1:] / mil[:, :-1]
        exact_ratio = exact[:, 1:] / exact[:, :-1]
        gaps.append(float(np.mean(np.abs(ratio - exact_ratio))))
    for coarse, fine in pairwise(gaps):
        assert 0.25 < fine / coarse < 0.45


# ---------------------------------------------------------------------------
# (c, first half) the CIR exact transition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected_df"), [("feller_ok", 8.0), ("feller_violated", 0.08)]
)
def test_cir_transition_parameters_and_moments(label: str, expected_df: float) -> None:
    """CLOSED_FORM: the ncx2 parameters and the moments they imply.

    ``d = 4 kappa theta / xi^2`` is the Feller number: ``d >= 2`` is exactly
    ``2 kappa theta >= xi^2``. The moments are recomputed here from the CIR
    conditional formulas rather than from ``c (d + lambda)``, so the test is a
    cross-check of the derivation in `cir_transition_parameters`, not a
    restatement of it.
    """
    t = 1.0
    params = CIR_PARAMS[label]
    df, nc, scale = cir_transition_parameters(CIR_V0, t, **params)
    assert math.isclose(df, expected_df, rel_tol=1e-12)
    assert (df >= 2.0) == (2.0 * params["kappa"] * params["theta"] >= params["xi"] ** 2)

    kappa, theta, xi = params["kappa"], params["theta"], params["xi"]
    decay = math.exp(-kappa * t)
    mean_direct = theta + (CIR_V0 - theta) * decay
    var_direct = CIR_V0 * (xi**2 / kappa) * (decay - decay**2) + theta * xi**2 * (
        1.0 - decay
    ) ** 2 / (2.0 * kappa)

    mean, var = cir_moments(CIR_V0, t, **params)
    assert math.isclose(mean, mean_direct, rel_tol=1e-12)
    assert math.isclose(var, var_direct, rel_tol=1e-12)
    assert math.isclose(mean, float(scale * (df + nc)), rel_tol=1e-12)


@pytest.mark.parametrize("label", ["feller_ok", "feller_violated"])
def test_cir_exact_sampler_reproduces_the_transition_moments(label: str) -> None:
    """STATISTICAL: sampler mean and variance against the closed forms.

    200 000 paths, four steps (so the sampler is exercised as a *transition*,
    not only as a one-shot terminal draw -- the composition of four exact
    transitions must have the same law as one). Tolerances are four standard
    errors of the corresponding sample statistic, the variance's computed from
    the sample's own fourth moment because the terminal law is far from normal
    in the Feller-violated case (``d = 0.08`` puts a large mass near zero).
    """
    n_paths = 200_000
    t = 1.0
    params = CIR_PARAMS[label]
    model = cir_sde(**params)
    terminal = simulate(
        model, CIR_V0, uniform_time_grid(t, 4), n_paths, scheme="exact", seed=5
    ).terminal

    assert np.all(terminal >= 0.0)
    mean, var = cir_moments(CIR_V0, t, **params)

    sample_mean = float(np.mean(terminal))
    mean_se = float(np.std(terminal, ddof=1) / math.sqrt(n_paths))
    assert abs(sample_mean - mean) <= 4.0 * mean_se

    sample_var = float(np.var(terminal, ddof=1))
    centred = terminal - sample_mean
    fourth = float(np.mean(centred**4))
    var_se = math.sqrt(max(fourth - sample_var**2, 0.0) / n_paths)
    assert abs(sample_var - var) <= 4.0 * var_se


@pytest.mark.parametrize("label", ["feller_ok", "feller_violated"])
def test_cir_expected_excess_matches_the_exact_sampler(label: str) -> None:
    """STATISTICAL: the quadrature reference against a sample of the same law.

    `cir_expected_excess` integrates the noncentral chi-square tail; this test
    checks it against 200 000 draws from the sampler built on the same
    parameters, at four standard errors. The two routes share the
    `cir_transition_parameters` derivation but nothing else -- one integrates
    the survival function, the other draws from it.
    """
    t, strike, n_paths = 1.0, 0.05, 200_000
    params = CIR_PARAMS[label]
    reference = cir_expected_excess(CIR_V0, t, strike=strike, **params)
    terminal = simulate(
        cir_sde(**params), CIR_V0, uniform_time_grid(t, 1), n_paths, scheme="exact", seed=8
    ).terminal
    payoff = np.maximum(terminal - strike, 0.0)
    stderr = float(np.std(payoff, ddof=1) / math.sqrt(n_paths))
    assert abs(float(np.mean(payoff)) - reference) <= 4.0 * stderr
    assert reference > 0.0


# ---------------------------------------------------------------------------
# (c, first half) the boundary that Euler cannot see
# ---------------------------------------------------------------------------

_NEGATIVE_CASES = tuple(
    c for c in SDE_NEGATIVITY_CASES if c.payoff == "negative_fraction"
)
_NAN_CASES = tuple(c for c in SDE_NEGATIVITY_CASES if c.payoff == "nan_fraction")


def _row_id(case: SDECase) -> str:
    return case.row.id


def _negative_row(regime: str) -> SDECase:
    return next(c for c in _NEGATIVE_CASES if c.regime == regime)


def _euler_cir_paths(case: SDECase, truncation: str) -> np.ndarray:
    spec = case.cir_spec
    return simulate(
        cir_sde(**spec.params),
        spec.v0,
        uniform_time_grid(spec.expiry, CIR_NEGATIVITY_STEPS),
        CIR_NEGATIVITY_PATHS,
        scheme="euler",
        truncation=truncation,
        seed=CIR_NEGATIVITY_SEED,
    ).values


@pytest.mark.parametrize("case", _NEGATIVE_CASES, ids=_row_id)
def test_plain_euler_on_cir_goes_negative_and_then_undefined(case: SDECase) -> None:
    """NEGATIVE_FINDING: plain Euler does not merely go negative, it *stops*.

    ``b(v) = xi sqrt(v)`` cannot be evaluated at a negative state, so the step
    after the first negative one produces NaN and the path is lost. Measured
    fractions of paths that reach a negative state: 2.6e-04 with the Feller
    condition satisfied, 0.9117 with it violated. The NaN fraction is slightly
    *lower* than the negative fraction (0.9097) for the only reason it can be:
    a path whose first negative state is the terminal one is never evaluated
    again.
    """
    paths = _euler_cir_paths(case, "none")
    nan_row = next(c for c in _NAN_CASES if c.regime == case.regime).row

    negative = float(np.mean(np.nanmin(paths, axis=1) < 0.0))
    undefined = float(np.mean(np.isnan(paths).any(axis=1)))
    assert abs(negative - case.row.expected) <= case.row.tolerance
    assert abs(undefined - nan_row.expected) <= nan_row.tolerance
    assert undefined <= negative


@pytest.mark.parametrize("case", _NEGATIVE_CASES, ids=_row_id)
def test_full_truncation_keeps_cir_defined_but_not_positive(case: SDECase) -> None:
    """NEGATIVE_FINDING, and the slice statement's expectation corrected.

    The slice said full truncation "does not" produce negative values. It
    produces **exactly as many** as plain Euler, and it must: the two schemes
    are pathwise identical until the first negative state (``max(v, 0) = v``
    while ``v >= 0``), and after that plain Euler has no next state at all.
    What full truncation buys is that the recursion stays *defined* -- no NaN,
    ever -- which is what makes a convergence study possible; positivity is
    not on offer and would need a different scheme (Andersen's QE, or the
    exact sampler).
    """
    plain = _euler_cir_paths(case, "none")
    full = _euler_cir_paths(case, "full")

    assert np.all(np.isfinite(full))
    negative_full = float(np.mean(np.min(full, axis=1) < 0.0))
    negative_plain = float(np.mean(np.nanmin(plain, axis=1) < 0.0))
    assert negative_full == negative_plain
    assert abs(negative_full - case.row.expected) <= case.row.tolerance
    # Pathwise identity up to the first negative state, which is what forces
    # the frequencies above to be equal.
    safe = ~np.isnan(plain)
    assert np.array_equal(plain[safe], full[safe])


def test_milstein_on_cir_is_finite_at_the_boundary() -> None:
    """NEGATIVE_FINDING guard: ``0 * inf`` at ``v = 0``.

    The Milstein correction ``(1/2) b db/dx`` is the constant ``xi^2/4`` for
    CIR, but its two factors are ``0`` and ``inf`` at the boundary, which full
    truncation makes a common state. `cir_sde` therefore supplies the product
    directly; without it every path that touches zero becomes NaN. Building
    the correction from the factors instead (by passing ``db_dx=``) is the
    failure this guards, and it is asserted here so the field cannot be
    deleted as unused.
    """
    params = CIR_PARAMS["feller_violated"]
    model = cir_sde(**params)
    grid = uniform_time_grid(1.0, 50)
    good = simulate(
        model, CIR_V0, grid, 5_000, scheme="milstein", truncation="full", seed=1
    ).values
    assert np.all(np.isfinite(good))

    def db_dx(x: np.ndarray, t: float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return params["xi"] / (2.0 * np.sqrt(x))

    from_factors = simulate(
        model,
        CIR_V0,
        grid,
        5_000,
        scheme="milstein",
        truncation="full",
        seed=1,
        db_dx=db_dx,
    ).values
    assert np.isnan(from_factors).any()


# ---------------------------------------------------------------------------
# coupling and validation
# ---------------------------------------------------------------------------


def test_coarsen_normals_preserves_the_terminal_brownian_increment() -> None:
    """EXACT_IDENTITY: the property the strong-error study depends on.

    Summing fine increments and rescaling gives coarse standard normals whose
    *total* Brownian increment is unchanged, so the exact GBM terminal value
    is the same at every level -- which is what makes ``E|X^h_T - X_T|`` a
    comparison against one fixed reference rather than against five different
    ones.
    """
    rng = np.random.default_rng(99)
    fine = rng.normal(size=(500, 64))
    total_fine = fine.sum(axis=1) * math.sqrt(GBM_T / 64)
    model = gbm_sde(GBM_MU, GBM_SIGMA)
    reference = None
    for n_steps in (1, 2, 4, 8, 16, 32, 64):
        coarse = coarsen_normals(fine, n_steps)
        assert coarse.shape == (500, n_steps)
        total = coarse.sum(axis=1) * math.sqrt(GBM_T / n_steps)
        assert np.allclose(total, total_fine, rtol=1e-12, atol=0.0)
        terminal = simulate(
            model,
            GBM_S0,
            uniform_time_grid(GBM_T, n_steps),
            500,
            scheme="exact",
            normals=coarse,
        ).terminal
        if reference is None:
            reference = terminal
        else:
            assert np.allclose(terminal, reference, rtol=1e-12, atol=0.0)


def test_simulate_validation_errors() -> None:
    model = gbm_sde(GBM_MU, GBM_SIGMA)
    grid = uniform_time_grid(GBM_T, 4)
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 10, scheme="heun", seed=1)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 10, truncation="partial", seed=1)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 0, seed=1)
    with pytest.raises(InvalidInputError):
        simulate(model, -1.0, grid, 10, seed=1)
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 10)  # no seed, rng or normals
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 10, seed=1, normals=np.zeros((10, 4)))
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 10, normals=np.zeros((10, 3)))
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, [0.5, 1.0], 10, seed=1)  # grid must start at 0
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, [0.0, 1.0, 0.5], 10, seed=1)
    with pytest.raises(InvalidInputError):
        simulate(_log_gbm_model(), 0.0, grid, 10, truncation="full", seed=1)
    with pytest.raises(InvalidInputError):
        simulate(model, GBM_S0, grid, 10, scheme="exact", truncation="full", seed=1)
    with pytest.raises(InvalidInputError):
        simulate(
            SDEModel(name="no_exact", drift=model.drift, diffusion=model.diffusion),
            GBM_S0,
            grid,
            10,
            scheme="exact",
            seed=1,
        )
    with pytest.raises(InvalidInputError):
        simulate(
            SDEModel(name="no_db", drift=model.drift, diffusion=model.diffusion),
            GBM_S0,
            grid,
            10,
            scheme="milstein",
            seed=1,
        )
    with pytest.raises(InvalidInputError):
        simulate(cir_sde(**CIR_PARAMS["feller_ok"]), CIR_V0, grid, 10,
                 scheme="exact", normals=np.zeros((10, 4)))


def test_helper_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        uniform_time_grid(0.0, 4)
    with pytest.raises(InvalidInputError):
        uniform_time_grid(1.0, 0)
    with pytest.raises(InvalidInputError):
        coarsen_normals(np.zeros((4, 10)), 3)
    with pytest.raises(InvalidInputError):
        coarsen_normals(np.zeros(10), 2)
    with pytest.raises(InvalidInputError):
        gbm_sde(float("nan"), 0.2)
    with pytest.raises(InvalidInputError):
        gbm_sde(0.1, -0.2)
    with pytest.raises(InvalidInputError):
        cir_sde(0.0, 0.04, 0.2)
    with pytest.raises(InvalidInputError):
        cir_sde(2.0, 0.0, 0.2)
    with pytest.raises(InvalidInputError):
        cir_sde(2.0, 0.04, 0.0)
    with pytest.raises(InvalidInputError):
        cir_transition_parameters(-1.0, 1.0, **CIR_PARAMS["feller_ok"])
    with pytest.raises(InvalidInputError):
        cir_transition_parameters(0.04, 0.0, **CIR_PARAMS["feller_ok"])
    with pytest.raises(InvalidInputError):
        cir_expected_excess(0.04, 1.0, strike=-1.0, **CIR_PARAMS["feller_ok"])
