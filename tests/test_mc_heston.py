"""The Heston path generator: moments, positivity, correlation, martingale.

What each family of test is evidence for, and of what class:

- **EXACT_IDENTITY** -- the QE branch parameters reproduce the *exact* CIR
  conditional moments to round-off, on both sides of the switching level
  `psi_c`. This is checked at the formula level (`qe_branch_moments` computes
  the branch's own mean and variance from `(a, b)` or `(p, beta)`), so it tests
  the moment-matching algebra rather than restating the target.
- **STATISTICAL** -- the sampler reproduces those moments, and its higher
  moments are *reported*: QE is a two-moment fit, so a skewness or kurtosis gap
  against the exact noncentral chi-square is a property of the scheme, not a
  bug, and the gap is pinned as a measured number.
- **NEGATIVE_FINDING** -- the log-spot drift without Andersen's martingale
  correction fails `E[e^{-(r-q)T} S_T] = S_0` by 1.10% of spot at `dt = 1/4`,
  84 standard errors from zero, and the correction removes it at every step
  size.
- **CONVERGENCE_ORDER** -- the realised correlation between the log-spot and
  variance increments approaches `rho` at measured order ~1 in `dt`, for every
  scheme and every `rho` tested. That is the sharp check that the correlation
  is wired correctly: a price comparison at `rho = 0` is *not* (see
  `tests/test_mc_heston_pricing.py`, where the `rho = 0` discretisation bias is
  measured to be larger than the `rho = -0.5` one).

Path counts here are deliberately modest; the full-`N` tables live in
`examples/heston_mc_qe.py` and `docs/notes/heston_monte_carlo_qe.md`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from heston_points import FELLER_VIOLATED, LEWIS
from scipy.stats import ncx2

from qpl.engines.mc.heston import (
    EULER_FULL_TRUNCATION,
    EXACT_VARIANCE_EULER_LOG_SPOT,
    HESTON_SCHEMES,
    QE,
    QE_PSI_C,
    ConditionalTerminalLaw,
    conditional_forward,
    conditional_vanilla_values,
    heston_variance_moments,
    qe_branch_moments,
    qe_variance_step,
    simulate_heston,
)
from qpl.engines.mc.sde import cir_moments, cir_transition_parameters, uniform_time_grid
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.models.heston import HestonModel
from qpl.validation import EvidenceClass, fit_convergence_order

# --------------------------------------------------------------------------
# (a) The moment match, at the formula level.
# --------------------------------------------------------------------------

# `(v_t, dt, kappa, theta, xi)` chosen to land on both branches: the first
# three are quadratic (`psi` well below 1.5) and the last two exponential
# (`psi` above it), which is what "both sides of psi_c" means concretely.
MOMENT_POINTS = (
    (0.04, 0.25, 4.0, 0.25, 1.0),
    (0.25, 1.0 / 64.0, 4.0, 0.25, 1.0),
    (0.5, 0.01, 4.0, 0.25, 1.0),
    (1e-06, 0.25, 0.5, 0.04, 1.0),
    (1e-08, 1.0, 0.5, 0.04, 1.0),
)


@pytest.mark.parametrize("point", MOMENT_POINTS, ids=lambda p: f"v{p[0]:g}_dt{p[1]:g}")
def test_qe_branch_moments_match_the_exact_cir_moments(point) -> None:
    """Evidence class: EXACT_IDENTITY.

    `heston_variance_moments` derives `(m, s2)` by taking expectations in the
    square-root SDE; `qpl.engines.mc.sde.cir_moments` derives them from the
    noncentral chi-square transition's degrees of freedom and non-centrality.
    They agree to round-off, and so does the mean and variance of whichever
    law QE actually samples. Tolerance is a relative round-off budget, not a
    model tolerance.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    v, dt, kappa, theta, xi = point
    m, s2 = heston_variance_moments(
        np.array([v]), dt, kappa=kappa, theta=theta, xi=xi
    )
    exact_mean, exact_var = cir_moments(v, dt, kappa=kappa, theta=theta, xi=xi)
    assert math.isclose(float(m[0]), exact_mean, rel_tol=1e-12)
    assert math.isclose(float(s2[0]), exact_var, rel_tol=1e-12)

    _, branch_mean, branch_var = qe_branch_moments(
        np.array([v]), dt, kappa=kappa, theta=theta, xi=xi
    )
    assert math.isclose(float(branch_mean[0]), exact_mean, rel_tol=1e-12)
    assert math.isclose(float(branch_var[0]), exact_var, rel_tol=1e-12)


def test_the_moment_points_exercise_both_qe_branches() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the test data).

    Without this the parametrisation above could drift onto one branch and the
    "both sides of psi_c" claim would be silently false.
    """
    branches = set()
    for v, dt, kappa, theta, xi in MOMENT_POINTS:
        branch, _, _ = qe_branch_moments(
            np.array([v]), dt, kappa=kappa, theta=theta, xi=xi
        )
        branches.add(int(branch[0]))
    assert branches == {0, 1}


def test_psi_c_sits_inside_the_window_where_both_branches_exist() -> None:
    """Evidence class: EXACT_IDENTITY.

    The quadratic branch needs `psi <= 2` for its `b^2` to be real and the
    exponential branch needs `psi >= 1` for its atom to be a probability.
    """
    assert 1.0 <= QE_PSI_C <= 2.0


# --------------------------------------------------------------------------
# (b) The sampler against the exact transition, and the higher-moment gap.
# --------------------------------------------------------------------------

_SAMPLER_PATHS = 200_000
_SAMPLER_SEED = 7

# Measured at 200k draws, seed 7, against the exact ncx2 transition sampled at
# the same count. `skew` and `kurtosis` are the QE value and the exact value:
# QE matches two moments and nothing more, and these are the gaps.
HIGHER_MOMENT_ROWS = (
    # (v, dt, kappa, theta, xi, qe_skew, exact_skew, qe_kurtosis, exact_kurtosis)
    (0.04, 0.25, 4.0, 0.25, 1.0, 1.103, 1.421, 4.620, 6.183),
    (0.04, 0.25, 0.5, 0.04, 1.0, 3.542, 3.674, 19.646, 21.128),
)


@pytest.mark.parametrize(
    "row", HIGHER_MOMENT_ROWS, ids=["feller_ok", "feller_violated"]
)
def test_qe_samples_reproduce_the_exact_transition_moments(row) -> None:
    """Evidence class: STATISTICAL for the first two moments.

    The tolerance is a multiple of the sample standard error, not a fixed
    number: the claim is that the sampler is unbiased for the mean and the
    variance it was constructed to match.
    """
    v, dt, kappa, theta, xi, *_ = row
    rng = np.random.default_rng(_SAMPLER_SEED)
    z = rng.normal(size=_SAMPLER_PATHS)
    draws = qe_variance_step(
        np.full(_SAMPLER_PATHS, v), dt, kappa=kappa, theta=theta, xi=xi, z=z
    )
    exact_mean, exact_var = cir_moments(v, dt, kappa=kappa, theta=theta, xi=xi)

    mean_stderr = float(np.std(draws, ddof=1) / math.sqrt(draws.size))
    assert abs(float(np.mean(draws)) - exact_mean) < 4.0 * mean_stderr
    # A variance estimate's own standard error is dominated by the fourth
    # central moment; taking it from the sample keeps the claim statistical.
    centred = draws - np.mean(draws)
    fourth = float(np.mean(centred**4))
    var_stderr = math.sqrt(
        max(fourth - float(np.var(draws, ddof=1)) ** 2, 0.0) / draws.size
    )
    assert abs(float(np.var(draws, ddof=1)) - exact_var) < 4.0 * var_stderr


@pytest.mark.parametrize(
    "row", HIGHER_MOMENT_ROWS, ids=["feller_ok", "feller_violated"]
)
def test_qe_higher_moments_differ_from_the_exact_transition(row) -> None:
    """Evidence class: NEGATIVE_FINDING, pinned as a measurement.

    QE matches two moments by construction and nothing else. On the
    Feller-satisfying set its skewness is **22% low** (1.104 against 1.415) and
    its kurtosis **25% low** (4.615 against 6.141); on the Feller-violating set
    the gaps shrink to 3% and 5%, because there the exponential branch's own
    heavy tail is close to the true law's. Neither is a failure and neither is
    allowed to drift silently.
    """
    v, dt, kappa, theta, xi, qe_skew, exact_skew, qe_kurt, exact_kurt = row
    rng = np.random.default_rng(_SAMPLER_SEED)
    draws = qe_variance_step(
        np.full(_SAMPLER_PATHS, v),
        dt,
        kappa=kappa,
        theta=theta,
        xi=xi,
        z=rng.normal(size=_SAMPLER_PATHS),
    )
    df, nc, scale = cir_transition_parameters(
        np.array([v]), dt, kappa=kappa, theta=theta, xi=xi
    )
    exact = ncx2.rvs(
        df, float(nc[0]), scale=scale, size=_SAMPLER_PATHS, random_state=rng
    )

    def _shape(sample: np.ndarray) -> tuple[float, float]:
        centred = sample - sample.mean()
        sd = sample.std(ddof=1)
        return float(np.mean(centred**3) / sd**3), float(np.mean(centred**4) / sd**4)

    qe_s, qe_k = _shape(draws)
    ex_s, ex_k = _shape(exact)
    # Third and fourth sample moments are themselves noisy estimators, so the
    # slack is 5% rather than a round-off budget; what is pinned is the SIZE of
    # the gap, not the last digit of either side.
    assert qe_s == pytest.approx(qe_skew, rel=0.05)
    assert ex_s == pytest.approx(exact_skew, rel=0.05)
    assert qe_k == pytest.approx(qe_kurt, rel=0.05)
    assert ex_k == pytest.approx(exact_kurt, rel=0.05)
    # The direction of the gap is the finding: QE's tail is the thinner one.
    assert qe_s < ex_s
    assert qe_k < ex_k


def test_qe_variance_is_non_negative_everywhere_including_feller_violation() -> None:
    """Evidence class: EXACT_IDENTITY (a property of the sampler's algebra).

    A scaled square on one branch and an atom-plus-exponential on the other:
    neither can produce a negative number, at any step size, in either Feller
    regime. Contrast `qpl.engines.mc.sde`'s measured 68-73% negative terminal
    variances for full-truncation Euler.
    """
    model = HestonModel(
        v0=FELLER_VIOLATED.v0,
        kappa=FELLER_VIOLATED.kappa,
        theta=FELLER_VIOLATED.theta,
        xi=FELLER_VIOLATED.xi,
        rho=FELLER_VIOLATED.rho,
    )
    assert model.feller_number < 2.0
    for n_steps in (4, 32):
        paths = simulate_heston(
            model,
            s0=100.0,
            mu=-0.01,
            t_grid=uniform_time_grid(1.0, n_steps),
            n_paths=20_000,
            seed=3,
            scheme=QE,
        )
        assert np.all(paths.variance >= 0.0)
        assert np.all(np.isfinite(paths.log_spot))


def test_full_truncation_euler_does_go_negative_on_the_same_set() -> None:
    """Evidence class: NEGATIVE_FINDING.

    The contrast that makes the row above worth asserting. Full truncation
    keeps the recursion *defined*; it does not keep it positive, and the
    engine reports the frequency rather than hiding it.
    """
    model = HestonModel(
        v0=FELLER_VIOLATED.v0,
        kappa=FELLER_VIOLATED.kappa,
        theta=FELLER_VIOLATED.theta,
        xi=FELLER_VIOLATED.xi,
        rho=FELLER_VIOLATED.rho,
    )
    paths = simulate_heston(
        model,
        s0=100.0,
        mu=-0.01,
        t_grid=uniform_time_grid(1.0, 8),
        n_paths=20_000,
        seed=3,
        scheme=EULER_FULL_TRUNCATION,
    )
    fraction = float(paths.meta["negative_variance_fraction"])
    assert 0.4 < fraction < 0.8
    assert np.any(paths.variance < 0.0)


# --------------------------------------------------------------------------
# (d) The martingale property, with and without the correction.
# --------------------------------------------------------------------------

_MARTINGALE_PATHS = 200_000
_MARTINGALE_SEED = 101


def _forward_defect(*, martingale_correction: bool, n_steps: int) -> tuple[float, float]:
    """`(E[e^{-(r-q)T} S_T] - S_0, stderr)` for QE on the reference set."""
    model = LEWIS.model()
    mu = LEWIS.rate - LEWIS.dividend
    paths = simulate_heston(
        model,
        s0=LEWIS.spot,
        mu=mu,
        t_grid=uniform_time_grid(1.0, n_steps),
        n_paths=_MARTINGALE_PATHS,
        seed=_MARTINGALE_SEED,
        scheme=QE,
        antithetic=True,
        martingale_correction=martingale_correction,
        store_paths=False,
    )
    discounted = math.exp(-mu * 1.0) * paths.terminal_spot
    half = _MARTINGALE_PATHS // 2
    units = 0.5 * (discounted[:half] + discounted[half:])
    return (
        float(np.mean(units)) - LEWIS.spot,
        float(np.std(units, ddof=1) / math.sqrt(units.size)),
    )


@pytest.mark.parametrize("n_steps", (4, 16))
def test_the_martingale_correction_holds_the_forward(n_steps: int) -> None:
    """Evidence class: STATISTICAL.

    With the correction the conditional expectation of each step is exactly
    `e^{(r-q) dt}` by construction, so the only thing between the estimate and
    `S_0` is sampling error.
    """
    defect, stderr = _forward_defect(martingale_correction=True, n_steps=n_steps)
    assert abs(defect) < 4.0 * stderr


def test_the_uncorrected_drift_fails_the_martingale_property() -> None:
    """Evidence class: NEGATIVE_FINDING, pinned.

    Measured over 12 seeds at 200k antithetic paths each (see
    `examples/heston_mc_qe.py`): the uncorrected QE drift overstates
    `E[e^{-(r-q)T} S_T]` by **+1.098** on a spot of 100 at `dt = 1/4` (84
    standard errors), +0.272 at `dt = 1/8` and +0.058 at `dt = 1/16` -- a
    defect that falls at roughly order 2 in `dt` and is therefore invisible on
    a fine grid and expensive on a coarse one. Andersen section 4.3.
    """
    defect, stderr = _forward_defect(martingale_correction=False, n_steps=4)
    assert defect > 20.0 * stderr
    assert defect == pytest.approx(1.1, abs=0.2)

    corrected, corrected_stderr = _forward_defect(
        martingale_correction=True, n_steps=4
    )
    assert abs(corrected) < 4.0 * corrected_stderr
    assert abs(corrected) < 0.05 * abs(defect)


def test_meta_reports_whether_the_correction_ran() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the metadata)."""
    model = LEWIS.model()
    grid = uniform_time_grid(1.0, 4)
    euler = simulate_heston(
        model, s0=100.0, mu=-0.01, t_grid=grid, n_paths=64, seed=1,
        scheme=EULER_FULL_TRUNCATION,
    )
    assert euler.meta["martingale_correction"] is False
    assert euler.meta["martingale_correction_applies"] is False

    qe = simulate_heston(
        model, s0=100.0, mu=-0.01, t_grid=grid, n_paths=64, seed=1, scheme=QE
    )
    assert qe.meta["martingale_correction"] is True
    assert qe.meta["martingale_correction_fallbacks"] == 0


# --------------------------------------------------------------------------
# (f) Correlation recovery.
# --------------------------------------------------------------------------

_CORRELATION_PATHS = 40_000
_CORRELATION_STEPS = 64
_CORRELATION_TOLERANCE = 0.025
"""Absolute slack on `corr(dX, dv) - rho` at `dt = 1/64`.

Measured at 400k paths: the residual for QE is +0.0163 (rho = -0.9), +0.0083
(-0.5), -0.0002 (0) and -0.0068 (+0.5) at `dt = 1/64`, falling to +0.0042,
+0.0020, -0.0003 and -0.0019 at `dt = 1/256`, with a sampling standard error
of 2e-05 -- so the residual is a *discretisation* effect, not noise, and it
falls at measured order ~1 in `dt` (see the order test below). Euler's is the
largest of the three schemes at the coarse end (+0.0177 at rho = -0.9). The
tolerance keeps a factor of about 1.4 on the worst measured cell."""


@pytest.mark.parametrize("rho", (-0.9, -0.5, 0.0, 0.5))
@pytest.mark.parametrize("scheme", HESTON_SCHEMES)
def test_the_increment_correlation_recovers_rho(rho: float, scheme: str) -> None:
    """Evidence class: STATISTICAL.

    The sharp check that the correlation is wired correctly, and the one the
    slice statement's "price at `rho = 0`" test is *not*: a price comparison
    confounds the correlation with the time-discretisation bias, which at
    `rho = 0` is measured to be larger than at `rho = -0.5`.
    """
    model = HestonModel(v0=0.04, kappa=4.0, theta=0.25, xi=1.0, rho=rho)
    paths = simulate_heston(
        model,
        s0=100.0,
        mu=-0.01,
        t_grid=uniform_time_grid(1.0, _CORRELATION_STEPS),
        n_paths=_CORRELATION_PATHS,
        seed=5,
        scheme=scheme,
    )
    d_log_spot = np.diff(paths.log_spot, axis=1).ravel()
    d_variance = np.diff(paths.variance, axis=1).ravel()
    realised = float(np.corrcoef(d_log_spot, d_variance)[0, 1])
    assert realised == pytest.approx(rho, abs=_CORRELATION_TOLERANCE)


def test_the_correlation_residual_vanishes_with_the_step_size() -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Measured order ~1 in `dt` for QE at `rho = -0.9`: residual +0.0594 at
    `dt = 1/16`, +0.0163 at 1/64 and +0.0042 at 1/256 (400k paths), i.e. a
    factor of about 3.8 per fourfold refinement. The fit here runs on a
    smaller sample and is asserted loosely; the claim is that the residual is
    a discretisation term and not a wiring error.
    """
    model = HestonModel(v0=0.04, kappa=4.0, theta=0.25, xi=1.0, rho=-0.9)
    levels = (8, 16, 32, 64)
    residuals = []
    for n_steps in levels:
        paths = simulate_heston(
            model,
            s0=100.0,
            mu=-0.01,
            t_grid=uniform_time_grid(1.0, n_steps),
            n_paths=40_000,
            seed=11,
            scheme=QE,
        )
        realised = float(
            np.corrcoef(
                np.diff(paths.log_spot, axis=1).ravel(),
                np.diff(paths.variance, axis=1).ravel(),
            )[0, 1]
        )
        residuals.append(abs(realised - model.rho))
    fit = fit_convergence_order(
        np.array([1.0 / n for n in levels]), np.array(residuals)
    )
    assert fit.order == pytest.approx(1.0, abs=0.2)
    assert fit.residual < 0.1


# --------------------------------------------------------------------------
# The conditional terminal law.
# --------------------------------------------------------------------------


def test_the_conditional_law_reproduces_the_realised_paths() -> None:
    """Evidence class: EXACT_IDENTITY.

    The conditional mean and variance are accumulated from the same quantities
    the path recursion uses, so the realised `ln S_T` must have the right law:
    the sample mean of `exp(mean + variance/2)` and of `S_T` estimate the same
    number, and the conditional estimator of a call estimates the same price as
    the plain one. Checked here as an agreement within the *plain* estimator's
    standard error, which is the only tolerance available -- the two estimators
    are not path-by-path equal, they share a mean.
    """
    model = LEWIS.model()
    mu = LEWIS.rate - LEWIS.dividend
    paths = simulate_heston(
        model,
        s0=LEWIS.spot,
        mu=mu,
        t_grid=uniform_time_grid(1.0, 8),
        n_paths=80_000,
        seed=17,
        scheme=QE,
        store_paths=False,
    )
    law = paths.conditional
    assert isinstance(law, ConditionalTerminalLaw)

    plain_spot = paths.terminal_spot
    conditional_spot = conditional_forward(law)
    spot_stderr = float(np.std(plain_spot, ddof=1) / math.sqrt(plain_spot.size))
    assert abs(float(np.mean(conditional_spot)) - float(np.mean(plain_spot))) < (
        4.0 * spot_stderr
    )

    plain_call = np.maximum(plain_spot - 100.0, 0.0)
    conditional_call = conditional_vanilla_values(law, strike=100.0, kind="call")
    call_stderr = float(np.std(plain_call, ddof=1) / math.sqrt(plain_call.size))
    assert abs(
        float(np.mean(conditional_call)) - float(np.mean(plain_call))
    ) < 4.0 * call_stderr
    # And it is the quieter estimator, which is the whole point of carrying it.
    assert float(np.std(conditional_call, ddof=1)) < 0.5 * float(
        np.std(plain_call, ddof=1)
    )


def test_conditional_put_call_parity_holds_path_by_path() -> None:
    """Evidence class: EXACT_IDENTITY.

    `C(m, V) - P(m, V) = e^{m + V/2} - K` for every path, which is parity
    inside the conditioning and pins the two closed forms against each other.
    """
    rng = np.random.default_rng(2)
    law = ConditionalTerminalLaw(
        mean=rng.normal(loc=math.log(100.0), scale=0.3, size=500),
        variance=rng.uniform(0.0, 0.5, size=500),
    )
    call = conditional_vanilla_values(law, strike=95.0, kind="call")
    put = conditional_vanilla_values(law, strike=95.0, kind="put")
    residual = call - put - (conditional_forward(law) - 95.0)
    assert float(np.max(np.abs(residual))) < 1e-10


def test_a_degenerate_zero_variance_conditional_law_is_intrinsic() -> None:
    """Evidence class: EXACT_IDENTITY (the `V = 0` limit of the formula)."""
    law = ConditionalTerminalLaw(
        mean=np.array([math.log(120.0), math.log(80.0)]), variance=np.zeros(2)
    )
    call = conditional_vanilla_values(law, strike=100.0, kind="call")
    assert call == pytest.approx([20.0, 0.0], abs=1e-12)


# --------------------------------------------------------------------------
# Input contract.
# --------------------------------------------------------------------------


def test_antithetic_is_refused_for_the_exact_variance_scheme() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    with pytest.raises(NotSupportedError, match="noncentral chi-square"):
        simulate_heston(
            LEWIS.model(),
            s0=100.0,
            mu=0.0,
            t_grid=uniform_time_grid(1.0, 2),
            n_paths=8,
            seed=1,
            scheme=EXACT_VARIANCE_EULER_LOG_SPOT,
            antithetic=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"scheme": "milstein"}, "scheme must be one of"),
        ({"s0": 0.0}, "s0 must be finite and > 0"),
        ({"n_paths": 0}, "n_paths must be >= 1"),
        ({"n_paths": 5, "antithetic": True}, "n_paths must be even"),
    ),
)
def test_invalid_inputs_are_refused(kwargs, match) -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    call = {
        "s0": 100.0,
        "mu": 0.0,
        "t_grid": uniform_time_grid(1.0, 2),
        "n_paths": 8,
        "seed": 1,
    }
    call.update(kwargs)
    with pytest.raises(InvalidInputError, match=match):
        simulate_heston(LEWIS.model(), **call)


def test_seed_and_rng_are_mutually_exclusive() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    with pytest.raises(InvalidInputError, match="exactly one of seed or rng"):
        simulate_heston(
            LEWIS.model(),
            s0=100.0,
            mu=0.0,
            t_grid=uniform_time_grid(1.0, 2),
            n_paths=8,
            seed=1,
            rng=np.random.default_rng(0),
        )


def test_the_same_seed_reproduces_the_same_paths() -> None:
    """Evidence class: EXACT_IDENTITY (determinism)."""
    kwargs = {
        "s0": 100.0,
        "mu": -0.01,
        "t_grid": uniform_time_grid(1.0, 4),
        "n_paths": 256,
        "seed": 99,
        "scheme": QE,
    }
    first = simulate_heston(LEWIS.model(), **kwargs)
    second = simulate_heston(LEWIS.model(), **kwargs)
    assert np.array_equal(first.log_spot, second.log_spot)
    assert np.array_equal(first.variance, second.variance)


def test_store_paths_false_keeps_only_the_terminal_column() -> None:
    """Evidence class: EXACT_IDENTITY.

    The terminal state must not depend on whether the intermediate columns
    were kept, so the two runs at one seed agree bit for bit.
    """
    kwargs = {
        "s0": 100.0,
        "mu": -0.01,
        "t_grid": uniform_time_grid(1.0, 6),
        "n_paths": 512,
        "seed": 42,
        "scheme": QE,
    }
    full = simulate_heston(LEWIS.model(), store_paths=True, **kwargs)
    terminal = simulate_heston(LEWIS.model(), store_paths=False, **kwargs)
    assert terminal.log_spot.shape == (512, 1)
    assert np.array_equal(full.terminal_log_spot, terminal.terminal_log_spot)
    assert np.array_equal(full.terminal_variance, terminal.terminal_variance)


def test_return_normals_gives_back_the_driving_block() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    paths = simulate_heston(
        LEWIS.model(),
        s0=100.0,
        mu=0.0,
        t_grid=uniform_time_grid(1.0, 4),
        n_paths=100,
        seed=1,
        return_normals=True,
        antithetic=True,
    )
    assert paths.normals is not None
    z_variance, z_spot = paths.normals
    assert z_variance.shape == (100, 4)
    assert np.allclose(z_variance[:50], -z_variance[50:])
    assert np.allclose(z_spot[:50], -z_spot[50:])
