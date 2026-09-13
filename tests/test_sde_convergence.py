"""Measured strong and weak convergence orders for the SDE schemes.

Every number here is measured in this repository. The reference results are
Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential
Equations*, chapters 9-10 (Euler: strong 1/2, weak 1; Milstein: strong 1,
weak 1) and Glasserman (2003) sections 6.1-6.2; the experiment design is
Higham (2001), SIAM Review 43(3), section 5 (coarse increments built by
summing the fine ones). No table or code from any of those is reproduced.

Evidence: ``CONVERGENCE_ORDER`` for every fitted slope, ``STATISTICAL`` for
the moment identities, ``NEGATIVE_FINDING`` for the two regimes in which the
experiment does **not** resolve an order and says so.

The design decisions that make these measurable
-----------------------------------------------
1. **Strong error** needs a coupling. All five step counts are driven by the
   same fine Brownian path (`coarsen_normals`), so the exact terminal value is
   literally the same array at every level and ``E|X^h_T - X_T|`` compares
   against one fixed reference.

2. **Weak error** is measured as ``mean(f(X^h_T) - f(X_T))`` on those same
   coupled paths, not as a difference of two independently estimated means.
   The estimator is unbiased either way; the coupled form's standard error is
   set by the spread of the *difference*, which is the scheme's own strong
   error, instead of by the spread of the payoff.

3. That trade has a consequence the slice statement did not anticipate, and it
   is the single most important thing measured here: **the coupled estimator's
   noise floor is the scheme's strong error.** Signal over noise is
   ``(C_weak / c_strong) sqrt(N) h^{p_weak - p_strong}``. For Milstein
   ``p_weak = p_strong = 1`` and the ratio is flat in ``h`` (measured |z| 211,
   209, 207, 207, 207 on the call payoff). For Euler
   ``p_weak - p_strong = 1/2`` and it *falls* as the grid is refined --
   measured |z| 73, 55, 40, 29, 20 on the same ladder -- so refining further
   makes the measurement worse, and only ``N`` helps.

4. Point 3 is why the GBM drift is ``mu = 0.2``. The signal-to-noise ratio is
   proportional to ``mu^2 / sigma^2`` (the weak-error constant is driven by the
   drift, the strong error by the diffusion), so at ``mu = 0.05, sigma = 0.2``
   -- an ordinary Black-Scholes point -- the Euler weak error is below the
   noise floor at 100 000 paths and no order can be fitted at all. That failure
   is pinned too, in `test_euler_weak_error_is_unmeasurable_at_a_small_drift`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.cases import (
    CIR_LEVELS,
    CIR_PATHS,
    CIR_SEED,
    GBM_LEVELS,
    GBM_PATHS,
    GBM_SEED,
    GBM_SMALL_DRIFT_MU,
    GBM_STUDY_SPEC,
    SDE_CIR_CALL_CASES,
    SDE_CIR_MEAN_CASES,
    SDE_PRICE_BIAS_CASES,
    SDE_STRONG_CASES,
    SDE_WEAK_CALL_CASES,
    SDE_WEAK_IDENTITY_CASES,
    CIRSpec,
    SDECase,
)
from qpl.cases.sde_discretization import CIR_FELLER_OK, CIR_FELLER_VIOLATED
from qpl.engines.mc.sde import (
    cir_expected_excess,
    cir_moments,
    cir_sde,
    coarsen_normals,
    gbm_sde,
    simulate,
    uniform_time_grid,
)
from qpl.models.black_scholes import bs_price
from qpl.validation import fit_convergence_order, strong_error, weak_error

# The study settings and every expected order, band and note live in
# `qpl.cases.sde_discretization`; each test below re-measures one row. The
# bands are derived there from the seed-to-seed spread of the fit and from the
# pre-asymptotic gap to the theoretical order, whichever dominates.
GBM = GBM_STUDY_SPEC
GBM_S0, GBM_MU, GBM_SIGMA = GBM.s0, GBM.mu, GBM.sigma
GBM_T, GBM_STRIKE = GBM.expiry, GBM.strike
SMALL_DRIFT_MU = GBM_SMALL_DRIFT_MU
CIR_V0 = CIR_FELLER_OK.v0
CIR_T = CIR_FELLER_OK.expiry
CIR_STRIKE = CIR_FELLER_OK.strike


def _row_id(case: SDECase) -> str:
    return case.row.id


def _gbm_study(mu: float, n_paths: int, seed: int) -> dict[str, object]:
    """Coupled Euler / Milstein / exact terminal samples at every level."""
    model = gbm_sde(mu, GBM_SIGMA)
    fine = np.random.default_rng(seed).normal(size=(n_paths, max(GBM_LEVELS)))
    out: dict[str, object] = {"h": [GBM_T / n for n in GBM_LEVELS]}
    for scheme in ("euler", "milstein"):
        terminal, exact = [], []
        for n_steps in GBM_LEVELS:
            z = coarsen_normals(fine, n_steps)
            grid = uniform_time_grid(GBM_T, n_steps)
            terminal.append(
                simulate(model, GBM_S0, grid, n_paths, scheme=scheme, normals=z).terminal
            )
            exact.append(
                simulate(model, GBM_S0, grid, n_paths, scheme="exact", normals=z).terminal
            )
        out[scheme] = terminal
        out[f"{scheme}_exact"] = exact
    return out


@pytest.fixture(scope="module")
def gbm_study() -> dict[str, object]:
    return _gbm_study(GBM_MU, GBM_PATHS, GBM_SEED)


# ---------------------------------------------------------------------------
# (a) strong convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SDE_STRONG_CASES, ids=_row_id)
def test_strong_order_on_gbm(gbm_study: dict[str, object], case: SDECase) -> None:
    """CONVERGENCE_ORDER: ``E|X^h_T - X_T|`` on a shared Brownian path.

    100 000 paths, five levels ``h = 1/8 ... 1/128``, coarse increments summed
    from a 128-step fine path. Measured: Euler 0.5154 (residual 0.0062,
    constant 2.9624), Milstein 0.9932 (residual 0.0025, constant 4.2643).
    Every level's estimate is between 317 and 389 of its own standard errors
    from zero, so the fit is not reading noise; the residual is what says it is
    a power law.

    The Milstein *constant* is larger than the Euler one (4.2643 against
    2.9624): at ``h = 1/8`` the two schemes are only 1.90x apart, and the whole
    benefit is the exponent. Worth stating, because "Milstein is better" is not
    a statement about any single grid.
    """
    errors = [
        strong_error(approx, exact)
        for approx, exact in zip(
            gbm_study[case.scheme], gbm_study[f"{case.scheme}_exact"], strict=True
        )
    ]
    assert min(abs(e.z) for e in errors) > 100.0
    fit = fit_convergence_order(gbm_study["h"], [e.value for e in errors])
    assert abs(fit.order - case.row.expected) < case.row.tolerance
    assert fit.residual < 0.03


def test_milstein_beats_euler_strongly_at_every_level(
    gbm_study: dict[str, object],
) -> None:
    """CONVERGENCE_ORDER, read as a ratio rather than a slope.

    The strong-error ratio Euler/Milstein must grow like ``h^{-1/2}``: measured
    1.90, 2.59, 3.61, 5.05, 7.12 across the ladder, i.e. a factor of about
    ``sqrt(2) = 1.41`` per halving (measured 1.36, 1.39, 1.40, 1.41).
    """
    ratios = [
        strong_error(e, ex).value / strong_error(m, mx).value
        for e, ex, m, mx in zip(
            gbm_study["euler"],
            gbm_study["euler_exact"],
            gbm_study["milstein"],
            gbm_study["milstein_exact"],
            strict=True,
        )
    ]
    for coarse, fine in pairwise(ratios):
        assert 1.30 < fine / coarse < 1.55


# ---------------------------------------------------------------------------
# (a) weak convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_first_moment_of_both_schemes_is_exactly_the_same_geometric_series(
    gbm_study: dict[str, object], scheme: str
) -> None:
    """EXACT_IDENTITY (checked STATISTICALLY): ``E[X^h_T] = S0 (1 + mu h)^n``.

    Taking conditional expectations through either recursion gives
    ``E[X_{i+1}] = (1 + mu h) E[X_i]``: the Euler diffusion term has mean zero,
    and so does the Milstein correction, since ``E[dW^2 - h] = 0``. So the two
    schemes have **identical** first moments at every step, and the weak error
    in ``f(x) = x`` is the deterministic

        S0 ((1 + mu h)^{T/h} - e^{mu T}) = -S0 e^{mu T} mu^2 T h / 2 + O(h^2),

    with constant 2.4428 at this point. This is why Milstein's extra strong
    order buys no weak order: the term it adds is a martingale increment.
    """
    for n_steps, sample in zip(GBM_LEVELS, gbm_study[scheme], strict=True):
        h = GBM_T / n_steps
        predicted = GBM_S0 * (1.0 + GBM_MU * h) ** n_steps
        stderr = float(np.std(sample, ddof=1) / math.sqrt(sample.size))
        assert abs(float(np.mean(sample)) - predicted) <= 4.0 * stderr


@pytest.mark.parametrize("case", SDE_WEAK_IDENTITY_CASES, ids=_row_id)
def test_weak_order_identity_payoff(
    gbm_study: dict[str, object], case: SDECase
) -> None:
    """CONVERGENCE_ORDER: ``|E X^h_T - E X_T|``, ``f(x) = x``.

    Measured order 1.0041 (Euler) and 0.9944 (Milstein), fitted constants
    2.414 and 2.392 against the closed-form leading constant
    ``S0 e^{mu T} mu^2 T / 2 = 2.4428``. Both schemes, one answer, as the
    identity above requires.
    """
    errors = [
        weak_error(approx, reference_values=exact)
        for approx, exact in zip(
            gbm_study[case.scheme], gbm_study[f"{case.scheme}_exact"], strict=True
        )
    ]
    assert all(e.value < 0.0 for e in errors)  # the geometric series undershoots
    fit = fit_convergence_order(gbm_study["h"], [abs(e.value) for e in errors])
    assert abs(fit.order - case.row.expected) < case.row.tolerance
    closed_form_constant = GBM_S0 * math.exp(GBM_MU * GBM_T) * GBM_MU**2 * GBM_T / 2.0
    assert abs(math.exp(fit.log_constant) / closed_form_constant - 1.0) < 0.08


_CALL_MIN_Z = {"euler": 15.0, "milstein": 150.0}


@pytest.mark.parametrize("case", SDE_WEAK_CALL_CASES, ids=_row_id)
def test_weak_order_call_payoff(gbm_study: dict[str, object], case: SDECase) -> None:
    """CONVERGENCE_ORDER: ``|E (X^h_T - K)^+ - E (X_T - K)^+|``.

    Measured order 1.0088 (Euler, residual 0.0109, constant 2.5642) and 0.9946
    (Milstein, residual 0.0018, constant 3.1096). Both order one: the kink in
    the payoff does not cost either scheme a weak order here, because the GBM
    transition density is smooth and smooths it. Both errors are negative --
    the schemes under-price -- which the geometric-series identity above
    explains for the first moment and the payoff inherits.

    ``min_z`` is the noise floor made an assertion. The Milstein signal sits at
    207 to 211 standard errors at *every* level; the Euler one falls from 73 to
    20 across the same ladder, which is the ``h^{p_weak - p_strong}`` decay in
    point 3 of the module docstring.
    """
    strike = GBM_STRIKE
    errors = [
        weak_error(
            np.maximum(approx - strike, 0.0),
            reference_values=np.maximum(exact - strike, 0.0),
        )
        for approx, exact in zip(
            gbm_study[case.scheme], gbm_study[f"{case.scheme}_exact"], strict=True
        )
    ]
    assert min(abs(e.z) for e in errors) > _CALL_MIN_Z[case.scheme]
    fit = fit_convergence_order(gbm_study["h"], [abs(e.value) for e in errors])
    assert abs(fit.order - case.row.expected) < case.row.tolerance
    assert fit.residual < 0.05


def test_euler_weak_error_is_unmeasurable_at_a_small_drift() -> None:
    """NEGATIVE_FINDING: the experiment fails at an ordinary market drift.

    Same design, same 100 000 paths, ``mu = 0.05`` instead of ``0.2``. The weak
    constant falls like ``mu^2`` (a factor of 16) while the strong error, and
    so the noise floor, is unchanged: the measured z-scores are -5.8, -3.3,
    -2.8, -1.3, -0.2 across the ladder and the fitted order is 1.5742 with a
    log-space residual of 0.4231 -- noise dressed as a power law.

    This is pinned because it is the honest reading of the slice statement's
    "design the experiment so this is measurable": the drift used in the tests
    above is not cosmetic, and a reader who swaps in a 5% rate and sees a
    plausible-looking slope is being fooled. `fit_convergence_order` will
    always return a number; only the residual and the z-scores say whether it
    means anything.
    """
    study = _gbm_study(SMALL_DRIFT_MU, GBM_PATHS, GBM_SEED)
    errors = [
        weak_error(
            np.maximum(approx - GBM_STRIKE, 0.0),
            reference_values=np.maximum(exact - GBM_STRIKE, 0.0),
        )
        for approx, exact in zip(study["euler"], study["euler_exact"], strict=True)
    ]
    assert max(abs(e.z) for e in errors) < 10.0
    assert abs(errors[-1].z) < 3.0
    fit = fit_convergence_order(study["h"], [abs(e.value) for e in errors])
    assert fit.residual > 0.2


# ---------------------------------------------------------------------------
# (d) the discretisation bias in a European call price
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SDE_PRICE_BIAS_CASES, ids=_row_id)
def test_euler_pricing_bias_is_order_one_in_dt(
    gbm_study: dict[str, object], case: SDECase
) -> None:
    """CONVERGENCE_ORDER: the price bias, and the constant it carries.

    Read the GBM study as a risk-neutral market: ``r = mu = 0.2``, ``q = 0``,
    ``sigma = 0.2``, ``S0 = K = 100``, ``T = 1``, analytic Black-Scholes price
    19.6346. The Euler price at step count ``n`` is
    ``e^{-rT} mean((X^h_T - K)^+)``; its bias is the discounted weak error
    above, so it must be order 1 with constant ``e^{-rT} * 2.5642 = 2.0994``.
    Measured order 1.0088 (residual 0.0109), constant 2.0994, with the bias
    running -0.2560, -0.1277, -0.0643, -0.0321, -0.0155 -- 1.30% of the price
    at ``n = 8`` and 0.08% at ``n = 128``, always **negative**.

    The measurable design is the exact-sampler payoff used as a control variate
    with its *known* mean (the analytic price): that estimator's standard error
    runs 3.5e-03 down to 7.8e-04 across the ladder, against the plain Euler
    estimator's flat 5.6e-02 to 5.8e-02 -- a variance factor of 250 to 5500. The
    assertion below makes that explicit -- the uncontrolled estimator's own
    standard error exceeds the bias at the fine end, so the bias would be
    invisible without the control.
    """
    r = GBM_MU
    df = math.exp(-r * GBM_T)
    analytic = float(
        bs_price(S=GBM_S0, K=GBM_STRIKE, T=GBM_T, r=r, sigma=GBM_SIGMA, q=0.0, kind="call")
    )

    biases, controlled, plain_stderrs = [], [], []
    for approx, exact in zip(gbm_study["euler"], gbm_study["euler_exact"], strict=True):
        euler_payoff = np.maximum(approx - GBM_STRIKE, 0.0)
        exact_payoff = np.maximum(exact - GBM_STRIKE, 0.0)
        est = weak_error(euler_payoff, reference_values=exact_payoff)
        biases.append(df * est.value)
        controlled.append(df * est.stderr)
        plain_stderrs.append(
            df * float(np.std(euler_payoff, ddof=1) / math.sqrt(euler_payoff.size))
        )

    fit = fit_convergence_order(gbm_study["h"], [abs(b) for b in biases])
    assert abs(fit.order - case.row.expected) < case.row.tolerance
    assert 1.8 < math.exp(fit.log_constant) < 2.4

    # The controlled estimator resolves the bias at every level ...
    assert all(abs(b) > 4.0 * s for b, s in zip(biases, controlled, strict=True))
    # ... and the plain one does not, at the fine end.
    assert abs(biases[-1]) < plain_stderrs[-1]

    # The raw Euler price and "analytic plus the measured bias" are the same
    # number to within the raw estimator's own sampling error, at every level:
    # the control variate moves the error bar, not the answer.
    for approx, bias, plain in zip(
        gbm_study["euler"], biases, plain_stderrs, strict=True
    ):
        raw_price = df * float(np.mean(np.maximum(approx - GBM_STRIKE, 0.0)))
        assert abs(raw_price - (analytic + bias)) <= 4.0 * plain
    assert abs(biases[0]) / analytic > 0.01  # 1.3% at n = 8: worth removing


# ---------------------------------------------------------------------------
# (c, second half) full-truncation Euler on CIR
# ---------------------------------------------------------------------------


def _cir_terminal(spec: CIRSpec, n_steps: int) -> np.ndarray:
    return simulate(
        cir_sde(**spec.params),
        spec.v0,
        uniform_time_grid(spec.expiry, n_steps),
        CIR_PATHS,
        scheme="euler",
        truncation="full",
        seed=CIR_SEED,
    ).terminal


@pytest.fixture(scope="module")
def cir_terminals() -> dict[str, list[np.ndarray]]:
    return {
        "feller_ok": [_cir_terminal(CIR_FELLER_OK, n) for n in CIR_LEVELS],
        "feller_violated": [_cir_terminal(CIR_FELLER_VIOLATED, n) for n in CIR_LEVELS],
    }


def test_full_truncation_has_no_measurable_mean_bias_when_feller_holds(
    cir_terminals: dict[str, list[np.ndarray]],
) -> None:
    """NEGATIVE_FINDING: there is no order to fit here, and that is the result.

    Start at ``v0 = theta``, so the exact ``E[v_T] = theta`` for every ``T``.
    The *untruncated* Euler recursion has the same fixed point --
    ``E[v_{i+1}] = (1 - kappa h) E[v_i] + kappa theta h`` leaves ``theta``
    invariant -- so any bias in ``E[v_T]`` is the truncation's doing alone, and
    with the Feller condition satisfied the truncation fires on 2.6e-04 of
    paths. Measured at 400 000 paths, z-scores -2.78, +0.19, -0.37, +0.76: only
    the coarsest level shows anything at all, and it has the same sign at every
    seed tried (-2.8, -6.2, -3.4, -5.0, -3.6 over five). From ``h = 1/8`` on,
    the bias is under the noise floor. The fitted order is 0.5340 here and
    ranges 0.53 to 1.05 over those five seeds with log-space residuals 0.17 to
    0.94, against 0.07 for the call below and 0.002-0.011 for the GBM fits:
    that residual is what says the slope is reading noise. Reported as "no
    measurable bias beyond the coarsest step", not as an order.
    """
    mean, _ = cir_moments(CIR_V0, CIR_T, **CIR_FELLER_OK.params)
    assert math.isclose(mean, CIR_FELLER_OK.theta, rel_tol=1e-12)
    errors = [
        weak_error(sample, reference_mean=mean) for sample in cir_terminals["feller_ok"]
    ]
    assert errors[0].value < 0.0
    assert max(abs(e.z) for e in errors[1:]) < 4.0
    fit = fit_convergence_order(
        [CIR_T / n for n in CIR_LEVELS], [abs(e.value) for e in errors]
    )
    assert fit.residual > 0.1


@pytest.mark.parametrize("case", SDE_CIR_MEAN_CASES, ids=_row_id)
def test_full_truncation_mean_bias_is_degraded_when_feller_fails(
    cir_terminals: dict[str, list[np.ndarray]], case: SDECase
) -> None:
    """CONVERGENCE_ORDER: measured 0.68, **below** the order-1 expectation.

    Same start, same exact mean ``theta``; now the truncation fires on 91% of
    paths and it biases ``E[v_T]`` **downwards** (-1.139e-02 at ``n = 4``, still
    -2.728e-03 at ``n = 32``, against a true mean of 0.04 -- a 28% error at the
    coarse end). The fitted order is 0.6834 with residual 0.1261 (0.6906 and
    0.7037 at two other seeds), i.e. worse
    than first order and not a clean power law either. This is the slice's
    expected "degraded in the violated case", measured.
    """
    spec = case.cir_spec
    mean, _ = cir_moments(spec.v0, spec.expiry, **spec.params)
    errors = [
        weak_error(sample, reference_mean=mean) for sample in cir_terminals[case.regime]
    ]
    assert all(e.value < 0.0 for e in errors)
    assert min(abs(e.z) for e in errors) > 5.0
    fit = fit_convergence_order(
        [spec.expiry / n for n in CIR_LEVELS], [abs(e.value) for e in errors]
    )
    assert abs(fit.order - case.row.expected) < case.row.tolerance
    assert fit.order < 0.85  # the claim is that it is BELOW first order
    assert abs(errors[0].value / mean) > 0.25


@pytest.mark.parametrize("case", SDE_CIR_CALL_CASES, ids=_row_id)
def test_full_truncation_weak_order_on_a_variance_call(
    cir_terminals: dict[str, list[np.ndarray]], case: SDECase
) -> None:
    """CONVERGENCE_ORDER: ``E[(v_T - 0.05)^+]`` against the exact law.

    The reference is `cir_expected_excess`, which integrates the noncentral
    chi-square tail rather than sampling it: the slice said "against the exact
    sampler", and replacing the sampler by an integral of the same law removes
    the reference's own Monte Carlo error, leaving only the scheme's.

    Measured, 400 000 paths, ``h = 1/4 ... 1/32``: order 0.9729 (residual
    0.0705, constant 3.8638e-03) with the Feller condition satisfied and 0.9389
    (residual 0.0606, constant 0.1329) with it violated. The *constant* is 34x
    larger in the violated regime -- the damage shows up there, not in the
    exponent, which is the opposite of where the mean bias showed it.

    The bands differ because the two regimes are measured to different
    precision: over seeds {4242, 77, 5150} the Feller-satisfied order moves
    0.9729 / 1.0404 / 1.0850 (its levels are only 8 to 55 standard errors from
    zero) while the violated one moves 0.9389 / 0.9472 / 0.9407 (19 to 118
    standard errors). A 0.06 band on the violated fit is therefore a real
    claim that it is **below** one; a 0.15 band on the other is an honest
    statement that this budget cannot distinguish 0.97 from 1.09.
    """
    spec = case.cir_spec
    reference = cir_expected_excess(
        spec.v0, spec.expiry, strike=spec.strike, **spec.params
    )
    errors = [
        weak_error(np.maximum(sample - spec.strike, 0.0), reference_mean=reference)
        for sample in cir_terminals[case.regime]
    ]
    assert all(e.value > 0.0 for e in errors)  # truncation over-prices the call
    assert min(abs(e.z) for e in errors) > 5.0
    fit = fit_convergence_order(
        [spec.expiry / n for n in CIR_LEVELS], [abs(e.value) for e in errors]
    )
    assert abs(fit.order - case.row.expected) < case.row.tolerance
    assert fit.residual < 0.12
