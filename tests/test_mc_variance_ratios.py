"""How much variance each estimator actually removes, measured.

`tests/test_mc_variance_reduction.py` establishes that every estimator here is
unbiased and reports a calibrated standard error. This file answers the other
question: what is it worth?

The cost axis is **normal draws**, not paths. Antithetic sampling evaluates the
payoff twice per draw, so an antithetic run with `2N` paths and a plain run
with `N` paths cost the same on this axis and are what gets compared; quoting a
gain at equal *paths* silently credits antithetic with free normals. Where the
distinction matters the docstring gives both conventions.

The variance being compared is the variance of the *estimator*: 50 independent
seeds, each producing one price, and the sample variance of those 50 numbers.
That is the quantity a user cares about and it is measured rather than inferred
from the reported standard errors -- which would be circular, since the
standard-error formula is one of the things under test.

Precision of the ratios themselves: each variance is a 49-degree-of-freedom
estimate, so the 99% band for a ratio of two independent ones is [0.47, 2.11].
The bands asserted below are a factor of 1.7 either way, which is inside that
for the theory-backed rows and is why they are not tighter. The seeds are fixed,
so the numbers reproduce exactly; the band states what the claim would survive,
not what this run produced.

Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
sections 4.1-4.3. Every number below was measured in this repository.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.cases import (
    MC_VARIANCE_REDUCTION_CASES,
    MC_VR_BAND,
    MC_VR_DRAWS,
    MC_VR_N_STRATA,
    MC_VR_POINTS,
    MC_VR_SEEDS,
    DigitalBSSpec,
    paths_for_draws,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.variance_reduction import terminal_spots_from_normals
from qpl.instruments.payoffs import call_payoff, digital_payoff
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

ATM_CALL = MC_VR_POINTS["atm_call"]
OTM_CALL = MC_VR_POINTS["otm_call"]
DIGITAL_CALL = MC_VR_POINTS["digital_call"]

MARKET = ATM_CALL.market()
MODEL = ATM_CALL.model()
SIGMA = ATM_CALL.sigma
EXPIRY = ATM_CALL.expiry
MU = MARKET.rate(EXPIRY) - MARKET.dividend_yield(EXPIRY)
DISCOUNT = MARKET.df_r(EXPIRY)
"""Read back from the market rather than written as literals: `Market.rate(t)`
recovers the rate from the curve's discount factor and is not bit-for-bit the
`0.05` that was passed in. See `tests/test_mc_variance_reduction.py`."""

SEEDS = MC_VR_SEEDS
N_DRAWS = MC_VR_DRAWS
N_STRATA = MC_VR_N_STRATA
BAND = MC_VR_BAND
"""Study constants live in `qpl.cases.mc_variance_reduction` next to the rows
they parametrise, so a test and a row cannot drift apart."""


def _price(spec, vr, seed: int, n_strata: int = N_STRATA, n_draws: int = N_DRAWS):
    cfg = MCConfig(
        n_paths=paths_for_draws(vr, n_draws),
        n_steps=1,
        seed=seed,
        variance_reduction=vr,
        n_strata=n_strata,
    )
    return price(spec.option(), spec.model(), spec.market(), method="mc", cfg=cfg).value


def _estimator_variance(spec, vr, n_strata: int = N_STRATA) -> float:
    values = np.array([_price(spec, vr, seed, n_strata) for seed in SEEDS])
    return float(values.var(ddof=1))


def _payoff_of(spec):
    if isinstance(spec, DigitalBSSpec):
        return lambda s_t: np.asarray(
            digital_payoff(s_t, spec.strike, spec.cash, spec.kind), dtype=float
        )
    return lambda s_t: call_payoff(s_t, spec.strike)


def _pilot_correlations(spec, n: int = 400_000, seed: int = 7) -> tuple[float, float]:
    """`(corr(Y(Z), Y(-Z)), corr(Y, X))` from one large independent pilot run.

    The theoretical predictions below are derived from these two correlations
    rather than hard-coded, so the test compares a measured ratio against a
    measured prediction of it -- which is the comparison that can fail
    informatively. A hard-coded prediction would only re-assert this file.
    """
    payoff = _payoff_of(spec)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 1))
    spots = terminal_spots_from_normals(z, s0=100.0, mu=MU, sigma=SIGMA, t=EXPIRY)
    spots_reflected = terminal_spots_from_normals(
        -z, s0=100.0, mu=MU, sigma=SIGMA, t=EXPIRY
    )
    y = DISCOUNT * payoff(spots)
    y_reflected = DISCOUNT * payoff(spots_reflected)
    x = DISCOUNT * spots
    return (
        float(np.corrcoef(y, y_reflected)[0, 1]),
        float(np.corrcoef(y, x)[0, 1]),
    )


# ---------------------------------------------------------------------------
# (b) The table.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", MC_VARIANCE_REDUCTION_CASES, ids=[c.row.id for c in MC_VARIANCE_REDUCTION_CASES]
)
def test_measured_variance_factor(case) -> None:
    """STATISTICAL: variance of plain / variance of reduced, at equal normal draws.

    Fifty seeds, 20 480 normal draws each, `K = 64` strata. The full table
    (factor, with the theoretical prediction where one exists):

        instrument     antithetic   control   stratified   anti+ctrl  strat+ctrl
        atm_call        4.59 (4.01)  7.58 (6.89)   110.40       90.80     577.80
        otm_call        3.04 (2.33)  2.71 (2.30)    42.41       92.35      84.61
        digital_call    9.32 (9.38)  2.11 (2.45)   101.26        9.73      82.32

    Three things in that table are worth reading twice.

    1. **Antithetic is worth most on the digital and least off the money.** The
       slice was written expecting the opposite ordering -- "the gain is bounded
       and can be small for convex payoffs" -- and the convexity story is right
       for the two calls (4.59 at the money, 3.04 at K = 120) but the digital
       gets 9.32, the largest antithetic gain here. The reason is not
       convexity: for a cash-or-nothing call with the in-the-money boundary at
       `z* = -0.15`, the pair `1{Z > z*} + 1{-Z > z*}` is exactly 1 unless
       `|Z| < 0.15`, so the pair average is a constant on 88% of the sample and
       the antithetic correlation is -0.7868. A bounded payoff whose reflection
       nearly complements it is the best case for the method, and a digital
       near the money is exactly that.
    2. **The control variate is worth much less off the money.** 7.58 at the
       money against 2.71 at K = 120: the correlation between the call payoff
       and the terminal spot falls from 0.9246 to 0.7522 once most of the
       sample pays nothing, and `1/(1-rho**2)` is very sensitive up there.
    3. **The combinations are not the product of their parts** and are not even
       ordered the same way. On the ATM call, stratified alone (110) beats
       antithetic+control (91); on the OTM call, antithetic+control (92) beats
       both of its parts by far more than their product (3.04 x 2.71 = 8.2) --
       because the control variate is much better correlated with the
       pair-averaged payoff than with the raw one (0.9643 against 0.7522),
       which is a real interaction and not an accident of the seeds.

    Every cell is asserted within a factor of 1.7 of the recorded value. Since
    the seeds are fixed the measurement is reproducible exactly; the band is the
    statement of how much of it is signal (see the module docstring).
    """
    assert case.row.evidence is EvidenceClass.STATISTICAL
    plain = _estimator_variance(case.spec, "none")
    reduced = _estimator_variance(case.spec, case.variance_reduction)
    factor = plain / reduced
    assert factor > 1.0
    assert abs(math.log(factor) - case.row.expected) <= case.row.tolerance, (
        f"{case.row.id}: measured {factor:.2f}, "
        f"expected {math.exp(case.row.expected):.2f} "
        f"within a factor of {math.exp(case.row.tolerance):.2f}"
    )


@pytest.mark.parametrize("point_id", list(MC_VR_POINTS))
def test_antithetic_factor_matches_two_over_one_plus_rho(point_id) -> None:
    """STATISTICAL: the measured antithetic gain against `2/(1 + rho_a)`.

    Glasserman 4.2: with `m` normals the plain estimator has variance
    `Var(Y)/m` and the antithetic one `Var(Y)(1 + rho_a)/(2m)`, so at equal
    normal draws the ratio is `2/(1 + rho_a)` with
    `rho_a = corr(Y(Z), Y(-Z))`. At equal payoff *evaluations* the same
    correlation gives `1/(1 + rho_a)`, which is the form in which the method's
    floor of 1 (it can never lose) is usually stated.

    Pilot at 400 000 draws, seed 7:

        instrument      rho_a      2/(1+rho_a)   1/(1+rho_a)   measured
        atm_call       -0.5009        4.01          2.00         4.59
        otm_call       -0.1399        2.33          1.16         3.04
        digital_call   -0.7868        9.38          4.69         9.32

    The measured gains run 1.15x, 1.31x and 0.99x the prediction. The two calls
    land above it for a reason that is visible in the sample and not in the
    formula: the plain estimator's realised variance over these 50 seeds is
    1.08 to 1.14 times what its own reported standard errors predict, which
    inflates every ratio that has it in the numerator. That is within the
    spread of a 49-degree-of-freedom variance estimate (relative standard
    deviation 20%) and is the reason the band is 1.7 rather than 1.2.
    """
    assert EvidenceClass.STATISTICAL
    spec = MC_VR_POINTS[point_id]
    rho_a, _ = _pilot_correlations(spec)
    assert rho_a < 0.0, "a monotone payoff must have a negative antithetic correlation"
    predicted = 2.0 / (1.0 + rho_a)
    measured = _estimator_variance(spec, "none") / _estimator_variance(spec, "antithetic")
    assert predicted / BAND < measured < predicted * BAND, (
        f"measured {measured:.2f} against predicted {predicted:.2f}"
    )


@pytest.mark.parametrize("point_id", list(MC_VR_POINTS))
def test_control_factor_matches_one_over_one_minus_rho_squared(point_id) -> None:
    """STATISTICAL: the measured control-variate gain against `1/(1 - rho**2)`.

    Pilot at 400 000 draws, seed 7:

        instrument      rho      1/(1-rho^2)   measured
        atm_call       0.9246       6.89         7.58
        otm_call       0.7522       2.30         2.71
        digital_call   0.7697       2.45         2.11

    The correlation is the whole story and it is a property of the *pair*
    (payoff, control), not of the payoff alone: the discounted terminal spot is
    a near-perfect control for a deep in-the-money call, where the payoff is
    affine in it, and a poor one for anything whose dependence on `S_T` is
    mostly a step or mostly zero. The digital's 0.7697 is not far below the
    OTM call's 0.7522 and buys a similar factor.

    The engine reports this prediction in `meta` as
    `control_variance_factor_predicted`, computed from the *same* sample it
    prices with; this test computes it from an independent pilot, which is what
    makes the comparison a check rather than a restatement.
    """
    assert EvidenceClass.STATISTICAL
    spec = MC_VR_POINTS[point_id]
    _, rho = _pilot_correlations(spec)
    predicted = 1.0 / (1.0 - rho * rho)
    measured = _estimator_variance(spec, "none") / _estimator_variance(
        spec, "control_variate"
    )
    assert predicted / BAND < measured < predicted * BAND, (
        f"measured {measured:.2f} against predicted {predicted:.2f}"
    )


@pytest.mark.parametrize(
    ("point_id", "expected_exponent"),
    [("atm_call", 1.0157), ("otm_call", 1.0384)],
)
def test_stratified_gain_grows_like_k_for_a_vanilla(point_id, expected_exponent) -> None:
    """STATISTICAL + CONVERGENCE_ORDER: the gain is `O(K)`, not `O(K**2)`.

    Measured factors over 50 seeds at 20 480 draws:

        K            8     16     32      64      128     256
        atm_call   14.0   33.2   58.3   110.4   249.2   506.5
        otm_call    4.9   12.0   22.1    42.4    94.2   193.8

    Least-squares slope of `log(1/gain)` on `log(1/K)`: **1.0157** (residual
    0.0652) and **1.0384** (0.0581). Doubling the strata doubles the gain.

    This contradicts the naive expectation, which the slice statement also
    carried: for a *smooth* integrand, stratifying into `K` equal-probability
    bins leaves a within-stratum variance of `O(1/K**2)` and the gain should be
    `O(K**2)`. It is not, and the reason is the tail. Strata are
    equal-probability, so the outermost one covers `Z > Phi^{-1}(1 - 1/K)` and
    all the way out; the call payoff is unbounded there, its conditional
    variance does not shrink with `K` faster than its weight `1/K` grows, and
    that single stratum dominates the residual variance. The gain is therefore
    governed by `Var(Y) / Var(Y | tail)` times `K`, which is linear.

    An estimator that concentrated draws where the variance is (optimal rather
    than proportional allocation, Glasserman 4.3.1) is the standard repair and
    is not implemented here: proportional allocation is what the slice asked
    for, and knowing the within-stratum variances in advance is a different
    problem. The linear-in-K behaviour is the honest description of what this
    sampler does.
    """
    assert EvidenceClass.CONVERGENCE_ORDER
    spec = MC_VR_POINTS[point_id]
    plain = _estimator_variance(spec, "none")
    strata = (8, 16, 32, 64, 128, 256)
    gains = [plain / _estimator_variance(spec, "stratified", k) for k in strata]
    assert gains == sorted(gains)
    fit = fit_convergence_order([1.0 / k for k in strata], [1.0 / g for g in gains])
    assert abs(fit.order - expected_exponent) < 0.20, f"exponent {fit.order:.4f}"
    assert fit.residual < 0.15


def test_stratified_gain_for_a_digital_is_set_by_where_the_jump_falls() -> None:
    """STATISTICAL: gain `= K p(1-p) / (f(1-f))`, and it is NOT monotone in `K`.

    For a cash-or-nothing call the payoff is constant inside every stratum
    except the one containing the in-the-money boundary `u* = Phi(z*)`, so all
    the residual variance sits in that single stratum. If `f = frac(K u*)` is
    the position of the boundary inside it, the conditional in-the-money
    probability there is `1 - f`, the stratum's variance is `f(1 - f)` and the
    gain is `K p(1-p) / (f(1-f))` with `p = N(d2)`.

    Measured over 50 seeds at 20 480 draws, `u* = 0.440382`, `p = 0.559618`:

        K        f       predicted   measured   ratio
        8      0.5231       7.90        9.00    1.14
        16     0.0461      89.64      108.82    1.21
        32     0.0922      94.19      115.32    1.22
        64     0.1845     104.84      101.26    0.97
        128    0.3689     135.49      150.07    1.11
        256    0.7379     326.19      385.85    1.18

    Two consequences, both of which contradict the slice statement's
    expectation of "a very large gain for the digital" that grows with `K`:

    - **The gain is not monotone in `K`.** `K = 16` (108.8) beats `K = 64`
      (101.3), because 16 strata happen to put the jump 4.6% into a stratum
      while 64 put it 18.4% in. Doubling the strata can make a digital *worse*.
    - **It is a lottery on the contract, not a property of the method.** Move
      the strike and the same `K` gives a different factor; the quantity that
      controls it is `f`, which depends on the strike, the drift, the
      volatility and the maturity through `u*`.

    The vanilla has no such structure -- its payoff varies inside every stratum
    -- which is why its gain is a clean `O(K)` and the digital's is not.
    """
    assert EvidenceClass.STATISTICAL
    from scipy.stats import norm

    z_star = (
        math.log(DIGITAL_CALL.strike / DIGITAL_CALL.spot)
        - (MU - 0.5 * SIGMA * SIGMA) * EXPIRY
    ) / (SIGMA * math.sqrt(EXPIRY))
    u_star = float(norm.cdf(z_star))
    p_itm = 1.0 - u_star

    plain = _estimator_variance(DIGITAL_CALL, "none")
    gains = {}
    for k in (8, 16, 32, 64, 128, 256):
        f = (k * u_star) % 1.0
        predicted = k * p_itm * (1.0 - p_itm) / (f * (1.0 - f))
        measured = plain / _estimator_variance(DIGITAL_CALL, "stratified", k)
        gains[k] = measured
        assert predicted / BAND < measured < predicted * BAND, (
            f"K={k}: measured {measured:.2f} against predicted {predicted:.2f}"
        )
    # The non-monotonicity, pinned: more strata is not always better.
    assert gains[16] > gains[64]


# ---------------------------------------------------------------------------
# (f) The Greeks path inherits the reduction.
# ---------------------------------------------------------------------------

# (label, variance_reduction, measured sd of the CRN bump delta over 20 seeds)
_DELTA_NOISE = (
    ("none", "none", 0.004481),
    ("antithetic", "antithetic", 0.001175),
    ("control", "control_variate", 0.001639),
    ("stratified", "stratified", 0.000254),
    ("antithetic+control", ("antithetic", "control_variate"), 0.001060),
    ("stratified+control", ("stratified", "control_variate"), 0.000367),
)


def test_crn_bump_delta_noise_falls_with_the_estimator() -> None:
    """STATISTICAL: the CRN bump delta inherits the variance reduction, and more.

    `greeks_european` revalues at bumped inputs through `price_european` with
    the same `cfg`, hence the same seed, hence the same normals -- common
    random numbers, unchanged by this slice. Standard deviation of the
    estimated delta over 20 seeds, ATM call, 20 480 normal draws, spot bump
    `h = 1e-2` (analytic delta 0.636830):

        estimator             sd(delta)   sd ratio   variance ratio   bias
        none                   0.004481      1.0          1.0       +8.7e-04
        antithetic             0.001175      3.8         14.5       -6.0e-05
        control_variate        0.001639      2.7          7.5       +4.7e-04
        stratified             0.000254     17.6        311.2       -3.9e-05
        antithetic+control     0.001060      4.2         17.9       -2.5e-05
        stratified+control     0.000367     12.2        149.2       +5.1e-05

    Two results that were not in the slice statement.

    - **Stratification helps the Greek more than it helps the price** (311x
      against 110x on the same configuration). Under common random numbers the
      difference quotient is driven by the paths whose payoff changes when the
      spot moves, i.e. by the ones near the strike; stratification controls
      exactly where those paths land, so the numerator of the quotient is far
      better behaved than the price itself.
    - **Adding the control variate to the stratified estimator makes delta
      *worse*** (0.000367 against 0.000254) while making the price better
      (578x against 110x). The coefficient is refitted on each bumped sample,
      so `b` is itself a random function of the bump, and the difference
      quotient picks up `(b_up - b_dn)` noise that common random numbers cannot
      cancel. A fixed coefficient, or one fitted once and reused across the
      bumps, would remove it; that is a real design option and is not taken
      here, because a fixed `b` is a different estimator with a different bias
      and this slice ships one.
    """
    assert EvidenceClass.STATISTICAL
    option = ATM_CALL.option()
    analytic_delta = greeks(option, MODEL, MARKET, method="analytic").delta
    sds = {}
    for label, vr, recorded in _DELTA_NOISE:
        deltas = []
        for seed in range(1, 21):
            cfg = MCConfig(
                n_paths=paths_for_draws(vr, N_DRAWS),
                n_steps=1,
                seed=seed,
                variance_reduction=vr,
                n_strata=N_STRATA,
            )
            deltas.append(
                greeks(
                    option, MODEL, MARKET, method="mc", cfg=cfg, bumps={"spot": 1e-2}
                ).delta
            )
        deltas = np.array(deltas)
        sd = float(deltas.std(ddof=1))
        sds[label] = sd
        assert recorded / BAND < sd < recorded * BAND, f"{label}: sd {sd:.6f}"
        # Still the same Greek: the reduced estimators must not have moved it.
        z = (float(deltas.mean()) - analytic_delta) / (sd / math.sqrt(20))
        assert abs(z) < 4.0, f"{label}: delta z {z:+.3f}"

    assert sds["antithetic"] < sds["none"] / 2.5
    assert sds["stratified"] < sds["none"] / 8.0
    # The one that goes the wrong way, pinned rather than smoothed over.
    assert sds["stratified+control"] > sds["stratified"]


def test_every_variance_reduction_row_states_its_evidence_and_source() -> None:
    """The row set's own metadata contract, and the fourth id space's disjointness.

    `qpl.cases.mc_variance_reduction` is the first case module keyed by an
    *estimator* rather than by an instrument, so it carries a vanilla and a
    digital point in the same table and its ids must not collide with any of
    the three instrument id spaces. This is the same check
    `tests/cases/test_european_black_scholes_cases.py` runs on `ALL_CASES`,
    extended across all four.
    """
    from qpl.cases import ALL_AMERICAN_CASES, ALL_CASES, ALL_DIGITAL_CASES

    ids = [case.row.id for case in MC_VARIANCE_REDUCTION_CASES]
    assert len(set(ids)) == len(ids)
    others = {
        case.row.id
        for case in (*ALL_CASES, *ALL_AMERICAN_CASES, *ALL_DIGITAL_CASES)
    }
    assert others.isdisjoint(ids)

    for case in MC_VARIANCE_REDUCTION_CASES:
        row = case.row
        assert isinstance(row.evidence, EvidenceClass)
        assert row.evidence is EvidenceClass.STATISTICAL
        assert row.description.strip()
        assert row.source.strip()
        assert row.notes.strip()
        assert row.tolerance > 0.0
        assert case.point_id in MC_VR_POINTS
        assert case.spec is MC_VR_POINTS[case.point_id]
        # Cost bookkeeping: antithetic runs twice the paths for the same draws.
        expected_paths = 2 * N_DRAWS if "antithetic" in case.variance_reduction else N_DRAWS
        assert case.n_paths() == expected_paths
