"""Benchmark cases for the Monte Carlo variance-reduction estimators.

A fourth id space, alongside the European, American and digital ones, and the
first that is keyed by an *estimator* rather than by an instrument: the claim
"antithetic sampling buys a factor of 4.6 on this call" belongs to the sampler,
and the same row shape has to carry a vanilla and a digital point side by side
for the comparison to mean anything. Splitting these rows between
`european_black_scholes` and `digital_black_scholes` would have put the two
halves of one table in two files.

What a row asserts
------------------
The measured quantity is `Var(plain) / Var(reduced)`, where each variance is
the sample variance of the price over `MC_VR_SEEDS` independent seeds and both
estimators are run at the same number of **normal draws** (`MC_VR_DRAWS`), not
the same number of paths -- antithetic sampling evaluates the payoff twice per
draw, so equal paths would hand it free normals.

`expected` and `tolerance` are in **log space**: `expected = log(factor)` and
`tolerance = log(MC_VR_BAND)`, so the comparison
`abs(log(measured) - expected) <= tolerance` is a symmetric multiplicative
band, which is the natural shape for a ratio. An absolute tolerance on a
quantity that ranges from 2 to 578 would have to be either meaningless at the
bottom of that range or vacuous at the top.

Where the expectation comes from
--------------------------------
Two of the three estimators have a closed-form prediction and the rows use it:

- antithetic: `2 / (1 + rho_a)` at equal normal draws, with
  `rho_a = corr(Y(Z), Y(-Z))` (Glasserman 4.2);
- control variate: `1 / (1 - rho**2)` with `rho = corr(Y, X)` and
  `X = e^{-rT} S_T` (Glasserman 4.1).

Both correlations are themselves measured, from an independent pilot sample, so
the "prediction" is a measured prediction; the rows record the value that pilot
produced. Stratified sampling and the two combinations have no closed form
here, and those rows say so: their `expected` is the measurement itself and the
evidence is `STATISTICAL` with an in-repo source. A row whose expectation is a
recorded measurement pins the measurement; it does not explain it.

Reference for the estimators: Glasserman (2003), *Monte Carlo Methods in
Financial Engineering*, chapter 4. Every number here was measured in this
repository; none is quoted from that book. Derivation and the full tables:
`docs/notes/mc_variance_reduction.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..validation import BenchmarkRow, EvidenceClass
from .digital_black_scholes import DIGITAL_REFERENCE_ATM_CALL, DigitalBSSpec
from .european_black_scholes import REFERENCE_ATM_CALL, EuropeanBSSpec

__all__ = [
    "MC_CROSS_ENGINE_DRAWS",
    "MC_CROSS_ENGINE_N_STRATA",
    "MC_CROSS_ENGINE_SEED",
    "MC_CROSS_ENGINE_STDERR_MULTIPLE",
    "MC_CROSS_ENGINE_VARIANCE_REDUCTION",
    "MC_GREEKS_VARIANCE_BAND",
    "MC_GREEKS_VARIANCE_CASES",
    "MC_VARIANCE_REDUCTION_CASES",
    "MC_VR_BAND",
    "MC_VR_DIGITAL_CALL",
    "MC_VR_DRAWS",
    "MC_VR_N_STRATA",
    "MC_VR_OTM_CALL",
    "MC_VR_POINTS",
    "MC_VR_SEEDS",
    "MCGreeksVarianceCase",
    "MCVarianceCase",
    "paths_for_draws",
]

MC_VR_SEEDS: tuple[int, ...] = tuple(range(101, 151))
"""Fifty independent seeds. Fifty because a variance estimated on `n` seeds has
relative standard deviation `sqrt(2/(n-1))`: 20% here, which is what sets
`MC_VR_BAND`. Independence of consecutive seeds is measured in
`tests/test_mc_variance_reduction.py`, not assumed."""

MC_VR_DRAWS = 20_480
"""Standard normal draws per run, the cost axis every ratio is measured on.
`64 * 320`, so each stratum count used in the studies divides it, and the
antithetic runs use `2 * MC_VR_DRAWS` paths to spend the same number of
normals."""

MC_VR_N_STRATA = 64
"""Strata for the `stratified` rows. Not a tuned value: the gain on a vanilla
is measured to grow like `K`, so there is no optimum to find, and on a digital
it is a lottery on where the jump falls between stratum boundaries (both
measured in `tests/test_mc_variance_ratios.py`)."""

MC_VR_BAND = 1.7
"""Multiplicative tolerance on every ratio row.

Derived, not chosen: each variance is a 49-degree-of-freedom estimate, so the
99% band for a ratio of two independent ones is [0.47, 2.11]. A factor of 1.7
sits inside that, and the measured-over-predicted ratios across the rows below
run 0.86 to 1.31, so the band is about five times the spread actually seen
while still being far too tight to hide an estimator that had lost its
reduction."""

MC_CROSS_ENGINE_VARIANCE_REDUCTION = "stratified"
MC_CROSS_ENGINE_DRAWS = 12_800
MC_CROSS_ENGINE_N_STRATA = 64
MC_CROSS_ENGINE_SEED = 123
MC_CROSS_ENGINE_STDERR_MULTIPLE = 4.0
"""Settings for the Monte Carlo leg of the European cross-engine test.

The leg used to be 200 000 plain paths at four standard errors. Stratifying the
terminal normal buys a measured factor of about 110 on this point, so 12 800
paths -- **one sixteenth** of the work -- give a standard error of 1.107e-02
against the plain run's 3.283e-02: three times tighter for 6% of the cost. The
tolerance is unchanged at four of the estimator's own standard errors, which is
the only honest way to state a Monte Carlo agreement; what changed is that the
interval it has to fall inside is now much smaller, so the leg is a stronger
check than it was. Measured |z| at seed 123: 0.954 (call) and 0.905 (put).
"""

MC_VR_OTM_CALL = EuropeanBSSpec(100.0, 120.0, 1.0, 0.05, 0.0, 0.20, "call")
"""The same market as the reference ATM call with the strike moved to 120, so
that only moneyness changes between the two vanilla points. About 19% of the
sample finishes in the money, which is what makes both the antithetic and the
control-variate gains fall."""

MC_VR_DIGITAL_CALL = DIGITAL_REFERENCE_ATM_CALL
"""The digital reference point, identical in market and maturity to the vanilla
ATM call; the payoff is the only difference between those two columns."""

MC_VR_POINTS: dict[str, EuropeanBSSpec | DigitalBSSpec] = {
    "atm_call": REFERENCE_ATM_CALL,
    "otm_call": MC_VR_OTM_CALL,
    "digital_call": MC_VR_DIGITAL_CALL,
}


def paths_for_draws(variance_reduction: str | tuple[str, ...], n_draws: int) -> int:
    """Paths that spend exactly `n_draws` standard normals under this estimator.

    Antithetic sampling draws `n_paths / 2` normals and uses each twice, so
    matching the cost means doubling the path count. Everything else spends one
    normal per path (per step, for a multi-step path, which these rows do not
    use).
    """
    return 2 * n_draws if "antithetic" in variance_reduction else n_draws


@dataclass(frozen=True)
class MCVarianceCase:
    """One variance-ratio claim, with the point and the estimator it is about."""

    row: BenchmarkRow
    point_id: str
    variance_reduction: str | tuple[str, ...]
    spec: EuropeanBSSpec | DigitalBSSpec

    def n_paths(self, n_draws: int = MC_VR_DRAWS) -> int:
        return paths_for_draws(self.variance_reduction, n_draws)


_THEORY_SOURCE = (
    "derived in this module from a measured correlation: the equal-cost "
    "variance ratios 2/(1+rho_a) (antithetic) and 1/(1-rho^2) (control "
    "variate) are re-derived in "
    "src/qpl/engines/mc/variance_reduction.py; the ideas are Glasserman "
    "(2003), 'Monte Carlo Methods in Financial Engineering', sections 4.1 and "
    "4.2. The correlations themselves were measured in-repo on a 400 000-draw "
    "pilot (tests/test_mc_variance_ratios.py). No number is quoted from that "
    "book."
)

_MEASURED_SOURCE = (
    "derived in-repo: 50-seed variance ratio measured by "
    "tests/test_mc_variance_ratios.py at MC_VR_DRAWS normal draws. There is no "
    "closed-form prediction for this cell, so the row pins the measurement "
    "rather than a theory. Stratified sampling with proportional allocation: "
    "Glasserman (2003), section 4.3."
)

# (point id, estimator, measured factor, predicted factor or None, note)
_ROWS: tuple[tuple[str, str | tuple[str, ...], float, float | None, str], ...] = (
    (
        "atm_call",
        "antithetic",
        4.59,
        4.01,
        "rho_a = -0.5009 on the pilot, so 2/(1+rho_a) = 4.01 at equal normal "
        "draws and 1/(1+rho_a) = 2.00 at equal payoff evaluations. Measured "
        "4.59, i.e. 1.15x the prediction; the plain estimator's realised "
        "variance over these 50 seeds is 1.08x what its own reported standard "
        "errors predict, which inflates every ratio with it in the numerator.",
    ),
    (
        "atm_call",
        "control_variate",
        7.58,
        6.89,
        "rho = +0.9246 between the discounted call payoff and the discounted "
        "terminal spot, so 1/(1-rho^2) = 6.89. Measured 7.58. This is the "
        "cell where the control is worth most: at the money roughly half the "
        "sample is in the region where the payoff is affine in S_T.",
    ),
    (
        "atm_call",
        "stratified",
        110.40,
        None,
        "K = 64. No closed form: the gain is Var(Y)/Var_within, and for an "
        "unbounded payoff the outermost equal-probability stratum sets it. "
        "Measured gains 14.0, 33.2, 58.3, 110.4, 249.2, 506.5 for K = 8 ... "
        "256, i.e. a fitted exponent of 1.0157 in K -- linear, not quadratic.",
    ),
    (
        "atm_call",
        ("antithetic", "control_variate"),
        90.80,
        None,
        "Not the product of its parts (4.59 x 7.58 = 34.8): the control is "
        "much better correlated with the pair-averaged payoff (+0.9643) than "
        "with the raw one (+0.9246), because pair-averaging removes most of "
        "the payoff's asymmetry before the regression sees it.",
    ),
    (
        "atm_call",
        ("stratified", "control_variate"),
        577.80,
        None,
        "The best cell in the table, and the only one above 500. The two "
        "methods attack different things -- stratification the placement of "
        "the draws, the control variate the residual dependence on S_T -- so "
        "they compose better here than either does with antithetic.",
    ),
    (
        "otm_call",
        "antithetic",
        3.04,
        2.33,
        "rho_a = -0.1399: with the strike at 120 only about a fifth of the "
        "sample pays, so a reflected pair rarely has one leg in the money and "
        "the negative correlation is weak. 2/(1+rho_a) = 2.33, measured 3.04.",
    ),
    (
        "otm_call",
        "control_variate",
        2.71,
        2.30,
        "rho falls from +0.9246 at the money to +0.7522 here, and 1/(1-rho^2) "
        "is steep up there: the gain falls from 6.89 to 2.30 for a "
        "correlation drop of 0.17.",
    ),
    (
        "otm_call",
        "stratified",
        42.41,
        None,
        "K = 64. Lower than the ATM call's 110.4 for the same reason the "
        "other two methods are lower -- more of the payoff's variance lives "
        "in the tail stratum, which stratification does not subdivide. "
        "Fitted exponent in K: 1.0384.",
    ),
    (
        "otm_call",
        ("antithetic", "control_variate"),
        92.35,
        None,
        "The largest interaction in the table: 92.35 against a product of "
        "8.24. Off the money the pair average is a far smoother function of "
        "the driving normal than either leg, and the discounted spot then "
        "explains almost all of what is left.",
    ),
    (
        "otm_call",
        ("stratified", "control_variate"),
        84.61,
        None,
        "Below antithetic+control here, and above it at the money: the "
        "ordering of the two combinations is not stable across moneyness, so "
        "'which combination is best' is a question about the contract.",
    ),
    (
        "digital_call",
        "antithetic",
        9.32,
        9.38,
        "The largest antithetic gain in the table, and the one the slice "
        "statement expected to be smallest. rho_a = -0.7868: for a "
        "cash-or-nothing call with the in-the-money boundary at z* = -0.15, "
        "1{Z > z*} + 1{-Z > z*} is exactly 1 unless |Z| < 0.15, so the pair "
        "average is constant on 88% of the sample. Measured 9.32 against a "
        "predicted 9.38 -- the closest agreement in the table.",
    ),
    (
        "digital_call",
        "control_variate",
        2.11,
        2.45,
        "rho = +0.7697, barely above the OTM call's 0.7522 and worth about "
        "the same. The discounted spot is a mediocre control for a step "
        "payoff: it explains where S_T lands, not which side of the strike.",
    ),
    (
        "digital_call",
        "stratified",
        101.26,
        None,
        "K = 64. Predicted by K p(1-p) / (f(1-f)) with "
        "f = frac(K Phi(z*)) = 0.1845: all the residual variance sits in the "
        "single stratum containing the jump, so the gain depends on where "
        "inside that stratum the jump falls. It is therefore NOT monotone in "
        "K, and the non-monotonicity is predicted rather than noise: K = 16 "
        "measures 108.8 and K = 20 measures 48.5 (predicted 89.6 and 31.7), "
        "and K = 320 measures 1014.2 against K = 512's 491.0 (predicted "
        "1101.0 and 505.9). More strata can mean less gain.",
    ),
    (
        "digital_call",
        ("antithetic", "control_variate"),
        9.73,
        None,
        "The control variate adds almost nothing to antithetic here (9.32 -> "
        "9.73): after pair-averaging an indicator, what is left is the "
        "|Z| < 0.15 band, and the discounted spot barely distinguishes "
        "inside it.",
    ),
    (
        "digital_call",
        ("stratified", "control_variate"),
        82.32,
        None,
        "*Worse* than stratified alone (101.26). The residual variance lives "
        "in one stratum; a coefficient fitted over the whole sample is fitted "
        "mostly on strata that have no variance left to explain, and the "
        "fitted correction adds noise where it cannot help.",
    ),
)


def _row_id(point_id: str, vr: str | tuple[str, ...]) -> str:
    name = vr if isinstance(vr, str) else "+".join(vr)
    return f"mc_vr_{point_id}_{name}"


MC_VARIANCE_REDUCTION_CASES: tuple[MCVarianceCase, ...] = tuple(
    MCVarianceCase(
        row=BenchmarkRow(
            id=_row_id(point_id, vr),
            description=(
                f"variance of plain / variance of "
                f"{vr if isinstance(vr, str) else '+'.join(vr)} at "
                f"{MC_VR_DRAWS} normal draws, {point_id} "
                f"(log-space claim: log {measured:.2f})"
            ),
            expected=math.log(predicted if predicted is not None else measured),
            tolerance=math.log(MC_VR_BAND),
            evidence=EvidenceClass.STATISTICAL,
            source=_THEORY_SOURCE if predicted is not None else _MEASURED_SOURCE,
            notes=note,
        ),
        point_id=point_id,
        variance_reduction=vr,
        spec=MC_VR_POINTS[point_id],
    )
    for point_id, vr, measured, predicted, note in _ROWS
)
"""Fifteen rows: five estimators at three points. `expected` is the log of the
theoretical factor where one exists and of the measured factor where it does
not; `source` says which."""


# --------------------------------------------------------------------------
# Slice 10: the same question asked about Greek *estimators* rather than about
# samplers. "Which estimator" is a variance question exactly as "which
# sampler" is, and the rows have the same shape -- a log-space ratio with a
# multiplicative band -- so they live here rather than being invented again.
# --------------------------------------------------------------------------

MC_GREEKS_VARIANCE_BAND = 2.0
"""Multiplicative band for the Greek-estimator ratios.

Wider than `MC_VR_BAND` (1.7) and the reason is arithmetic, not caution: these
rows compare two *estimators* over the same 50 seeds, so each variance carries
a relative standard deviation of `sqrt(2/49) = 0.202` and their ratio about
`0.29`. A factor of 2.0 is 2.4 of those, against the 1.8 that 1.7 would be.
The sampler rows can afford the tighter band because most of them have a
closed-form prediction to sit against; none of these does.
"""


@dataclass(frozen=True)
class MCGreeksVarianceCase:
    """One "estimator A is noisier than estimator B" claim, for one Greek.

    Parameters
    ----------
    row
        The claim. `expected` is `log(variance ratio)` and `tolerance` is
        `log(MC_GREEKS_VARIANCE_BAND)`, as for :class:`MCVarianceCase`.
    point_id
        Key into :data:`MC_VR_POINTS`, so the Greek rows and the sampler rows
        are measured at the same points.
    greek
        Which Greek the ratio is about. It matters: at the ATM call the
        likelihood-ratio estimator is 6.5 times noisier than the pathwise one
        in delta and 13.0 times in gamma.
    numerator, denominator
        `MCConfig.greeks_estimator` values. The ratio is
        `Var(numerator) / Var(denominator)`, so a row's factor is always the
        cost of choosing the numerator.
    """

    row: BenchmarkRow
    point_id: str
    greek: str
    numerator: str
    denominator: str
    spec: EuropeanBSSpec | DigitalBSSpec


_GREEKS_MEASURED_SOURCE = (
    "derived in-repo: 50-seed variance ratio between two Greek estimators at "
    "MC_VR_DRAWS normal draws, measured by tests/test_mc_greeks_variance.py. "
    "The estimators are re-derived in src/qpl/engines/mc/greeks.py from the "
    "terminal lognormal law; the ideas are Glasserman (2003), 'Monte Carlo "
    "Methods in Financial Engineering', sections 7.2-7.4, and Broadie & "
    "Glasserman (1996), Management Science 42(2), 269-285. There is no "
    "closed-form prediction for any of these ratios, so each row pins a "
    "measurement. No number is quoted from either source."
)

# (point id, greek, numerator, denominator, measured factor, note)
_GREEKS_ROWS: tuple[tuple[str, str, str, str, float, str], ...] = (
    (
        "atm_call",
        "delta",
        "likelihood_ratio",
        "pathwise",
        6.46,
        "Glasserman's rule in the direction it predicts: the payoff is smooth, "
        "so differentiating it beats weighting it. The likelihood-ratio "
        "estimator keeps the payoff -- an O(10) quantity -- and multiplies it "
        "by a mean-zero score; the pathwise one replaces the payoff by an "
        "indicator bounded by 1. Multiplying by a mean-zero weight cannot "
        "reduce variance. Standard deviations 1.1244e-02 against 4.4236e-03.",
    ),
    (
        "atm_call",
        "gamma",
        "likelihood_ratio",
        "pathwise",
        13.03,
        "The 'pathwise' gamma is really the mixed LR-PW estimator (the payoff "
        "has no second derivative), so this row compares two density "
        "derivatives of different orders: one score against two. Twice the "
        "penalty of the delta row, at the same point and the same draws.",
    ),
    (
        "atm_call",
        "vega",
        "likelihood_ratio",
        "pathwise",
        13.03,
        "Exactly the gamma row's factor, to three decimals, and not by "
        "coincidence: the two likelihood-ratio weights are proportional path "
        "by path, so the Black-Scholes identity vega = S**2 sigma T gamma is "
        "reproduced in the *sample* and not merely in the mean. Asserted "
        "directly in tests/test_mc_greeks.py. Two rows, one measurement.",
    ),
    (
        "atm_call",
        "gamma",
        "bump",
        "pathwise",
        559.64,
        "What a *second* difference costs: the variance carries 1/(N h**4) "
        "because the estimator divides by h**2 rather than h, and at the "
        "default h = 0.01 that is a factor of 10 000 against a signal of "
        "0.0189. Standard deviations 6.7609e-03 against 2.8579e-04. This is "
        "the row that makes the mixed estimator worth having.",
    ),
    (
        "digital_call",
        "delta",
        "bump",
        "likelihood_ratio",
        864.73,
        "The same rule with the payoff's smoothness removed, and the ordering "
        "reverses: on a jump the bump's two legs differ only on the O(h) "
        "fraction of paths that cross the strike, each by a full cash amount. "
        "The pathwise estimator is absent from this row because it does not "
        "exist here -- its almost-everywhere payoff derivative is identically "
        "zero (tests/test_digital_mc.py computes it).",
    ),
)


MC_GREEKS_VARIANCE_CASES: tuple[MCGreeksVarianceCase, ...] = tuple(
    MCGreeksVarianceCase(
        row=BenchmarkRow(
            id=f"mc_greeks_var_{point_id}_{greek}_{numerator}_over_{denominator}",
            description=(
                f"Var({numerator}) / Var({denominator}) for {greek} at "
                f"{point_id}, {MC_VR_DRAWS} normal draws "
                f"(log-space claim: log {measured:.2f})"
            ),
            expected=math.log(measured),
            tolerance=math.log(MC_GREEKS_VARIANCE_BAND),
            evidence=EvidenceClass.STATISTICAL,
            source=_GREEKS_MEASURED_SOURCE,
            notes=note,
        ),
        point_id=point_id,
        greek=greek,
        numerator=numerator,
        denominator=denominator,
        spec=MC_VR_POINTS[point_id],
    )
    for point_id, greek, numerator, denominator, measured, note in _GREEKS_ROWS
)
"""Five rows: which Greek estimator costs what, at the two Slice 7 points.

Both of Glasserman's rules appear here, in opposite directions on the same
axis: the likelihood ratio loses by 6.5 on a smooth payoff's delta and the bump
loses by 865 on a discontinuous one's. That pair is the slice's headline and it
is two rows of the same table."""
