"""Benchmark cases for American options under Black-Scholes.

Same pattern as `qpl.cases.european_black_scholes`: each case pairs a
market/model/instrument specification with one
:class:`~qpl.validation.BenchmarkRow` stating what is expected, how tight the
comparison is, which :class:`~qpl.validation.EvidenceClass` justifies it, and
where the expectation comes from. The lists contain no assertions and no
pytest dependency; `tests/cases/test_american_black_scholes_cases.py`
evaluates them.

There is no closed form for an American option, so the evidence here is of
four different kinds and the difference matters:

- an **exact identity** that holds on the lattice itself (the no-dividend
  American call, the zero-rate put premium);
- a **closed-form bound** on the early-exercise premium, derived below;
- the one **published benchmark** available to this slice, Longstaff and
  Schwartz (2001) Table 1 row 1 -- which turns out not to be the number this
  engine computes, for a reason worth reading (see `_LS_SOURCE` and the
  negative-finding row);
- an **in-repo reference value** produced by this engine and pinned, which is
  cross-checked by an independent engine in `tests/oracle/`.

None of the expected values is copied from a published table except the single
cited Longstaff-Schwartz figure, and that row says exactly what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..engines.tree.lattice import crr_parameters
from ..exceptions import InvalidInputError
from ..instruments.options import AmericanOption, EuropeanOption
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_AMERICAN_CASES",
    "AMERICAN_BRACKETED_LIMIT",
    "AMERICAN_CROSS_ENGINE_CASES",
    "AMERICAN_CROSS_ENGINE_TREE_N_STEPS",
    "AMERICAN_IDENTITY_CASES",
    "AMERICAN_LR_CASES",
    "AMERICAN_LR_LEVELS",
    "AMERICAN_LSM_ATM_BERMUDAN_GAP",
    "AMERICAN_LSM_ATM_DATES",
    "AMERICAN_LSM_ATM_FINITE_SAMPLE_BIAS",
    "AMERICAN_LSM_ATM_PATHS",
    "AMERICAN_LSM_ATM_SEED",
    "AMERICAN_LSM_ATM_STDERR",
    "AMERICAN_LSM_BOUNDARY_TIMES",
    "AMERICAN_LSM_BOUNDARY_TOLERANCE",
    "AMERICAN_LSM_CASES",
    "AMERICAN_PDE_N",
    "AMERICAN_PDE_STRIKE_ALIGNMENT",
    "AMERICAN_PDE_TIME_STEPPING",
    "AMERICAN_PREMIUM_CASES",
    "AMERICAN_REFERENCE_CASES",
    "AMERICAN_REFERENCE_N_STEPS",
    "AMERICAN_REFERENCE_SPEC",
    "AMERICAN_REFERENCE_VALUE",
    "LS2001_BERMUDAN_EXERCISES_PER_YEAR",
    "LS2001_BRACKETED_LIMIT",
    "LS2001_CASES",
    "LS2001_LSM_PATHS",
    "LS2001_LSM_SEED",
    "LS2001_N_STEPS",
    "LS2001_PUBLISHED_LSM_STDERR",
    "LS2001_PUBLISHED_LSM_VALUE",
    "LS2001_ROW1",
    "LSM_BASIS",
    "LSM_DEGREE",
    "LSM_EXERCISE_FREQUENCIES",
    "LSM_LATTICE_N_STEPS",
    "LSM_STDERR_MULTIPLE",
    "PREMIUM_STRIKE_LADDER",
    "AmericanBSCase",
    "AmericanBSSpec",
    "bermudan_value_on_lattice",
]


@dataclass(frozen=True)
class AmericanBSSpec:
    """A single American Black-Scholes pricing point.

    Holds only scalars, so a case list is plain comparable data; the domain
    objects are built on demand.
    """

    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float
    kind: Literal["call", "put"] = "put"

    def option(self) -> AmericanOption:
        return AmericanOption(kind=self.kind, strike=self.strike, expiry=self.expiry)

    def european_option(self) -> EuropeanOption:
        """The otherwise identical European option, for premium comparisons."""
        return EuropeanOption(kind=self.kind, strike=self.strike, expiry=self.expiry)

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def with_strike(self, strike: float) -> AmericanBSSpec:
        return replace(self, strike=strike)


@dataclass(frozen=True)
class AmericanBSCase:
    """One benchmark row plus the point(s) it is evaluated at."""

    row: BenchmarkRow
    specs: tuple[AmericanBSSpec, ...]

    @property
    def spec(self) -> AmericanBSSpec:
        if len(self.specs) != 1:
            raise ValueError(f"case {self.row.id} holds {len(self.specs)} specs, not 1")
        return self.specs[0]


def bermudan_value_on_lattice(
    spec: AmericanBSSpec, *, n_steps: int, n_exercise: int
) -> float:
    """Price `spec` as a Bermudan with `n_exercise` equally spaced dates.

    Same CRR lattice and the same continuation step as
    `qpl.engines.tree.price_american`; the only change is that the Bellman
    maximum against the intrinsic value is taken at every
    `n_steps / n_exercise`-th level instead of at every level. `n_steps` must
    be divisible by `n_exercise`, so the exercise dates land exactly on lattice
    levels and nothing is interpolated. The dates are
    `t_i = i T / n_exercise`, `i = 1 ... n_exercise`, which is the grid
    `qpl.engines.mc.american` simulates on: `t = 0` is not an exercise date in
    either.

    This lives here rather than in `qpl.engines` because `qpl` still has no
    Bermudan *instrument*. Slice 2 wrote it inside
    `tests/cases/test_american_black_scholes_cases.py` to check one citation,
    with a note that a second case would be when it earned a home. Slice 11 is
    that second case -- the least-squares Monte Carlo engine prices a Bermudan
    by construction and needs a reference value at several exercise
    frequencies -- so the function moved here, where both test modules can
    read it and where its contract is documented once.

    Raises
    ------
    InvalidInputError
        If `n_exercise` does not divide `n_steps`, or either is non-positive.
    """
    if n_exercise < 1 or n_steps < 1:
        raise InvalidInputError("n_steps and n_exercise must both be >= 1")
    step, remainder = divmod(n_steps, n_exercise)
    if remainder:
        raise InvalidInputError(
            f"n_exercise={n_exercise} must divide n_steps={n_steps} so the "
            "exercise dates fall on lattice levels"
        )

    lattice = crr_parameters(
        sigma=spec.sigma,
        expiry=spec.expiry,
        rate=spec.rate,
        dividend_yield=spec.dividend,
        n_steps=n_steps,
    )
    counts = np.arange(n_steps + 1, dtype=float)
    up_powers = lattice.up**counts
    down_powers = lattice.down**counts

    def spots(level: int) -> np.ndarray:
        return spec.spot * up_powers[: level + 1] * down_powers[level::-1]

    def intrinsic(values: np.ndarray) -> np.ndarray:
        if spec.kind == "call":
            return np.maximum(values - spec.strike, 0.0)
        return np.maximum(spec.strike - values, 0.0)

    exercise_levels = {n_steps - i * step for i in range(n_exercise)}
    values = intrinsic(spots(n_steps))
    for level in range(n_steps - 1, -1, -1):
        values = lattice.discount * (
            lattice.p * values[1:] + (1.0 - lattice.p) * values[:-1]
        )
        if level in exercise_levels:
            values = np.maximum(intrinsic(spots(level)), values)
    return float(values[0])


# --------------------------------------------------------------------------
# (i) The one published benchmark, and what it actually prices.
#
# Longstaff & Schwartz (2001), "Valuing American options by simulation: a
# simple least-squares approach", Review of Financial Studies 14(1), 113-147.
# Table 1, first row: S = 36, sigma = 0.20, T = 1, K = 40, r = 0.06, no
# dividends. The table reports 4.478 by finite differences (and 4.472 for
# their own least-squares Monte Carlo estimate).
#
# The options in that table are **not** continuously exercisable: they are
# exercisable 50 times per year. That distinction is the whole story here.
# Pricing the same specification as a continuously-exercisable American option
# gives 4.4867, which is 8.7e-03 away from 4.478 -- forty times the 2e-03 that
# a naive reading of the table would suggest, and not something a finer tree
# fixes, since 4.4867 is where this engine converges (QuantLib's finite
# difference and binomial American engines land there too; see
# `tests/oracle/test_american_vs_quantlib.py`).
#
# Restricting exercise to 50 equally spaced dates on this same CRR lattice
# reproduces the published figure: 4.477922 at n = 5000 and 4.477826 at
# n = 40000, against 4.478. That is the check the PUBLISHED_BENCHMARK row
# below performs; the continuous-exercise row next to it is a NEGATIVE_FINDING
# recording that the published number does not apply to `AmericanOption`.
#
# `qpl` has no Bermudan instrument, and this slice is not the place to add one:
# the 50-date restriction is applied in the test, on the shared lattice, as
# twenty lines of backward induction.
# --------------------------------------------------------------------------

_LS_SOURCE = (
    "Longstaff & Schwartz (2001), 'Valuing American options by simulation: a "
    "simple least-squares approach', Review of Financial Studies 14(1), "
    "113-147, Table 1, first row (S=36, sigma=0.20, T=1; K=40, r=0.06, no "
    "dividends): finite-difference value 4.478, their LSM estimate 4.472. "
    "Used as a fixture with citation; no table, prose or code is reproduced."
)

LS2001_ROW1 = AmericanBSSpec(36.0, 40.0, 1.0, 0.06, 0.0, 0.20, "put")
"""The Longstaff-Schwartz Table 1 row-1 specification."""

LS2001_BERMUDAN_EXERCISES_PER_YEAR = 50
"""Exercise dates per year in Longstaff & Schwartz's Table 1."""

LS2001_N_STEPS = 5000
"""Lattice size used for both Longstaff-Schwartz rows.

Divisible by `LS2001_BERMUDAN_EXERCISES_PER_YEAR`, so the 50 exercise dates
fall exactly on lattice levels and the Bermudan restriction introduces no
date-interpolation error of its own.
"""

LS2001_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="ls2001_table1_row1_bermudan_50",
            description=(
                "L&S (2001) Table 1 row 1 as a 50-exercise-date Bermudan put on "
                "the CRR lattice: S=36, K=40, T=1, r=6%, q=0, sigma=20%"
            ),
            expected=4.478,
            tolerance=5e-4,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_LS_SOURCE,
            notes=(
                "Tolerance is dominated by the published figure's own quoting "
                "precision: 4.478 is three decimals, so nothing finer than "
                "5e-04 is meaningful to compare against. Measured gaps to "
                "4.478 on this lattice: 7.8e-05 at n=5000 and 1.74e-04 at "
                "n=40000 (the Bermudan value settles at 4.477826), so the "
                "tolerance keeps a factor of about three at the converged "
                "value."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="ls2001_table1_row1_is_not_a_continuous_american_value",
            description=(
                "the continuously-exercisable American put at the L&S row-1 "
                "specification does NOT match the published 4.478"
            ),
            expected=4.478,
            tolerance=2e-3,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_LS_SOURCE,
            notes=(
                "Pinned as a failure of an expectation, not as an agreement. "
                "This slice was specified to check that the tree's American "
                "put at n=5000 lands within 2e-03 of 4.478. It does not: the "
                "measured value is 4.486710 at n=5000, 8.71e-03 away, and the "
                "engine converges to 4.48668 rather than to 4.478. The reason "
                "is the instrument, not the engine -- L&S Table 1 prices "
                "options exercisable 50 times per year, and the companion row "
                "reproduces 4.478 under exactly that restriction. The "
                "assertion is that the gap EXCEEDS this tolerance, so the row "
                "fails if the discrepancy ever quietly disappears."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
)


# --------------------------------------------------------------------------
# (ii) Identities that hold exactly on the lattice.
# --------------------------------------------------------------------------

_MERTON_SOURCE = (
    "Merton (1973), 'Theory of rational option pricing', Bell Journal of "
    "Economics and Management Science 4(1), 141-183: an American call on a "
    "non-dividend-paying stock is never exercised early. Re-derived on the "
    "lattice in docs/notes/american_exercise_on_trees.md; no text or "
    "numbering is reproduced from that paper."
)

_ZERO_RATE_SOURCE = (
    "derived in docs/notes/american_exercise_on_trees.md: early exercise of a "
    "put is paid for out of interest earned on the strike received, so at "
    "r = 0 (with q >= 0, which makes waiting weakly better still) it is never "
    "optimal and the American value equals the European one"
)

AMERICAN_IDENTITY_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="american_call_equals_european_call_without_dividends",
            description=(
                "with q=0 the American call equals the European call on the "
                "same lattice; expected value is the difference, which is zero"
            ),
            expected=0.0,
            tolerance=0.0,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_MERTON_SOURCE,
            notes=(
                "Tolerance is exactly zero, not a round-off budget. The "
                "Bellman maximum is a no-op at every node, and `max(a, b)` "
                "returns `b` itself rather than a rounded copy, so the two "
                "engines agree bit for bit at every n."
            ),
        ),
        specs=(AmericanBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call"),),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="zero_rate_put_has_no_early_exercise_premium",
            description=(
                "at r=0 with q>0 the American put equals the European put on "
                "the same lattice; expected value is the premium, which is zero"
            ),
            expected=0.0,
            tolerance=0.0,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_ZERO_RATE_SOURCE,
            notes="Measured difference exactly 0.0 at n in {51, 200, 501}.",
        ),
        specs=(AmericanBSSpec(100.0, 105.0, 1.0, 0.0, 0.03, 0.30, "put"),),
    ),
)


# --------------------------------------------------------------------------
# (iii) Bounds on the early-exercise premium.
#
# Lower bound: the American holder can always decline to exercise, so
# `P_A >= P_E` and the premium is non-negative. On a lattice this is
# structural -- `max(intrinsic, continuation) >= continuation` at every node --
# so the permitted violation is zero, not a tolerance.
#
# Upper bound (no dividends): the premium cannot exceed the interest that can
# be earned on the strike over the option's life,
#
#     0 <= P_A - P_E <= K (1 - e^{-rT}).
#
# Why: exercising at time `t` hands the holder `K - S_t`, while continuing to
# hold the European put is worth at least its intrinsic-forward bound
# `K e^{-r(T-t)} - S_t` when `q = 0`. The excess of exercising over waiting is
# therefore at most `K (1 - e^{-r(T-t)}) <= K (1 - e^{-rT})`, and discounting
# that back to today only shrinks it. The bound is attained in the deep-in-the
# -money limit, where the American put is worth `K` and the European put is
# worth `K e^{-rT} - S`: measured premium 9.516258 against a bound of
# 9.516258 at `S = 10, K = 100, T = 1, r = 10%`.
# --------------------------------------------------------------------------

_PREMIUM_LOWER_SOURCE = (
    "derived: an American holder may always decline to exercise, so the "
    "American value dominates the European one; on a lattice this is "
    "max(intrinsic, continuation) >= continuation at every node"
)

_PREMIUM_UPPER_SOURCE = (
    "derived in docs/notes/american_exercise_on_trees.md: with q=0 the "
    "European put dominates K e^{-r(T-t)} - S_t at every t, so the gain from "
    "exercising early is at most the interest on the strike, K (1 - e^{-rT})"
)

_PREMIUM_POINTS: tuple[AmericanBSSpec, ...] = (
    AmericanBSSpec(36.0, 40.0, 1.0, 0.06, 0.0, 0.20, "put"),
    AmericanBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "put"),
    AmericanBSSpec(60.0, 100.0, 1.0, 0.08, 0.0, 0.30, "put"),
    AmericanBSSpec(100.0, 140.0, 2.0, 0.10, 0.0, 0.15, "put"),
    AmericanBSSpec(10.0, 100.0, 1.0, 0.10, 0.0, 0.20, "put"),
    AmericanBSSpec(100.0, 100.0, 0.1, 0.02, 0.0, 0.50, "put"),
)

PREMIUM_STRIKE_LADDER: tuple[float, ...] = (60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0)
"""Strikes for the "premium increases with K" comparative-statics row."""

AMERICAN_PREMIUM_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="put_early_exercise_premium_is_non_negative",
            description=(
                "American put >= European put on the same lattice; expected "
                "value is the worst permitted violation, which is zero"
            ),
            expected=0.0,
            tolerance=0.0,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_PREMIUM_LOWER_SOURCE,
        ),
        specs=_PREMIUM_POINTS,
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="put_early_exercise_premium_is_below_interest_on_the_strike",
            description=(
                "with q=0 the early-exercise premium is at most K (1 - e^{-rT}); "
                "expected value is the worst permitted overshoot"
            ),
            expected=0.0,
            tolerance=1e-9,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_PREMIUM_UPPER_SOURCE,
            notes=(
                "Tolerance is a round-off budget: the bound is attained, not "
                "merely approached, in the deep-in-the-money limit (measured "
                "premium 9.516258 against bound 9.516258 at S=10, K=100, T=1, "
                "r=10%), so the check has no slack to spare at that point and "
                "does not need any at the others (premium/bound ratios 0.28, "
                "0.11, 0.90, 0.86, 1.00, 0.05)."
            ),
        ),
        specs=_PREMIUM_POINTS,
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="put_early_exercise_premium_increases_with_strike",
            description=(
                "the put's early-exercise premium is non-decreasing in K; "
                "expected value is the worst permitted ordering violation"
            ),
            expected=0.0,
            tolerance=0.0,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=(
                "derived: the premium is the value of receiving K early, which "
                "scales with K; see docs/notes/american_exercise_on_trees.md"
            ),
            notes=(
                "Measured premia at S=100, T=1, r=5%, q=1%, sigma=20%, n=500: "
                "3.0e-04, 3.1e-02, 1.4e-01, 4.3e-01, 1.0e+00, 2.2e+00, "
                "5.1e+00 for K = 60 ... 140."
            ),
        ),
        specs=tuple(
            AmericanBSSpec(100.0, k, 1.0, 0.05, 0.01, 0.20, "put")
            for k in PREMIUM_STRIKE_LADDER
        ),
    ),
)


# --------------------------------------------------------------------------
# (iv) The in-repo reference point.
#
# S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1, American put, priced by this
# engine at n = 8001 and recorded to full double precision.
#
# This is NOT a published table value. Nobody published it; this repository
# computed it. What makes it usable as a reference is stated separately and
# checked separately:
#
# - CONVERGENCE_ORDER: the value sits on a measured order-1 sequence whose
#   odd and even subsequences bracket the limit
#   (`tests/test_tree_american_convergence.py`). Its own distance from the
#   limit is about 1.8e-04 -- measured, by comparing against the average of
#   n = 64000 and n = 64001, 6.090376463020103.
# - INDEPENDENT_ENGINE: QuantLib's `FdBlackScholesVanillaEngine` on a fine
#   grid reaches the same value from the other side
#   (`tests/oracle/test_american_vs_quantlib.py`).
#
# The pinned row's tolerance is a bit-equality budget for the engine, not an
# accuracy claim about the American put.
# --------------------------------------------------------------------------

AMERICAN_REFERENCE_SPEC = AmericanBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "put")
AMERICAN_REFERENCE_N_STEPS = 8001
"""Lattice size at which the reference value below was produced."""

AMERICAN_REFERENCE_VALUE = 6.0905564143067235
"""This engine's American put value at `AMERICAN_REFERENCE_SPEC`, n = 8001."""

_REFERENCE_SOURCE = (
    "derived in-repo: qpl.engines.tree.price_american at n=8001; NOT a "
    "published table value"
)

AMERICAN_REFERENCE_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="atm_1y_american_put_reference",
            description=(
                "S=K=100, r=5%, q=0, sigma=20%, T=1 American put, pinned at "
                f"n={AMERICAN_REFERENCE_N_STEPS}"
            ),
            expected=AMERICAN_REFERENCE_VALUE,
            tolerance=1e-10,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_REFERENCE_SOURCE,
            notes=(
                "The tolerance pins the engine, not the American put: it is a "
                "round-off budget. It is NOT a bit-equality budget, and cannot "
                "be: the digits were recorded on macOS/arm64 and an 8001-step "
                "rollback amplifies a one-ULP difference in "
                "u = exp(sigma sqrt(dt)) -- which is exactly the kind of "
                "difference a different libm produces -- into 6.7e-13 on the "
                "price (measured, by nudging u with math.nextafter). 1e-12 "
                "would therefore fail on a platform whose exp rounds two ULP "
                "the other way, so the budget is 1e-10, about 150 such ULP and "
                "still six orders of magnitude below the value's own accuracy. "
                "That accuracy is about 1.8e-04, measured against the average "
                "of n=64000 and n=64001 (6.090376463020103), which brackets "
                "the oscillation. The order-1 evidence behind that is in "
                "tests/test_tree_american_convergence.py."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="atm_1y_american_put_reference_vs_quantlib_fd",
            description=(
                "the same reference value against QuantLib's "
                "FdBlackScholesVanillaEngine on a fine grid"
            ),
            expected=AMERICAN_REFERENCE_VALUE,
            tolerance=1e-3,
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=(
                "QuantLib-Python FdBlackScholesVanillaEngine (optional [oracle] "
                "extra); evaluated in tests/oracle/test_american_vs_quantlib.py"
            ),
            notes=(
                "Tolerance derived from both engines' measured refinement in "
                "that test's docstring, not chosen: the tree at n=8001 sits "
                "+1.80e-04 above the limit and QuantLib's FD at "
                "tGrid=xGrid=3200 sits -1.91e-04 below it, so the two differ "
                "by 3.71e-04 and 1e-03 keeps a factor of 2.7."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
)


# --------------------------------------------------------------------------
# (v) The Leisen-Reimer lattice under early exercise.
#
# The European story -- order 2, no oscillation -- does not carry over, and
# the rows below say so rather than assuming it. The American value's dominant
# discretisation error is the location of the early-exercise boundary, which a
# lattice resolves only to its own node spacing; that error is O(1/n) and is
# indifferent to how the terminal grid was chosen. Measured order: 1.06.
#
# What does carry over is a better constant and a useful sign: on this grid
# Leisen-Reimer approaches the limit from BELOW at every n while CRR approaches
# from ABOVE, so the two schemes bracket the value at a shared n -- a cheap
# error bar that CRR alone can only get by pairing an odd and an even lattice.
#
# The reference is `AMERICAN_BRACKETED_LIMIT`, which comes from the CRR engine
# and is therefore independent of the scheme being measured.
# --------------------------------------------------------------------------

AMERICAN_BRACKETED_LIMIT = 6.090376463020103
"""The converged American put value at `AMERICAN_REFERENCE_SPEC`.

Derived in-repo, not published: the average of the CRR engine at `n = 64000`
and `n = 64001`, which bracket the limit from below and above (the odd/even
bracketing measured in `tests/test_tree_american_convergence.py`). Distinct
from `AMERICAN_REFERENCE_VALUE`, which pins this engine's output at a
particular `n = 8001` and sits 1.8e-04 above this number.
"""

AMERICAN_LR_LEVELS: tuple[int, ...] = (25, 51, 101, 201, 401, 801)
"""Refinement grid for the Leisen-Reimer American rows; odd throughout, which
the scheme requires. The same grid as `TREE_ODD_LEVELS`."""

_AMERICAN_LR_SOURCE = (
    "Leisen & Reimer (1996), 'Binomial models for option valuation - "
    "examining and improving convergence', Applied Mathematical Finance 3(4), "
    "319-346, for the lattice. The paper is about European convergence; every "
    "American number below was measured in-repo "
    "(tests/test_tree_lr_convergence.py) and nothing is quoted from it."
)

AMERICAN_LR_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="american_put_on_lr_lattice_is_first_order_not_second",
            description=(
                "the American put on a Leisen-Reimer lattice converges at "
                "order 1, not at the order 2 the European price gets"
            ),
            expected=1.0,
            tolerance=0.25,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_AMERICAN_LR_SOURCE,
            notes=(
                "Measured order 1.0641 with log-space RMS residual 0.0176 "
                "over n in (25, 51, 101, 201, 401, 801), against "
                "AMERICAN_BRACKETED_LIMIT. CRR on the same grid fits 0.9872 "
                "with residual 0.0034. The band is deliberately wide enough "
                "to admit the measured 1.0641 -- the boundary error and the "
                "smaller order-2 terminal error are both present and the "
                "second shrinks faster -- and deliberately too narrow to "
                "admit order 2. The signed errors at n=801 are -3.66e-04 "
                "(Leisen-Reimer) and +1.82e-03 (CRR)."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="american_lr_and_crr_bracket_the_limit_at_the_same_n",
            description=(
                "at a shared odd n the Leisen-Reimer American put is below the "
                "limit and the CRR one above it; expected value is the worst "
                "permitted violation of that ordering"
            ),
            expected=0.0,
            tolerance=1e-12,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_AMERICAN_LR_SOURCE,
            notes=(
                "Measured signed errors against AMERICAN_BRACKETED_LIMIT over "
                "n in (25 ... 801): Leisen-Reimer -1.47e-02 ... -3.66e-04, all "
                "negative; CRR +5.58e-02 ... +1.82e-03, all positive. The "
                "|CRR| / |LR| ratio runs 3.79, 4.05, 4.50, 4.55, 4.79, 4.97. "
                "Tolerance is a round-off budget on an ordering, not an "
                "accuracy claim: the two errors differ by 1e-02 to 1e-04, so "
                "nothing here is close to the boundary."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="american_lr_richardson_does_not_restore_order_two",
            description=(
                "Richardson extrapolation of consecutive odd n on the "
                "Leisen-Reimer lattice does not make the American sequence a "
                "power law; expected value is the order it fails to reach"
            ),
            expected=2.0,
            tolerance=0.2,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_AMERICAN_LR_SOURCE,
            notes=(
                "Pinned as a failure, so the assertion is that the fitted "
                "order lies OUTSIDE this band and the log-space residual is "
                "large. Measured: order 1.3423 with residual 0.7187, and "
                "non-monotone extrapolated errors 7.97e-04, 5.68e-04, "
                "2.87e-05, 6.70e-05, 2.23e-05 over the pairs (25,51) ... "
                "(401,801). Slice 2 measured the same failure on the CRR "
                "lattice (0.30 odd, 0.64 even), and Slice 3 confirms a better "
                "lattice does not fix it: the boundary error is C(n)/n with a "
                "constant that jumps as the node grid steps past the true "
                "boundary, not C/n."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
)


# --------------------------------------------------------------------------
# (vi) Three engines, three discretisations, one number.
#
# Slice 5 added a second engine that can price early exercise: the
# finite-difference PSOR solve in `qpl.engines.pde.american`. With the two
# lattice schemes that makes three genuinely different discretisations of the
# same free-boundary problem --
#
#   - CRR: a recombining binomial lattice, forward-matching probability,
#     geometric node spacing, converging to the limit from ABOVE at odd `n`;
#   - Leisen-Reimer: the same lattice shape with the Peizer-Pratt inversion,
#     converging from BELOW;
#   - PDE/PSOR: a grid uniform in spot with the strike at a half-integer node
#     position, Crank-Nicolson with Rannacher start-up, the exercise condition
#     enforced as a linear complementarity problem at every time step, also
#     converging from BELOW (measured, `tests/test_pde_american_convergence.py`).
#
# Nothing is shared between them except the model. The rows below state that
# the three agree, and their tolerances are derived from each engine's own
# measured error rather than chosen.
#
# At the ATM reference point, signed errors against `AMERICAN_BRACKETED_LIMIT`
# at the grids these rows use (PDE `n_s = n_t = 800`, both lattices
# `n = 8001`):
#
#     PDE  -4.240e-04     CRR  +1.800e-04     Leisen-Reimer  -3.888e-05
#
# so the worst pairwise gap is PDE-to-CRR, and it is the *sum* of two
# opposite-signed errors, 6.040e-04, not a cancellation. A tolerance of
# 1.5e-03 keeps a factor of 2.5. Anything below about 7e-04 would be asserting
# that the two errors partly cancel; anything above about 4e-03 would pass with
# one engine an order of magnitude out.
#
# At the Longstaff-Schwartz row-1 point the same three grids give
#
#     PDE  -1.786e-04     CRR  -1.613e-05     Leisen-Reimer  -5.734e-05
#
# against `LS2001_BRACKETED_LIMIT`; worst pairwise gap 1.624e-04, tolerance
# 5e-04, a factor of 3.1. Note that all three are on the same side here, which
# is why this point's budget is smaller than the ATM one's despite the same
# grids: a bracket is a feature of a point, not of an engine.
# --------------------------------------------------------------------------

AMERICAN_PDE_N = 800
"""`n_s = n_t` for the PDE leg of the cross-engine rows.

0.29 s per solve. Fine enough that the PDE's error (4.2e-04 at the ATM point)
is comparable with the lattices' at `n = 8001`, so the comparison is between
three engines of similar accuracy rather than between two good ones and a
coarse one.
"""

AMERICAN_PDE_STRIKE_ALIGNMENT = "midpoint"
AMERICAN_PDE_TIME_STEPPING = "rannacher"
"""Grid settings for that leg. Unaligned, the American PDE sequence is not a
power law at all (log-space residual 0.27 against 0.027); see
`tests/test_pde_american_convergence.py`."""

AMERICAN_CROSS_ENGINE_TREE_N_STEPS = 8001
"""Lattice size for both tree legs. Odd, which Leisen-Reimer requires, and the
same `n` the reference row already pins for CRR."""

LS2001_BRACKETED_LIMIT = 4.4866721476
"""The converged American put value at `LS2001_ROW1`.

Derived in-repo, not published: the average of the CRR lattice at `n = 64000`
and the Leisen-Reimer lattice at `n = 64001` (4.4866771455 and 4.4866671497),
which differ by 1.0e-05. Distinct from the published 4.478, which is a
50-exercise-date Bermudan value -- see `_LS_SOURCE` and the negative-finding
row above.
"""

_CROSS_ENGINE_SOURCE = (
    "derived in-repo: three discretisations of the same free-boundary problem "
    "-- CRR and Leisen-Reimer lattices (qpl.engines.tree.american) and a "
    "PSOR finite-difference solve (qpl.engines.pde.american). Tolerances "
    "derived from each engine's measured error in "
    "tests/test_pde_american_convergence.py and "
    "tests/test_tree_american_convergence.py; no published value is involved."
)

AMERICAN_CROSS_ENGINE_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="atm_1y_american_put_three_engine_agreement",
            description=(
                "CRR, Leisen-Reimer and PDE/PSOR agree on the ATM American "
                "put; expected value is the worst pairwise gap"
            ),
            expected=0.0,
            tolerance=1.5e-3,
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=_CROSS_ENGINE_SOURCE,
            notes=(
                "Measured values: PDE 6.08995244 at n_s=n_t=800, CRR "
                "6.09055641 and Leisen-Reimer 6.09033758 at n=8001. Pairwise "
                "gaps 6.040e-04 (PDE-CRR), 3.851e-04 (PDE-LR), 2.188e-04 "
                "(CRR-LR). The worst is the sum of two opposite-signed errors "
                "(-4.240e-04 and +1.800e-04 against "
                "AMERICAN_BRACKETED_LIMIT), so the PDE and the CRR lattice "
                "bracket the value at these settings -- asserted separately, "
                "because a bracket is a cheap error bar and losing it would be "
                "a real regression."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="ls2001_row1_three_engine_agreement",
            description=(
                "the same three engines at the Longstaff-Schwartz row-1 "
                "specification, priced as a continuously-exercisable American "
                "put; expected value is the worst pairwise gap"
            ),
            expected=0.0,
            tolerance=5e-4,
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=_CROSS_ENGINE_SOURCE,
            notes=(
                "Measured values: PDE 4.48649360, CRR 4.48665602, "
                "Leisen-Reimer 4.48661481; worst pairwise gap 1.624e-04 "
                "against a tolerance of 5e-04. All three sit BELOW "
                "LS2001_BRACKETED_LIMIT = 4.4866721476 (-1.786e-04, "
                "-1.613e-05, -5.734e-05), so there is no bracket here and the "
                "budget is correspondingly tighter than the ATM row's. The "
                "three-engine value is 4.4867, not the published 4.478, for "
                "the reason the negative-finding row above records."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
)


# --------------------------------------------------------------------------
# (vii) Least-squares Monte Carlo: the third discretisation, and the first
# American engine whose answer carries a standard error.
#
# Slice 11. `qpl.engines.mc.american` prices a **Bermudan** option on a chosen
# exercise grid by the Longstaff-Schwartz regression, so three things have to
# be kept apart and each row below says which it is checking:
#
#   - the published Longstaff-Schwartz *simulation* value, 4.472, which is a
#     Bermudan value produced by an estimator with a standard error of 0.010;
#   - the Bermudan value itself, which `bermudan_value_on_lattice` supplies
#     with no noise at all;
#   - the continuously-exercisable American value, `LS2001_BRACKETED_LIMIT`,
#     which no simulation on a finite grid is estimating.
#
# Slice 2 already pinned that the published 4.478 is a Bermudan finite-
# difference value and not a continuous American one. Slice 11 adds the
# companion: 4.472 is the same instrument priced by the same paper's own
# simulation, so it is the number an LSM reproduction should be compared with.
#
# Every tolerance below is a multiple of a measured standard error, never a
# round number. The multiple is `LSM_STDERR_MULTIPLE`.
# --------------------------------------------------------------------------

LSM_STDERR_MULTIPLE = 3.0
"""Standard errors allowed in every statistical LSM row.

Three, not two: the rows are evaluated at one fixed seed each, so the relevant
false-failure rate is per-row-per-run rather than an average over seeds, and a
two-sigma band on nine such rows would fail somewhere about once every three
full-suite runs on a different platform's RNG rounding."""

LSM_BASIS = "laguerre"
LSM_DEGREE = 3
"""The basis Longstaff & Schwartz (2001) section 2 describes: a constant plus
the first three weighted Laguerre functions, which is `lsm_degree = 3` in
`MCConfig` (the degree counts the non-constant functions)."""

LSM_LATTICE_N_STEPS = 5_000
"""Lattice size for every Bermudan reference in these rows.

Divisible by 10, 50 and 250, so each exercise grid lands exactly on lattice
levels. Its own discretisation error is 1.1e-04 (Longstaff-Schwartz point) and
1.8e-04 (ATM) against `n = 50000`, thirty times below the standard errors it is
compared against."""

LSM_EXERCISE_FREQUENCIES: tuple[int, ...] = (10, 50, 250)
"""Exercise counts for the Bermudan-gap study."""

LS2001_LSM_PATHS = 100_000
LS2001_LSM_SEED = 20260913
"""Longstaff & Schwartz's own sample size -- 100 000 paths drawn as 50 000
antithetic pairs -- and the seed the rows are evaluated at."""

LS2001_PUBLISHED_LSM_VALUE = 4.472
LS2001_PUBLISHED_LSM_STDERR = 0.010
"""Their Table 1 row-1 simulation value and the standard error printed with
it. Used as fixtures with the citation in `_LS_SOURCE`; no table is
reproduced."""

_LS2001_LSM_IN_SAMPLE_STDERR = 6.032e-3
_LS2001_LSM_OUT_OF_SAMPLE_STDERR = 6.057e-3
"""Standard errors this engine reports at `LS2001_LSM_SEED`, measured. The
in-sample one is 0.60 of the 0.010 printed in the table, at the same nominal
sample size -- the difference is that the table's estimator had a different
seed, not a different amount of information."""

AMERICAN_LSM_ATM_DATES = 250
AMERICAN_LSM_ATM_PATHS = 100_000
AMERICAN_LSM_ATM_SEED = 7
AMERICAN_LSM_ATM_STDERR = 1.309e-2
"""Settings and the measured standard error for the ATM cross-method row."""

AMERICAN_LSM_ATM_BERMUDAN_GAP = 2.46e-3
"""`AMERICAN_BRACKETED_LIMIT` minus the 250-date Bermudan on the lattice: the
part of the ATM cross-method gap that is the instrument, not the estimator."""

AMERICAN_LSM_ATM_FINITE_SAMPLE_BIAS = 1.89e-2
"""The part that is neither the instrument nor the noise.

Measured over ten seeds at `(250 dates, 100 000 antithetic paths)`: the mean
out-of-sample estimate is 6.06911 against a lattice Bermudan of 6.087958. It is
the low bias of a policy fitted on a finite sample, it decays like `N^{-1/2}`
(3.29e-02 at 20 000 paths, 2.30e-02 at 50 000, 1.89e-02 at 100 000), and it is
**point-dependent**: the same measurement at `LS2001_ROW1` gives -1.5e-03, an
order of magnitude smaller at the same settings. QuantLib's `MCAmericanEngine`
shows the same effect at the same point, so it is the method and not this
implementation (`tests/oracle/test_lsm_vs_quantlib.py`).

The slice this row was written for expected the cross-method budget to be
"the standard error plus the Bermudan gap". This term is larger than both and
is why that expectation is recorded here rather than quietly absorbed."""

AMERICAN_LSM_BOUNDARY_TIMES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
AMERICAN_LSM_BOUNDARY_TOLERANCE = 1.2
"""Absolute tolerance on the LSM exercise boundary, in strike units: 1.8 times
the worst deviation measured against the PSOR boundary over three seeds and
five dates (0.65 on a boundary of 33 to 36, i.e. 2.0%)."""

_LSM_SOURCE = (
    "Longstaff & Schwartz (2001), Review of Financial Studies 14(1), 113-147, "
    "sections 1-2 for the algorithm and the weighted Laguerre basis, and Table "
    "1 row 1 for the simulation value 4.472 with standard error 0.010; "
    "Glasserman (2003), Monte Carlo Methods in Financial Engineering, sections "
    "8.6 (in-sample bias of a fitted continuation value) and 8.7 (high- and "
    "low-biased estimators); Clement, Lamberton & Protter (2002), Finance and "
    "Stochastics 6, 449-471, for convergence in the number of paths and basis "
    "functions. Derived independently in qpl.engines.mc.american and "
    "docs/notes/lsm_american_monte_carlo.md; the only figures taken from a "
    "source are 4.472 and its 0.010, used as fixtures with this citation."
)

_LSM_IN_REPO_SOURCE = (
    "derived in-repo: qpl.engines.mc.american against "
    "qpl.cases.bermudan_value_on_lattice, qpl.engines.tree.american and "
    "qpl.engines.pde.american; measured in tests/test_lsm_american.py and "
    "tests/test_lsm_american_bias.py"
)

AMERICAN_LSM_CASES: tuple[AmericanBSCase, ...] = (
    AmericanBSCase(
        row=BenchmarkRow(
            id="ls2001_row1_lsm_reproduces_the_published_simulation_value",
            description=(
                "least-squares Monte Carlo at Longstaff & Schwartz's own "
                "settings (100 000 paths as 50 000 antithetic pairs, 50 "
                "exercise dates, constant + three weighted Laguerre functions, "
                "in-sample) against their published 4.472"
            ),
            expected=LS2001_PUBLISHED_LSM_VALUE,
            tolerance=LSM_STDERR_MULTIPLE * _LS2001_LSM_IN_SAMPLE_STDERR,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_LS_SOURCE + " " + _LSM_SOURCE,
            notes=(
                "Measured 4.467550 with standard error 6.032e-03 at seed "
                "20260913; the gap to 4.472 is 4.45e-03, or 0.74 of this run's "
                "own standard error and 0.44 of the 0.010 the table prints. "
                "The tolerance is three standard errors, so this is a "
                "statistical statement about a published estimate and not a "
                "claim that two Monte Carlo runs at different seeds agree to "
                "four figures. The value is a 50-date BERMUDAN value: the "
                "continuously-exercisable American put at this specification "
                "is LS2001_BRACKETED_LIMIT = 4.4867, and the companion "
                "negative-finding row above says so."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="ls2001_row1_lsm_out_of_sample_matches_the_50_date_bermudan",
            description=(
                "the same settings with the policy fitted on an independent "
                "path set (Glasserman section 8.7) against the 50-date "
                "Bermudan on the CRR lattice; expected value is the gap"
            ),
            expected=0.0,
            tolerance=LSM_STDERR_MULTIPLE * _LS2001_LSM_OUT_OF_SAMPLE_STDERR,
            evidence=EvidenceClass.STATISTICAL,
            source=_LSM_IN_REPO_SOURCE,
            notes=(
                "Measured 4.472996 with standard error 6.057e-03 against a "
                "lattice Bermudan of 4.477922 at n=5000: gap -4.93e-03, or "
                "0.81 standard errors. Over ten seeds the mean is 4.47634 "
                "(standard error of the mean 1.43e-03), so the residual low "
                "bias at 100 000 paths is about -1.5e-03 at this point. The "
                "reference is deliberately the lattice Bermudan and not the "
                "published 4.472: this estimator has a different bias "
                "direction from the one the table reports, so comparing it "
                "with a published simulation would be comparing two different "
                "estimators of the same quantity."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="lsm_bermudan_gap_is_first_order_in_the_exercise_count",
            description=(
                "the gap between the Bermudan and the continuously-exercisable "
                "value is O(1/m) in the number of exercise dates; expected "
                "value is the order"
            ),
            expected=1.0,
            tolerance=0.1,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_LSM_IN_REPO_SOURCE,
            notes=(
                "Measured on the lattice, where there is no noise: gaps "
                "4.4029e-02, 8.750e-03 and 1.672e-03 at m = 10, 50, 250 "
                "against LS2001_BRACKETED_LIMIT, successive ratios 5.03 and "
                "5.23 over a fivefold refinement, fitted order 1.0176 with a "
                "log-space RMS residual of 0.0080. The band excludes order 1/2 "
                "(which is what a barrier's discrete-monitoring bias gives, "
                "and the reason to measure rather than assume) and order 2. "
                "The same study CANNOT be run through the simulation: at "
                "m = 250 the gap is 0.2 of a single 50 000-path run's standard "
                "error, so the order would be a fit to the seed."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="atm_american_put_lsm_agrees_with_the_lattice_and_the_grid",
            description=(
                "the ATM American put from three discretisations -- "
                "Leisen-Reimer lattice, PSOR grid, and a path sample; expected "
                "value is the worst gap between the simulation and either"
            ),
            expected=0.0,
            tolerance=(
                AMERICAN_LSM_ATM_BERMUDAN_GAP
                + AMERICAN_LSM_ATM_FINITE_SAMPLE_BIAS
                + LSM_STDERR_MULTIPLE * AMERICAN_LSM_ATM_STDERR
            ),
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=_LSM_IN_REPO_SOURCE,
            notes=(
                "Measured: Leisen-Reimer 6.09033758 at n=8001, PSOR 6.08995244 "
                "at n_s=n_t=800, LSM 6.05330 with standard error 1.309e-02 at "
                "250 dates and 100 000 antithetic paths. The tolerance is the "
                "sum of three named terms -- Bermudan gap 2.46e-03, "
                "finite-sample low bias 1.89e-02, three standard errors "
                "3.93e-02 -- and the middle one is the largest thing the slice "
                "statement did not anticipate (see "
                "AMERICAN_LSM_ATM_FINITE_SAMPLE_BIAS). At 1.0% of the value "
                "this is a much weaker agreement than the 1.5e-03 the "
                "tree-and-grid row gets, which is the honest price of a third "
                "discretisation that is a sample rather than a mesh."
            ),
        ),
        specs=(AMERICAN_REFERENCE_SPEC,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="lsm_exercise_boundary_tracks_the_grid_and_the_lattice",
            description=(
                "the exercise boundary implied by the fitted regression "
                "against the PSOR and Leisen-Reimer boundaries at five dates; "
                "expected value is the worst deviation"
            ),
            expected=0.0,
            tolerance=AMERICAN_LSM_BOUNDARY_TOLERANCE,
            evidence=EvidenceClass.STATISTICAL,
            source=_LSM_IN_REPO_SOURCE,
            notes=(
                "The LSM boundary at a date is the largest sampled spot at "
                "which exercising beat the fitted continuation, which is an "
                "upper order statistic of a sample and therefore biased "
                "upward; it is also a Bermudan boundary and it carries the "
                "regression's approximation error at exactly the place where "
                "intrinsic and continuation nearly coincide. Measured against "
                "the PSOR boundary over seeds 1, 2, 3 at t = 0.10 ... 0.90: "
                "+0.65, +0.30, +0.49, +0.16, -0.36 worst-case, on a boundary "
                "running 33.08 to 36.31. The lattice's own boundary sits 0.10 "
                "to 0.28 below the grid's, which sets the scale. The sign is "
                "informative: the fitted policy exercises too eagerly early in "
                "the option's life, which is the same suboptimality the price's "
                "low bias measures."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
    AmericanBSCase(
        row=BenchmarkRow(
            id="lsm_in_sample_estimator_is_biased_high",
            description=(
                "fitting and valuing the exercise policy on the same paths "
                "biases the estimate upward; the compared quantity is each "
                "configuration's in-sample gap divided by its own standard "
                "error, and every one must EXCEED expected + tolerance"
            ),
            expected=0.0,
            tolerance=LSM_STDERR_MULTIPLE,
            evidence=EvidenceClass.STATISTICAL,
            source=_LSM_IN_REPO_SOURCE,
            notes=(
                "Measured with a PAIRED design at 2 000 paths over 24 seeds -- "
                "two path sets per seed, a policy fitted on each, both valued "
                "on the same set, so the valuation noise cancels. Gaps "
                "+0.03794 to +0.07436, every one at better than five of its "
                "own standard errors. The naive unpaired comparison (run "
                "in-sample, run out-of-sample, subtract) reads -0.0022 +- "
                "0.0028 over 20 seeds at 50 000 paths: the wrong sign and not "
                "significant. The bias decays like 1/N (fitted order 1.020 "
                "between 2 000 and 8 000 paths) and grows with the number of "
                "basis functions (+0.0283 +- 0.0068 going from four "
                "coefficients to six), which is Glasserman section 8.6. "
                "The row is shaped like the negative-finding rows above -- the "
                "assertion is that the measured quantity EXCEEDS "
                "expected + tolerance -- because the claim is a SIGN with a "
                "confidence attached, not a distance to a number. Measured "
                "z-scores 5.7, 7.5, 10.2, 7.2, 9.7 and 8.9 against the "
                "required 3.0; evaluated in tests/test_lsm_american_bias.py."
            ),
        ),
        specs=(LS2001_ROW1,),
    ),
)


ALL_AMERICAN_CASES: tuple[AmericanBSCase, ...] = (
    LS2001_CASES
    + AMERICAN_IDENTITY_CASES
    + AMERICAN_PREMIUM_CASES
    + AMERICAN_REFERENCE_CASES
    + AMERICAN_LR_CASES
    + AMERICAN_CROSS_ENGINE_CASES
    + AMERICAN_LSM_CASES
)
