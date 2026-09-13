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

from ..instruments.options import AmericanOption, EuropeanOption
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_AMERICAN_CASES",
    "AMERICAN_BRACKETED_LIMIT",
    "AMERICAN_IDENTITY_CASES",
    "AMERICAN_LR_CASES",
    "AMERICAN_LR_LEVELS",
    "AMERICAN_PREMIUM_CASES",
    "AMERICAN_REFERENCE_CASES",
    "AMERICAN_REFERENCE_N_STEPS",
    "AMERICAN_REFERENCE_SPEC",
    "AMERICAN_REFERENCE_VALUE",
    "LS2001_BERMUDAN_EXERCISES_PER_YEAR",
    "LS2001_CASES",
    "LS2001_N_STEPS",
    "LS2001_ROW1",
    "PREMIUM_STRIKE_LADDER",
    "AmericanBSCase",
    "AmericanBSSpec",
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


ALL_AMERICAN_CASES: tuple[AmericanBSCase, ...] = (
    LS2001_CASES
    + AMERICAN_IDENTITY_CASES
    + AMERICAN_PREMIUM_CASES
    + AMERICAN_REFERENCE_CASES
    + AMERICAN_LR_CASES
)
