"""Benchmark cases for European options under Black-Scholes.

Each case pairs a market/model/instrument specification with one
:class:`~qpl.validation.BenchmarkRow` stating what is expected, how tight the
comparison is, which :class:`~qpl.validation.EvidenceClass` justifies it, and
where the expectation comes from. Tests parametrise over these lists; the
lists themselves contain no assertions and no pytest dependency.

Expected values here are either exact identities (parity), limits recomputed
from first principles in this module, or closed-form values evaluated inside
this repository. None of them is copied from a published table; where a number
originates in-repo the ``source`` field says so explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..instruments.options import EuropeanOption
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_CASES",
    "FOURIER_KNOWN_VALUE_METHOD",
    "FOURIER_KNOWN_VALUE_N_TERMS",
    "FOURIER_KNOWN_VALUE_TOLERANCE",
    "FOURIER_KNOWN_VALUE_TRUNCATION_L",
    "KNOWN_VALUE_CASES",
    "LIMIT_CASES",
    "MC_GREEKS_ESTIMATORS",
    "MC_GREEKS_PATHS",
    "MC_GREEKS_SEED",
    "MC_GREEKS_STDERR_MULTIPLE",
    "MC_GREEK_CASES",
    "MC_GREEK_CASE_KEYS",
    "MONOTONICITY_CASES",
    "PARITY_CASES",
    "PDE_GREEKS_N",
    "PDE_GREEKS_STRIKE_ALIGNMENT",
    "PDE_GREEKS_TIME_STEPPING",
    "PDE_GREEK_CASES",
    "REFERENCE_ATM_CALL",
    "REFERENCE_ATM_PUT",
    "TREE_EVEN_LEVELS",
    "TREE_KNOWN_VALUE_TOLERANCE",
    "TREE_LR_KNOWN_VALUE_TOLERANCE",
    "TREE_LR_ORDER_CASES",
    "TREE_LR_REFERENCE_N_STEPS",
    "TREE_ODD_LEVELS",
    "TREE_ORDER_CASES",
    "TREE_REFERENCE_N_STEPS",
    "EuropeanBSCase",
    "EuropeanBSSpec",
]


@dataclass(frozen=True)
class EuropeanBSSpec:
    """A single European Black-Scholes pricing point.

    Holds only scalars so that a case list is plain, comparable data; the
    domain objects are built on demand.
    """

    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float
    kind: Literal["call", "put"] = "call"

    def option(self) -> EuropeanOption:
        return EuropeanOption(kind=self.kind, strike=self.strike, expiry=self.expiry)

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def flipped(self) -> EuropeanBSSpec:
        """The same point with call and put exchanged."""
        from dataclasses import replace

        return replace(self, kind="put" if self.kind == "call" else "call")


@dataclass(frozen=True)
class EuropeanBSCase:
    """One benchmark row plus the point(s) it is evaluated at.

    A single-point case (parity, a limit, a known value) carries one spec. A
    comparative-statics case carries the whole ordered family in ``specs``.
    """

    row: BenchmarkRow
    specs: tuple[EuropeanBSSpec, ...]

    @property
    def spec(self) -> EuropeanBSSpec:
        if len(self.specs) != 1:
            raise ValueError(f"case {self.row.id} holds {len(self.specs)} specs, not 1")
        return self.specs[0]


# --------------------------------------------------------------------------
# (i) Put-call parity: C - P = S e^{-qT} - K e^{-rT}.
#
# This is an exact identity in the model, not an approximation: it follows
# from static replication (long call, short put, and the forward they
# reproduce), so it holds for any volatility and any consistent pricer. The
# residual is therefore expected to be zero up to floating-point round-off,
# and the tolerance is a round-off budget rather than a model-error budget.
# Standard reference: Hull, "Options, Futures, and Other Derivatives", the
# chapter on properties of stock options.
# --------------------------------------------------------------------------

_PARITY_SOURCE = (
    "derived: static replication of C - P by a forward contract; standard "
    "result, e.g. Hull, 'Options, Futures, and Other Derivatives', chapter on "
    "properties of stock options"
)

_PARITY_POINTS: tuple[tuple[str, EuropeanBSSpec], ...] = (
    ("atm_1y", EuropeanBSSpec(100.0, 100.0, 1.0, 0.05, 0.00, 0.20)),
    ("otm_9m_with_div", EuropeanBSSpec(100.0, 110.0, 0.75, 0.03, 0.01, 0.25)),
    ("deep_otm_2y_high_vol", EuropeanBSSpec(80.0, 120.0, 2.0, 0.02, 0.03, 0.40)),
    ("deep_itm_3m_low_vol", EuropeanBSSpec(150.0, 90.0, 0.25, 0.06, 0.00, 0.15)),
    ("short_dated_atm", EuropeanBSSpec(100.0, 100.0, 0.05, 0.01, 0.00, 0.30)),
)

PARITY_CASES: tuple[EuropeanBSCase, ...] = tuple(
    EuropeanBSCase(
        row=BenchmarkRow(
            id=f"parity_{name}",
            description=(
                f"put-call parity residual at S={spec.spot}, K={spec.strike}, "
                f"T={spec.expiry}, r={spec.rate}, q={spec.dividend}, sigma={spec.sigma}"
            ),
            expected=0.0,
            tolerance=1e-10,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_PARITY_SOURCE,
            notes=(
                "Tolerance is a floating-point round-off budget, not a model-error "
                "budget: the identity is exact in the model."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _PARITY_POINTS
)


def parity_residual(call_price: float, put_price: float, spec: EuropeanBSSpec) -> float:
    """`C - P - (S e^{-qT} - K e^{-rT})`, which must vanish exactly."""
    forward_leg = spec.spot * math.exp(-spec.dividend * spec.expiry)
    strike_leg = spec.strike * math.exp(-spec.rate * spec.expiry)
    return (call_price - put_price) - (forward_leg - strike_leg)


# --------------------------------------------------------------------------
# (ii) Degenerate limits.
#
# T = 0: no time remains, so the price is the payoff itself.
# sigma = 0: the spot is deterministic and equals its forward, so the price is
# the discounted intrinsic value of that forward.
#
# The expected numbers below are written out from those two statements rather
# than read back from the pricer, so a sign or discounting error in the engine
# cannot hide behind them.
# --------------------------------------------------------------------------

_LIMIT_SOURCE_T0 = (
    "derived in this module: at T=0 the price is the payoff, max(S-K,0) or "
    "max(K-S,0)"
)
_LIMIT_SOURCE_SIGMA0 = (
    "derived in this module: at sigma=0 the terminal spot is its forward "
    "S e^{(r-q)T}, so the price is e^{-rT} times the intrinsic value of that "
    "forward"
)


def _intrinsic(spec: EuropeanBSSpec) -> float:
    if spec.kind == "call":
        return max(spec.spot - spec.strike, 0.0)
    return max(spec.strike - spec.spot, 0.0)


def _discounted_forward_intrinsic(spec: EuropeanBSSpec) -> float:
    forward = spec.spot * math.exp((spec.rate - spec.dividend) * spec.expiry)
    disc = math.exp(-spec.rate * spec.expiry)
    if spec.kind == "call":
        return disc * max(forward - spec.strike, 0.0)
    return disc * max(spec.strike - forward, 0.0)


_T0_POINTS: tuple[tuple[str, EuropeanBSSpec], ...] = (
    ("t0_call_itm", EuropeanBSSpec(105.0, 100.0, 0.0, 0.05, 0.01, 0.20, "call")),
    ("t0_call_otm", EuropeanBSSpec(95.0, 100.0, 0.0, 0.05, 0.01, 0.20, "call")),
    ("t0_put_itm", EuropeanBSSpec(95.0, 100.0, 0.0, 0.05, 0.01, 0.20, "put")),
    ("t0_put_otm", EuropeanBSSpec(105.0, 100.0, 0.0, 0.05, 0.01, 0.20, "put")),
)

_SIGMA0_POINTS: tuple[tuple[str, EuropeanBSSpec], ...] = (
    ("vol0_call_fwd_itm", EuropeanBSSpec(110.0, 100.0, 1.0, 0.05, 0.02, 0.0, "call")),
    ("vol0_call_fwd_otm", EuropeanBSSpec(100.0, 105.0, 1.0, 0.05, 0.02, 0.0, "call")),
    ("vol0_put_fwd_itm", EuropeanBSSpec(100.0, 105.0, 1.0, 0.05, 0.02, 0.0, "put")),
    ("vol0_put_fwd_otm", EuropeanBSSpec(110.0, 100.0, 1.0, 0.05, 0.02, 0.0, "put")),
)

LIMIT_CASES: tuple[EuropeanBSCase, ...] = tuple(
    EuropeanBSCase(
        row=BenchmarkRow(
            id=name,
            description=f"T=0 limit: {spec.kind} price equals intrinsic value",
            expected=_intrinsic(spec),
            tolerance=1e-12,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_LIMIT_SOURCE_T0,
        ),
        specs=(spec,),
    )
    for name, spec in _T0_POINTS
) + tuple(
    EuropeanBSCase(
        row=BenchmarkRow(
            id=name,
            description=(
                f"sigma=0 limit: {spec.kind} price equals discounted forward intrinsic"
            ),
            expected=_discounted_forward_intrinsic(spec),
            tolerance=1e-12,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_LIMIT_SOURCE_SIGMA0,
        ),
        specs=(spec,),
    )
    for name, spec in _SIGMA0_POINTS
)


# --------------------------------------------------------------------------
# (iii) The reference ATM point.
#
# S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1. The values below were
# produced by evaluating this repository's own closed-form implementation at
# that point and are recorded to full double precision so that any change to
# the formula shows up immediately. They are NOT a published benchmark table:
# the evidence class is CLOSED_FORM, and the only thing they pin is that the
# closed form keeps returning what it returned when the digits were taken.
# --------------------------------------------------------------------------

_REFERENCE_SOURCE = (
    "derived in-repo: qpl.models.black_scholes.bs_price evaluated at this point; "
    "not a published table"
)

REFERENCE_ATM_CALL = EuropeanBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call")
REFERENCE_ATM_PUT = REFERENCE_ATM_CALL.flipped()

KNOWN_VALUE_CASES: tuple[EuropeanBSCase, ...] = (
    EuropeanBSCase(
        row=BenchmarkRow(
            id="atm_1y_call_reference",
            description="S=K=100, r=5%, q=0, sigma=20%, T=1 European call",
            expected=10.450583572185565,
            tolerance=1e-12,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_REFERENCE_SOURCE,
        ),
        specs=(REFERENCE_ATM_CALL,),
    ),
    EuropeanBSCase(
        row=BenchmarkRow(
            id="atm_1y_put_reference",
            description="S=K=100, r=5%, q=0, sigma=20%, T=1 European put",
            expected=5.573526022256971,
            tolerance=1e-12,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_REFERENCE_SOURCE,
        ),
        specs=(REFERENCE_ATM_PUT,),
    ),
)


# --------------------------------------------------------------------------
# (iv) The transform methods (Slice 14).
#
# A fifth route to the same two numbers, and the one that is *least* like the
# other four: it never discretises the dynamics at all. The lattice, the grid
# and the simulation each approximate how the spot moves; a transform method
# takes the terminal law as given by the model's characteristic function and
# approximates only the pairing of that law with the payoff.
#
# Which makes the evidence class worth being careful about. The Fourier leg of
# the cross-engine test is CLOSED_FORM, not INDEPENDENT_ENGINE: the
# characteristic function it reads is the Gaussian transform of the same
# lognormal law the closed form integrates in the first place, so agreement
# says the payoff transform and the closed-form integral agree -- which is
# worth pinning, and is not a second opinion about the model.
# --------------------------------------------------------------------------

FOURIER_KNOWN_VALUE_METHOD = "cos"
"""Which transform method prices the known-value rows in the cross-engine test.

The Fourier-cosine expansion, because it is the only one of the four that
computes a *put* from its own payoff coefficients rather than from the call by
parity, so it is the only one whose agreement on both rows carries information
about both."""

FOURIER_KNOWN_VALUE_N_TERMS = 256
FOURIER_KNOWN_VALUE_TRUNCATION_L = 10.0
"""Package defaults. Both rows are at the floating-point floor well before this
(the error stops improving at `n_terms = 48`), so the settings are the defaults
rather than anything tuned for these two points."""

FOURIER_KNOWN_VALUE_TOLERANCE = 1e-13
"""Absolute tolerance for the transform leg of the cross-engine test.

Measured errors at the two reference rows: -2.665e-14 (call) and -2.665e-15
(put) for COS; +1.954e-14 / +1.421e-14 for Carr-Madan by direct quadrature;
-1.421e-14 / -1.421e-14 for Lewis; exactly zero for Gil-Pelaez. The tolerance
keeps a factor of about four over the worst of those. It is **four orders of
magnitude tighter than the order-2 Leisen-Reimer leg** at 2001 lattice steps
and nine tighter than the CRR leg at 2000, which is the comparison the leg
exists to make: a method that does not discretise the dynamics has no
discretisation error to converge away."""


# --------------------------------------------------------------------------
# (v) The CRR binomial tree: order, oscillation, and the tolerance the
#     cross-engine test uses.
#
# Unlike the rows above, these are claims about a *rate*, not about a price:
# the expected value is a convergence order and the tolerance is the band the
# fitted slope must fall in. They are evaluated at the reference ATM call,
# where the strike sits in the densest part of the terminal grid and the
# odd/even effect is cleanest.
#
# The order-1 behaviour and the odd/even oscillation of the CRR tree are the
# subject of Leisen & Reimer (1996); the numbers quoted in `notes` were
# measured in this repository, not copied from that paper.
# --------------------------------------------------------------------------

TREE_ODD_LEVELS: tuple[int, ...] = (25, 51, 101, 201, 401, 801)
"""Odd step counts for the convergence fit; `h = 1 / n`."""

TREE_EVEN_LEVELS: tuple[int, ...] = (26, 50, 100, 200, 400, 800)
"""Even step counts for the convergence fit; `h = 1 / n`."""

TREE_REFERENCE_N_STEPS = 2000
"""Step count at which the tree prices the known-value rows in the
cross-engine test."""

TREE_KNOWN_VALUE_TOLERANCE = 2.5e-3
"""Absolute tolerance for the tree leg of the cross-engine test.

Derived from the measured error constant rather than guessed. On the reference
ATM point the scaled error `n * |tree - closed form|` tends to about 1.9994 on
even `n`, so at `n = 2000` the predicted error is `1.9994 / 2000 = 1.00e-3`;
the measured error there is 9.998e-04 for both the call and the put. The
tolerance keeps a factor of 2.5 of headroom, which is enough to absorb the
neighbouring odd-`n` constant (1.7529, i.e. 8.8e-04) but not enough to hide a
tree that had fallen to half-order accuracy.
"""

_TREE_ORDER_SOURCE = (
    "order-1 convergence and the odd/even oscillation of the CRR tree: Leisen "
    "& Reimer (1996), 'Binomial models for option valuation - examining and "
    "improving convergence', Applied Mathematical Finance 3(4), 319-346. The "
    "fitted slopes and error constants in `notes` were measured in-repo "
    "(tests/test_tree_convergence.py); they are not quoted from that paper."
)

TREE_ORDER_CASES: tuple[EuropeanBSCase, ...] = (
    EuropeanBSCase(
        row=BenchmarkRow(
            id="tree_crr_order_one_odd_n",
            description=(
                "CRR tree error against the closed form decays like 1/n on odd "
                "step counts"
            ),
            expected=1.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_TREE_ORDER_SOURCE,
            notes=(
                "Measured order 1.0010, log-space RMS residual 0.0005 over "
                "n in (25, 51, 101, 201, 401, 801). The tree price is ABOVE "
                "Black-Scholes at every one of these n; scaled error n*|err| "
                "tends to 1.7529."
            ),
        ),
        specs=(REFERENCE_ATM_CALL,),
    ),
    EuropeanBSCase(
        row=BenchmarkRow(
            id="tree_crr_order_one_even_n",
            description=(
                "CRR tree error against the closed form decays like 1/n on even "
                "step counts"
            ),
            expected=1.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_TREE_ORDER_SOURCE,
            notes=(
                "Measured order 0.9987, log-space RMS residual 0.0007 over "
                "n in (26, 50, 100, 200, 400, 800). The tree price is BELOW "
                "Black-Scholes at every one of these n, so odd and even n "
                "bracket the true value; scaled error n*|err| tends to 1.9994. "
                "At the money an even n places a terminal node exactly on the "
                "strike, which is the source of the parity dependence."
            ),
        ),
        specs=(REFERENCE_ATM_CALL,),
    ),
)


# --------------------------------------------------------------------------
# (vi) The Leisen-Reimer tree: order two, and the margin over CRR.
#
# Same shape as the CRR rows above -- the expected value is a convergence
# order and the tolerance is the band the fitted slope must land in -- but
# three things differ and each is the point of the scheme:
#
# - there is one sequence, not two, because the construction has no even-`n`
#   form and the error therefore has no parity to oscillate with;
# - the rows are evaluated off the money as well as at it, at the two points
#   where `tests/oracle/test_tree_vs_quantlib.py` measured the CRR error
#   constant to be erratic in BOTH engines;
# - the fitted order is about 2 rather than about 1.
#
# The scheme is Leisen & Reimer (1996); the numbers in `notes` were measured
# in this repository (`tests/test_tree_lr_convergence.py`), not copied from
# that paper.
# --------------------------------------------------------------------------

TREE_LR_REFERENCE_N_STEPS = 2001
"""Step count at which the Leisen-Reimer tree prices the known-value rows in
the cross-engine test. Odd, which the scheme requires; chosen next to the CRR
leg's 2000 so the two legs are compared at essentially the same work."""

TREE_LR_KNOWN_VALUE_TOLERANCE = 2.5e-7
"""Absolute tolerance for the Leisen-Reimer leg of the cross-engine test.

Derived from the measurement, not guessed. At `n = 2001` the measured error at
the reference ATM point is -8.853e-08 for both the call and the put, so the
tolerance keeps a factor of 2.8 -- the same headroom the CRR leg's
`TREE_KNOWN_VALUE_TOLERANCE` keeps, at four orders of magnitude tighter. The
CRR leg at `n = 2000` misses by 9.998e-04, i.e. this leg is 11_000 times
closer at the same lattice size.
"""

_TREE_LR_ORDER_SOURCE = (
    "Leisen & Reimer (1996), 'Binomial models for option valuation - "
    "examining and improving convergence', Applied Mathematical Finance 3(4), "
    "319-346, whose construction this package implements. The fitted slopes, "
    "residuals and error ratios in `notes` were measured in-repo "
    "(tests/test_tree_lr_convergence.py); none is quoted from that paper."
)

LR_OTM_2Y_DIV = EuropeanBSSpec(100.0, 110.0, 2.0, 0.03, 0.01, 0.25, "call")
"""Off-the-money point 1: where the CRR order fit came out at 1.21-1.48 with a
log-space residual of 0.37-1.52 in both `qpl` and QuantLib."""

LR_ITM_1Y_DIV = EuropeanBSSpec(120.0, 90.0, 1.0, 0.03, 0.05, 0.35, "call")
"""Off-the-money point 2, same provenance; the CRR errors there run
2.8e-02, 1.3e-02, 2.8e-04, 5.2e-04, 2.0e-03 -- non-monotone."""

_LR_ORDER_POINTS: tuple[tuple[str, EuropeanBSSpec, str], ...] = (
    (
        "atm_1y",
        REFERENCE_ATM_CALL,
        "order 1.9840, log-space RMS residual 0.0087, |error| 5.52e-07 at "
        "n=801. CRR on the same odd grid: order 1.0010, |error| 2.19e-03. "
        "Error ratio CRR/LR 507x at n=101 and 3966x at n=801.",
    ),
    (
        "otm_2y_div",
        LR_OTM_2Y_DIV,
        "order 1.9842, log-space RMS residual 0.0086, |error| 1.06e-06 at "
        "n=801. This is one of the two points where the CRR fit is erratic "
        "(order 1.21-1.48, residual 0.37-1.52). Error ratio CRR/LR 35x at "
        "n=101 -- the weakest cell measured, because CRR's erratic constant "
        "happens to sit near a sign change there -- and 2673x at n=801.",
    ),
    (
        "itm_1y_div",
        LR_ITM_1Y_DIV,
        "order 1.9709, log-space RMS residual 0.0154, |error| 2.26e-07 at "
        "n=801. The second erratic-CRR point. Error ratio CRR/LR 1137x at "
        "n=101 and 5229x at n=801.",
    ),
)

TREE_LR_ORDER_CASES: tuple[EuropeanBSCase, ...] = tuple(
    EuropeanBSCase(
        row=BenchmarkRow(
            id=f"tree_lr_order_two_{name}",
            description=(
                "Leisen-Reimer tree error against the closed form decays like "
                f"1/n**2 on odd step counts at {name}"
            ),
            expected=2.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_TREE_LR_ORDER_SOURCE,
            notes=(
                f"Measured over n in {TREE_ODD_LEVELS}: {note} The fitted "
                "order sits just BELOW 2 at every point rather than "
                "straddling it: the Peizer-Pratt inversion matches the "
                "binomial tail to high but finite order, so a slowly decaying "
                "correction rides on the 1/n**2 term and drags the slope down "
                "by 0.02-0.03 over a sixfold refinement. The error is "
                "one-signed and monotone across the whole grid -- there is no "
                "parity to oscillate with, since the scheme has no even-n "
                "construction."
            ),
        ),
        specs=(spec,),
    )
    for name, spec, note in _LR_ORDER_POINTS
)


# --------------------------------------------------------------------------
# (vii) Greeks read off the finite-difference grid.
#
# These rows are a different shape again: the quantity checked is a *residual*
# against the closed-form Greek, so ``expected`` is 0 and ``tolerance`` is the
# accuracy claim. Each tolerance below is derived from a measurement at the
# stated grid, not chosen; the measurement is quoted in ``notes`` and can be
# reproduced with `tests/test_pde_greeks.py`.
#
# The grid is fixed for every row: Crank-Nicolson with Rannacher start-up,
# strike-aligned, n_s = n_t = 400. That combination is what Slice 4 measured
# to be second order in delta, gamma and theta; plain Crank-Nicolson is second
# order *here* too, but is not on grids where dt is large relative to ds**2,
# and the cases layer should carry the configuration that is safe on both.
# --------------------------------------------------------------------------

PDE_GREEKS_N = 400
"""`n_s = n_t` at which the cross-engine test reads the PDE Greeks."""

PDE_GREEKS_TIME_STEPPING = "rannacher"
"""Time stepping used for the PDE Greek rows; see `PDE_GREEK_CASES`."""

PDE_GREEKS_STRIKE_ALIGNMENT = "midpoint"
"""Strike placement used for the PDE Greek rows."""

_PDE_GREEKS_SOURCE = (
    "derived in-repo: residual of qpl.engines.pde grid Greeks against "
    "qpl.engines.analytic closed-form Greeks at the reference ATM point; the "
    "measured errors in `notes` come from tests/test_pde_greeks.py. The "
    "second-order stencils are standard (any finite-difference text; Tavella "
    "and Randall (2000), 'Pricing Financial Instruments: The Finite "
    "Difference Method'), and the Rannacher start-up is Rannacher (1984), "
    "Numerische Mathematik 43, 309-327, analysed in Giles and Carter (2006), "
    "Journal of Computational Finance 9(4), 89-112. No number here is quoted "
    "from any of those."
)

# (greek, tolerance, measured absolute error at n = 400, note)
_PDE_GREEK_ROWS: tuple[tuple[str, float, str], ...] = (
    (
        "delta",
        5.0e-4,
        "Measured |error| 1.430e-04 for both the call and the put -- the two "
        "are the same number, because the call and put grids differ only in "
        "their boundary data and delta_C - delta_P = e^{-qT} is exact on the "
        "grid. Second-order central stencil, linearly interpolated to the "
        "spot, which at S = K on a midpoint-aligned grid sits exactly halfway "
        "between two nodes. Tolerance keeps a factor of 3.5.",
    ),
    (
        "gamma",
        7.0e-6,
        "Measured |error| 1.720e-06, call and put identical (gamma_C = gamma_P "
        "exactly, by differentiating parity twice). Tolerance keeps a factor "
        "of 4.1. This is the row that plain Crank-Nicolson would fail on a "
        "grid with dt large relative to ds**2; see the NEGATIVE_FINDING in "
        "tests/test_pde_greeks.py.",
    ),
    (
        "vega",
        3.0e-3,
        "Measured |error| 7.887e-04, call and put identical (vega_C = vega_P). "
        "Bump-and-revalue with h = 1e-2, so this inherits the grid's own error "
        "rather than a stencil's -- it is no better than the price, whose "
        "error at this grid is 1.285e-04 on a value of 10.45. Tolerance keeps "
        "a factor of 3.8.",
    ),
    (
        "theta",
        4.0e-3,
        "Measured |error| 1.053e-03 (call) and 1.053e-03 (put). Read from the "
        "PDE identity, not from the last two time levels: the identity is "
        "second order, the time difference is first order and 20x-200x worse. "
        "Tolerance keeps a factor of 3.8.",
    ),
    (
        "rho",
        2.0e-2,
        "Measured |error| 5.674e-03 (call) and 5.645e-03 (put). "
        "Bump-and-revalue with h = 1e-4; like vega it inherits the grid's "
        "error, and rho is the largest Greek here in absolute terms (53.2), "
        "so its absolute residual is correspondingly the largest. Relative "
        "residual 1.07e-04. Tolerance keeps a factor of 3.5.",
    ),
)

PDE_GREEK_CASES: tuple[EuropeanBSCase, ...] = tuple(
    EuropeanBSCase(
        row=BenchmarkRow(
            id=f"pde_grid_{greek}_{'call' if spec is REFERENCE_ATM_CALL else 'put'}",
            description=(
                f"PDE grid {greek} residual against the closed form at "
                f"S=K=100, r=5%, q=0, sigma=20%, T=1 "
                f"({'call' if spec is REFERENCE_ATM_CALL else 'put'}), "
                f"Rannacher, aligned, n_s = n_t = {PDE_GREEKS_N}"
            ),
            expected=0.0,
            tolerance=tolerance,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_PDE_GREEKS_SOURCE,
            notes=note,
        ),
        specs=(spec,),
    )
    for greek, tolerance, note in _PDE_GREEK_ROWS
    for spec in (REFERENCE_ATM_CALL, REFERENCE_ATM_PUT)
)


MC_GREEKS_PATHS = 200_000
MC_GREEKS_SEED = 123
MC_GREEKS_STDERR_MULTIPLE = 4.0
MC_GREEKS_ESTIMATORS: tuple[str, ...] = ("bump", "pathwise", "likelihood_ratio")
"""Settings for the Monte Carlo Greeks leg of the cross-engine claims.

The tolerance is a multiple of **each Greek's own reported standard error**,
which is what makes the leg STATISTICAL rather than an accuracy claim: a Monte
Carlo Greek has no absolute error budget, it has an error bar, and the row
checks that the error bar is honest. Four standard errors is a two-sided
false-failure rate near 6e-05 per cell for a fixed seed.

All three estimators are held to the same budget on a *vanilla*, and they all
meet it: measured |z| at 200 000 paths, seed 123, over the ATM call and put and
all five Greeks, worst 1.364 (put delta, bump) and median 0.50. That is the
finding worth having next to the digital rows, where the same table for
`"bump"` reaches |z| = 531 -- the estimator is not universally bad, it is bad
exactly where the payoff jumps.
"""

_MC_GREEKS_SOURCE = (
    "derived in-repo: qpl.engines.mc Greek estimators against "
    "qpl.engines.analytic closed-form Greeks at the reference ATM points, "
    "each compared to its own reported standard error. The estimators are "
    "re-derived in src/qpl/engines/mc/greeks.py from the terminal lognormal "
    "law; the ideas are Glasserman (2003), 'Monte Carlo Methods in Financial "
    "Engineering', sections 7.1-7.4, and Broadie & Glasserman (1996), "
    "Management Science 42(2), 269-285. No number here is quoted from either."
)

_MC_GREEK_CELLS: tuple[tuple[str, str, EuropeanBSSpec], ...] = tuple(
    (estimator, greek, spec)
    for estimator in MC_GREEKS_ESTIMATORS
    for greek in ("delta", "gamma", "vega", "theta", "rho")
    for spec in (REFERENCE_ATM_CALL, REFERENCE_ATM_PUT)
)


def _mc_greek_row_id(estimator: str, greek: str, spec: EuropeanBSSpec) -> str:
    return f"mc_{estimator}_{greek}_{spec.kind}"


MC_GREEK_CASES: tuple[EuropeanBSCase, ...] = tuple(
    EuropeanBSCase(
        row=BenchmarkRow(
            id=_mc_greek_row_id(estimator, greek, spec),
            description=(
                f"Monte Carlo {estimator} {greek} against the closed form at "
                f"S=K=100, r=5%, q=0, sigma=20%, T=1 ({spec.kind}), "
                f"{MC_GREEKS_PATHS} paths, seed {MC_GREEKS_SEED}"
            ),
            expected=0.0,
            tolerance=MC_GREEKS_STDERR_MULTIPLE,
            evidence=EvidenceClass.STATISTICAL,
            source=_MC_GREEKS_SOURCE,
            notes=(
                "The residual is measured in units of the estimator's own "
                "reported standard error, so `tolerance` is a z-score and not "
                "a price. Gamma under `pathwise` is the mixed LR-PW estimator "
                "(the payoff has no second derivative) and the result records "
                "that in meta['estimator']; gamma and vega under "
                "`likelihood_ratio` carry proportional weights and therefore "
                "the same z. Measured |z| at this seed: worst 1.364, median "
                "0.50, over all thirty cells."
            ),
        ),
        specs=(spec,),
    )
    for estimator, greek, spec in _MC_GREEK_CELLS
)
MC_GREEK_CASE_KEYS: dict[str, tuple[str, str]] = {
    _mc_greek_row_id(estimator, greek, spec): (estimator, greek)
    for estimator, greek, spec in _MC_GREEK_CELLS
}
"""Row id -> `(greeks_estimator, greek)`.

Exported so the test reads the cell off a mapping instead of splitting the id
on underscores -- `"likelihood_ratio"` contains one, and a parser that gets
that wrong fails by testing the *wrong Greek*, silently."""

"""Thirty rows: three estimators x five Greeks x call and put."""


# --------------------------------------------------------------------------
# (iv) Comparative statics.
#
# A call is non-decreasing in spot (its payoff is), non-decreasing in
# volatility (more dispersion cannot hurt a convex payoff), and non-increasing
# in strike (its payoff is). These are ordering facts, so the "tolerance" below
# is the slack permitted on a violation, and the expected value is zero
# violation.
# --------------------------------------------------------------------------

_MONOTONICITY_SOURCE = (
    "derived: comparative statics of a convex, non-decreasing payoff; standard "
    "result, e.g. Hull, 'Options, Futures, and Other Derivatives', chapter on "
    "properties of stock options"
)

MONOTONICITY_CASES: tuple[EuropeanBSCase, ...] = (
    EuropeanBSCase(
        row=BenchmarkRow(
            id="call_nondecreasing_in_spot",
            description="call price is non-decreasing in spot",
            expected=0.0,
            tolerance=1e-12,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_MONOTONICITY_SOURCE,
            notes="expected/tolerance bound the permitted ordering violation",
        ),
        specs=tuple(
            EuropeanBSSpec(s, 100.0, 1.0, 0.01, 0.0, 0.20, "call")
            for s in (80.0, 90.0, 100.0, 110.0, 120.0)
        ),
    ),
    EuropeanBSCase(
        row=BenchmarkRow(
            id="call_nondecreasing_in_sigma",
            description="call price is non-decreasing in volatility",
            expected=0.0,
            tolerance=1e-12,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_MONOTONICITY_SOURCE,
            notes="expected/tolerance bound the permitted ordering violation",
        ),
        specs=tuple(
            EuropeanBSSpec(100.0, 100.0, 1.0, 0.01, 0.0, sigma, "call")
            for sigma in (0.10, 0.20, 0.30, 0.40)
        ),
    ),
    EuropeanBSCase(
        row=BenchmarkRow(
            id="call_nonincreasing_in_strike",
            description="call price is non-increasing in strike",
            expected=0.0,
            tolerance=1e-12,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_MONOTONICITY_SOURCE,
            notes=(
                "ordering is reversed relative to the other two families: the "
                "sequence must be non-increasing"
            ),
        ),
        specs=tuple(
            EuropeanBSSpec(100.0, k, 1.0, 0.01, 0.0, 0.20, "call")
            for k in (90.0, 100.0, 110.0, 120.0)
        ),
    ),
)


ALL_CASES: tuple[EuropeanBSCase, ...] = (
    PARITY_CASES
    + LIMIT_CASES
    + KNOWN_VALUE_CASES
    + TREE_ORDER_CASES
    + TREE_LR_ORDER_CASES
    + PDE_GREEK_CASES
    + MC_GREEK_CASES
    + MONOTONICITY_CASES
)
