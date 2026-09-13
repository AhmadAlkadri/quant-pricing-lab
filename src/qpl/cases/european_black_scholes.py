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
    "KNOWN_VALUE_CASES",
    "LIMIT_CASES",
    "MONOTONICITY_CASES",
    "PARITY_CASES",
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
    PARITY_CASES + LIMIT_CASES + KNOWN_VALUE_CASES + MONOTONICITY_CASES
)
