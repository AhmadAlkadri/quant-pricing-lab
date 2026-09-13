"""Benchmark cases for cash-or-nothing digitals under Black-Scholes.

Same shape as `qpl.cases.european_black_scholes`: each case pairs a
specification with one :class:`~qpl.validation.BenchmarkRow` stating what is
expected, how tight the comparison is, which
:class:`~qpl.validation.EvidenceClass` justifies it, and where it comes from.
The lists hold no assertions and no pytest dependency; the tests in
`tests/cases/test_digital_black_scholes_cases.py` parametrise over them.

What is different here is *which* claims are worth carrying, because a digital
stresses each method differently:

- the identities are stronger than a vanilla's (a digital call plus a digital
  put is a zero-coupon bond, exactly, with no dependence on volatility);
- the tree rows record that Leisen-Reimer is order 2 on a digital and CRR is
  order 1/2 off the money -- the opposite of what the slice expected, and the
  reason the cross-engine tree leg is the Leisen-Reimer one;
- the PDE rows carry both the pathology (plain Crank-Nicolson on an unaligned
  grid loses a full order in the **price**) and the remedy;
- the Monte Carlo leg is `STATISTICAL` and the Greeks leg does not exist.

Closed forms: Reiner and Rubinstein (1991), "Unscrambling the binary code",
Risk 4(9), 75-83. Payoff projection as a convergence remedy: Pooley, Forsyth
and Vetzal (2003), Journal of Computational Finance 6(4). Rannacher start-up:
Rannacher (1984), Numerische Mathematik 43, analysed in Giles and Carter
(2006), Journal of Computational Finance 9(4). Every number below was measured
in this repository; none is quoted from those sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from ..instruments.options import DigitalOption
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_DIGITAL_CASES",
    "DIGITAL_CROSS_ENGINE_CASES",
    "DIGITAL_FOURIER_METHODS",
    "DIGITAL_FOURIER_TOLERANCE",
    "DIGITAL_GREEKS_MC_PATHS",
    "DIGITAL_GREEKS_MC_STDERR_MULTIPLE",
    "DIGITAL_IDENTITY_CASES",
    "DIGITAL_KNOWN_VALUE_CASES",
    "DIGITAL_MC_GREEKS_CASES",
    "DIGITAL_MC_PATHS",
    "DIGITAL_MC_SEED",
    "DIGITAL_MC_STDERR_MULTIPLE",
    "DIGITAL_ODD_LEVELS",
    "DIGITAL_PDE_LEVELS",
    "DIGITAL_PDE_N",
    "DIGITAL_PDE_ORDER_CASES",
    "DIGITAL_PDE_PAYOFF_PROJECTION",
    "DIGITAL_PDE_STRIKE_ALIGNMENT",
    "DIGITAL_PDE_TIME_STEPPING",
    "DIGITAL_PDE_TOLERANCE",
    "DIGITAL_REFERENCE_ATM_CALL",
    "DIGITAL_REFERENCE_ATM_PUT",
    "DIGITAL_STRIKE_BUMP",
    "DIGITAL_STRIKE_DERIVATIVE_CASES",
    "DIGITAL_TREE_LR_N_STEPS",
    "DIGITAL_TREE_LR_TOLERANCE",
    "DIGITAL_TREE_ORDER_CASES",
    "DigitalBSCase",
    "DigitalBSSpec",
]


@dataclass(frozen=True)
class DigitalBSSpec:
    """A single cash-or-nothing digital pricing point.

    Scalars only, so a case list stays plain comparable data; the domain
    objects are built on demand. `cash` is carried because every price and
    Greek is linear in it, so a row that dropped the factor would still look
    plausible.
    """

    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float
    kind: Literal["call", "put"] = "call"
    cash: float = 1.0

    def option(self) -> DigitalOption:
        return DigitalOption(
            kind=self.kind, strike=self.strike, expiry=self.expiry, cash=self.cash
        )

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def flipped(self) -> DigitalBSSpec:
        """The same point with call and put exchanged."""
        return replace(self, kind="put" if self.kind == "call" else "call")

    def riskless(self) -> float:
        """`cash * e^{-rT}`: what the call and the put are worth together."""
        return self.cash * math.exp(-self.rate * self.expiry)


@dataclass(frozen=True)
class DigitalBSCase:
    """One benchmark row plus the point it is evaluated at."""

    row: BenchmarkRow
    specs: tuple[DigitalBSSpec, ...]

    @property
    def spec(self) -> DigitalBSSpec:
        if len(self.specs) != 1:
            raise ValueError(f"case {self.row.id} holds {len(self.specs)} specs, not 1")
        return self.specs[0]


# --------------------------------------------------------------------------
# The points.
# --------------------------------------------------------------------------

DIGITAL_REFERENCE_ATM_CALL = DigitalBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call", 1.0)
"""`S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`, one unit of cash. The same
specification the vanilla reference rows use, so the two can be read side by
side."""

DIGITAL_REFERENCE_ATM_PUT = DIGITAL_REFERENCE_ATM_CALL.flipped()

_OTM_9M_DIV = DigitalBSSpec(100.0, 110.0, 0.75, 0.03, 0.01, 0.25, "call", 1.0)
"""Out of the money with a dividend yield. One of the two points where CRR's
digital error is measured to be order 1/2 with a sawtooth."""

_ITM_1Y_DIV = DigitalBSSpec(120.0, 90.0, 1.0, 0.03, 0.05, 0.35, "call", 2.5)
"""In the money, `q > r`, and `cash = 2.5` so that a dropped cash factor shows
up as a factor of 2.5 rather than as nothing."""

_POINTS: tuple[tuple[str, DigitalBSSpec], ...] = (
    ("atm_1y", DIGITAL_REFERENCE_ATM_CALL),
    ("otm_9m_div", _OTM_9M_DIV),
    ("itm_1y_div", _ITM_1Y_DIV),
)


# --------------------------------------------------------------------------
# (i) Static replication: a digital call plus a digital put is a bond.
# --------------------------------------------------------------------------

_IDENTITY_SOURCE = (
    "derived: 1{S_T > K} + 1{S_T < K} = 1 almost surely, so the two digitals "
    "together are a zero-coupon bond paying `cash` at T; the boundary event "
    "has probability zero under a continuous law. Reiner & Rubinstein (1991), "
    "'Unscrambling the binary code', Risk 4(9), 75-83, give the closed forms "
    "this is checked against"
)

DIGITAL_IDENTITY_CASES: tuple[DigitalBSCase, ...] = tuple(
    DigitalBSCase(
        row=BenchmarkRow(
            id=f"digital_parity_{name}",
            description=(
                f"digital call + digital put - cash e^-rT at S={spec.spot}, "
                f"K={spec.strike}, T={spec.expiry}, r={spec.rate}, "
                f"q={spec.dividend}, sigma={spec.sigma}, cash={spec.cash}"
            ),
            expected=0.0,
            tolerance=1e-15,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_IDENTITY_SOURCE,
            notes=(
                "Stronger than put-call parity: the right-hand side does not "
                "depend on the spot, the strike or the volatility at all. The "
                "tolerance is a round-off budget. Measured residual at these "
                "points: 0.0 exactly for the analytic engine."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _POINTS
)


# --------------------------------------------------------------------------
# (ii) The digital as the strike-derivative of the vanilla.
# --------------------------------------------------------------------------

DIGITAL_STRIKE_BUMP = 1e-2
"""Central-difference step in the strike for the `-dC/dK` rows.

Derived: the residual against the closed form over five points and both kinds
is 1.375e-06 at `h = 1e-1`, 1.375e-08 at `h = 1e-2`, 1.424e-10 at `h = 1e-3`
and 1.367e-10 at `h = 1e-4` -- clean `O(h**2)` onto a round-off floor near
1.4e-10. `1e-2` is the largest step still two decades clear of that floor.
"""

_STRIKE_DERIVATIVE_SOURCE = (
    "derived: differentiating the Black-Scholes call in K, the phi(d1)/phi(d2) "
    "terms cancel and -dC/dK = e^{-rT} N(d2), which is the cash-or-nothing "
    "call per unit of cash; equivalently the digital is the limit of a short "
    "call spread. Reference for the closed forms: Reiner & Rubinstein (1991), "
    "Risk 4(9), 75-83. The residuals in `notes` were measured in-repo "
    "(tests/test_digital_analytic.py)"
)

DIGITAL_STRIKE_DERIVATIVE_CASES: tuple[DigitalBSCase, ...] = tuple(
    DigitalBSCase(
        row=BenchmarkRow(
            id=f"digital_strike_derivative_{name}",
            description=(
                "digital price minus the central difference of the vanilla "
                f"price in the strike at {name}"
            ),
            expected=0.0,
            tolerance=5e-8,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_STRIKE_DERIVATIVE_SOURCE,
            notes=(
                f"h = {DIGITAL_STRIKE_BUMP}; worst measured residual 1.375e-08 "
                "over five points and both kinds, so the tolerance keeps 3.6x. "
                "The reference is this package's *vanilla* engine, so the row "
                "ties two engines together rather than restating one."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _POINTS
)


# --------------------------------------------------------------------------
# (iii) Reference values.
# --------------------------------------------------------------------------

_REFERENCE_SOURCE = (
    "derived in-repo: qpl.engines.analytic.digital.digital_price evaluated at "
    "this point; not a published table"
)

DIGITAL_KNOWN_VALUE_CASES: tuple[DigitalBSCase, ...] = (
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_atm_1y_call_reference",
            description="S=K=100, r=5%, q=0, sigma=20%, T=1 cash-or-nothing call, cash=1",
            expected=0.5323248154537634,
            tolerance=1e-15,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_REFERENCE_SOURCE,
            notes=(
                "Equal to e^{-rT} N(d2) = 0.951229 x 0.559618. The vanilla "
                "call at the same point is 10.450584, and its K e^{-rT} N(d2) "
                "term is 100 x this number."
            ),
        ),
        specs=(DIGITAL_REFERENCE_ATM_CALL,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_atm_1y_put_reference",
            description="S=K=100, r=5%, q=0, sigma=20%, T=1 cash-or-nothing put, cash=1",
            expected=0.41890460904695065,
            tolerance=1e-15,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_REFERENCE_SOURCE,
            notes="Sums with the call row to e^{-0.05} = 0.9512294245007140.",
        ),
        specs=(DIGITAL_REFERENCE_ATM_PUT,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_otm_9m_div_call_reference",
            description="S=100, K=110, T=0.75, r=3%, q=1%, sigma=25% call, cash=1",
            expected=0.3088733086771533,
            tolerance=1e-15,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_REFERENCE_SOURCE,
        ),
        specs=(_OTM_9M_DIV,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_itm_1y_div_call_reference",
            description="S=120, K=90, T=1, r=3%, q=5%, sigma=35% call, cash=2.5",
            expected=1.7524781297375458,
            tolerance=1e-15,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_REFERENCE_SOURCE,
            notes="cash = 2.5, so a dropped cash factor shows as a factor of 2.5.",
        ),
        specs=(_ITM_1Y_DIV,),
    ),
)


# --------------------------------------------------------------------------
# (iv) The trees.
# --------------------------------------------------------------------------

DIGITAL_ODD_LEVELS: tuple[int, ...] = (25, 51, 101, 201, 401, 801)
"""Odd step counts for the tree order fits; `h = 1 / n`. Odd throughout,
because the Leisen-Reimer construction has no even-`n` form."""

DIGITAL_TREE_LR_N_STEPS = 2001
"""Step count at which the Leisen-Reimer leg of the cross-engine rows prices."""

DIGITAL_TREE_LR_TOLERANCE = 2.0e-8
"""Absolute tolerance for the Leisen-Reimer cross-engine leg.

Derived from the measurement: at `n = 2001` the errors are -7.152e-10 (atm),
+1.938e-09 (otm) and -5.284e-09 (itm, cash = 2.5). The tolerance keeps a factor
of 3.8 over the worst of those.
"""

_TREE_SOURCE = (
    "Leisen & Reimer (1996), 'Binomial models for option valuation - "
    "examining and improving convergence', Applied Mathematical Finance 3(4), "
    "319-346, and Cox, Ross & Rubinstein (1979), Journal of Financial "
    "Economics 7, 229-263. The fitted slopes, residuals and error ratios in "
    "`notes` were measured in-repo (tests/test_digital_tree_convergence.py); "
    "none is quoted from either paper"
)

DIGITAL_TREE_ORDER_CASES: tuple[DigitalBSCase, ...] = (
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_tree_lr_order_two_atm",
            description=(
                "Leisen-Reimer digital error against the closed form decays "
                "like 1/n**2 at the money"
            ),
            expected=2.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_TREE_SOURCE,
            notes=(
                "Measured 1.9844, log-space residual 0.0085, over n in "
                "(25, 51, 101, 201, 401, 801); errors one-signed and strictly "
                "decreasing. The mechanism is exact rather than asymptotic: "
                "the strike always falls strictly between the two central "
                "terminal nodes, so the lattice price IS "
                "cash e^{-rT} P(Bin(n, p) > n/2), and the Peizer-Pratt "
                "inversion chose p to make that N(d2). A digital is the "
                "contract this construction is best at, not worst -- which "
                "contradicts the expectation the slice was written with."
            ),
        ),
        specs=(DIGITAL_REFERENCE_ATM_CALL,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_tree_lr_order_two_otm",
            description=(
                "Leisen-Reimer digital error decays like 1/n**2 out of the "
                "money too"
            ),
            expected=2.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_TREE_SOURCE,
            notes=(
                "Measured 1.9839, residual 0.0088. The same point where CRR "
                "is order 1/2 -- the error ratio CRR/LR at n = 801 is 4.68e+05."
            ),
        ),
        specs=(_OTM_9M_DIV,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_tree_crr_half_order_otm",
            description=(
                "CRR digital error off the money decays like 1/sqrt(n) with a "
                "sawtooth, not like 1/n"
            ),
            expected=0.5,
            tolerance=0.2,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_TREE_SOURCE,
            notes=(
                "Measured 0.5020 on a block-RMS fit over every odd n from 25 "
                "to 801 (windows 25-50, 51-100, 101-200, 201-400, 401-801; "
                "RMS 3.84e-02 down to 8.39e-03), log-space residual 0.1254, "
                "with 10 sign changes and individual errors spanning 5.11e-02 "
                "to 3.35e-06. A single-n fit over the six standard levels "
                "gives 0.3569 with residual 0.5064, i.e. it is not a power "
                "law at all at fixed n. At the money on odd n CRR IS order 1 "
                "(1.0014, residual 0.0008), because u d = 1 makes the strike "
                "the geometric mean of the two central nodes; that is a "
                "symmetry, not a rule."
            ),
        ),
        specs=(_OTM_9M_DIV,),
    ),
)


# --------------------------------------------------------------------------
# (v) The finite-difference grid.
# --------------------------------------------------------------------------

DIGITAL_PDE_LEVELS: tuple[int, ...] = (100, 200, 400, 800)
"""`n_s = n_t = n` for the PDE order fits.

`n = 50` is deliberately excluded: with `s_max = 4 S = 400` the unaligned
spacing there is 8, which puts the strike at 12.5 spacings -- accidentally
midpoint-aligned, so the pathology row would start from its own best case.
Every level listed puts the strike exactly on a node instead.
"""

DIGITAL_PDE_N = 800
"""`n_s = n_t` at which the PDE leg of the cross-engine rows prices."""

DIGITAL_PDE_STRIKE_ALIGNMENT = "midpoint"
DIGITAL_PDE_TIME_STEPPING = "rannacher"
DIGITAL_PDE_PAYOFF_PROJECTION = "none"
"""The remedied configuration. `payoff_projection` is `"none"` here and that is
not an oversight: with midpoint alignment the strike lands exactly on a cell
face, so no cell straddles it and the cell average of the indicator equals its
point sample. The projection is a measured no-op on an aligned grid (agreement
1e-16 at `n = 100`, bit-for-bit at 200, 400, 800) and is the remedy to reach
for when the grid cannot be aligned."""

DIGITAL_PDE_TOLERANCE = 8.0e-5
"""Absolute tolerance for the PDE cross-engine leg.

Derived: at `n_s = n_t = 800`, aligned, Rannacher, the errors are +2.229e-07
(atm), -1.152e-05 (otm) and +2.331e-05 (itm, cash = 2.5). The tolerance keeps a
factor of 3.4 over the worst. It is two orders of magnitude looser than the
Leisen-Reimer leg at comparable work, which is the honest comparison: both are
order 2, and on this problem the lattice's constant is far smaller because the
digital price is exactly the quantity its construction matches.
"""

_PDE_SOURCE = (
    "derived in-repo: qpl.engines.pde.digital against "
    "qpl.engines.analytic.digital; the fitted orders in `notes` come from "
    "tests/test_digital_pde.py. The remedies are standard -- payoff projection "
    "from Pooley, Forsyth & Vetzal (2003), Journal of Computational Finance "
    "6(4), and the implicit start-up from Rannacher (1984), Numerische "
    "Mathematik 43, analysed in Giles & Carter (2006), Journal of "
    "Computational Finance 9(4) -- but no number here is quoted from them"
)

DIGITAL_PDE_ORDER_CASES: tuple[DigitalBSCase, ...] = (
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_pde_unaligned_loses_an_order",
            description=(
                "plain Crank-Nicolson on an unaligned grid is first order in "
                "the digital PRICE, not second"
            ),
            expected=1.0,
            tolerance=0.15,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_PDE_SOURCE,
            notes=(
                "Measured 1.0012, log-space residual 0.0007, over "
                "n_s = n_t in (100, 200, 400, 800); errors -3.761e-02, "
                "-1.876e-02, -9.377e-03, -4.689e-03, a clean halving. Delta "
                "0.8632 and gamma 1.0153 on the same grids. The difference "
                "from Slice 4's kink pathology is that the damage reaches the "
                "PRICE: 0.88% of the value at n = 800, where the remedied "
                "grid is out by 2.229e-07."
            ),
        ),
        specs=(DIGITAL_REFERENCE_ATM_CALL,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_pde_remedied_price_order_two",
            description=(
                "midpoint alignment plus Rannacher restores second-order "
                "convergence in the digital price"
            ),
            expected=2.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_PDE_SOURCE,
            notes=(
                "Measured 1.9248, residual 0.0233, over the same grids; "
                "errors +1.216e-05 down to +2.229e-07, monotone. Cell "
                "averaging on an UNALIGNED grid gives 2.0154 (residual 0.0063) "
                "-- the same remedy reached the other way. Which to use is "
                "settled by whether the grid can be aligned; stacking them "
                "does nothing, because alignment already puts the jump on a "
                "cell face."
            ),
        ),
        specs=(DIGITAL_REFERENCE_ATM_CALL,),
    ),
    DigitalBSCase(
        row=BenchmarkRow(
            id="digital_pde_remedied_delta_order_two",
            description="the same configuration is second order in delta",
            expected=2.0,
            tolerance=0.2,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_PDE_SOURCE,
            notes=(
                "Measured 2.0360, residual 0.0258; errors -5.162e-05 down to "
                "-7.413e-07. Gamma on the same grids is 2.0315 (residual "
                "0.0239) despite changing sign at d1 = 0; at the sign-change "
                "spot itself the absolute error still falls at order 2.1226, "
                "but relative accuracy there is undefined because the true "
                "gamma is zero."
            ),
        ),
        specs=(DIGITAL_REFERENCE_ATM_CALL,),
    ),
)


# --------------------------------------------------------------------------
# (vi) Four engines on one number.
# --------------------------------------------------------------------------

DIGITAL_MC_PATHS = 200_000
DIGITAL_MC_SEED = 123
DIGITAL_MC_STDERR_MULTIPLE = 4.0
"""The Monte Carlo leg's tolerance is a multiple of its own reported standard
error, not an absolute accuracy claim. Four standard errors is a two-sided
false-failure rate of about 6e-05 for a fixed seed; this seed lands at |z| of
1.218 (atm), 1.644 (otm) and 0.495 (itm)."""

DIGITAL_GREEKS_MC_PATHS = 200_000
DIGITAL_GREEKS_MC_STDERR_MULTIPLE = 4.0
"""Budget for the Monte Carlo **Greeks** leg of the cross-engine rows, in
multiples of each Greek's own reported standard error.

Only `greeks_estimator="likelihood_ratio"` is checked this way, and that is the
finding rather than a convenience: it is the one estimator here whose reported
standard error is an error bar. Measured |z| over the three points and five
Greeks at 200 000 paths, seed 123: worst 1.289 (itm_1y_div rho), median 0.59.
The `"pathwise"` estimator is refused (its almost-everywhere payoff derivative
is identically zero) and `"bump"` is available but is *not* held to this budget
-- its theta lands 531 standard errors out at the ATM point, because the paths
that cross the strike when the maturity moves by 1e-04 are too rare to appear
in a 200 000-path sample. See `tests/test_digital_mc.py`."""

DIGITAL_FOURIER_METHODS: tuple[str, ...] = ("cos", "gil_pelaez")
"""The two transform methods that price a cash-or-nothing payoff (Slice 14).

`"cos"` has its own payoff coefficient for an indicator (`cash * psi_k`), and
`"gil_pelaez"` gets the digital for free because `Pi_2` *is* the exercise
probability. `"carr_madan"` and `"lewis"` transform the call price rather than
the law and are refused."""

DIGITAL_FOURIER_TOLERANCE = 1e-14
"""Absolute tolerance for the transform legs of the digital cross-engine test.

Measured errors over the three points: 2.220e-16, 2.776e-16 and 0.0 for COS;
0.0, 1.110e-16 and 0.0 for Gil-Pelaez. This is the number worth reading next to
`DIGITAL_PDE_TOLERANCE` (8e-05) and `DIGITAL_TREE_LR_TOLERANCE` (2e-08): the
payoff's jump costs the finite-difference scheme a full order of convergence
(Slice 6) and costs a transform method **nothing**, because what gets expanded
is the density and the discontinuous part enters through an exact integral."""


_CROSS_ENGINE_SOURCE = (
    "derived in-repo: qpl.engines.analytic.digital, qpl.engines.tree.digital "
    "(Leisen-Reimer), qpl.engines.pde.digital (midpoint-aligned, Rannacher) "
    "and qpl.engines.mc.digital evaluated at this point; the per-engine "
    "tolerances in `notes` are derived from the measured errors"
)

DIGITAL_CROSS_ENGINE_CASES: tuple[DigitalBSCase, ...] = tuple(
    DigitalBSCase(
        row=BenchmarkRow(
            id=f"digital_cross_engine_{name}",
            description=(
                "analytic, Leisen-Reimer tree, remedied PDE grid, Monte "
                f"Carlo and the two transform methods agree on the digital "
                f"price at {name}"
            ),
            expected=0.0,
            tolerance=DIGITAL_PDE_TOLERANCE,
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=_CROSS_ENGINE_SOURCE,
            notes=(
                "Per-leg budgets, each derived from a measurement rather than "
                "chosen: Leisen-Reimer at n = 2001, "
                f"{DIGITAL_TREE_LR_TOLERANCE:g} (worst measured 5.284e-09); "
                f"PDE at n_s = n_t = {DIGITAL_PDE_N} aligned with Rannacher, "
                f"{DIGITAL_PDE_TOLERANCE:g} (worst measured 2.331e-05); Monte "
                f"Carlo at {DIGITAL_MC_PATHS} paths, "
                f"{DIGITAL_MC_STDERR_MULTIPLE:g} standard errors, which is "
                "STATISTICAL rather than an accuracy claim; the two transform "
                f"legs at {DIGITAL_FOURIER_TOLERANCE:g} (worst measured "
                "2.776e-16), which is the floating-point floor and not a "
                "discretisation error at all. The row's own tolerance is the "
                "loosest deterministic leg's, since that bounds the gap "
                "between any two of them."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _POINTS
)


_MC_GREEKS_SOURCE = (
    "derived in-repo: qpl.engines.mc.digital likelihood-ratio Greeks against "
    "qpl.engines.analytic.digital closed-form Greeks, each compared to its own "
    "reported standard error. The scores are re-derived in "
    "src/qpl/engines/mc/greeks.py from the lognormal density; the method is "
    "Glasserman (2003), 'Monte Carlo Methods in Financial Engineering', "
    "section 7.3, and Broadie & Glasserman (1996), Management Science 42(2), "
    "269-285. No number here is quoted from either."
)

DIGITAL_MC_GREEKS_CASES: tuple[DigitalBSCase, ...] = tuple(
    DigitalBSCase(
        row=BenchmarkRow(
            id=f"digital_mc_likelihood_ratio_{greek}_{name}",
            description=(
                f"Monte Carlo likelihood-ratio {greek} against the closed form "
                f"at {name}, {DIGITAL_GREEKS_MC_PATHS} paths, seed "
                f"{DIGITAL_MC_SEED}"
            ),
            expected=0.0,
            tolerance=DIGITAL_GREEKS_MC_STDERR_MULTIPLE,
            evidence=EvidenceClass.STATISTICAL,
            source=_MC_GREEKS_SOURCE,
            notes=(
                "The residual is in units of the estimator's own reported "
                "standard error, so `tolerance` is a z-score. Measured |z| at "
                "this seed across the three points and five Greeks: worst "
                "1.289 (itm_1y_div rho), median 0.59. The likelihood ratio is "
                "the only estimator held to this budget here: `pathwise` does "
                "not exist for an indicator payoff (its almost-everywhere "
                "derivative is identically zero) and `bump` reaches |z| = 531 "
                "on theta at the ATM point, because the paths that cross the "
                "strike when the maturity moves by 1e-04 are too rare to "
                "appear in the sample. See tests/test_digital_mc.py."
            ),
        ),
        specs=(spec,),
    )
    for greek in ("delta", "gamma", "vega", "theta", "rho")
    for name, spec in _POINTS
)
"""Fifteen rows: five Greeks at three points, likelihood ratio only."""


ALL_DIGITAL_CASES: tuple[DigitalBSCase, ...] = (
    DIGITAL_IDENTITY_CASES
    + DIGITAL_STRIKE_DERIVATIVE_CASES
    + DIGITAL_KNOWN_VALUE_CASES
    + DIGITAL_TREE_ORDER_CASES
    + DIGITAL_PDE_ORDER_CASES
    + DIGITAL_CROSS_ENGINE_CASES
    + DIGITAL_MC_GREEKS_CASES
)
