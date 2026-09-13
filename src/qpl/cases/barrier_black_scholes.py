"""Benchmark cases for single-barrier options under Black-Scholes.

Same shape as the other case modules: each case pairs a specification with one
:class:`~qpl.validation.BenchmarkRow` stating what is expected, how tight the
comparison is, which :class:`~qpl.validation.EvidenceClass` justifies it, and
where it comes from. The lists hold no assertions and no pytest dependency;
`tests/cases/test_barrier_black_scholes_cases.py` parametrises over them.

What is different here is that the barrier is the first contract in this
repository whose **monitoring convention** is part of the specification, so a
row has to say which of two contracts it is about. Three of the four groups
below are about the continuously monitored contract (the closed form, the
published values, the lattice); the Monte Carlo leg prices the discrete one and
uses the Brownian-bridge estimator to bring it back to the continuous contract,
which is unbiased at any `m` and is the reason a cross-engine comparison is
possible at all.

The published values are Haug, E.G. (2007), *The Complete Guide to Option
Pricing Formulas*, 2nd ed., section 4.17, the standard tabulation of the
Reiner-Rubinstein forms. Three rows of that table are embedded in QuantLib's
own test suite and are used here as fixtures with that citation; the table
itself is not reproduced, and the closed forms are derived independently in
`qpl.engines.analytic.barrier`. Reiner, E. and Rubinstein, M. (1991), "Breaking
down the barriers", Risk 4(8), 28-35, is the source of the formulas; Merton
(1973), Bell Journal of Economics and Management Science 4(1), section 8, of
the down-and-out call alone; Broadie, Glasserman and Kou (1997), Mathematical
Finance 7(4), 325-348, of the continuity correction; Boyle and Lau (1994),
Journal of Derivatives 1(4), 6-14, of the lattice step counts. Every number in
`notes` below was measured in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..instruments.options import BarrierOption, uniform_monitoring_times
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_BARRIER_CASES",
    "BARRIER_BOYLE_LAU_ENVELOPE",
    "BARRIER_CROSS_ENGINE_CASES",
    "BARRIER_DISCRETE_CROSS_ENGINE_CASES",
    "BARRIER_HAUG_CASES",
    "BARRIER_HAUG_SPEC",
    "BARRIER_IDENTITY_CASES",
    "BARRIER_LIMIT_CASES",
    "BARRIER_MC_MONITORING",
    "BARRIER_MC_ORDER_CASES",
    "BARRIER_MC_PATHS",
    "BARRIER_MC_SEED",
    "BARRIER_MC_STDERR_MULTIPLE",
    "BARRIER_MC_VARIANCE_REDUCTION",
    "BARRIER_PDE_CONCENTRATION",
    "BARRIER_PDE_DISCRETE_STDERR_MULTIPLE",
    "BARRIER_PDE_GRID",
    "BARRIER_PDE_N",
    "BARRIER_PDE_ORDER_CASES",
    "BARRIER_PDE_STRIKE_ALIGNMENT",
    "BARRIER_PDE_TIME_STEPPING",
    "BARRIER_PDE_TOLERANCE",
    "BARRIER_TREE_ORDER_CASES",
    "BarrierBSCase",
    "BarrierBSSpec",
]


@dataclass(frozen=True)
class BarrierBSSpec:
    """A single single-barrier pricing point.

    Scalars only, so a case list stays plain comparable data; the domain
    objects are built on demand. `boyle_lau_layer` travels with the point
    rather than with the test because the step count it produces depends on
    `log(S/H) / (sigma sqrt(T))`, so "the same lattice resolution" is a
    different `n` at every point.
    """

    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float
    barrier: float
    barrier_type: Literal[
        "down-and-out", "down-and-in", "up-and-out", "up-and-in"
    ] = "down-and-out"
    kind: Literal["call", "put"] = "call"
    rebate: float = 0.0
    boyle_lau_layer: int = 14

    def option(self, *, monitoring: int | None = None) -> BarrierOption:
        """The instrument, continuously monitored unless `monitoring` is given."""
        schedule = (
            "continuous"
            if monitoring is None
            else uniform_monitoring_times(self.expiry, monitoring)
        )
        return BarrierOption(
            kind=self.kind,
            strike=self.strike,
            expiry=self.expiry,
            barrier=self.barrier,
            barrier_type=self.barrier_type,
            rebate=self.rebate,
            monitoring=schedule,
        )

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def flipped_knock(self) -> BarrierBSSpec:
        """The same point with knock-in and knock-out exchanged."""
        other = (
            self.barrier_type.replace("out", "in")
            if self.barrier_type.endswith("out")
            else self.barrier_type.replace("in", "out")
        )
        return replace(self, barrier_type=other)  # type: ignore[arg-type]


@dataclass(frozen=True)
class BarrierBSCase:
    """One benchmark row plus the point it is evaluated at."""

    row: BenchmarkRow
    specs: tuple[BarrierBSSpec, ...]

    @property
    def spec(self) -> BarrierBSSpec:
        if len(self.specs) != 1:
            raise ValueError(f"case {self.row.id} holds {len(self.specs)} specs, not 1")
        return self.specs[0]


# --------------------------------------------------------------------------
# The points.
# --------------------------------------------------------------------------

BARRIER_HAUG_SPEC = BarrierBSSpec(
    spot=100.0, strike=100.0, expiry=0.5, rate=0.08, dividend=0.04, sigma=0.25,
    barrier=95.0, barrier_type="down-and-out", kind="call", rebate=3.0,
    boyle_lau_layer=14,
)
"""`S = 100, K = 100, H = 95, rebate 3, r = 8%, q = 4%, sigma = 25%, T = 0.5`.

The specification of the published table rows below, and of the Monte Carlo
monitoring study. Worth one warning, because it is an easy mis-reading and this
slice was handed it: in that table the varying column is the **strike**, at a
fixed spot of 100. A row described as "down-and-out call 90" is `K = 90` with
`S = 100`, not `S = 90` -- which would put the spot *below* a down barrier at
95 and settle the contract at inception for 3.00 rather than 9.02.
"""

_ATM_6M_DOWN_OUT = replace(BARRIER_HAUG_SPEC, rebate=0.0)
"""The same point with no rebate, so that in-out parity is exact."""

_ATM_1Y_UP_OUT = BarrierBSSpec(
    spot=100.0, strike=100.0, expiry=1.0, rate=0.05, dividend=0.0, sigma=0.20,
    barrier=120.0, barrier_type="up-and-out", kind="call", rebate=0.0,
    boyle_lau_layer=45,
)
"""An up-and-out call whose barrier is 20% away. `boyle_lau_layer` is 45 rather
than 14 because `lambda_0 = |log(S/H)| / (sigma sqrt(T))` is 0.912 here against
0.290 at the Haug point, so `n_k = floor(k^2 / lambda_0^2)` grows nine times
more slowly: layer 14 is `n = 235`, and layer 45 is `n = 2436`."""

_OTM_1Y_DOWN_IN = BarrierBSSpec(
    spot=100.0, strike=105.0, expiry=1.0, rate=0.03, dividend=0.01, sigma=0.30,
    barrier=90.0, barrier_type="down-and-in", kind="put", rebate=0.0,
    boyle_lau_layer=16,
)
"""A knock-**in** put, so the cross-engine row exercises the three-roll-back
assembly rather than only the direct knock-out induction."""

_CROSS_ENGINE_POINTS: tuple[tuple[str, BarrierBSSpec], ...] = (
    ("atm_6m_down_out", _ATM_6M_DOWN_OUT),
    ("atm_1y_up_out", _ATM_1Y_UP_OUT),
    ("otm_1y_down_in", _OTM_1Y_DOWN_IN),
)

_PARITY_POINTS: tuple[tuple[str, BarrierBSSpec], ...] = (
    ("atm_6m_down", _ATM_6M_DOWN_OUT),
    ("atm_1y_up", _ATM_1Y_UP_OUT),
    ("otm_1y_down", replace(_OTM_1Y_DOWN_IN, barrier_type="down-and-out")),
    (
        "itm_1y_up_put",
        BarrierBSSpec(
            spot=100.0, strike=110.0, expiry=1.0, rate=0.04, dividend=0.06,
            sigma=0.35, barrier=115.0, barrier_type="up-and-out", kind="put",
        ),
    ),
)


# --------------------------------------------------------------------------
# Study settings.
# --------------------------------------------------------------------------

BARRIER_MC_MONITORING = 50
"""Monitoring dates for the Monte Carlo leg of the cross-engine rows.

Any count would do: the Brownian-bridge estimator is unbiased for the
continuous contract at every `m`, measured down to `m = 1`. 50 is chosen so the
leg also exercises a realistic schedule rather than the degenerate one.
"""

BARRIER_MC_PATHS = 100_000
BARRIER_MC_SEED = 20_260_913
BARRIER_MC_VARIANCE_REDUCTION: tuple[str, ...] = ("antithetic", "control_variate")
BARRIER_MC_STDERR_MULTIPLE = 4.0
"""The Monte Carlo leg's tolerance is a multiple of its own reported standard
error, not an absolute accuracy claim. Four standard errors is a two-sided
false-failure rate of about 6e-05 at a fixed seed; at 100 000 paths this seed
lands at |z| of 0.008, 0.055 and 0.440 across the three points."""

BARRIER_PDE_N = 400
"""Spatial and temporal resolution of the finite-difference leg, `n_s = n_t`.

Chosen so that the leg's worst error over the three cross-engine points is
3.836e-05 -- an order of magnitude inside the lattice leg's budget at the same
points -- at a cost of 0.02 s to 0.11 s per point.
"""

BARRIER_PDE_GRID = "sinh"
BARRIER_PDE_CONCENTRATION = 0.05
"""The `sinh` mesh concentrated at the strike and the barrier, and its strength.

`concentration = 0.05` is the package default and is deliberately **not** the
best value on these points: at `n = 100` on the Haug point the errors are
2.114e-04, 2.594e-04, 4.355e-04 and 9.397e-04 at 0.02, 0.05, 0.10 and 0.20, so
0.02 would be 1.2x better here -- and worse on a plain vanilla, where a mesh
that starves the tails to feed the barrier pays for it. See
`tests/test_pde_barrier.py`.
"""

BARRIER_PDE_STRIKE_ALIGNMENT = "midpoint"
BARRIER_PDE_TIME_STEPPING = "rannacher"

BARRIER_PDE_TOLERANCE = 2.0e-4
"""Budget for the finite-difference leg, about 5x its measured worst error.

Measured errors against the closed form at `BARRIER_PDE_N` on the `sinh` grid:
-1.665e-05 (`atm_6m_down_out`), -3.588e-06 (`atm_1y_up_out`) and -3.836e-05
(`otm_1y_down_in`). Unlike the lattice leg's, this constant is *not* erratic --
the barrier is a boundary rather than a node the scheme rounds to -- so the
budget can be read off the errors rather than off an envelope.
"""

BARRIER_PDE_DISCRETE_STDERR_MULTIPLE = 4.0
"""The discrete cross-engine rows hold the grid to this many standard errors of
the simulation it is compared with.

The simulation is the plain estimator (`barrier_correction='none'`), which is
unbiased for the **discrete** contract, so the two legs estimate the same
number by genuinely different routes. Measured `z` at the study seed and
100 000 paths: -0.287, +0.132, +0.347.
"""

BARRIER_BOYLE_LAU_ENVELOPE = 8.0
"""`max |error| * n` along the Boyle-Lau subsequence: the constant that sets the
lattice leg's tolerance.

Choosing `n` so that a layer of nodes lands on the barrier makes the lattice
error first order (block-RMS fit 1.1258 at the Haug point), but with a constant
that is **erratic rather than monotone**: `floor` leaves a residual
misalignment `1 - H_eff/H` that behaves like a uniform draw on `[0, 1)`, so a
tolerance read off the error at one chosen layer would be cherry-picked. The
honest budget is the measured *envelope*.

Measured `|error| * n` over the Boyle-Lau layers at the three cross-engine
points: up to 1.70 at `atm_6m_down_out` (thirteen layers, `n` 190 to 3040),
5.54 at `atm_1y_up_out` (seven layers, `n` 481 to 3008) and 3.00 at
`otm_1y_down_in`. The envelope keeps about 1.4x over the worst, and each row's
tolerance is `BARRIER_BOYLE_LAU_ENVELOPE / n` at that point's own `n`.
"""


# --------------------------------------------------------------------------
# (i) The published table rows.
# --------------------------------------------------------------------------

_HAUG_SOURCE = (
    "Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., "
    "section 4.17, the standard tabulation of the Reiner & Rubinstein (1991) "
    "single-barrier formulas (Risk 4(8), 28-35). The three values used here "
    "are the rows embedded in QuantLib's own barrier-option test suite. The "
    "table is not reproduced; the formulas are derived independently in "
    "qpl.engines.analytic.barrier"
)

_HAUG_TOLERANCE = 1.0e-4
"""The published values carry four decimals, so agreement can be asserted no
more tightly than half a unit in the last place. Measured residuals: 3.23e-05,
3.66e-05 and 1.98e-05 -- all inside the rounding of the published figure
itself, which is the most that can be claimed from a four-decimal table."""

BARRIER_HAUG_CASES: tuple[BarrierBSCase, ...] = (
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_haug_down_out_call_k90",
            description=(
                "down-and-out call, S=100, K=90, H=95, rebate=3, r=8%, q=4%, "
                "sigma=25%, T=0.5"
            ),
            expected=9.0246,
            tolerance=_HAUG_TOLERANCE,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_HAUG_SOURCE,
            notes=(
                "In-repo value 9.024567694966874; residual 3.23e-05, inside "
                "the published figure's own rounding. The strike is 90 at a "
                "spot of 100 -- reading it the other way round puts the spot "
                "below the barrier, which settles the contract at inception "
                "for the rebate 3.00."
            ),
        ),
        specs=(replace(BARRIER_HAUG_SPEC, strike=90.0),),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_haug_down_out_call_k100",
            description=(
                "down-and-out call, S=100, K=100, H=95, rebate=3, r=8%, q=4%, "
                "sigma=25%, T=0.5"
            ),
            expected=6.7924,
            tolerance=_HAUG_TOLERANCE,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_HAUG_SOURCE,
            notes="In-repo value 6.792436575025215; residual 3.66e-05.",
        ),
        specs=(BARRIER_HAUG_SPEC,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_haug_up_out_call_k100",
            description=(
                "up-and-out call, S=100, K=100, H=105, rebate=3, r=8%, q=4%, "
                "sigma=25%, T=0.5"
            ),
            expected=2.3580,
            tolerance=_HAUG_TOLERANCE,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_HAUG_SOURCE,
            notes=(
                "In-repo value 2.358019790844047; residual 1.98e-05. This is "
                "the row that caught the one sign error in the assembly: the "
                "rebate block F's second term is N(eta z - 2 eta lambda v), "
                "and writing it as N(eta (z - 2 eta lambda v)) -- which is the "
                "same thing for a DOWN barrier and not for an up one -- left "
                "the two down-and-out rows correct and this one out by 1.04."
            ),
        ),
        specs=(
            replace(BARRIER_HAUG_SPEC, barrier=105.0, barrier_type="up-and-out"),
        ),
    ),
)


# --------------------------------------------------------------------------
# (ii) In-out parity, and what a rebate does to it.
# --------------------------------------------------------------------------

_PARITY_SOURCE = (
    "derived: a path either touches the barrier before expiry or it does not, "
    "and the knock-in and knock-out pay the vanilla payoff on exactly those "
    "two complementary events, so with a zero rebate their sum is the vanilla "
    "pathwise. Formulas: Reiner & Rubinstein (1991), Risk 4(8), 28-35"
)

BARRIER_IDENTITY_CASES: tuple[BarrierBSCase, ...] = tuple(
    BarrierBSCase(
        row=BenchmarkRow(
            id=f"barrier_in_out_parity_{name}",
            description=(
                f"knock-in + knock-out - vanilla at S={spec.spot}, "
                f"K={spec.strike}, H={spec.barrier}, T={spec.expiry}, "
                f"sigma={spec.sigma}, {spec.kind}, zero rebate"
            ),
            expected=0.0,
            tolerance=1.0e-13,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_PARITY_SOURCE,
            notes=(
                "A round-off budget, not a model tolerance. Worst measured "
                "residual over five points, both kinds and both barrier sides: "
                "1.421e-14 absolute, 1.810e-15 relative. The identity is what "
                "catches a sign error in the reflected blocks C and D, since "
                "the two assemblies use them with opposite signs; it is FALSE "
                "with a rebate, by exactly E + F."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _PARITY_POINTS
)


# --------------------------------------------------------------------------
# (iii) The two barrier limits.
# --------------------------------------------------------------------------

_LIMIT_SOURCE = (
    "derived: a barrier that cannot be reached leaves the vanilla, and a "
    "barrier at the spot has already been reached, so a knock-out is worth "
    "exactly its rebate. Both are checked against this package's own vanilla "
    "engine and against the rebate respectively"
)

BARRIER_LIMIT_CASES: tuple[BarrierBSCase, ...] = (
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_vanishing_barrier_is_the_vanilla",
            description="down-and-out call at H = 1e-6 S equals the Black-Scholes call",
            expected=0.0,
            tolerance=1.0e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_LIMIT_SOURCE,
            notes=(
                "Worst measured residual 1.421e-14 over five points and both "
                "kinds, and the same for an up-and-out at H = 1e+6 S. The "
                "reflected blocks do not merely become small: (H/S)^{2mu} can "
                "grow without bound when mu < 0, and it is the normal tail "
                "multiplying it that decays fast enough to win."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_at_the_spot_is_the_rebate",
            description="every knock-out with H = S is worth exactly its rebate",
            expected=0.0,
            tolerance=1.0e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_LIMIT_SOURCE,
            notes=(
                "Exact at H = S, not a limit: the block F equals the rebate "
                "there (z = lambda v and the two tails sum to 1) and the "
                "option part cancels -- block by block for a call (A = C, "
                "B = D) and as a difference of sums for a put. Worst measured "
                "residual of the formula route 1.421e-14 over three points and "
                "four knock-out types; the inception guard returns the rebate "
                "exactly."
            ),
        ),
        specs=(replace(_ATM_6M_DOWN_OUT, rebate=3.0),),
    ),
)


# --------------------------------------------------------------------------
# (iv) The monitoring bias.
# --------------------------------------------------------------------------

_MC_SOURCE = (
    "derived in-repo: qpl.engines.mc.barrier against "
    "qpl.engines.analytic.barrier; the fitted orders in `notes` come from "
    "tests/test_barrier_mc.py. The continuity correction is Broadie, "
    "Glasserman & Kou (1997), Mathematical Finance 7(4), 325-348, and the "
    "bridge crossing probability is standard (Glasserman 2003, sections 6.4 "
    "and 3.1); no number here is quoted from either"
)

BARRIER_MC_ORDER_CASES: tuple[BarrierBSCase, ...] = (
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_monitoring_bias_order_one_half",
            description=(
                "the discrete-monitoring bias against the continuous closed "
                "form decays like m**-1/2"
            ),
            expected=0.5,
            tolerance=0.15,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_MC_SOURCE,
            notes=(
                "Measured 0.5045 (log-space residual 0.0687) at 40 000 paths, "
                "antithetic plus the vanilla control variate, over m in "
                "(25, 50, 100, 200, 400); biases +1.1211, +0.8164, +0.6221, "
                "+0.4490, +0.2631 against standard errors near 0.030. Over ten "
                "seeds the same fit gives 0.4764 +- 0.0329, and the "
                "lower-noise paired estimator (plain minus bridge on the same "
                "paths) gives 0.4603 +- 0.0125 -- significantly below 0.5, "
                "which is the o(1/sqrt(m)) term still visible at m = 25 and "
                "not seed noise. This is the prediction Slice 11 handed "
                "forward: an exercise frequency costs O(1/m), a monitoring "
                "frequency costs O(1/sqrt(m))."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_brownian_bridge_removes_the_bias",
            description=(
                "the Brownian-bridge estimator agrees with the continuous "
                "closed form within its own noise at every m"
            ),
            expected=0.0,
            tolerance=3.0,
            evidence=EvidenceClass.STATISTICAL,
            source=_MC_SOURCE,
            notes=(
                "The residual is in units of the estimator's own reported "
                "standard error, so `tolerance` is a z-score. Measured z at "
                "the study seed: +0.14, +0.13, -0.26, +0.71, -1.43 over m in "
                "(25, 50, 100, 200, 400), while the plain estimator on the "
                "SAME paths is 8 to 40 standard errors out. Unbiasedness is "
                "checked by coverage rather than closeness: over 40 seeds at "
                "20 000 paths the 95% interval covers the closed form 39/40 at "
                "m = 25 and 38/40 at m = 100, against a Binomial(40, 0.95) "
                "mean of 38. The surprise is that it holds at m = 1 (z = "
                "-0.10): the estimator conditions on the sampled points, so "
                "its MEAN does not depend on how many there are."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_bgk_order_is_below_the_noise_floor",
            description=(
                "the BGK-corrected estimator's residual bias is not resolvable "
                "at this cost, so its order is not measurable"
            ),
            expected=0.0,
            tolerance=2.0,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_MC_SOURCE,
            notes=(
                "Measured, paired against the bridge estimator on the same "
                "paths: the residual is +0.0346, +0.0028, -0.0010, -0.0036, -0.0041 "
                "against standard errors 0.0109 to 0.0063: resolved only at "
                "m = 25 (3.2 sigma, reproduced as +0.0008 to +0.0377 over ten "
                "seeds) and inside two standard errors from m = 50 on, with a "
                "wandering sign. A fit through that returns 0.578 with a "
                "log-space residual of 1.02, which is the diagnostic saying "
                "the fit is meaningless. What IS measurable: from m = 25 to "
                "50 the plain bias falls by 1.37x and the BGK residual by at "
                "least 12x. `tolerance` is a z-score on the tail levels."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
)


# --------------------------------------------------------------------------
# (v) The lattice.
# --------------------------------------------------------------------------

_TREE_SOURCE = (
    "derived in-repo: qpl.engines.tree.barrier against "
    "qpl.engines.analytic.barrier; the fitted orders and amplitudes in "
    "`notes` come from tests/test_barrier_tree_convergence.py. The step-count "
    "remedy is Boyle & Lau (1994), Journal of Derivatives 1(4), 6-14, and the "
    "interpolation alternative is Derman, Kani, Ergener & Bardhan (1995), "
    "Risk 8(6), which is cited and deliberately not implemented; the lattices "
    "are Cox, Ross & Rubinstein (1979) and Leisen & Reimer (1996). No number "
    "here is quoted from any of them"
)

BARRIER_TREE_ORDER_CASES: tuple[BarrierBSCase, ...] = (
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_tree_sawtooth_order_one_half",
            description=(
                "the CRR barrier error sawtooths in n with an amplitude that "
                "decays like n**-1/2, not n**-1"
            ),
            expected=0.5,
            tolerance=0.12,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_TREE_SOURCE,
            notes=(
                "Measured 0.4566 (log-space residual 0.030) on amplitudes "
                "taken over a FULL sawtooth period at each level -- 0.93510 "
                "(n0 = 100, period 70), 0.63967 (200, 99), 0.49654 (400, 139). "
                "A fixed-width window would report decay that is only the "
                "period growing like sqrt(n). The mechanism: the lattice knocks "
                "out at the first node beyond the barrier, so the effective "
                "barrier is displaced by O(sigma sqrt(dt)) and the price is "
                "locally linear in it. Same one-half as the Monte Carlo "
                "monitoring bias, seen from the other side. At n0 = 100 the "
                "error swings over 21% of the option's value as n walks "
                "through 70 consecutive step counts."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_tree_leisen_reimer_does_not_help",
            description=(
                "Leisen-Reimer removes 2% of the barrier sawtooth's amplitude "
                "and none of its order"
            ),
            expected=1.0,
            tolerance=0.05,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_TREE_SOURCE,
            notes=(
                "`expected` is the LR/CRR amplitude ratio: measured 0.977, "
                "0.978, 0.980 at n0 = 100, 200, 400, with fitted orders 0.4542 "
                "(LR) against 0.4566 (CRR). Leisen-Reimer buys two decimal "
                "orders on a European vanilla and a full order on a digital; "
                "on a barrier it buys nothing, because its construction places "
                "the terminal grid around the STRIKE and a barrier is a second "
                "level it never looks at. Since u d != 1 its nodes are not on "
                "a single geometric ladder, so no step count puts a layer on "
                "the barrier by construction either -- and the Boyle-Lau "
                "counts, applied to an LR lattice, give errors three decimal "
                "orders worse than CRR's at the identical n (block order "
                "0.4254)."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_tree_boyle_lau_first_order",
            description=(
                "along the Boyle-Lau step counts the CRR error is first order, "
                "but not monotone"
            ),
            expected=1.0,
            tolerance=0.25,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_TREE_SOURCE,
            notes=(
                "Measured block-RMS fit 1.1258 (log-space residual 0.0041) over three "
                "groups of thirteen layers, n from 190 to 3040; errors "
                "+1.38e-04 (k=4), -6.38e-05 (7), +1.43e-03 (10), +4.92e-04 "
                "(13), +3.37e-04 (16) against +0.64 for the unaligned n = 200. "
                "NOT monotone, which the slice expected: the signs alone "
                "disprove it, because `floor` leaves a residual misalignment "
                "1 - H_eff/H spanning 1.14e-07 to 8.14e-05 that behaves like a "
                "uniform draw on the fractional part. `|error| * n` stays "
                "inside [0.025, 1.70] across all thirteen levels, which is the "
                "assumption-free form of 'order 1 with an erratic constant' "
                "and is what BARRIER_BOYLE_LAU_ENVELOPE is taken from. A "
                "per-n fit returns 0.5716 with a residual of 1.49, i.e. it is "
                "not a power law at fixed n at all."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
)


# --------------------------------------------------------------------------
# (vi) Three engines on one number.
# --------------------------------------------------------------------------

_CROSS_ENGINE_SOURCE = (
    "derived in-repo: qpl.engines.analytic.barrier, qpl.engines.tree.barrier "
    "at a Boyle-Lau step count, qpl.engines.mc.barrier with the "
    "Brownian-bridge estimator, and (Slice 13) qpl.engines.pde.barrier on a "
    "sinh grid truncated at the barrier, evaluated at this point; the "
    "per-engine tolerances in `notes` are derived from the measured errors"
)

BARRIER_CROSS_ENGINE_CASES: tuple[BarrierBSCase, ...] = tuple(
    BarrierBSCase(
        row=BenchmarkRow(
            id=f"barrier_cross_engine_{name}",
            description=(
                "closed form, Boyle-Lau lattice, Brownian-bridge Monte Carlo "
                f"and the finite-difference grid agree on the barrier price "
                f"at {name}"
            ),
            expected=0.0,
            tolerance=BARRIER_BOYLE_LAU_ENVELOPE,
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=_CROSS_ENGINE_SOURCE,
            notes=(
                "`tolerance` is the lattice leg's envelope constant: the row's "
                "actual budget is BARRIER_BOYLE_LAU_ENVELOPE / n at this "
                "point's own Boyle-Lau step count, because the same layer "
                "index is a different n at every point. The Monte Carlo leg is "
                f"held to {BARRIER_MC_STDERR_MULTIPLE:g} of its own standard "
                "error, which is STATISTICAL rather than an accuracy claim, "
                "and it prices the DISCRETELY monitored contract on "
                f"{BARRIER_MC_MONITORING} dates -- the Brownian-bridge "
                "estimator is what makes that an unbiased estimate of the "
                "continuous price the other two legs compute. Measured "
                "lattice errors -1.075e-05, +1.730e-04 and +1.055e-04 against "
                "budgets of 3.44e-03, 3.28e-03 and 3.86e-03, and Monte Carlo "
                "|z| of 0.008, 0.055 and 0.440. Slice 13 adds the fourth leg, "
                "a finite-difference solve on a sinh grid truncated at the "
                "barrier: measured errors -1.665e-05, -3.588e-06 and "
                "-3.836e-05 against a flat BARRIER_PDE_TOLERANCE of 2.0e-4. "
                "Its budget is a plain constant rather than an envelope, "
                "because the barrier is a boundary here and not a node the "
                "scheme rounds to, so the constant is not erratic."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _CROSS_ENGINE_POINTS
)


# --------------------------------------------------------------------------
# (vi) The grid: what a node on the barrier is worth.
# --------------------------------------------------------------------------

_PDE_SOURCE = (
    "derived in-repo: qpl.engines.pde.barrier against "
    "qpl.engines.analytic.barrier; the fitted orders and ratios in `notes` "
    "come from tests/test_pde_barrier.py. The barrier-on-a-node requirement is "
    "Zvan, Vetzal & Forsyth (2000), Journal of Economic Dynamics and Control "
    "24, 1563-1590; the non-uniform stencil's consistency order is Duffy "
    "(2006), Finite Difference Methods in Financial Engineering, and the "
    "coordinate-transformation view is Tavella & Randall (2000) chapter 5; the "
    "sinh mesh formula is In 't Hout & Foulon (2010), IJNAM 7(2), section 3, "
    "cited for the mesh alone. No number here is quoted from any of them"
)

BARRIER_PDE_ORDER_CASES: tuple[BarrierBSCase, ...] = (
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_pde_on_node_order_two",
            description=(
                "with the barrier as the domain boundary the finite-difference "
                "knock-out is second order in n_s = n_t"
            ),
            expected=2.0,
            tolerance=0.15,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_PDE_SOURCE,
            notes=(
                "Measured 2.0668 (log-space residual 0.0883) over n = 100, "
                "200, 400, 800 on the uniform grid, errors -4.676e-03, "
                "-8.861e-04, -2.568e-04, -5.959e-05, all of one sign. This is "
                "the order both Slice 12 discretisations could not reach: the "
                "lattice and the simulation are each order ONE HALF on this "
                "contract, because both displace the barrier by "
                "O(sigma sqrt(dt)). A grid can put a node on the barrier for "
                "every time step at once, which is why the mesh was deferred "
                "until a barrier forced it."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_pde_off_node_is_first_order",
            description=(
                "rounding the barrier to the nodes costs a full order and two "
                "to three decimal orders of accuracy"
            ),
            expected=1.0,
            tolerance=0.35,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_PDE_SOURCE,
            notes=(
                "Measured with PDEConfig(barrier_alignment='none'): errors "
                "+8.515e-01, +1.182e+00, +3.924e-01, +2.396e-01 over n = 100, "
                "200, 400, 800, a fit of 0.7079 with a log-space residual of "
                "0.3068 -- not a power law at fixed n, because the constant is "
                "set by the fractional part of H / ds. |error| * n stays in "
                "[85.2, 245.7] through n = 1600, which is the assumption-free "
                "form of the claim. Every error is POSITIVE: the scheme kills "
                "at the largest node at or below H, so it prices a barrier "
                "further from the spot, which is worth more. Against the "
                "aligned grid at the same node count the errors are 182x to "
                "4021x larger."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_pde_sinh_mesh_shrinks_the_constant",
            description=(
                "the sinh mesh concentrated at the barrier and the strike has "
                "a 13x-18x smaller error constant at equal node count"
            ),
            expected=15.0,
            tolerance=7.0,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_PDE_SOURCE,
            notes=(
                "`expected` is the uniform/sinh error ratio. Measured 18.0, "
                "13.0, 15.4 and 14.5 at n = 100, 200, 400, 800, with the sinh "
                "order 1.9981 (residual 0.0195) against the uniform grid's "
                "2.0668 -- same order, smaller constant, which is all a mesh "
                "can buy. The ratio is NOT monotone in n and is not expected "
                "to be: both sequences carry their own pre-asymptotic wobble. "
                "The concentration scan at n = 100 gives 22.1, 18.0, 10.7 and "
                "5.0 at 0.02, 0.05, 0.10 and 0.20, reported rather than tuned: "
                "the package default 0.05 is not the best cell here."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_pde_discrete_gap_order_one_half",
            description=(
                "the grid's discrete-to-continuous gap decays like m**-1/2 and "
                "agrees with the Broadie-Glasserman-Kou shift"
            ),
            expected=0.5,
            tolerance=0.1,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_PDE_SOURCE,
            notes=(
                "Measured 0.4431 (log-space residual 0.0101) over m = 10, 20, "
                "40, 80, 160, gaps +1.625746, +1.213371, +0.900876, +0.658822, "
                "+0.475119. Below 0.5 for a reason that is measured rather "
                "than assumed: the BGK closed-form shift, on the same m "
                "ladder, has fitted order 0.4361, so the deficit is the "
                "correction's own o(1/sqrt(m)) at finite m and not the grid's "
                "-- the same effect Slice 12 saw from the simulation side "
                "(0.4603 +- 0.0125 paired). The gap/BGK-gap ratio is 1.0147, "
                "0.9981, 0.9974, 0.9963, 0.9915, i.e. the continuity "
                "correction predicts this engine's own gap to better than 1% "
                "from m = 20 on, and nothing in the engine knows about "
                "beta = -zeta(1/2)/sqrt(2 pi)."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
    BarrierBSCase(
        row=BenchmarkRow(
            id="barrier_pde_node_sampled_projection_is_first_order",
            description=(
                "sampling the monitoring projection at nodes costs a full "
                "order and biases the price low"
            ),
            expected=1.0,
            tolerance=0.15,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_PDE_SOURCE,
            notes=(
                "Not anticipated by the slice. Observing a barrier makes the "
                "value function discontinuous, so each monitoring date repeats "
                "the Slice 6 digital pathology; and on a grid where H IS a "
                "node the node's cell is half alive while the node is set to "
                "the rebate, so the scheme discards half a cell of value per "
                "date. Measured on ONE grid (s_max = 380, so H = 95 is node "
                "n/4 for every n) at m = 10: node-sampled errors -6.339e-01, "
                "-3.365e-01, -1.711e-01, -8.604e-02, order 0.9619 (residual "
                "0.0140), every sign negative; cell-weighted +3.639e-02, "
                "+7.067e-03, +1.304e-03, +3.261e-04, order 2.2844. The "
                "cell-weighted projection is the L2 one and needs no node "
                "placement at all, which is why barrier_alignment stops "
                "mattering for the discrete contract."
            ),
        ),
        specs=(_ATM_6M_DOWN_OUT,),
    ),
)


# --------------------------------------------------------------------------
# (vii) Four engines on one number, and two on the other contract.
# --------------------------------------------------------------------------

_DISCRETE_CROSS_ENGINE_SOURCE = (
    "derived in-repo: qpl.engines.pde.barrier (projection at the monitoring "
    "dates) against qpl.engines.mc.barrier with barrier_correction='none' "
    "(unbiased for the discrete contract); the measured z-scores in `notes` "
    "come from tests/cases/test_barrier_black_scholes_cases.py"
)

BARRIER_DISCRETE_CROSS_ENGINE_CASES: tuple[BarrierBSCase, ...] = tuple(
    BarrierBSCase(
        row=BenchmarkRow(
            id=f"barrier_discrete_cross_engine_{name}",
            description=(
                "the finite-difference grid and the plain Monte Carlo "
                f"estimator agree on the DISCRETELY monitored contract at {name}"
            ),
            expected=0.0,
            tolerance=BARRIER_PDE_DISCRETE_STDERR_MULTIPLE,
            evidence=EvidenceClass.INDEPENDENT_ENGINE,
            source=_DISCRETE_CROSS_ENGINE_SOURCE,
            notes=(
                "`tolerance` is a z-score on the simulation's own standard "
                "error, so the row is INDEPENDENT_ENGINE resting on a "
                "STATISTICAL leg. Measured z at the study seed and "
                f"{BARRIER_MC_PATHS} paths: -0.287, +0.132, +0.347 across the "
                "three points. There is deliberately no lattice leg: "
                "qpl.engines.tree.barrier refuses a discrete schedule, because "
                "knocking out only at the time levels that coincide with "
                "monitoring dates needs n to be a multiple of m AND "
                "barrier-aligned at once. The grid has no such conflict -- its "
                "time levels are built to contain the monitoring dates -- "
                "which is the whole reason this row exists."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _CROSS_ENGINE_POINTS
)


ALL_BARRIER_CASES: tuple[BarrierBSCase, ...] = (
    BARRIER_HAUG_CASES
    + BARRIER_IDENTITY_CASES
    + BARRIER_LIMIT_CASES
    + BARRIER_MC_ORDER_CASES
    + BARRIER_TREE_ORDER_CASES
    + BARRIER_PDE_ORDER_CASES
    + BARRIER_CROSS_ENGINE_CASES
    + BARRIER_DISCRETE_CROSS_ENGINE_CASES
)
