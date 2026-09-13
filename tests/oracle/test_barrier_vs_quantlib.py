"""The barrier engines against QuantLib's, analytic, simulated and on a lattice.

Three comparisons with very different characters, and one of them is the most
interesting oracle result in this slice.

**Analytic.** QuantLib's `AnalyticBarrierEngine` evaluates the same
Reiner-Rubinstein construction this package derives in
`qpl.engines.analytic.barrier`, arranged differently. Agreement is therefore
expected at machine precision and measured at **2.886e-14 or better over 96
cells** -- all eight single-barrier types, both kinds, three strikes and both a
zero and a non-zero rebate. That leg checks the derivation and the assembly
table, not a discretisation, and it is what caught the one sign error in the
rebate block.

**Monte Carlo, discrete monitoring.** `ql.MCBarrierEngine(..., isBiased=True)`
checks the barrier at its own time-grid points, which is the *discretely
monitored* contract this package's `method="mc"` prices. Both estimators are
statistical, so this is an agreement-within-combined-noise check and is
labelled as such: measured `z = -0.743` at `m = 25` and `-1.132` at `m = 50`.
`isBiased=False` switches QuantLib to a Brownian-bridge crossing probability,
which is the same estimator as `barrier_correction="brownian_bridge"` here, and
the two agree at `z = -1.426` and `-0.745` while both sit within noise of the
continuous closed form (|z| of 0.806 / 1.897 here and 1.676 / 0.257 there).

**Lattice, and the result worth reading.** QuantLib's
`BinomialCRRBarrierEngine` shows **no Boyle-Lau sawtooth at all**. Over
`n = 200 ... 239` its error against the closed form spans +1e-05 to +2.0e-03
(amplitude 1.98e-03) where this package's plain knock-out-at-nodes lattice spans
+9.4e-03 to +1.24 (amplitude 1.229) -- a factor of **620** in amplitude, and
QuantLib is also ahead of this package's *Boyle-Lau-aligned* prices at the same
step counts (-7.3e-06 against +6.1e-04 at `n = 2375`).

That is not QuantLib being lucky. Its binomial barrier engine applies a barrier
adjustment to the nodes straddling the barrier rather than knocking out at the
first node beyond it -- the interpolation remedy whose standard reference is
Derman, Kani, Ergener and Bardhan (1995), "Enhanced numerical methods for
options with barriers", Risk 8(6). The attribution here is an inference from
the measured behaviour and not a claim about their source code; what is
asserted below is only the behaviour. Slice 12 cited that remedy and
deliberately implemented the other one (choose `n`), so that one remedy could
be measured properly rather than two badly; this leg is the record of what that
choice cost, and it is the strongest argument for implementing the
interpolation in a later slice.

Day count: `Actual365Fixed` with 365 days makes `T = 1.0` exactly in double
precision, so every comparison below is at the same maturity with no day-count
residual. `T = 0.5` -- the maturity of the published Haug rows -- is *not* an
exact `Actual365Fixed` fraction (182/365 = 0.4986) and is deliberately not used
here; those rows are checked against the published figures in
`tests/cases/test_barrier_black_scholes_cases.py` instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.analytic.barrier import barrier_price  # noqa: E402
from qpl.engines.mc.pricers import MCConfig  # noqa: E402
from qpl.engines.tree.barrier import boyle_lau_steps  # noqa: E402
from qpl.engines.tree.pricers import TreeConfig  # noqa: E402
from qpl.instruments.options import (  # noqa: E402
    BARRIER_TYPES,
    BarrierOption,
    uniform_monitoring_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import price  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()
_ONE_YEAR_DAYS = 365
_EXPIRY = 1.0

_SPOT, _RATE, _DIV, _SIGMA = 100.0, 0.08, 0.04, 0.25
_DOWN_BARRIER, _UP_BARRIER = 95.0, 105.0

_ANALYTIC_TOLERANCE = 1e-13
"""Both engines evaluate the same closed form; measured worst residual 2.886e-14
over 96 cells."""

_QL_BARRIER = {
    "down-and-out": ql.Barrier.DownOut,
    "down-and-in": ql.Barrier.DownIn,
    "up-and-out": ql.Barrier.UpOut,
    "up-and-in": ql.Barrier.UpIn,
}

_MC_MONITORING_LEVELS = (25, 50)
_QPL_MC_PATHS = 400_000
_QPL_MC_SEED = 20_260_913
_QL_MC_SAMPLES = 30_000
_QL_MC_SEED = 42
_MC_Z_TOLERANCE = 3.0
"""Both legs are statistical, so the comparison is a z-score on the combined
standard error. Measured |z| at 30 000 QuantLib samples: 0.743 and 1.132 for the
discrete contract, 1.426 and 0.745 for the bridged one. QuantLib's leg carries
four times this package's standard error at a quarter the effective cost, which
is the control variate and the pair reduction doing their work; the combined
error is therefore set almost entirely by QuantLib's."""


def _process(spot: float = _SPOT):
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, _DIV, _DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, _RATE, _DAY_COUNT)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, _SIGMA, _DAY_COUNT)
        ),
    )


def _ql_option(barrier_type: str, barrier: float, rebate: float, kind: str, strike: float):
    maturity = _EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)
    return ql.BarrierOption(
        _QL_BARRIER[barrier_type],
        barrier,
        rebate,
        ql.PlainVanillaPayoff(
            ql.Option.Call if kind == "call" else ql.Option.Put, strike
        ),
        ql.EuropeanExercise(maturity),
    )


def _market(spot: float = _SPOT) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=_SIGMA)


def _barrier_for(barrier_type: str) -> float:
    return _DOWN_BARRIER if barrier_type.startswith("down") else _UP_BARRIER


# --------------------------------------------------------------------------
# Analytic against analytic: all eight types.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rebate", [0.0, 3.0])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_all_eight_types_agree_to_machine_precision(
    barrier_type: str, kind: str, strike: float, rebate: float
) -> None:
    """Evidence class: INDEPENDENT_ENGINE, at round-off.

    Two arrangements of the same algebra, so the only thing that can differ is
    the order of the additions. Measured worst residual over these 96 cells:
    **2.886e-14**.

    The rebate cells are load-bearing, not padding. They are the only ones that
    exercise the blocks `E` and `F`, and `F`'s second normal argument is
    `N(eta z - 2 eta lambda v)`: writing it as `N(eta (z - 2 eta lambda v))` is
    the *same expression* for a down barrier and a different one for an up
    barrier, so a version with that error passes every down-barrier cell and
    fails the up-and-out ones by 1.04. Agreement here also pins that the two
    packages use the same rebate conventions -- paid at the touch time for a
    knock-out, at expiry for a knock-in -- which is not something a zero-rebate
    comparison can check.
    """
    barrier = _barrier_for(barrier_type)
    option = _ql_option(barrier_type, barrier, rebate, kind, strike)
    option.setPricingEngine(ql.AnalyticBarrierEngine(_process()))

    ours = barrier_price(
        S=_SPOT, K=strike, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV, H=barrier,
        rebate=rebate, barrier_type=barrier_type, kind=kind,
    )
    assert ours == pytest.approx(option.NPV(), abs=_ANALYTIC_TOLERANCE)

    # The dispatcher route returns the same number to within one ulp:
    # `Market.rate(t)` recovers the rate as `-log(df)/t` rather than storing it.
    through_dispatcher = price(
        BarrierOption(kind, strike, _EXPIRY, barrier, barrier_type, rebate),
        _model(), _market(),
    ).value
    assert through_dispatcher == pytest.approx(ours, abs=1e-13)


def test_the_analytic_agreement_is_not_vacuous() -> None:
    """A barrier is not a vanilla, and 1e-13 is not a wide net.

    The same maturity and strike priced without the barrier is 11.37 against
    the knock-out's 5.08 -- thirteen decimal orders above the tolerance the
    rows above use -- so the machine-precision agreement is a statement about
    the barrier formulas and not about two engines both returning something
    plausible. The knock-in at the same point is 6.29, a further 1.21 away, so
    the two assemblies are not accidentally the same number either.
    """
    knock_out = barrier_price(
        S=_SPOT, K=100.0, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
        H=_DOWN_BARRIER, rebate=0.0, barrier_type="down-and-out", kind="call",
    )
    knock_in = barrier_price(
        S=_SPOT, K=100.0, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
        H=_DOWN_BARRIER, rebate=0.0, barrier_type="down-and-in", kind="call",
    )
    vanilla = _ql_option("down-and-out", 1e-8, 0.0, "call", 100.0)
    vanilla.setPricingEngine(ql.AnalyticBarrierEngine(_process()))
    assert abs(vanilla.NPV() - knock_out) > 6.0
    assert abs(knock_in - knock_out) > 1.0


def test_the_day_count_makes_one_year_exact() -> None:
    """`Actual365Fixed` over 365 days is `1.0` in double precision.

    Pinned because the whole file depends on it: the published Haug rows use
    `T = 0.5`, which is 182/365 = 0.49863 under this day count, and comparing
    the two engines there would confound a formula difference with a maturity
    difference of 1.4e-03 years.
    """
    maturity = _EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)
    assert _DAY_COUNT.yearFraction(_EVALUATION_DATE, maturity) == _EXPIRY
    half = _EVALUATION_DATE + ql.Period(182, ql.Days)
    assert _DAY_COUNT.yearFraction(_EVALUATION_DATE, half) != 0.5


# --------------------------------------------------------------------------
# Monte Carlo: the discrete contract, and the bridged one.
# --------------------------------------------------------------------------


def _ql_mc_price(m: int, *, biased: bool) -> tuple[float, float]:
    option = _ql_option("down-and-out", _DOWN_BARRIER, 0.0, "call", 100.0)
    option.setPricingEngine(
        ql.MCBarrierEngine(
            _process(),
            "pseudorandom",
            timeSteps=m,
            brownianBridge=False,
            antitheticVariate=True,
            requiredSamples=_QL_MC_SAMPLES,
            maxSamples=4 * _QL_MC_SAMPLES,
            isBiased=biased,
            seed=_QL_MC_SEED,
        )
    )
    return option.NPV(), option.errorEstimate()


def _our_mc_price(m: int, *, correction: str):
    option = BarrierOption(
        "call", 100.0, _EXPIRY, _DOWN_BARRIER, "down-and-out", 0.0,
        uniform_monitoring_times(_EXPIRY, m),
    )
    return price(
        option, _model(), _market(), method="mc",
        cfg=MCConfig(
            n_paths=_QPL_MC_PATHS,
            seed=_QPL_MC_SEED,
            variance_reduction=("antithetic", "control_variate"),
            barrier_correction=correction,  # type: ignore[arg-type]
        ),
    )


@pytest.mark.parametrize("m", _MC_MONITORING_LEVELS)
def test_the_discretely_monitored_price_agrees_with_quantlibs(m: int) -> None:
    """Evidence class: INDEPENDENT_ENGINE and STATISTICAL at once.

    `isBiased=True` puts QuantLib's barrier check on its own time grid, which is
    the same contract this package prices when the instrument names `m` equally
    spaced monitoring dates. Both legs are simulations, so the comparison is a
    z-score on the combined standard error -- **not** an accuracy claim, and
    labelled that way because an MC-versus-MC agreement can only ever rule out
    a bias larger than the noise.

    Measured z: -0.743 at `m = 25`, -1.132 at `m = 50`. Both are far from the
    continuous closed form (7.10 and 6.53 against 5.0838), which is what makes
    this a check on the discrete contract rather than on the continuous one:
    two engines agreeing on 7.10 when the continuous answer is 5.08 is evidence
    they are pricing the same *contract*.
    """
    ours = _our_mc_price(m, correction="none")
    theirs, their_error = _ql_mc_price(m, biased=True)
    z = (ours.value - theirs) / math.hypot(ours.stderr, their_error)
    assert abs(z) < _MC_Z_TOLERANCE, f"m={m} z={z}"

    continuous = barrier_price(
        S=_SPOT, K=100.0, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
        H=_DOWN_BARRIER, rebate=0.0, barrier_type="down-and-out", kind="call",
    )
    assert ours.value > continuous + 1.0
    assert theirs > continuous + 1.0


@pytest.mark.parametrize("m", _MC_MONITORING_LEVELS)
def test_the_bridged_price_agrees_with_quantlibs_and_with_the_closed_form(
    m: int,
) -> None:
    """Evidence class: INDEPENDENT_ENGINE, STATISTICAL, and CLOSED_FORM.

    `isBiased=False` switches QuantLib to the Brownian-bridge crossing
    probability, which is the construction `barrier_correction="brownian_bridge"`
    implements here. Three numbers have to line up: the two simulations with
    each other (measured z -1.426 and -0.745), and each of them with the
    continuous closed form (|z| 0.806 and 1.897 here, 1.676 and 0.257 there) --
    which is the point, since a bridged estimator is unbiased for the
    continuous contract and a discrete one is not.
    """
    ours = _our_mc_price(m, correction="brownian_bridge")
    theirs, their_error = _ql_mc_price(m, biased=False)
    continuous = barrier_price(
        S=_SPOT, K=100.0, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
        H=_DOWN_BARRIER, rebate=0.0, barrier_type="down-and-out", kind="call",
    )

    assert ours.meta["estimates"] == "continuous"
    z_pair = (ours.value - theirs) / math.hypot(ours.stderr, their_error)
    assert abs(z_pair) < _MC_Z_TOLERANCE, f"m={m} pair z={z_pair}"
    assert abs(ours.value - continuous) < _MC_Z_TOLERANCE * ours.stderr
    assert abs(theirs - continuous) < _MC_Z_TOLERANCE * their_error


# --------------------------------------------------------------------------
# The lattice, and what this package left on the table.
# --------------------------------------------------------------------------

_LATTICE_WINDOW = range(200, 240)


def test_quantlibs_binomial_barrier_has_no_sawtooth_and_this_ones_does() -> None:
    """Evidence class: NEGATIVE_FINDING, against an independent implementation.

    Over `n = 200 ... 239` at the reference down-and-out call:

        engine                        min err     max err    amplitude
        qpl  (knock out at nodes)    +9.41e-03   +1.239      1.229
        QuantLib BinomialCRRBarrier  +1.09e-05   +1.99e-03   1.98e-03

    a factor of **620** in amplitude. QuantLib's binomial barrier engine
    adjusts the value at the nodes straddling the barrier instead of knocking
    out at the first node beyond it -- the interpolation remedy whose standard
    reference is Derman, Kani, Ergener and Bardhan (1995), Risk 8(6). The
    attribution is inferred from the behaviour measured here, not read from
    their source; only the behaviour is asserted.

    Slice 12 cited that remedy and implemented the other one (choose `n` so a
    layer lands on the barrier), so that one remedy could be measured properly
    rather than two badly. This test is the record of what that choice cost,
    and the argument for the later slice that closes it.
    """
    continuous = barrier_price(
        S=_SPOT, K=100.0, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
        H=_DOWN_BARRIER, rebate=0.0, barrier_type="down-and-out", kind="call",
    )
    option = BarrierOption("call", 100.0, _EXPIRY, _DOWN_BARRIER, "down-and-out", 0.0)
    ql_option = _ql_option("down-and-out", _DOWN_BARRIER, 0.0, "call", 100.0)
    process = _process()

    ours, theirs = [], []
    for n in _LATTICE_WINDOW:
        ours.append(
            price(option, _model(), _market(), method="tree",
                  cfg=TreeConfig(n_steps=n)).value - continuous
        )
        ql_option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n))
        theirs.append(ql_option.NPV() - continuous)

    our_amplitude = max(ours) - min(ours)
    their_amplitude = max(theirs) - min(theirs)
    assert our_amplitude > 1.0
    assert their_amplitude < 0.01
    assert our_amplitude / their_amplitude > 100.0
    # Both are one-signed: the effective barrier is never nearer the spot than
    # the contractual one, so both engines over-value the knock-out.
    assert min(ours) > 0.0
    assert min(theirs) > -1e-9


def test_quantlibs_lattice_also_beats_the_boyle_lau_subsequence() -> None:
    """Evidence class: NEGATIVE_FINDING -- choosing `n` is the weaker remedy.

    At the Boyle-Lau step counts for layers 10, 12, 14 and 16 (`n` = 2375,
    3420, 4656, 6081 at this specification) the errors are

        layer        10          12          14          16
        qpl      +6.10e-04   +4.50e-04   +1.44e-05   +1.76e-04
        QuantLib -7.33e-06   -8.35e-06   -8.41e-06   +5.35e-05

    -- so even where this package's lattice is at its best, the interpolating
    engine is one to two decimal orders ahead and, unlike this one, is *stable*
    across the four counts rather than varying by a factor of 42. Choosing `n`
    removes the sawtooth's amplitude; it does not remove the leftover
    misalignment, and interpolation does.
    """
    continuous = barrier_price(
        S=_SPOT, K=100.0, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
        H=_DOWN_BARRIER, rebate=0.0, barrier_type="down-and-out", kind="call",
    )
    option = BarrierOption("call", 100.0, _EXPIRY, _DOWN_BARRIER, "down-and-out", 0.0)
    ql_option = _ql_option("down-and-out", _DOWN_BARRIER, 0.0, "call", 100.0)
    process = _process()

    ours, theirs = [], []
    for layer in (10, 12, 14, 16):
        n = boyle_lau_steps(
            layer, spot=_SPOT, barrier=_DOWN_BARRIER, sigma=_SIGMA, expiry=_EXPIRY
        )
        ours.append(
            abs(price(option, _model(), _market(), method="tree",
                      cfg=TreeConfig(n_steps=n)).value - continuous)
        )
        ql_option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n))
        theirs.append(abs(ql_option.NPV() - continuous))

    assert np.mean(ours) > 5.0 * np.mean(theirs)
    # Stability, which is the half of the finding a mean hides.
    assert max(ours) / min(ours) > 10.0
    assert max(theirs) / min(theirs) < 15.0
