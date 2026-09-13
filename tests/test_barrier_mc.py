"""Monte Carlo on a discretely monitored barrier, and the monitoring bias.

The engine's contract, the three estimators, and the measurement that is the
point of the slice. Every assertion names its `EvidenceClass`.

1. **EXACT_IDENTITY** -- at a fixed seed the knock-in and knock-out estimators
   run on the *same paths*, and their payoffs partition each path, so their
   estimates sum to the plain vanilla estimate on those paths **bit for bit**.
   Measured difference: 0.0 exactly.
2. **CONVERGENCE_ORDER** -- the plain estimator's bias against the continuous
   closed form decays like `m**-0.5`. Measured 0.5045 at the study seed, with a
   lower-noise paired estimate of **0.4603 +- 0.0125** over ten seeds.
3. **STATISTICAL** -- the Brownian-bridge estimator agrees with the continuous
   closed form within its own noise at every `m`, including `m = 1`.
4. **NEGATIVE_FINDING** -- the BGK correction's residual bias is below the
   paired noise floor for `m >= 50`, so its *order* is not measurable at this
   cost. What is measurable is the size of the drop.
5. Contract checks: what the engine refuses, and why.

Sources: Broadie, Glasserman and Kou (1997), "A continuity correction for
discrete barrier options", Mathematical Finance 7(4), 325-348; Glasserman
(2003), Monte Carlo Methods in Financial Engineering, sections 6.4 and 3.1. The
constructions are re-derived in `qpl.engines.mc.barrier`; every number below was
measured in this repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpl.engines.analytic.barrier import barrier_price
from qpl.engines.mc.barrier import (
    barrier_terminal_sample,
    bridge_survival,
    normalise_barrier_correction,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.variance_reduction import estimate_from_sample
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import (
    BarrierOption,
    EuropeanOption,
    uniform_monitoring_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

# The study point: a down-and-out call with the barrier 5% below the spot, so
# the knock-out probability is large (0.71 at m = 50) and the monitoring bias
# is a first-order effect rather than a rounding difference.
_SPOT, _STRIKE, _EXPIRY = 100.0, 100.0, 0.5
_RATE, _DIV, _SIGMA, _BARRIER = 0.08, 0.04, 0.25, 95.0

_MONITORING_LEVELS: tuple[int, ...] = (25, 50, 100, 200, 400)
"""`m` ladder for the bias fit; `h = 1 / m`, spanning a factor of 16."""

_STUDY_PATHS = 40_000
_STUDY_SEED = 20_260_913
_STUDY_VR: tuple[str, ...] = ("antithetic", "control_variate")
"""40 000 paths with antithetic sampling and the vanilla control variate.

The control's measured correlation with the barrier payoff is 0.755-0.821 over
the ladder, a variance factor of about 2.5 -- modest, because the barrier
payoff *is* the vanilla payoff times an indicator and the indicator is exactly
the part the control cannot explain. It is kept because it is free, and because
the same sample is reused for all three estimators.
"""

_BIAS_ORDER_EXPECTED = 0.5
_BIAS_ORDER_BAND = 0.15
"""Band on the fitted bias order, derived from two separate spreads.

Over ten seeds the fit against the closed form gives 0.4764 +- 0.0329 (min
0.4350, max 0.5288), so seed noise alone needs about +-0.10. On top of that
there is a *systematic* shortfall: the much lower-noise paired estimator (the
plain estimator minus the Brownian-bridge estimator on the same paths) gives
0.4603 +- 0.0125 over the same ten seeds, significantly below 0.5, which is the
`o(1/sqrt(m))` term of the Broadie-Glasserman-Kou expansion still visible at
`m = 25`. Fitting `a/sqrt(m) + b/m` to the measured biases gives `b < 0`, which
is the sign that drags a finite-window fit below 0.5. The band therefore has to
cover 0.46 as well as 0.50, and 0.15 does with room to spare in both
directions.
"""

_NOISE_MULTIPLE = 3.0
"""How many of its own standard errors a "within noise" row is allowed."""


def _market() -> Market:
    return Market(
        spot=_SPOT,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=_SIGMA)


def _option(m: int, *, barrier_type: str = "down-and-out", rebate: float = 0.0):
    return BarrierOption(
        "call", _STRIKE, _EXPIRY, _BARRIER, barrier_type, rebate,
        uniform_monitoring_times(_EXPIRY, m),
    )


_CONTINUOUS = barrier_price(
    S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV, H=_BARRIER,
    rebate=0.0, barrier_type="down-and-out", kind="call",
)
"""4.512598607823680 -- the continuous closed form at the study point."""


@pytest.fixture(scope="module")
def ladder() -> dict[int, dict[str, object]]:
    """One sample per `m`, reused by every estimator and every test below.

    The three corrections differ only in how the payoff is read off the path,
    so at a fixed seed, path count and grid they run on **identical** draws.
    Building the sample once and reading it three ways is what makes the paired
    differences below legitimate: `plain - bridge` is a per-path difference,
    not a difference of two independent runs, and its standard error is
    correspondingly small.
    """
    market, model = _market(), _model()
    out: dict[int, dict[str, object]] = {}
    for m in _MONITORING_LEVELS:
        option = _option(m)
        cfg = MCConfig(
            n_paths=_STUDY_PATHS, seed=_STUDY_SEED, variance_reduction=_STUDY_VR
        )
        samples, estimates = {}, {}
        for correction in ("none", "bgk", "brownian_bridge"):
            sample, diagnostics = barrier_terminal_sample(
                option, model, market, cfg=cfg, methods=_STUDY_VR,
                correction=correction,
            )
            samples[correction] = sample
            estimates[correction] = estimate_from_sample(sample, methods=_STUDY_VR)
            if correction == "none":
                out.setdefault(m, {})["knock_fraction"] = diagnostics["knock_fraction"]
        out[m]["samples"] = samples
        out[m]["estimates"] = estimates
    return out


def _paired(ladder_row: dict[str, object], a: str, b: str) -> tuple[float, float]:
    samples = ladder_row["samples"]  # type: ignore[index]
    d = samples[a].y - samples[b].y
    return float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(d.size))


# --------------------------------------------------------------------------
# (5) The contract.
# --------------------------------------------------------------------------


def test_mc_refuses_a_continuously_monitored_barrier() -> None:
    """Simulation observes a schedule; a continuous contract supplies none."""
    option = BarrierOption("call", _STRIKE, _EXPIRY, _BARRIER, "down-and-out")
    with pytest.raises(NotSupportedError) as excinfo:
        price(option, _model(), _market(), method="mc", cfg=MCConfig(n_paths=100))
    message = str(excinfo.value)
    assert "brownian_bridge" in message
    assert "method='analytic'" in message


def test_mc_refuses_stratification_and_names_the_confusion() -> None:
    """The Brownian-bridge *sampler* and the Brownian-bridge *correction* differ.

    One changes which normals are drawn, the other changes how the payoff reads
    a path that was drawn the usual way. The refusal says so, because a reader
    who has just configured `barrier_correction="brownian_bridge"` has every
    reason to think stratification would now work.
    """
    with pytest.raises(NotSupportedError) as excinfo:
        price(
            _option(10), _model(), _market(), method="mc",
            cfg=MCConfig(n_paths=1024, variance_reduction="stratified", n_strata=64),
        )
    message = str(excinfo.value)
    assert "SAMPLER" in message
    assert "barrier_correction='brownian_bridge'" in message


def test_mc_validates_its_own_settings() -> None:
    """`n_steps`, the correction name, and the two schedule-shape rules."""
    market, model = _market(), _model()
    with pytest.raises(InvalidInputError, match="n_steps must be 1"):
        price(_option(10), model, market, method="mc", cfg=MCConfig(n_steps=4))
    with pytest.raises(InvalidInputError, match="n_paths must be >= 2"):
        price(_option(10), model, market, method="mc", cfg=MCConfig(n_paths=1))
    with pytest.raises(InvalidInputError, match="barrier_correction must be one of"):
        price(
            _option(10), model, market, method="mc",
            cfg=MCConfig(n_paths=100, barrier_correction="shift"),  # type: ignore[arg-type]
        )
    with pytest.raises(InvalidInputError, match="n_paths must be even"):
        price(
            _option(10), model, market, method="mc",
            cfg=MCConfig(n_paths=101, variance_reduction="antithetic"),
        )

    # `bgk` names a single `dt`, so an uneven schedule has no correction.
    uneven = BarrierOption(
        "call", _STRIKE, _EXPIRY, _BARRIER, "down-and-out", 0.0, (0.1, 0.2, 0.5)
    )
    with pytest.raises(InvalidInputError, match="equally spaced"):
        price(
            uneven, model, market, method="mc",
            cfg=MCConfig(n_paths=100, barrier_correction="bgk"),
        )
    # ... but the bridge needs no uniformity and prices it.
    assert price(
        uneven, model, market, method="mc",
        cfg=MCConfig(n_paths=1000, barrier_correction="brownian_bridge"),
    ).value > 0.0

    assert normalise_barrier_correction("BGK") == "bgk"
    with pytest.raises(InvalidInputError, match="barrier_correction must be one of"):
        normalise_barrier_correction(3)


def test_the_bridge_refuses_a_knock_out_rebate_but_not_a_knock_in_one() -> None:
    """Evidence class: none -- a statement about what the estimator knows.

    A knock-out rebate is paid at the *touch time*, and the bridge supplies the
    probability that the path crossed between two samples without supplying the
    distribution of when. A knock-in rebate is paid at expiry, which the bridge
    knows exactly, so it is allowed. The asymmetry is the whole point: the
    refusal is about the payment date, not about the rebate.
    """
    market, model = _market(), _model()
    cfg = MCConfig(n_paths=2_000, barrier_correction="brownian_bridge")
    with pytest.raises(NotSupportedError, match="touch time"):
        price(_option(10, rebate=3.0), model, market, method="mc", cfg=cfg)
    knock_in = price(
        _option(10, barrier_type="down-and-in", rebate=3.0), model, market,
        method="mc", cfg=cfg,
    )
    assert knock_in.value > 0.0
    # Both corrections that do track the touch time will price the knock-out.
    for correction in ("none", "bgk"):
        assert price(
            _option(10, rebate=3.0), model, market, method="mc",
            cfg=MCConfig(n_paths=2_000, barrier_correction=correction),  # type: ignore[arg-type]
        ).value > 0.0


def test_mc_barrier_greeks_are_refused_with_a_reason() -> None:
    """Registered-but-raising, so the message is about barriers and not lookup."""
    with pytest.raises(NotSupportedError) as excinfo:
        greeks(_option(10), _model(), _market(), method="mc", cfg=MCConfig(n_paths=100))
    message = str(excinfo.value)
    assert "pathwise" in message and "likelihood-ratio" in message
    assert "method='analytic'" in message


def test_mc_is_deterministic_and_reports_what_it_priced() -> None:
    cfg = MCConfig(n_paths=5_000, seed=3)
    first = price(_option(50), _model(), _market(), method="mc", cfg=cfg)
    second = price(_option(50), _model(), _market(), method="mc", cfg=cfg)
    assert first.value == second.value
    assert first.meta["estimates"] == "discrete"
    assert first.meta["n_monitoring"] == 50
    assert first.meta["barrier_correction"] == "none"
    assert first.meta["effective_barrier"] == _BARRIER

    bridged = price(
        _option(50), _model(), _market(), method="mc",
        cfg=MCConfig(n_paths=5_000, seed=3, barrier_correction="brownian_bridge"),
    )
    assert bridged.meta["estimates"] == "continuous"
    shifted = price(
        _option(50), _model(), _market(), method="mc",
        cfg=MCConfig(n_paths=5_000, seed=3, barrier_correction="bgk"),
    )
    assert shifted.meta["estimates"] == "continuous"
    # Toward the spot for a down barrier, which is what makes the discretely
    # monitored simulation estimate the continuous price.
    assert _BARRIER < shifted.meta["effective_barrier"] < _SPOT


@pytest.mark.parametrize("barrier_type", ["down-and-out", "down-and-in"])
def test_a_touched_spot_gives_the_same_answer_as_the_closed_form(
    barrier_type: str,
) -> None:
    """Evidence class: EXACT_IDENTITY across engines.

    A spot already beyond the barrier settles the contract, so the Monte Carlo
    engine must return the analytic number exactly and must not spend a single
    path getting there.
    """
    market = Market(
        spot=90.0,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )
    option = BarrierOption(
        "call", _STRIKE, _EXPIRY, _BARRIER, barrier_type, 3.0,
        uniform_monitoring_times(_EXPIRY, 50),
    )
    mc = price(option, _model(), market, method="mc", cfg=MCConfig(n_paths=10))
    analytic = price(option.__class__(
        option.kind, option.strike, option.expiry, option.barrier,
        option.barrier_type, option.rebate,
    ), _model(), market)
    assert mc.value == analytic.value
    assert mc.stderr == 0.0
    assert mc.meta["degenerate"] == "touched_at_inception"


def test_zero_volatility_observes_the_schedule_and_not_the_path() -> None:
    """Evidence class: CLOSED_FORM, and a place the two contracts differ.

    At `sigma = 0` the forward is deterministic. With `r > q` it rises through
    an up barrier at `t* = log(H/S)/(r-q) = 0.1247`; a discrete contract only
    notices at the first observation date **at or after** `t*`, here `0.25`.

    Both contracts knock out and both pay the same rebate, so the only
    difference is the payment date -- and the discrete one is therefore worth
    **less**, not more: 2.94060 against 2.97022, exactly `3 e^{-r/4}` against
    `3 e^{-r t*}`. That is the opposite of the usual reading that discrete
    monitoring favours the holder of a knock-out, and it is worth pinning
    because both statements are true of different cash flows: a later knock-out
    preserves more *option* value and less *rebate* value, and with `sigma = 0`
    there is no option value left to preserve.
    """
    market = _market()
    model = BlackScholesModel(sigma=0.0)
    schedule = uniform_monitoring_times(_EXPIRY, 2)
    option = BarrierOption(
        "call", _STRIKE, _EXPIRY, 100.5, "up-and-out", 3.0, schedule
    )
    discrete = price(option, model, market, method="mc", cfg=MCConfig(n_paths=10))
    continuous = barrier_price(
        S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=0.0, q=_DIV, H=100.5,
        rebate=3.0, barrier_type="up-and-out", kind="call",
    )
    assert discrete.stderr == 0.0
    assert discrete.meta["degenerate"] == "sigma=0"
    # Both knock out; the discrete one pays at t = 0.25 rather than at t*.
    assert discrete.value < continuous
    assert discrete.value == pytest.approx(3.0 * np.exp(-_RATE * 0.25), abs=1e-14)
    t_star = np.log(100.5 / _SPOT) / (_RATE - _DIV)
    assert continuous == pytest.approx(3.0 * np.exp(-_RATE * t_star), abs=1e-14)


# --------------------------------------------------------------------------
# (1) The pathwise partition.
# --------------------------------------------------------------------------


def test_knock_in_plus_knock_out_is_the_vanilla_on_the_same_paths() -> None:
    """Evidence class: EXACT_IDENTITY, bit for bit.

    The two barrier payoffs partition each path, so on a fixed sample their
    means add to the mean of the vanilla payoff on that same sample --
    identically, not within noise. Measured difference: **0.0 exactly**.

    This checks the indicator bookkeeping in a way no comparison with a closed
    form can, because it does not depend on the sample being large: a path
    counted as knocked in both legs, or in neither, breaks it immediately.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    market, model = _market(), _model()
    cfg = MCConfig(n_paths=20_000, seed=11)
    schedule = uniform_monitoring_times(_EXPIRY, 50)
    common = ("call", _STRIKE, _EXPIRY, _BARRIER)

    knock_in = price(
        BarrierOption(*common, "down-and-in", 0.0, schedule), model, market,
        method="mc", cfg=cfg,
    )
    knock_out = price(
        BarrierOption(*common, "down-and-out", 0.0, schedule), model, market,
        method="mc", cfg=cfg,
    )
    sample, _ = barrier_terminal_sample(
        BarrierOption(*common, "down-and-out", 0.0, schedule), model, market,
        cfg=cfg, methods=(), correction="none",
    )
    # `sample.x` is the discounted vanilla payoff on exactly these paths.
    assert knock_in.value + knock_out.value == float(np.mean(sample.x))

    # And the vanilla mean is itself close to the Black-Scholes price, so the
    # identity is not being satisfied by two wrong numbers.
    assert float(np.mean(sample.x)) == pytest.approx(
        bs_price(S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
                 kind="call"),
        rel=5e-3,
    )
    # The same identity through the dispatcher's own vanilla engine, within its
    # noise rather than exactly: a different sampler, a different sample.
    vanilla = price(
        EuropeanOption("call", _STRIKE, _EXPIRY), model, market, method="mc",
        cfg=MCConfig(n_paths=20_000, seed=11),
    )
    assert knock_in.value + knock_out.value == pytest.approx(
        vanilla.value, abs=6.0 * vanilla.stderr
    )


# --------------------------------------------------------------------------
# The bridge formula itself.
# --------------------------------------------------------------------------


def test_bridge_survival_is_zero_once_a_sample_lands_beyond_the_barrier() -> None:
    """Evidence class: EXACT_IDENTITY on the degenerate branch.

    If a sampled point is on the far side of the barrier the path has certainly
    crossed, and the formula must say so rather than returning
    `exp(-2 (+ve)(+ve)/...)`, which is what the raw expression evaluates to
    when *both* endpoints are beyond. It never has to: the initial spot is on
    the live side (the engine guards that), so a path that ends up beyond the
    barrier has a straddling interval earlier, and that interval alone zeroes
    the product. Both branches are checked here.
    """
    sigma, dt = 0.25, np.array([0.1, 0.1])
    # Straddling interval: the product of the two log-ratios is negative.
    straddle = np.array([[0.05, -0.02, -0.10]])
    assert bridge_survival(straddle, sigma=sigma, dt=dt)[0] == 0.0
    # A path that stays well clear survives with high probability.
    clear = np.array([[0.30, 0.32, 0.35]])
    survival = bridge_survival(clear, sigma=sigma, dt=dt)[0]
    assert 0.0 < survival < 1.0
    assert survival > 0.5
    # Exactly on the barrier counts as a crossing.
    touching = np.array([[0.05, 0.0, 0.10]])
    assert bridge_survival(touching, sigma=sigma, dt=dt)[0] == 0.0


@pytest.mark.parametrize("m", [1, 2, 4])
def test_one_bridged_interval_already_prices_the_continuous_barrier(m: int) -> None:
    """Evidence class: STATISTICAL, and the slice's most surprising row.

    The bridge estimator is `E[payoff * 1{no crossing} | sampled points]`, so it
    is unbiased for the **continuous** contract at *any* number of sampling
    dates -- conditioning on more of the path changes its variance and not its
    mean. At `m = 1` the option is sampled once, at expiry, and the estimate is
    already the continuous price: z = -0.10, -0.58, -0.68 at m = 1, 2, 4 over
    200 000 paths. The plain estimator at `m = 1` is not remotely close (it
    cannot knock out at all, so it returns the vanilla, 7.85 against 4.51).
    """
    market, model = _market(), _model()
    bridged = price(
        _option(m), model, market, method="mc",
        cfg=MCConfig(n_paths=200_000, seed=7, barrier_correction="brownian_bridge"),
    )
    z = (bridged.value - _CONTINUOUS) / bridged.stderr
    assert abs(z) < 2.5, f"m={m} z={z}"

    if m == 1:
        plain = price(
            _option(m), model, market, method="mc",
            cfg=MCConfig(n_paths=200_000, seed=7),
        )
        # One observation at expiry: the barrier can only be "touched" by the
        # terminal spot, so the discrete contract is nearly the vanilla.
        assert plain.value > _CONTINUOUS + 2.0


# --------------------------------------------------------------------------
# (2), (3), (4): the ladder.
# --------------------------------------------------------------------------


def test_the_monitoring_bias_is_order_one_half(ladder) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    The plain estimator is unbiased for the **discrete** contract and biased
    for the continuous one, and the bias is the contract's, not the estimator's.
    Measured at the study seed, biases against the closed form:

        m          25        50       100       200       400
        bias    +1.1211   +0.8164   +0.6221   +0.4490   +0.2631
        stderr   0.0280    0.0299    0.0307    0.0314    0.0320

    fitted order **0.5045** (log-space residual 0.0687). The lower-noise paired
    version (plain minus bridge on the same paths) gives 0.4603 +- 0.0125 over
    ten seeds, and that 0.46 rather than 0.50 is real rather than noise; see
    `_BIAS_ORDER_BAND`.

    This is the prediction Slice 11 handed forward -- an exercise frequency
    costs `O(1/m)` and a monitoring frequency costs `O(1/sqrt(m))` -- and the
    difference is large: at `m = 25` the bias is 25% of the continuous price.
    """
    biases, errors = [], []
    for m in _MONITORING_LEVELS:
        estimate = ladder[m]["estimates"]["none"]  # type: ignore[index]
        biases.append(estimate.value - _CONTINUOUS)
        errors.append(estimate.stderr)

    assert all(b > 0.0 for b in biases)  # a knock-out is worth MORE discretely
    assert all(b > _NOISE_MULTIPLE * e for b, e in zip(biases, errors, strict=True))

    fit = fit_convergence_order(
        np.array([1.0 / m for m in _MONITORING_LEVELS]), np.abs(biases)
    )
    assert fit.order == pytest.approx(_BIAS_ORDER_EXPECTED, abs=_BIAS_ORDER_BAND)
    assert fit.residual < 0.15
    # Not an `O(1/m)` effect: that would predict a 16x drop over this ladder.
    assert biases[0] / biases[-1] < 8.0
    assert biases[0] / biases[-1] > 2.0


def test_the_bridge_removes_the_monitoring_bias_at_every_m(ladder) -> None:
    """Evidence class: STATISTICAL.

    The bridged estimate agrees with the continuous closed form within
    `_NOISE_MULTIPLE` of its own standard error at every `m`, while the plain
    estimate on the *same paths* is 8 to 40 standard errors out. Measured
    z-scores of the bridge at the study seed: +0.14, +0.13, -0.26, +0.71,
    -1.43.

    Unbiasedness is checked separately by coverage rather than by closeness:
    over 40 seeds at 20 000 paths the reported 95% interval covers the closed
    form 37/40 at `m = 25` and 38/40 at `m = 100` (Binomial(40, 0.95) mean 38),
    with a mean z of -0.13 and -0.09. That measurement is recorded here and not
    re-run, because 80 more simulations would triple this file's runtime to
    confirm a number that has already been taken.
    """
    for m in _MONITORING_LEVELS:
        estimates = ladder[m]["estimates"]  # type: ignore[index]
        bridge = estimates["brownian_bridge"]
        plain = estimates["none"]
        z_bridge = (bridge.value - _CONTINUOUS) / bridge.stderr
        z_plain = (plain.value - _CONTINUOUS) / plain.stderr
        assert abs(z_bridge) < _NOISE_MULTIPLE, f"m={m} z={z_bridge}"
        assert abs(z_plain) > 8.0, f"m={m} z={z_plain}"


def test_the_bgk_correction_drops_below_the_noise_floor(ladder) -> None:
    """Evidence class: NEGATIVE_FINDING -- the order is not measurable here.

    Broadie, Glasserman and Kou prove the shifted-barrier price equals the
    continuous price plus `o(1/sqrt(m))`. Measuring that order needs the
    residual to stay above the noise, and it does not. Paired against the
    Brownian-bridge estimator on the same paths (which removes almost all of
    the common variance), the residual bias is

        m            25        50       100       200       400
        bgk-bridge +0.0346   +0.0028   -0.0010   -0.0036   -0.0041
        stderr      0.0109    0.0098    0.0083    0.0067    0.0063

    -- resolved only at `m = 25` (3.2 standard errors, and +0.0346 +- 0.0109 is
    reproduced across ten seeds as +0.0008 to +0.0377), and indistinguishable
    from zero from `m = 50` on, with a sign that wanders. A fit through that is
    a fit through noise: it returns 0.578 with a log-space residual of 1.02,
    which is the diagnostic saying so.

    What *is* measurable is the size of the drop, and it is the real result:
    from `m = 25` to `m = 50` the plain bias falls by a factor of 1.37 and the
    BGK residual by at least 12. Resolving the order would need roughly 16x the
    paths at `m = 400` -- 256 million normals -- to halve an error bar that is
    already smaller than the quantity it is measuring.
    """
    paired = {m: _paired(ladder[m], "bgk", "brownian_bridge") for m in _MONITORING_LEVELS}
    plain = {m: _paired(ladder[m], "none", "brownian_bridge") for m in _MONITORING_LEVELS}

    # At the coarsest grid the BGK residual is real but already 30x smaller
    # than the uncorrected bias.
    coarse_bgk, coarse_se = paired[_MONITORING_LEVELS[0]]
    coarse_plain = plain[_MONITORING_LEVELS[0]][0]
    assert coarse_bgk > 2.0 * coarse_se
    assert abs(coarse_bgk) < coarse_plain / 20.0

    # From m = 50 on it is inside two standard errors and changes sign.
    tail = [paired[m] for m in _MONITORING_LEVELS[1:]]
    assert all(abs(value) < 2.0 * stderr for value, stderr in tail)
    assert min(v for v, _ in tail) < 0.0 < max(v for v, _ in tail)

    # The fit is reported as uninformative rather than suppressed.
    fit = fit_convergence_order(
        np.array([1.0 / m for m in _MONITORING_LEVELS]),
        np.abs([paired[m][0] for m in _MONITORING_LEVELS]),
    )
    assert fit.residual > 0.5  # an order-of-magnitude worse than the plain fit

    # And the drop from the first to the second level is the measurable claim.
    assert abs(paired[_MONITORING_LEVELS[1]][0]) < abs(coarse_bgk) / 10.0
    assert plain[_MONITORING_LEVELS[0]][0] / plain[_MONITORING_LEVELS[1]][0] < 1.5


def test_the_control_variate_is_real_but_modest(ladder) -> None:
    """Evidence class: STATISTICAL, with the mechanism stated.

    The control is the discounted vanilla payoff on the same path. Its measured
    correlation with the barrier payoff is 0.755 to 0.821 down the ladder,
    predicting a variance factor `1/(1-rho^2)` of 2.3 to 3.1 -- two orders of
    magnitude less than the Kemna-Vorst control buys on an Asian (Slice 8,
    rho = 0.9996). The reason is structural rather than unlucky: a barrier
    payoff *is* the vanilla payoff times an indicator, so what the control
    cannot explain is exactly the indicator, and the indicator is the whole
    contract. The correlation falls as `m` rises, because more monitoring dates
    mean more knock-outs and a larger residual.
    """
    correlations = [
        ladder[m]["estimates"]["none"].meta["control_correlation"]  # type: ignore[index]
        for m in _MONITORING_LEVELS
    ]
    assert all(0.70 < rho < 0.85 for rho in correlations)
    assert correlations[0] > correlations[-1]
    predicted = [1.0 / (1.0 - rho * rho) for rho in correlations]
    assert 2.0 < min(predicted) and max(predicted) < 4.0
    # The knock-out probability rises with m, which is the same fact.
    fractions = [ladder[m]["knock_fraction"] for m in _MONITORING_LEVELS]
    assert fractions[0] < fractions[-1]
    assert 0.6 < fractions[0] < fractions[-1] < 0.9
