"""The four Monte Carlo engines under `HestonModel`, and what changes.

This module is the pricing half of Slice 16. `qpl.engines.mc.heston` produces
`(ln S, v)` paths; here they become prices for `EuropeanOption`,
`DigitalOption`, `AsianOption` and `BarrierOption`, registered on the same
`method="mc"` keys the Black-Scholes engines use.

Why these are separate engines rather than a branch inside the existing ones
---------------------------------------------------------------------------
Every Black-Scholes Monte Carlo engine in this package rests on one property:
`qpl.engines.mc.processes.gbm_paths_from_normals` samples the path at the
contract's own dates **exactly**, so there is no time grid to choose and no
discretisation bias to measure. Slices 8 and 12 both say so in their module
docstrings, and both refuse `n_steps != 1` because of it.

Under Heston that property is gone. The path *must* be built on a time grid
finer than the contract's dates, the answer depends on how fine, and `n_steps`
stops being a cost knob and becomes the discretisation. Routing the Heston
model through the existing engines would therefore mean inverting the meaning
of `n_steps` inside each of them, replacing the exact sampler with a scheme,
and -- for the Asian -- replacing the Kemna-Vorst control variate, which has no
Heston analogue. That is four rewrites of engines whose current numbers are
pinned bit for bit by `tests/test_mc_pricing.py` and friends.

So the path *source* is the seam, and it is a `PathSampler` -- see
`heston_path_sample`, which is the single function every engine below calls to
get `(grid, spots, variances, conditional law)`. The Black-Scholes engines keep
their own exact sampler, unchanged and unreached from here;
`tests/test_mc_heston_pricing.py` pins that their output is bit-for-bit what it
was before this slice.

What `n_steps` means here
-------------------------
`MCConfig.n_steps` is the number of **uniform** steps over `[0, T]`. For a
path-dependent contract the simulation grid is the *union* of that uniform grid
with the contract's own fixing or monitoring dates, so the contract is always
observed exactly where it says and the discretisation is always at least as
fine as `n_steps` asks. `n_steps = 1` (the `MCConfig` default, and the value
every Black-Scholes engine wants) is **refused**: none of the three schemes is
exact in one step, so a one-step answer is a discretisation error wearing a
price's clothes.

Variance reduction, and what composes
-------------------------------------
- **Antithetic** reflects both driving normals and composes with everything
  except `heston_scheme="exact_variance_euler_log_spot"`, whose variance draw
  is not a function of a normal. Measured factor on the conditional European
  estimator: **7.7x** in variance.
- **Control variate**: the discounted terminal spot, whose mean
  `S_0 e^{-qT}` is exact *in the model*. Measured correlation with the plain
  European payoff **0.79 to 0.89** (factor 2.7 to 5.0). Two warnings, both
  measured, both in `docs/notes/heston_monte_carlo_qe.md`:
    1. On the **conditional** estimator the same control is nearly worthless
       for QE (rho 0.34 to 0.49, factor 1.1 to 1.3) -- conditioning has already
       integrated out the spot diffusion, which is exactly what the control was
       correlated with. The two variance reductions are substitutes, not
       complements.
    2. Its mean is the **model's**, not the **scheme's**. When the scheme's own
       `E[e^{-(r-q)T} S_T]` is off -- which is every scheme without the
       martingale correction, and full-truncation Euler always -- the control
       silently removes the part of the discretisation bias collinear with that
       defect. Measured shift on full-truncation Euler at `dt = 1/4`:
       **-5.2e-03** on the conditional estimator. That is a legitimate
       bias reduction and an illegitimate bias *measurement*, so the bias study
       in `examples/heston_mc_qe.py` runs without it.
- **Stratified** is refused, as it is for every path-dependent engine here:
  there is no single scalar driving the terminal law to partition.

Greeks
------
`greeks_estimator="bump"` (the default) gives all five by common-random-numbers
revaluation, with vega taken with respect to **`sqrt(v0)`** so that it is
comparable with the Black-Scholes vega and with the Heston transform vega
(`qpl.models.heston.HestonModel.with_volatility_bump`); the metadata says so.
`"pathwise"` supplies **delta only** and falls back to the bump for the rest:
`S_T` is homogeneous of degree one in `S_0` under Heston -- the variance
dynamics do not see the spot -- so `dS_T/dS_0 = S_T/S_0` exactly and the
pathwise delta of a Lipschitz payoff costs one multiplication. `"likelihood_
ratio"` is **refused**, with the reason in `_LIKELIHOOD_RATIO_REFUSAL`.

Greeks are refused outright for the digital, the Asian and the barrier under
Heston, for the reasons those engines already give under Black-Scholes, plus
one that is new here: a bumped Heston Greek costs a full re-simulation of a
multi-step scheme, which is the expensive half of every number on this page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import (
    AsianOption,
    BarrierOption,
    DigitalOption,
    EuropeanOption,
)
from ...instruments.payoffs import (
    arithmetic_average,
    asian_payoff,
    call_payoff,
    digital_payoff,
    geometric_average,
    put_payoff,
)
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.heston import HestonModel
from ..base import GreeksResult, PriceResult
from .barrier import (
    BGK,
    BROWNIAN_BRIDGE,
    NO_CORRECTION,
    bridge_survival_from_step_variance,
    normalise_barrier_correction,
)
from .greeks import (
    BUMP,
    GREEK_NAMES,
    LIKELIHOOD_RATIO,
    PATHWISE,
    GreekEstimate,
    Scenario,
    bump_estimates,
    bump_sizes,
    greeks_result,
    normalise_greeks_estimator,
    one_sided_estimate,
    reduce_to_units,
)
from .heston import (
    EXACT_VARIANCE_EULER_LOG_SPOT,
    HESTON_SCHEMES,
    ConditionalTerminalLaw,
    conditional_digital_values,
    conditional_forward,
    conditional_vanilla_values,
    simulate_heston,
)
from .pricers import MCConfig
from .variance_reduction import (
    ANTITHETIC,
    CONTROL_VARIATE,
    STRATIFIED,
    TerminalSample,
    control_variate_coefficient,
    estimate_from_sample,
    normalise_variance_reduction,
)

__all__ = [
    "HestonPathSample",
    "greeks_asian",
    "greeks_barrier",
    "greeks_digital",
    "greeks_european",
    "heston_path_sample",
    "heston_time_grid",
    "price_asian",
    "price_barrier",
    "price_digital",
    "price_european",
]

_TERMINAL_SPOT_CONTROL = "discounted_terminal_spot"

_N_STEPS_REFUSAL = (
    "n_steps must be >= 2 under HestonModel: it is the TIME DISCRETISATION "
    "here, not a cost knob. None of the three schemes "
    f"{HESTON_SCHEMES} is exact in one step -- the variance transition is "
    "exact only in 'exact_variance_euler_log_spot', and even there the "
    "log-spot step approximates the integrated variance by a two-point rule -- "
    "so a one-step answer reports a discretisation error as a price. Choose "
    "n_steps from the step size you want: n_steps = T / dt, with dt = 1/32 or "
    "finer on the reference parameters (measured bias 5e-03 there against "
    "1e-01 at dt = 1/4). The default MCConfig(n_steps=1) is the Black-Scholes "
    "engines' terminal-sampling value and has no Heston meaning."
)

_STRATIFIED_REFUSAL = (
    "variance_reduction 'stratified' is not available under HestonModel: it "
    "stratifies the single normal that drives a terminal price, and a Heston "
    "path is driven by 2 * n_steps normals with no single scalar to partition. "
    "Use 'antithetic', 'control_variate' (the discounted terminal spot, whose "
    "mean is exact in the model), or MCConfig(heston_conditional=True), which "
    "integrates the spot diffusion out exactly and is measured to be worth 10x "
    "to 54x in variance."
)

_LIKELIHOOD_RATIO_REFUSAL = (
    "greeks_estimator='likelihood_ratio' is not available under HestonModel. "
    "The score needs the transition density of (ln S, v), and the model's is "
    "not available in closed form: Broadie & Kaya (2006) obtain the exact "
    "joint transition only by inverting the characteristic function of the "
    "INTEGRATED variance numerically, which is the cost that made the QE "
    "scheme necessary in the first place. The QE step's own conditional law IS "
    "Gaussian and does have a score, but differentiating it estimates the "
    "derivative of the DISCRETISED law -- which is a legitimate thing to want "
    "and is not what an unbiased-likelihood-ratio claim means, so it is not "
    "offered under that name. Use greeks_estimator='bump' (all five Greeks by "
    "common random numbers) or 'pathwise' (delta only: S_T is homogeneous of "
    "degree one in S_0, so dS_T/dS_0 = S_T/S_0 exactly)."
)

_DIGITAL_GREEKS_REFUSAL = (
    "Monte Carlo Greeks are not available for a digital under HestonModel. "
    "The pathwise derivative of an indicator is zero almost everywhere (the "
    "Slice 6 finding, unchanged by the model), the likelihood-ratio score "
    "needs a transition density Heston does not have in closed form (see the "
    "European engine's refusal), and a common-random-numbers bump on a "
    "discontinuous payoff has a standard deviation that GROWS as the bump "
    "shrinks -- measured coverage 36/40 at best under Black-Scholes, and a "
    "Heston bump additionally re-simulates a multi-step scheme per leg. Use "
    "method='fourier', whose digital Greeks are exact in the transform."
)

_PATH_GREEKS_REFUSAL = (
    "Monte Carlo Greeks are not available for a path-dependent option under "
    "HestonModel. The Black-Scholes engines refuse the barrier for the same "
    "reason (the payoff carries an indicator of the running extremum, so its "
    "pathwise derivative is a surface measure) and support the Asian only "
    "through estimators built on the exact lognormal path, which does not "
    "exist here. A common-random-numbers bump would work and is deliberately "
    "not wired: it costs a full re-simulation of a multi-step scheme per leg "
    "per Greek, and this slice has no measurement that would justify the "
    "surface."
)


def heston_time_grid(expiry: float, n_steps: int, *, dates: tuple[float, ...] = ()) -> np.ndarray:
    """Uniform grid of `n_steps` steps over `[0, T]`, unioned with `dates`.

    The contract's own dates are always grid points, so a fixing or a barrier
    observation is read where the contract says it is and never interpolated;
    `n_steps` only guarantees that the discretisation is at least that fine.
    Duplicates are removed with `numpy.unique`, which also sorts, so a schedule
    that already lies on the uniform grid costs nothing.
    """
    uniform = expiry * np.arange(n_steps + 1, dtype=float) / float(n_steps)
    if not dates:
        return uniform
    schedule = np.asarray(dates, dtype=float)
    merged = np.unique(np.concatenate([uniform, schedule, [0.0]]))
    # Two ways of computing `i T / n` can differ in the last bit, and
    # `numpy.unique` then keeps both -- leaving a sub-ulp interval on which
    # `e^{-kappa dt}` rounds to 1 and the QE variance step degenerates. Collapse
    # near-duplicates, keeping the CONTRACT's date where the pair contains one,
    # so that `_column_indices` still finds every fixing exactly.
    tolerance = 1e-12 * max(float(expiry), 1.0)

    def _is_contract_date(value: float) -> bool:
        return bool(np.any(np.abs(schedule - value) <= tolerance))

    kept = [float(merged[0])]
    for value in merged[1:]:
        if value - kept[-1] <= tolerance:
            if _is_contract_date(value) and not _is_contract_date(kept[-1]):
                kept[-1] = float(value)
            continue
        kept.append(float(value))
    return np.asarray(kept, dtype=float)


@dataclass(frozen=True)
class HestonPathSample:
    """What every engine below reads: one simulated block plus its bookkeeping.

    Parameters
    ----------
    times
        The simulation grid (see `heston_time_grid`).
    spots, variances
        `(n_paths, len(times))` when the engine asked for paths, and
        `(n_paths, 1)` terminal columns otherwise.
    conditional
        The terminal law given the variance driver
        (`qpl.engines.mc.heston.ConditionalTerminalLaw`).
    n_normal_draws
        Standard normals actually drawn: `2 * n_paths * n_steps` plain,
        `n_paths * n_steps` antithetic (each drawn normal serves two paths).
        The cost axis the variance ratios are measured on, matching
        `qpl.engines.mc.variance_reduction.TerminalSample`.
    meta
        The simulator's own metadata.
    """

    times: np.ndarray
    spots: np.ndarray
    variances: np.ndarray
    conditional: ConditionalTerminalLaw
    n_normal_draws: int
    meta: dict[str, Any]


def heston_path_sample(
    model: HestonModel,
    market: Market,
    *,
    expiry: float,
    cfg: MCConfig,
    methods: tuple[str, ...],
    dates: tuple[float, ...] = (),
    store_paths: bool = False,
) -> HestonPathSample:
    """The `PathSampler` seam: one simulated block for any Heston engine.

    Every engine in this module gets its paths here and nowhere else, so the
    choice of scheme, the grid construction, the antithetic reflection and the
    normal-draw accounting exist once. The Black-Scholes engines keep their own
    exact sampler and never reach this function, which is what makes their
    output bit-for-bit unchanged by this slice.
    """
    antithetic = ANTITHETIC in methods
    grid = heston_time_grid(expiry, cfg.n_steps, dates=dates)
    paths = simulate_heston(
        model,
        s0=market.spot,
        mu=market.rate(expiry) - market.dividend_yield(expiry),
        t_grid=grid,
        n_paths=cfg.n_paths,
        seed=cfg.seed,
        scheme=cfg.heston_scheme,
        martingale_correction=True,
        antithetic=antithetic,
        store_paths=store_paths,
    )
    n_steps = grid.size - 1
    per_path = cfg.n_paths // 2 if antithetic else cfg.n_paths
    return HestonPathSample(
        times=grid,
        spots=paths.spots,
        variances=paths.variance,
        conditional=paths.conditional,
        n_normal_draws=2 * per_path * n_steps,
        meta=dict(paths.meta),
    )


def _validate(cfg: MCConfig, *, model: HestonModel) -> tuple[str, ...]:
    """Shared entry checks; returns the normalised variance-reduction methods."""
    if not isinstance(model, HestonModel):  # pragma: no cover - registry guarantees it
        raise InvalidInputError("model must be a HestonModel")
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps < 2:
        raise InvalidInputError(_N_STEPS_REFUSAL)
    if cfg.heston_scheme not in HESTON_SCHEMES:
        raise InvalidInputError(f"heston_scheme must be one of {HESTON_SCHEMES}")
    methods = normalise_variance_reduction(cfg.variance_reduction)
    if STRATIFIED in methods:
        raise NotSupportedError(_STRATIFIED_REFUSAL)
    if ANTITHETIC in methods:
        if cfg.n_paths % 2 != 0:
            raise InvalidInputError(
                "n_paths must be even for variance_reduction 'antithetic': the "
                "estimator's unit is a (Z, -Z) pair"
            )
        if cfg.heston_scheme == EXACT_VARIANCE_EULER_LOG_SPOT:
            raise NotSupportedError(
                "variance_reduction 'antithetic' does not compose with "
                f"heston_scheme='{EXACT_VARIANCE_EULER_LOG_SPOT}': its variance "
                "transition is a noncentral chi-square draw, not a function of "
                "one normal per step, so a reflected pair would reflect only "
                "the spot half of the driver."
            )
    return methods


def _base_meta(
    model: HestonModel, cfg: MCConfig, methods: tuple[str, ...], *, instrument: str
) -> dict[str, Any]:
    return {
        "method": "mc",
        "model": "Heston",
        "instrument": instrument,
        "heston_scheme": cfg.heston_scheme,
        "heston_conditional": bool(cfg.heston_conditional),
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
        "variance_reduction": methods if methods else "none",
        "feller_number": model.feller_number,
        "feller_satisfied": model.feller_satisfied,
    }


def _pair_average(values: np.ndarray, *, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return values
    half = values.shape[0] // 2
    return 0.5 * (values[:half] + values[half:])


def _terminal_sample(
    sample: HestonPathSample,
    *,
    payoff_values: np.ndarray,
    discount: float,
    market: Market,
    expiry: float,
    methods: tuple[str, ...],
    conditional: bool,
    n_paths: int,
) -> TerminalSample:
    """Wrap a realised payoff and the terminal-spot control as a Slice 7 sample.

    The control is the discounted terminal spot when the estimator is the
    realised one and its **conditional expectation** when the estimator is the
    conditional one, because a control has to live on the same sigma-field as
    the estimate it corrects.
    """
    antithetic = ANTITHETIC in methods
    y = discount * np.asarray(payoff_values, dtype=float)
    spot_like = (
        conditional_forward(sample.conditional)
        if conditional
        else sample.spots[:, -1]
    )
    x = discount * np.asarray(spot_like, dtype=float)
    mu = market.rate(expiry) - market.dividend_yield(expiry)
    return TerminalSample(
        y=_pair_average(y, antithetic=antithetic),
        x=_pair_average(x, antithetic=antithetic),
        x_mean=discount * market.spot * math.exp(mu * expiry),
        stratum=None,
        n_strata=0,
        n_normal_draws=sample.n_normal_draws,
        n_paths=n_paths,
        control_name=_TERMINAL_SPOT_CONTROL,
    )


# --------------------------------------------------------------------------
# European and digital: terminal payoffs.
# --------------------------------------------------------------------------


def _european_payoff_values(
    option: EuropeanOption, sample: HestonPathSample, *, conditional: bool
) -> np.ndarray:
    if conditional:
        return conditional_vanilla_values(
            sample.conditional, strike=option.strike, kind=option.kind
        )
    terminal = sample.spots[:, -1]
    return (
        call_payoff(terminal, option.strike)
        if option.kind == "call"
        else put_payoff(terminal, option.strike)
    )


def price_european(
    option: EuropeanOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a European vanilla under Heston by simulation.

    `cfg.n_steps` is the time discretisation and must be at least 2; see the
    module docstring and `_N_STEPS_REFUSAL`. `cfg.heston_scheme` selects among
    the three discretisations and `cfg.heston_conditional` selects the
    estimator.

    Returns
    -------
    PriceResult
        Value, standard error and `meta` carrying the scheme, the grid, the
        estimator, the Feller number, and -- under `control_variate` -- the
        Slice 7 regression bookkeeping.

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps < 2`, an odd `n_paths` under antithetic, or an
        unknown `heston_scheme`.
    NotSupportedError
        `variance_reduction` including `"stratified"`, or antithetic with the
        exact-variance scheme.
    """
    methods = _validate(cfg, model=model)
    expiry = option.expiry
    meta = _base_meta(model, cfg, methods, instrument="european")
    discount = market.df_r(expiry)

    sample = heston_path_sample(
        model, market, expiry=expiry, cfg=cfg, methods=methods
    )
    meta.update(
        {
            "martingale_correction": sample.meta["martingale_correction"],
            "martingale_correction_fallbacks": sample.meta[
                "martingale_correction_fallbacks"
            ],
            "n_normal_draws": sample.n_normal_draws,
        }
    )
    if "negative_variance_fraction" in sample.meta:
        meta["negative_variance_fraction"] = sample.meta["negative_variance_fraction"]

    terminal = _terminal_sample(
        sample,
        payoff_values=_european_payoff_values(
            option, sample, conditional=cfg.heston_conditional
        ),
        discount=discount,
        market=market,
        expiry=expiry,
        methods=methods,
        conditional=cfg.heston_conditional,
        n_paths=cfg.n_paths,
    )
    estimate = estimate_from_sample(terminal, methods=methods)
    meta.update(estimate.meta)
    if CONTROL_VARIATE in methods:
        meta["control_variate_caveat"] = (
            "the control's mean is the MODEL's forward, not the SCHEME's, so "
            "it also removes the part of the discretisation bias collinear "
            "with the scheme's martingale defect"
        )
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def price_digital(
    option: DigitalOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a cash-or-nothing digital under Heston by simulation.

    Nothing about the discontinuity troubles the estimator -- a path either
    finishes in the money or it does not, exactly as under Black-Scholes
    (Slice 6). `cfg.heston_conditional` replaces the indicator by its
    conditional probability `Phi(d2)`, which is the payoff conditioning helps
    most: it turns a Bernoulli variable into a smooth one.
    """
    methods = _validate(cfg, model=model)
    expiry = option.expiry
    meta = _base_meta(model, cfg, methods, instrument="digital")
    discount = market.df_r(expiry)

    sample = heston_path_sample(
        model, market, expiry=expiry, cfg=cfg, methods=methods
    )
    meta["n_normal_draws"] = sample.n_normal_draws
    if cfg.heston_conditional:
        values = conditional_digital_values(
            sample.conditional,
            strike=option.strike,
            cash=option.cash,
            kind=option.kind,
        )
    else:
        values = np.asarray(
            digital_payoff(
                sample.spots[:, -1], option.strike, option.cash, option.kind
            ),
            dtype=float,
        )

    terminal = _terminal_sample(
        sample,
        payoff_values=values,
        discount=discount,
        market=market,
        expiry=expiry,
        methods=methods,
        conditional=cfg.heston_conditional,
        n_paths=cfg.n_paths,
    )
    estimate = estimate_from_sample(terminal, methods=methods)
    meta.update(estimate.meta)
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


# --------------------------------------------------------------------------
# Asian and barrier: path payoffs.
# --------------------------------------------------------------------------


def _column_indices(grid: np.ndarray, dates: tuple[float, ...]) -> np.ndarray:
    """Where each contract date sits in the simulation grid.

    `heston_time_grid` puts every date on the grid, but the *representative* it
    keeps for a pair that differs in the last bit may be the uniform grid's
    point rather than the contract's -- so the lookup is nearest-neighbour with
    a relative tolerance, and a miss larger than that would mean the grid
    construction is broken, which is asserted rather than papered over.
    """
    wanted = np.asarray(dates, dtype=float)
    index = np.abs(grid[None, :] - wanted[:, None]).argmin(axis=1)
    tolerance = 1e-12 * max(float(grid[-1]), 1.0)
    if np.any(np.abs(grid[index] - wanted) > tolerance):
        raise InvalidInputError(
            "contract dates are not on the simulation grid"
        )  # pragma: no cover - guarded by heston_time_grid
    return index


def price_asian(
    option: AsianOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a fixed-strike Asian under Heston by simulation.

    **There is no geometric control variate here.** Under Black-Scholes the
    geometric average of a lognormal path is itself lognormal and its option
    has a closed form, which is what makes Kemna-Vorst the textbook control
    (Slice 8, measured correlation 0.9996 and a factor of 1277). Under Heston
    the geometric average is `exp(mean of ln S_{t_i})` and the joint law of
    those log-spots is not Gaussian -- it is Gaussian only *conditional on the
    variance path* -- so there is no closed form to use as the control's mean.
    The fallback is the same one the Black-Scholes **geometric** Asian uses: the
    discounted terminal spot, whose mean is exact because it is a martingale.
    `meta["control_variate_available"]` says which one is in play and
    `meta["geometric_control_unavailable"]` says why the good one is not.

    `cfg.n_steps` is the time discretisation; the simulation grid is its union
    with the fixing schedule, so every fixing is read exactly.
    """
    methods = _validate(cfg, model=model)
    expiry = option.expiry
    meta = _base_meta(model, cfg, methods, instrument="asian")
    meta.update(
        {
            "averaging": option.averaging,
            "n_fixings": option.n_fixings,
            "control_variate_available": _TERMINAL_SPOT_CONTROL,
            "geometric_control_unavailable": (
                "the geometric average of a Heston path is lognormal only "
                "conditional on the variance path, so its option has no closed "
                "form and cannot supply a control variate's exact mean"
            ),
        }
    )
    if cfg.heston_conditional:
        raise NotSupportedError(
            "MCConfig(heston_conditional=True) prices a TERMINAL payoff by its "
            "conditional expectation given the variance driver. An Asian reads "
            "the path at every fixing, and conditioning on the variance driver "
            "leaves the fixings jointly Gaussian but their average lognormal -- "
            "there is no closed form for the conditional expectation of an "
            "arithmetic average's payoff, which is the original Asian problem "
            "one dimension in. Use heston_conditional=False."
        )

    sample = heston_path_sample(
        model,
        market,
        expiry=expiry,
        cfg=cfg,
        methods=methods,
        dates=option.fixing_times,
        store_paths=True,
    )
    meta["n_normal_draws"] = sample.n_normal_draws
    meta["n_grid"] = int(sample.times.size - 1)

    columns = _column_indices(sample.times, option.fixing_times)
    fixings = sample.spots[:, columns]
    average = (
        geometric_average(fixings)
        if option.averaging == "geometric"
        else arithmetic_average(fixings)
    )
    payoff = np.asarray(
        asian_payoff(average, option.strike, option.kind), dtype=float
    )
    terminal = _terminal_sample(
        sample,
        payoff_values=payoff,
        discount=market.df_r(expiry),
        market=market,
        expiry=expiry,
        methods=methods,
        conditional=False,
        n_paths=cfg.n_paths,
    )
    estimate = estimate_from_sample(terminal, methods=methods)
    meta.update(estimate.meta)
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


_BGK_REFUSAL = (
    "barrier_correction 'bgk' is not available under HestonModel. The "
    "Broadie-Glasserman-Kou shift is exp(-+ beta sigma sqrt(dt)) and names a "
    "single constant sigma, which a stochastic-volatility model does not have; "
    "substituting sqrt(v0), sqrt(theta) or a realised average would each give "
    "a different barrier and none of them is the correction. Use "
    "barrier_correction='none' for the discrete contract or 'brownian_bridge' "
    "for the continuous one -- and read the bridge's own caveat, which this "
    "engine reports in meta['brownian_bridge_caveat']."
)

_BRIDGE_CAVEAT = (
    "the Brownian-bridge crossing probability conditions on the two endpoints "
    "of a bridge with CONSTANT volatility. Under Heston the volatility varies "
    "inside the step, so feeding the trapezoidal integrated variance "
    "dt (v_i + v_{i+1}) / 2 is a one-term approximation and the estimator is "
    "NOT unbiased for the continuous contract, as it is under Black-Scholes. "
    "The residual is measured against a fine-grid reference in "
    "docs/notes/heston_monte_carlo_qe.md and pinned in "
    "tests/test_mc_heston_path_dependent.py"
)


def price_barrier(
    option: BarrierOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a discretely monitored barrier under Heston by simulation.

    `cfg.barrier_correction` selects `"none"` (observe the barrier at the
    contract's dates; unbiased for the **discrete** contract *given the
    scheme*) or `"brownian_bridge"` (weight each path by its conditional
    survival probability). `"bgk"` is refused: its shift names a constant
    volatility and this model has none.

    The bridge uses the step's own integrated variance,
    `dt (v_i + v_{i+1}) / 2`, in place of `sigma^2 dt`. That is an
    approximation and not an identity -- see `_BRIDGE_CAVEAT`, which travels on
    every result that uses it.
    """
    methods = _validate(cfg, model=model)
    correction = normalise_barrier_correction(cfg.barrier_correction)
    if option.is_continuous:
        raise NotSupportedError(
            "method='mc' prices a barrier observed on a schedule: give the "
            "instrument a `monitoring` tuple. Under Heston there is no "
            "continuously monitored closed form to fall back on either, so "
            "barrier_correction='brownian_bridge' on a fine grid is the route "
            "to the continuous contract."
        )
    if correction == BGK:
        raise NotSupportedError(_BGK_REFUSAL)
    if cfg.heston_conditional:
        raise NotSupportedError(
            "MCConfig(heston_conditional=True) prices a TERMINAL payoff. A "
            "barrier payoff depends on the whole path, and conditioning on the "
            "variance driver does not make the running extremum of a Gaussian "
            "path analytically available at every monitoring date. Use "
            "heston_conditional=False; barrier_correction='brownian_bridge' is "
            "the conditioning that does apply here."
        )
    if (
        correction == BROWNIAN_BRIDGE
        and option.is_knock_out
        and option.rebate > 0.0
    ):
        raise NotSupportedError(
            "barrier_correction 'brownian_bridge' cannot price a knock-out "
            "REBATE: the rebate is paid at the touch time and the bridge "
            "supplies the crossing probability, not the crossing time. Same "
            "refusal as the Black-Scholes engine."
        )

    expiry = option.expiry
    meta = _base_meta(model, cfg, methods, instrument="barrier")
    meta.update(
        {
            "barrier_type": option.barrier_type,
            "barrier": option.barrier,
            "rebate": option.rebate,
            "monitoring": "discrete",
            "n_monitoring": option.n_monitoring,
            "barrier_correction": correction,
            "estimates": "discrete" if correction == NO_CORRECTION else "continuous",
            "touched_at_inception": option.is_touched(market.spot),
        }
    )
    if option.is_touched(market.spot):
        raise NotSupportedError(
            "the barrier is already touched at inception: the contract settles "
            "immediately for a rebate (knock-out) or becomes a vanilla "
            "(knock-in), and the vanilla leg under Heston is a transform price, "
            "not a simulation. Use method='fourier' for that leg."
        )

    dates = option.monitoring_times
    if float(dates[-1]) < expiry:
        # Expiry always ends the grid: the terminal payoff is read there
        # whatever the last observation was.
        dates = (*dates, float(expiry))
    sample = heston_path_sample(
        model,
        market,
        expiry=expiry,
        cfg=cfg,
        methods=methods,
        dates=dates,
        store_paths=True,
    )
    meta["n_normal_draws"] = sample.n_normal_draws
    meta["n_grid"] = int(sample.times.size - 1)

    terminal_spots = sample.spots[:, -1]
    vanilla = np.asarray(
        call_payoff(terminal_spots, option.strike)
        if option.kind == "call"
        else put_payoff(terminal_spots, option.strike),
        dtype=float,
    )
    barrier = float(option.barrier)
    discount = market.df_r(expiry)

    if correction == BROWNIAN_BRIDGE:
        # Every interval of the SIMULATION grid, not just the monitoring dates:
        # the contract being estimated is the continuous one.
        log_ratio = np.log(sample.spots / barrier)
        dt = np.diff(sample.times)
        step_variance = 0.5 * (
            sample.variances[:, :-1] + sample.variances[:, 1:]
        ) * dt[None, :]
        step_variance = np.maximum(step_variance, np.finfo(float).tiny)
        survival = bridge_survival_from_step_variance(
            log_ratio, step_variance=step_variance
        )
        weight = survival if option.is_knock_out else 1.0 - survival
        payoff = vanilla * weight
        if option.rebate > 0.0:
            payoff = payoff + option.rebate * survival
        meta["mean_survival_probability"] = float(np.mean(survival))
        meta["knock_fraction"] = float(np.mean(1.0 - survival))
        meta["brownian_bridge_caveat"] = _BRIDGE_CAVEAT
    else:
        columns = _column_indices(sample.times, option.monitoring_times)
        observed = sample.spots[:, columns]
        touched_step = (
            observed <= barrier if option.is_down else observed >= barrier
        )
        touched = np.any(touched_step, axis=1)
        if option.is_knock_out:
            payoff = vanilla * (~touched)
            if option.rebate > 0.0:
                first = np.argmax(touched_step, axis=1)
                touch_time = np.where(
                    touched, np.asarray(option.monitoring_times)[first], 0.0
                )
                rate = market.rate(expiry)
                payoff = payoff + np.where(
                    touched,
                    option.rebate * np.exp(-rate * touch_time) / discount,
                    0.0,
                )
        else:
            payoff = vanilla * touched
            if option.rebate > 0.0:
                payoff = payoff + np.where(touched, 0.0, option.rebate)
        meta["knock_fraction"] = float(np.mean(touched))

    terminal = _terminal_sample(
        sample,
        payoff_values=payoff,
        discount=discount,
        market=market,
        expiry=expiry,
        methods=methods,
        conditional=False,
        n_paths=cfg.n_paths,
    )
    estimate = estimate_from_sample(terminal, methods=methods)
    meta.update(estimate.meta)
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


# --------------------------------------------------------------------------
# Greeks.
# --------------------------------------------------------------------------


def _european_scenario(
    option: EuropeanOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
    methods: tuple[str, ...],
    expiry: float,
    discount: float,
) -> Scenario:
    """One repriced state and the per-unit sample behind it, at one seed."""
    sample = heston_path_sample(
        model, market, expiry=expiry, cfg=cfg, methods=methods
    )
    y = discount * np.asarray(
        _european_payoff_values(
            option, sample, conditional=cfg.heston_conditional
        ),
        dtype=float,
    )
    if CONTROL_VARIATE not in methods:
        units = reduce_to_units(y, methods=methods)
        return Scenario(value=float(np.mean(units)), units=units)
    control = _terminal_sample(
        sample,
        payoff_values=_european_payoff_values(
            option, sample, conditional=cfg.heston_conditional
        ),
        discount=discount,
        market=market,
        expiry=expiry,
        methods=methods,
        conditional=cfg.heston_conditional,
        n_paths=cfg.n_paths,
    )
    reduced = estimate_from_sample(control, methods=methods)
    beta = control_variate_coefficient(control.y, control.x)
    return Scenario(
        value=reduced.value,
        units=control.y - beta * (control.x - control.x_mean),
    )


def _european_bump_estimates(
    option: EuropeanOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None,
    methods: tuple[str, ...],
    names: tuple[str, ...],
    meta: dict[str, Any],
) -> dict[str, GreekEstimate]:
    """Common-random-numbers bumps for whichever Greeks are still needed.

    Every leg reuses `cfg.seed`, hence the same normals, hence the same
    variance paths: the difference quotient is paired and its standard error is
    the standard error of a mean of differences, not of a difference of means.
    Vega bumps `sqrt(v0)`, matching `HestonModel.with_volatility_bump` and the
    transform engine's convention.
    """
    expiry = option.expiry
    spot = market.spot
    steps = bump_sizes(
        bumps=bumps, s0=spot, sigma=model.spot_volatility, t=expiry
    )
    extra = {"ddof": 2} if CONTROL_VARIATE in methods else {}
    meta.setdefault("bumps", {}).update(steps)
    rate = market.rate(expiry)
    dividend = market.dividend_yield(expiry)

    def _market_at(*, new_spot: float | None = None, new_rate: float | None = None) -> Market:
        return Market(
            spot=spot if new_spot is None else new_spot,
            rate_curve=FlatRateCurve(
                rate if new_rate is None else new_rate, allow_negative=True
            ),
            dividend_curve=FlatDividendCurve(dividend, allow_negative=True),
        )

    def _at(
        *,
        new_spot: float | None = None,
        new_model: HestonModel | None = None,
        new_rate: float | None = None,
        new_expiry: float | None = None,
    ) -> Scenario:
        mkt = _market_at(new_spot=new_spot, new_rate=new_rate)
        t = expiry if new_expiry is None else new_expiry
        return _european_scenario(
            EuropeanOption(kind=option.kind, strike=option.strike, expiry=t),
            model if new_model is None else new_model,
            mkt,
            cfg=cfg,
            methods=methods,
            expiry=t,
            discount=mkt.df_r(t),
        )

    base = _at()
    out: dict[str, GreekEstimate] = {}
    if "delta" in names or "gamma" in names:
        up = _at(new_spot=spot + steps["spot"])
        down = _at(new_spot=spot - steps["spot"])
        if "delta" in names:
            out["delta"] = bump_estimates(
                base=base, up=up, down=down, step=steps["spot"], order=1, **extra
            )
        if "gamma" in names:
            out["gamma"] = bump_estimates(
                base=base, up=up, down=down, step=steps["spot"], order=2, **extra
            )
    if "vega" in names:
        out["vega"] = bump_estimates(
            base=base,
            up=_at(new_model=model.with_volatility_bump(steps["sigma"])),
            down=_at(new_model=model.with_volatility_bump(-steps["sigma"])),
            step=steps["sigma"],
            order=1,
            **extra,
        )
    if "rho" in names:
        out["rho"] = bump_estimates(
            base=base,
            up=_at(new_rate=rate + steps["r"]),
            down=_at(new_rate=rate - steps["r"]),
            step=steps["r"],
            order=1,
            **extra,
        )
    if "theta" in names:
        out["theta"] = one_sided_estimate(
            base=base,
            shifted=_at(new_expiry=expiry - steps["time"]),
            step=steps["time"],
            **extra,
        )
    return out


def greeks_european(
    option: EuropeanOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """European Heston Greeks by common-random-numbers bump, plus a pathwise delta.

    `cfg.greeks_estimator="bump"` (the default) gives all five.
    `"pathwise"` replaces delta by `e^{-rT} f'(S_T) S_T / S_0`, which is exact
    because the variance dynamics do not see the spot, so `S_T` is homogeneous
    of degree one in `S_0`; the other four still come from the bump and
    `meta["estimator"]` says so per Greek. `"likelihood_ratio"` is refused.

    **Vega is `dV / d sqrt(v0)`**, the same unit
    `qpl.engines.fourier.pricers` reports under Heston, so the two are
    comparable; `meta["vega_convention"]` records it.

    Raises
    ------
    InvalidInputError
        The shared entry checks, or an unknown estimator.
    NotSupportedError
        `greeks_estimator="likelihood_ratio"`, or the shared refusals.
    """
    methods = _validate(cfg, model=model)
    estimator = normalise_greeks_estimator(cfg.greeks_estimator)
    if estimator == LIKELIHOOD_RATIO:
        raise NotSupportedError(_LIKELIHOOD_RATIO_REFUSAL)
    if estimator != BUMP and bumps is not None:
        raise InvalidInputError(
            f"bumps are meaningless for greeks_estimator '{estimator}' except "
            "for the Greeks it falls back to; pass greeks_estimator='bump' to "
            "size every bump, or none at all"
        )

    meta = _base_meta(model, cfg, methods, instrument="european")
    meta["greeks_estimator"] = estimator
    meta["vega_convention"] = "d/d sqrt(v0) (spot volatility), not d/d v0"

    estimates: dict[str, GreekEstimate] = {}
    remaining = GREEK_NAMES
    if estimator == PATHWISE:
        expiry = option.expiry
        discount = market.df_r(expiry)
        sample = heston_path_sample(
            model, market, expiry=expiry, cfg=cfg, methods=methods
        )
        terminal = sample.spots[:, -1]
        derivative = (
            np.asarray(terminal > option.strike, dtype=float)
            if option.kind == "call"
            else -np.asarray(terminal < option.strike, dtype=float)
        )
        units = reduce_to_units(
            discount * derivative * terminal / market.spot, methods=methods
        )
        estimates["delta"] = GreekEstimate(
            value=float(np.mean(units)),
            stderr=float(np.std(units, ddof=1) / math.sqrt(units.size)),
            estimator=PATHWISE,
        )
        remaining = ("gamma", "vega", "theta", "rho")

    estimates.update(
        _european_bump_estimates(
            option,
            model,
            market,
            cfg=cfg,
            bumps=bumps,
            methods=methods,
            names=remaining,
            meta=meta,
        )
    )
    return greeks_result(estimates, meta)


def greeks_digital(
    option: DigitalOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises `NotSupportedError`; see `_DIGITAL_GREEKS_REFUSAL`.

    Registered rather than left out so the caller gets a reason instead of
    "Unsupported instrument/model/market combination", which would be false
    when the price engine next to it prices exactly that combination.
    """
    raise NotSupportedError(_DIGITAL_GREEKS_REFUSAL)


def greeks_asian(
    option: AsianOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises `NotSupportedError`; see `_PATH_GREEKS_REFUSAL`."""
    raise NotSupportedError(_PATH_GREEKS_REFUSAL)


def greeks_barrier(
    option: BarrierOption,
    model: HestonModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises `NotSupportedError`; see `_PATH_GREEKS_REFUSAL`."""
    raise NotSupportedError(_PATH_GREEKS_REFUSAL)
