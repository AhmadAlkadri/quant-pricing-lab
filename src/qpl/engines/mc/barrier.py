"""Discretely monitored barrier options by Monte Carlo, and three estimators.

What the estimator is
---------------------
The instrument carries its monitoring schedule, so this engine samples the path
at those dates and nowhere else, by exact lognormal stepping
(`qpl.engines.mc.processes.gbm_paths_from_normals`). Every monitored spot is an
exact draw from its own marginal with the right joint law, so there is **no
time-discretisation bias**: the plain estimator is an unbiased estimator of the
*discretely monitored contract*, and its standard error is the whole story.

That is worth stating precisely, because the thing this slice measures is a
bias and it would be easy to attribute it to the wrong place. The plain
estimator is unbiased for what the contract says. It is *biased* relative to
the continuous closed form, and by a lot -- `O(1/sqrt(m))` in the number of
monitoring dates, which at `m = 25` is a percent-level error on a knock-out
near its barrier. The bias belongs to the **contract**, not to the estimator.

`expiry` is appended to the simulation grid when the last monitoring date falls
short of it, because the terminal payoff is read at expiry whatever the last
observation was. That extra column is never used as an observation.

Three ways to treat the barrier
-------------------------------
`MCConfig.barrier_correction` selects, and the three are genuinely different
estimators of different things:

- **`"none"`**: observe the barrier at the monitoring dates exactly as the
  contract says. Unbiased for the discrete contract. This is what the `m`-ladder
  in `docs/notes/barrier_options_monitoring_bias.md` measures the bias *of*.

- **`"bgk"`**: shift the barrier **toward the spot** by
  `exp(+- beta sigma sqrt(dt))`, `beta = -zeta(1/2)/sqrt(2 pi) ~ 0.5826`, and
  then monitor discretely as before. Broadie, Glasserman and Kou (1997) show
  that a discretely monitored barrier at `H` behaves like a continuous one at
  `H exp(-+ beta sigma sqrt(dt))` to `o(1/sqrt(m))`; read backwards, simulating
  the shifted barrier discretely estimates the *continuous* price. It needs a
  uniformly spaced schedule, since a single `dt` is what `beta sigma sqrt(dt)`
  means, and a non-uniform one is refused rather than given an average.

- **`"brownian_bridge"`**: do not kill the path at all. Between two consecutive
  sampled points a geometric Brownian bridge crosses a level `H` with the
  conditional probability

      p_i = exp( -2 log(S_{t_i} / H) log(S_{t_{i+1}} / H) / (sigma^2 dt_i) )

  (both endpoints on the live side; `p_i = 1` otherwise), which is the same
  reflection-principle computation the closed forms rest on, applied to one
  interval instead of to `[0, T]`. The survival probability of the path is
  `prod_i (1 - p_i)`, and the estimator is `payoff * survival` for a knock-out
  and `payoff * (1 - survival)` for a knock-in. That is `E[payoff 1{no touch} |
  sampled points]`, so it is an **unbiased estimator of the continuous
  contract** at any number of sampling dates -- conditioning on more of the
  path reduces its variance but cannot change its mean. The first interval runs
  from `t = 0` at the known spot, so the whole of `[0, T]` is covered.

  The one thing the bridge does not supply is the *time* of the crossing, only
  its probability. A knock-out rebate is paid at the touch time, so a non-zero
  rebate is refused on this route rather than silently discounted to expiry. A
  knock-**in** rebate is paid at expiry and is perfectly well defined here, so
  it is allowed.

The control variate
-------------------
The discounted vanilla payoff `e^{-rT} max(S_T - K, 0)` on the same path, whose
mean is the Black-Scholes price. It is the natural control for every barrier
type, because a barrier payoff *is* the vanilla payoff times an indicator, so
the two agree exactly on the surviving paths and the residual is the indicator
alone. Estimation routes through the Slice 7 layer
(`qpl.engines.mc.variance_reduction`): this module builds the sample, it does
not re-implement the regression, the `ddof=2` residual standard error, or the
antithetic pair bookkeeping.

What is refused, and why
------------------------
`variance_reduction="stratified"` raises `NotSupportedError`, for the reason
Slice 8 gives on the Asian: stratifying one terminal normal stratifies the
terminal price, and a path observed at `m` dates is driven by `m` normals with
no single scalar to partition. Brownian-bridge *stratification* -- stratify a
projection of the path, fill the rest in with a bridge -- is a different
sampler and a later slice, and is not to be confused with the Brownian-bridge
*correction* above, which changes the estimator and not the draws.

Greeks are refused outright, with a message naming what would be needed.

References: Broadie, M., Glasserman, P. and Kou, S. (1997), "A continuity
correction for discrete barrier options", *Mathematical Finance* 7(4),
325-348, for the barrier shift; Glasserman (2003), *Monte Carlo Methods in
Financial Engineering*, section 6.4 for the discretisation error of barrier
simulation and sections 6.4 / 3.1 for the Brownian-bridge crossing probability.
Both constructions are re-derived here and every number in the notes was
measured in this repository.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import BarrierOption
from ...instruments.payoffs import call_payoff, put_payoff
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel, bs_price
from ..analytic.barrier import barrier_price, shifted_barrier
from ..base import GreeksResult, PriceResult
from .pricers import MCConfig
from .processes import gbm_paths_from_normals
from .variance_reduction import (
    ANTITHETIC,
    STRATIFIED,
    TerminalSample,
    estimate_from_sample,
    normalise_variance_reduction,
    validate_sampler,
)

__all__ = [
    "BARRIER_CORRECTIONS",
    "BGK",
    "BROWNIAN_BRIDGE",
    "NO_CORRECTION",
    "barrier_terminal_sample",
    "bridge_survival",
    "greeks_barrier",
    "normalise_barrier_correction",
    "price_barrier",
]

NO_CORRECTION = "none"
BGK = "bgk"
BROWNIAN_BRIDGE = "brownian_bridge"

BARRIER_CORRECTIONS: tuple[str, ...] = (NO_CORRECTION, BGK, BROWNIAN_BRIDGE)
"""The three treatments of the barrier, as data, for validation and messages."""

_VANILLA_CONTROL = "discounted_vanilla_payoff"

_UNIFORM_TOLERANCE = 1e-12
"""Relative slack when checking that a monitoring schedule is equally spaced."""

_STRATIFIED_REFUSAL = (
    "variance_reduction 'stratified' is not available for a barrier option: it "
    "stratifies the single normal that drives a terminal price, and a path "
    "observed at m monitoring dates is driven by m normals with no single "
    "scalar to partition. The standard construction stratifies a linear "
    "projection of the Brownian path and fills the remainder in with a "
    "Brownian bridge; that is a different SAMPLER and a later slice, and it is "
    "not the same thing as barrier_correction='brownian_bridge', which changes "
    "the estimator and not the draws. Use 'control_variate' (the vanilla "
    "payoff, whose mean is the Black-Scholes price) or 'antithetic'."
)

_CONTINUOUS_REFUSAL = (
    "method='mc' prices a barrier observed on a schedule: it simulates the "
    "monitoring dates the contract names, and a continuously monitored "
    "contract has none. Give the instrument a `monitoring` tuple and, to "
    "approximate the continuous contract on that grid, set "
    "MCConfig(barrier_correction='brownian_bridge') -- which is unbiased for "
    "the continuous price at any m -- or 'bgk'. For the continuous price "
    "itself use method='analytic'."
)

_GREEKS_REFUSAL = (
    "Monte Carlo Greeks are not available for a barrier option in this slice. "
    "The payoff carries an indicator of the running extremum, so its pathwise "
    "derivative is a surface measure on the barrier rather than a function "
    "(the same obstruction the cash-or-nothing digital has, one dimension up), "
    "and the likelihood-ratio score has to be taken over the joint density of "
    "all m monitored spots rather than over one terminal normal. Use "
    "method='analytic', whose Greeks are central differences of the closed "
    "form."
)


def normalise_barrier_correction(value: object) -> str:
    """Validate `MCConfig.barrier_correction` and return it in canonical form."""
    if not isinstance(value, str):
        raise InvalidInputError(
            "barrier_correction must be one of "
            + ", ".join(repr(name) for name in BARRIER_CORRECTIONS)
        )
    name = value.lower()
    if name not in BARRIER_CORRECTIONS:
        raise InvalidInputError(
            "barrier_correction must be one of "
            + ", ".join(repr(n) for n in BARRIER_CORRECTIONS)
        )
    return name


def _uniform_step(times: np.ndarray) -> float:
    """`dt` of an equally spaced schedule starting one step after zero.

    Raises `InvalidInputError` on any other schedule: `beta sigma sqrt(dt)` has
    no meaning without a single `dt`, and averaging an uneven schedule would
    produce a number that looks like a correction and is not one.
    """
    dt = float(times[0])
    steps = np.diff(times, prepend=0.0)
    if dt <= 0.0 or np.any(np.abs(steps - dt) > _UNIFORM_TOLERANCE * dt):
        raise InvalidInputError(
            "barrier_correction 'bgk' needs an equally spaced monitoring "
            "schedule t_i = i T / m: the correction shifts the barrier by "
            "beta * sigma * sqrt(dt), which names a single dt. Use "
            "qpl.instruments.uniform_monitoring_times, or "
            "barrier_correction='brownian_bridge', which needs no uniformity."
        )
    return dt


def bridge_survival(
    log_ratio: np.ndarray, *, sigma: float, dt: np.ndarray
) -> np.ndarray:
    """Probability that the path never crossed, given the sampled points.

    Parameters
    ----------
    log_ratio
        `log(S_{t_i} / H)` at the grid points, `(n_paths, n_grid + 1)` with
        column 0 the (known, common) initial spot. Its **sign** is what
        identifies a crossing: the live side is positive for a down barrier and
        negative for an up one, so a product of two consecutive entries that is
        non-positive means the interval's endpoints straddle the barrier and
        the crossing probability is 1.
    sigma, dt
        Volatility and the `(n_grid,)` vector of interval lengths.

    Returns
    -------
    numpy.ndarray
        `(n_paths,)` survival probabilities `prod_i (1 - p_i)`, with

            p_i = exp(-2 log(S_i/H) log(S_{i+1}/H) / (sigma^2 dt_i)).

    Derived from the reflection principle for a Brownian bridge: conditional on
    its two endpoints, a Brownian bridge's minimum has an explicit law, and the
    probability of reaching a level below both endpoints is the expression
    above with the log-spot in place of the Brownian motion. It is the same
    computation the closed forms use over `[0, T]`, applied to one interval.
    Because it conditions on the endpoints and nothing else, multiplying the
    per-interval survivals is exact: the increments are independent given the
    sampled points.
    """
    a = log_ratio[:, :-1]
    b = log_ratio[:, 1:]
    product = a * b
    with np.errstate(over="ignore", under="ignore"):
        crossing = np.exp(-2.0 * product / (sigma * sigma * dt[None, :]))
    # Endpoints on opposite sides (or exactly on the barrier): certain crossing.
    crossing = np.where(product <= 0.0, 1.0, crossing)
    return np.prod(1.0 - crossing, axis=1)


def _simulation_grid(option: BarrierOption) -> tuple[np.ndarray, bool]:
    """`(grid, terminal_is_monitored)` -- the monitoring dates plus expiry."""
    times = np.asarray(option.monitoring_times, dtype=float)
    if times[-1] >= option.expiry:
        return times, True
    return np.append(times, float(option.expiry)), False


def _vanilla_payoff(spots: np.ndarray, option: BarrierOption) -> np.ndarray:
    if option.kind == "call":
        return np.asarray(call_payoff(spots, option.strike), dtype=float)
    return np.asarray(put_payoff(spots, option.strike), dtype=float)


def barrier_terminal_sample(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    methods: tuple[str, ...],
    correction: str,
) -> tuple[TerminalSample, dict[str, object]]:
    """One realised sample of the barrier payoff and its control, plus diagnostics.

    Exposed (rather than inlined into :func:`price_barrier`) because the
    measurement tests read `y` and `x` directly to compute the realised
    correlation and the knock fraction, and computing them from a second
    sampler would measure a different thing.

    The unit is the path, or the antithetic **pair** when antithetic sampling
    is on; `y` and `x` are reduced identically so they stay paired unit by
    unit, which is what makes the control-variate regression legitimate.
    """
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    sigma = model.sigma
    df_r = market.df_r(t)
    s0 = market.spot
    grid, terminal_is_monitored = _simulation_grid(option)
    n_grid = grid.size
    n_monitored = option.n_monitoring

    barrier = float(option.barrier)
    if correction == BGK:
        barrier = shifted_barrier(
            H=barrier,
            sigma=sigma,
            dt=_uniform_step(np.asarray(option.monitoring_times, dtype=float)),
            down=option.is_down,
            toward_spot=True,
        )

    rng = np.random.default_rng(cfg.seed)
    if ANTITHETIC in methods:
        n_pairs = cfg.n_paths // 2
        base = rng.normal(size=(n_pairs, n_grid))
        z = np.concatenate([base, -base], axis=0)
        n_draws = n_pairs * n_grid
    else:
        z = rng.normal(size=(cfg.n_paths, n_grid))
        n_draws = cfg.n_paths * n_grid

    paths = gbm_paths_from_normals(z, s0=s0, mu=r - q, sigma=sigma, times=grid)
    terminal = paths[:, -1]
    vanilla = _vanilla_payoff(terminal, option)

    diagnostics: dict[str, object] = {"effective_barrier": barrier}

    if correction == BROWNIAN_BRIDGE:
        # The bridge covers every interval of the grid, including the terminal
        # one when it is not a monitoring date: the contract being estimated
        # here is the CONTINUOUS one, which is observed up to expiry.
        log_ratio = np.empty((paths.shape[0], n_grid + 1), dtype=float)
        log_ratio[:, 0] = math.log(s0 / barrier)
        log_ratio[:, 1:] = np.log(paths / barrier)
        survival = bridge_survival(
            log_ratio, sigma=sigma, dt=np.diff(grid, prepend=0.0)
        )
        weight = survival if option.is_knock_out else 1.0 - survival
        y = df_r * vanilla * weight
        if option.rebate > 0.0:
            # Knock-in only; a knock-out rebate is refused upstream.
            y = y + option.rebate * df_r * survival
        diagnostics["mean_survival_probability"] = float(np.mean(survival))
        diagnostics["knock_fraction"] = float(np.mean(1.0 - survival))
    else:
        observed = paths[:, :n_monitored] if not terminal_is_monitored else paths
        touched_step = (
            observed <= barrier if option.is_down else observed >= barrier
        )
        touched = np.any(touched_step, axis=1)
        if option.is_knock_out:
            y = df_r * vanilla * (~touched)
            if option.rebate > 0.0:
                first = np.argmax(touched_step, axis=1)
                touch_time = np.where(touched, grid[:n_monitored][first], 0.0)
                y = y + np.where(
                    touched, option.rebate * np.exp(-r * touch_time), 0.0
                )
        else:
            y = df_r * vanilla * touched
            if option.rebate > 0.0:
                y = y + np.where(touched, 0.0, option.rebate * df_r)
        diagnostics["knock_fraction"] = float(np.mean(touched))

    x = df_r * vanilla
    x_mean = bs_price(
        S=s0, K=option.strike, T=t, r=r, sigma=sigma, q=q, kind=option.kind
    )

    if ANTITHETIC in methods:
        n_pairs = cfg.n_paths // 2
        y = 0.5 * (y[:n_pairs] + y[n_pairs:])
        x = 0.5 * (x[:n_pairs] + x[n_pairs:])

    sample = TerminalSample(
        y=np.asarray(y, dtype=float),
        x=np.asarray(x, dtype=float),
        x_mean=float(x_mean),
        stratum=None,
        n_strata=0,
        n_normal_draws=n_draws,
        n_paths=cfg.n_paths,
        control_name=_VANILLA_CONTROL,
    )
    return sample, diagnostics


def _degenerate_value(
    option: BarrierOption, market: Market, *, barrier: float
) -> float:
    """Price at `sigma = 0` with the barrier observed on the schedule.

    Deliberately *not* the continuous-monitoring answer: at `sigma = 0` the
    forward path may cross the barrier and return between two observation
    dates only if the drift changes sign, which it cannot, but it may cross
    *after* the last observation, and then the discrete contract survives where
    the continuous one does not. The two agree except in that case, and the
    case is real, so it is computed rather than delegated.
    """
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    s0 = market.spot
    times = np.asarray(option.monitoring_times, dtype=float)
    observed = s0 * np.exp((r - q) * times)
    touched_step = observed <= barrier if option.is_down else observed >= barrier
    touched = bool(np.any(touched_step))

    terminal = s0 * math.exp((r - q) * t)
    intrinsic = (
        max(terminal - option.strike, 0.0)
        if option.kind == "call"
        else max(option.strike - terminal, 0.0)
    )
    df_r = market.df_r(t)
    if option.is_knock_out:
        if not touched:
            return df_r * intrinsic
        touch_time = float(times[int(np.argmax(touched_step))])
        return option.rebate * math.exp(-r * touch_time)
    if touched:
        return df_r * intrinsic
    return option.rebate * df_r


def price_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a discretely monitored barrier option by Monte Carlo.

    `cfg.n_steps` must be `1`, its default: the time grid of a barrier is its
    monitoring schedule, which the instrument supplies, and exact lognormal
    stepping means sub-dividing between observations would change the cost and
    not the answer -- the contract is not observed in between. Passing anything
    else raises rather than being silently ignored.

    `cfg.barrier_correction` selects one of the three estimators described in
    the module docstring. `"none"` estimates the **discrete** contract without
    bias; `"brownian_bridge"` estimates the **continuous** contract without
    bias; `"bgk"` estimates the continuous contract with an `o(1/sqrt(m))`
    bias. `meta["estimates"]` says which of the two contracts the returned
    number is an estimate of, so a caller never has to infer it from the
    configuration.

    Returns
    -------
    PriceResult
        Value, standard error and `meta` carrying `barrier_correction`,
        `effective_barrier`, `knock_fraction`, `n_monitoring`, the monitoring
        grid, and the Slice 7 estimator's own bookkeeping.

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps != 1`, an odd `n_paths` under antithetic, an
        unknown `barrier_correction`, or `"bgk"` on a non-uniform schedule.
    NotSupportedError
        Continuous monitoring, `variance_reduction` including `"stratified"`,
        or a knock-out rebate under `"brownian_bridge"`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps != 1:
        raise InvalidInputError(
            "n_steps must be 1 for a barrier option: the simulation grid is the "
            "option's monitoring schedule, and exact lognormal stepping means "
            "sub-dividing between observations changes nothing but the cost "
            "(the contract is not observed in between). To approximate "
            "continuous monitoring use barrier_correction, not more steps."
        )
    correction = normalise_barrier_correction(cfg.barrier_correction)
    if option.is_continuous:
        raise NotSupportedError(_CONTINUOUS_REFUSAL)
    methods = normalise_variance_reduction(cfg.variance_reduction)
    if STRATIFIED in methods:
        raise NotSupportedError(_STRATIFIED_REFUSAL)
    if (
        correction == BROWNIAN_BRIDGE
        and option.is_knock_out
        and option.rebate > 0.0
    ):
        raise NotSupportedError(
            "barrier_correction 'brownian_bridge' cannot price a knock-out "
            "REBATE: the rebate is paid at the touch time, and the bridge "
            "supplies the probability that the path crossed between two "
            "samples but not the distribution of when it did. A knock-IN "
            "rebate is paid at expiry and is supported. Use "
            "barrier_correction='none' or 'bgk' for a knock-out rebate, and "
            "read the result as the discrete contract's value."
        )
    validate_sampler(
        methods=methods, n_paths=cfg.n_paths, n_steps=1, n_strata=cfg.n_strata
    )

    t = option.expiry
    grid, _ = _simulation_grid(option)
    meta: dict[str, object] = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "barrier",
        "barrier_type": option.barrier_type,
        "barrier": option.barrier,
        "rebate": option.rebate,
        "monitoring": "discrete",
        "n_monitoring": option.n_monitoring,
        "monitoring_dt": float(option.monitoring_times[0]),
        "barrier_correction": correction,
        "estimates": (
            "discrete" if correction == NO_CORRECTION else "continuous"
        ),
        "n_paths": cfg.n_paths,
        "n_steps": int(grid.size),
        "seed": cfg.seed,
        "variance_reduction": methods if methods else "none",
        "n_normal_draws": cfg.n_paths * int(grid.size),
        "touched_at_inception": option.is_touched(market.spot),
    }

    if option.is_touched(market.spot):
        # Settled at inception, exactly as the closed form reports it: a fact
        # about the contract, so every engine must return the same number and
        # none of them may burn a path count on it.
        value = (
            float(option.rebate)
            if option.is_knock_out
            else bs_price(
                S=market.spot, K=option.strike, T=t, r=market.rate(t),
                sigma=model.sigma, q=market.dividend_yield(t), kind=option.kind,
            )
        )
        meta["degenerate"] = "touched_at_inception"
        return PriceResult(value=value, stderr=0.0, meta=meta)

    if model.sigma == 0.0:
        barrier = float(option.barrier)
        if correction == BGK:
            # `shifted_barrier` is the identity at sigma = 0, but going through
            # it keeps the reported `effective_barrier` honest.
            barrier = shifted_barrier(
                H=barrier, sigma=0.0,
                dt=_uniform_step(np.asarray(option.monitoring_times, dtype=float)),
                down=option.is_down, toward_spot=True,
            )
        meta["degenerate"] = "sigma=0"
        meta["effective_barrier"] = barrier
        return PriceResult(
            value=_degenerate_value(option, market, barrier=barrier),
            stderr=0.0,
            meta=meta,
        )

    sample, diagnostics = barrier_terminal_sample(
        option, model, market, cfg=cfg, methods=methods, correction=correction
    )
    estimate = estimate_from_sample(sample, methods=methods)
    meta.update(diagnostics)
    meta.update(estimate.meta)
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def greeks_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises `NotSupportedError`; see :data:`_GREEKS_REFUSAL`.

    Registered rather than left out so that the caller gets a reason. An
    unregistered key would report "Unsupported instrument/model/market
    combination", which is misleading when the price engine sitting next to it
    prices exactly that combination.
    """
    raise NotSupportedError(_GREEKS_REFUSAL)


def continuous_reference(
    option: BarrierOption, model: BlackScholesModel, market: Market
) -> float:
    """The continuously monitored closed form at the same specification.

    A one-line convenience that the bias studies and the example both use, so
    that "the continuous price of this contract" is written once. It is the
    same `BarrierOption` with its monitoring schedule ignored.
    """
    t = option.expiry
    return barrier_price(
        S=market.spot,
        K=option.strike,
        T=t,
        r=market.rate(t),
        sigma=model.sigma,
        q=market.dividend_yield(t),
        H=option.barrier,
        rebate=option.rebate,
        barrier_type=option.barrier_type,
        kind=option.kind,
    )
