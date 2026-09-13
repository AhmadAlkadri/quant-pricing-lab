"""Asian options by Monte Carlo, with the geometric average as the control.

What the estimator is
---------------------
Sample the path at the fixing times and nowhere else. Exact lognormal stepping
(`qpl.engines.mc.processes.gbm_paths_from_normals`) makes every fixing an exact
draw from its own marginal with the right joint law, so there is **no
time-discretisation bias** in this engine: the only error in the reported price
is statistical, and the standard error is the whole story. That is not true of
an Euler path, and it is the reason this slice has no time-step refinement
study -- there is nothing to refine.

The payoff reduces the `(n_paths, n_fixings)` sample to one average per path
and then applies a vanilla payoff to it, so the arithmetic and geometric
contracts differ in exactly one line of code (`qpl.instruments.payoffs`).

The control variate
-------------------
Kemna & Vorst (1990) observed that the geometric average of the same path is
extremely highly correlated with the arithmetic one and has an exact closed
form; using it as a control variate is the standard way to price an arithmetic
Asian by simulation, and it is the textbook example of a control variate
(Glasserman 2003, section 4.1). This engine implements exactly that:

    Y = e^{-rT} max(A - K, 0),      X = e^{-rT} max(G - K, 0),
    E[X] = the discrete geometric closed form (qpl.engines.analytic.asian),

estimated as `mean(Y) - b (mean(X) - E[X])` with `b` the ordinary least-squares
slope, by the Slice 7 estimator layer -- this module builds the sample, it does
not re-implement the regression, the `ddof=2` residual standard error, or the
antithetic pair bookkeeping.

For a *geometric* Asian the geometric payoff is the payoff itself, so the
control would be perfectly correlated with it and the "estimate" would be the
closed form plus zero times noise -- true, useless, and a division by a
vanishing residual variance away from being a bug. The fallback control there is
the discounted spot at the last fixing, `X = e^{-rT} S_{t_n}`, whose mean
`e^{-rT} S_0 e^{mu t_n}` is also exact. It is a much weaker control (measured
rho ~ 0.9 against the geometric control's ~0.999) and it is chosen because it is
*honest*, not because it is good.

What is refused, and why
------------------------
`variance_reduction="stratified"` raises `NotSupportedError`. Stratifying a
single terminal normal stratifies the terminal price; an Asian payoff depends on
`n` normals and there is no single scalar whose strata partition it. The
standard construction stratifies a linear projection of the Brownian path and
fills in the rest with a Brownian bridge -- a different sampler, not different
bookkeeping of this one, and a later slice. Slice 7 already refuses stratified
sampling with `n_steps > 1` for the same reason; this engine states it in terms
of the fixing schedule, because an Asian with one fixing would otherwise slip
through a path-count test and be silently stratified.

Greeks raise `NotSupportedError` (Phase 3's pathwise / likelihood-ratio item).

Reference: Kemna, A.G.Z. and Vorst, A.C.F. (1990), "A pricing method for options
based on average asset values", *Journal of Banking and Finance* 14, 113-129;
Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, sections 4.1
(control variates, with the Asian as the worked example) and 3.2 (generating
paths at a given set of dates). Both were re-derived here; no number below is
quoted from either.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import AsianOption
from ...instruments.payoffs import (
    arithmetic_average,
    asian_payoff,
    geometric_average,
)
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..analytic.asian import discrete_geometric_price
from ..base import GreeksResult, PriceResult
from .pricers import MCConfig
from .processes import gbm_paths_from_normals
from .variance_reduction import (
    ANTITHETIC,
    CONTROL_VARIATE,
    STRATIFIED,
    TerminalSample,
    estimate_from_sample,
    normalise_variance_reduction,
    validate_sampler,
)

__all__ = ["asian_terminal_sample", "greeks_asian", "price_asian"]

_GEOMETRIC_CONTROL = "discounted_geometric_average_payoff"
_SPOT_CONTROL = "discounted_last_fixing_spot"

_STRATIFIED_REFUSAL = (
    "variance_reduction 'stratified' is not available for an Asian option: it "
    "stratifies the single normal that drives a terminal price, and an average "
    "over n fixing dates is driven by n normals with no single scalar to "
    "partition. The standard construction stratifies a linear projection of "
    "the Brownian path (usually its terminal value) and fills the remainder in "
    "with a Brownian bridge; that is a different sampler and a later slice. "
    "Use 'control_variate' (the geometric average with its exact closed form) "
    "or 'antithetic'."
)


def asian_terminal_sample(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    methods: tuple[str, ...],
) -> TerminalSample:
    """One realised sample of the Asian payoff and its control, reduced to units.

    Exposed (rather than inlined into :func:`price_asian`) because the
    measurement tests read `y` and `x` directly to compute the realised
    correlation and the variance factor, and computing them from a second
    sampler would measure a different thing.

    The unit is the path, or the antithetic **pair** when antithetic sampling is
    on; `y` and `x` are reduced identically so that they stay paired unit by
    unit, which is what makes the regression legitimate.
    """
    times = np.asarray(option.fixing_times, dtype=float)
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    mu = r - q
    sigma = model.sigma
    df_r = market.df_r(t)
    s0 = market.spot
    n_fixings = times.size

    rng = np.random.default_rng(cfg.seed)
    if ANTITHETIC in methods:
        n_pairs = cfg.n_paths // 2
        base = rng.normal(size=(n_pairs, n_fixings))
        z = np.concatenate([base, -base], axis=0)
        n_draws = n_pairs * n_fixings
    else:
        z = rng.normal(size=(cfg.n_paths, n_fixings))
        n_draws = cfg.n_paths * n_fixings

    fixings = gbm_paths_from_normals(z, s0=s0, mu=mu, sigma=sigma, times=times)

    geometric = geometric_average(fixings)
    average = geometric if option.averaging == "geometric" else arithmetic_average(fixings)
    y = df_r * np.asarray(
        asian_payoff(average, option.strike, option.kind), dtype=float
    )

    if option.averaging == "arithmetic":
        # Kemna-Vorst: the same path's geometric average, whose discounted
        # payoff has a known mean -- the closed form, which is the price of the
        # geometric Asian and is therefore already discounted.
        control_name = _GEOMETRIC_CONTROL
        x = df_r * np.asarray(
            asian_payoff(geometric, option.strike, option.kind), dtype=float
        )
        x_mean = discrete_geometric_price(
            S=s0,
            K=option.strike,
            T=t,
            r=r,
            sigma=sigma,
            fixing_times=option.fixing_times,
            q=q,
            kind=option.kind,
        )
    else:
        # The geometric control would BE the payoff. Fall back to the discounted
        # spot at the last fixing; its mean is exact and it is a much weaker
        # control (see the module docstring).
        control_name = _SPOT_CONTROL
        x = df_r * fixings[:, -1]
        x_mean = df_r * s0 * math.exp(mu * float(times[-1]))

    if ANTITHETIC in methods:
        n_pairs = cfg.n_paths // 2
        y = 0.5 * (y[:n_pairs] + y[n_pairs:])
        x = 0.5 * (x[:n_pairs] + x[n_pairs:])

    return TerminalSample(
        y=y,
        x=x,
        x_mean=float(x_mean),
        stratum=None,
        n_strata=0,
        n_normal_draws=n_draws,
        n_paths=cfg.n_paths,
        control_name=control_name,
    )


def price_asian(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a fixed-strike Asian option by Monte Carlo.

    Both averagings; with `MCConfig(variance_reduction="control_variate")` the
    arithmetic case uses the geometric-average payoff and its exact closed form
    as the control (Kemna & Vorst 1990).

    `cfg.n_steps` must be `1`, its default. The time grid of an Asian is its
    fixing schedule, which the instrument supplies; there is no second grid to
    choose, and exact lognormal stepping means sub-dividing between fixings
    would buy nothing but work. Passing anything else raises rather than being
    silently ignored -- a config field that is read on one instrument and
    discarded on another is the kind of thing that is discovered by a wrong
    number, not by a message.

    Returns
    -------
    PriceResult
        Value, standard error and `meta` carrying `n_fixings`, the estimator's
        own bookkeeping, and -- under `control_variate` -- `control_beta`,
        `control_correlation`, `control_mean` and
        `control_variance_factor_predicted` (`1/(1-rho^2)`).

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps != 1`, an odd `n_paths` under antithetic, or an
        unusable `variance_reduction` combination.
    NotSupportedError
        `variance_reduction` including `"stratified"`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps != 1:
        raise InvalidInputError(
            "n_steps must be 1 for an Asian option: the simulation grid is the "
            "option's fixing schedule, and exact lognormal stepping means "
            "sub-dividing between fixings changes nothing but the cost"
        )
    methods = normalise_variance_reduction(cfg.variance_reduction)
    if STRATIFIED in methods:
        raise NotSupportedError(_STRATIFIED_REFUSAL)
    # Reuses the Slice 7 sampler checks (the even-`n_paths` rule for
    # antithetic); `n_steps=1` here is the *config* value, not the fixing count.
    validate_sampler(
        methods=methods, n_paths=cfg.n_paths, n_steps=1, n_strata=cfg.n_strata
    )

    t = option.expiry
    meta: dict[str, object] = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "asian",
        "averaging": option.averaging,
        "n_fixings": option.n_fixings,
        "n_paths": cfg.n_paths,
        "n_steps": option.n_fixings,
        "seed": cfg.seed,
        "variance_reduction": methods if methods else "none",
        "n_normal_draws": cfg.n_paths * option.n_fixings,
        "control_variate_available": (
            _GEOMETRIC_CONTROL if option.averaging == "arithmetic" else _SPOT_CONTROL
        ),
    }

    if model.sigma == 0.0:
        # The path is deterministic, so every "sample" is the same number and a
        # sample standard deviation would be an exactly-zero estimate of an
        # exactly-zero quantity. Return the deterministic value instead of
        # burning n_paths draws on it, as the vanilla engine does.
        r = market.rate(t)
        q = market.dividend_yield(t)
        forwards = market.spot * np.exp((r - q) * np.asarray(option.fixing_times))
        average = (
            float(geometric_average(forwards))
            if option.averaging == "geometric"
            else float(arithmetic_average(forwards))
        )
        value = market.df_r(t) * float(
            asian_payoff(average, option.strike, option.kind)
        )
        meta["degenerate"] = "sigma=0"
        return PriceResult(value=value, stderr=0.0, meta=meta)

    sample = asian_terminal_sample(
        option, model, market, cfg=cfg, methods=methods
    )
    estimate = estimate_from_sample(sample, methods=methods)
    meta.update(estimate.meta)
    if CONTROL_VARIATE in methods:
        meta["control_variate"] = sample.control_name
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def greeks_asian(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises: Asian Monte Carlo Greeks are Phase 3's own item.

    Raises
    ------
    NotSupportedError
        Always. The pathwise estimator does exist here (unlike for a digital --
        the Asian payoff is Lipschitz in the average and the average is smooth
        in the path), so this is a scope refusal rather than an impossibility,
        and it is registered as a raising callable so the message can say so
        instead of the registry reporting a generic unsupported combination.
    """
    raise NotSupportedError(
        "Monte Carlo Greeks are not available for Asian options in this slice. "
        "Unlike the digital case the pathwise estimator is well defined here "
        "(the payoff is Lipschitz in the average and the average is smooth in "
        "the path), so this is a scope boundary and not an impossibility: it "
        "lands with the Phase 3 pathwise/likelihood-ratio Greeks item "
        "(docs/CURRICULUM.md). Until then, bump the spot and reprice with the "
        "same seed -- common random numbers make that difference quotient "
        "usable for an Asian, which is exactly what they cannot do for a "
        "digital."
    )
