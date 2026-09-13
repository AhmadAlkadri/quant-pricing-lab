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

Greeks
------
`cfg.greeks_estimator` selects, and on an Asian the three families do not cover
the same ground, so a result **mixes** them and `meta["estimator"]` says which
Greek came from where.

*Pathwise* is the natural estimator here and it is available for delta and
vega. Both averages are homogeneous of degree one in the spot -- every fixing
is proportional to `S_0`, so `dA/dS_0 = A / S_0` for the arithmetic mean and for
the geometric mean alike -- which makes the pathwise delta

    delta_pw = e^{-rT} f'(A) A / S_0.

For vega, recover the Brownian path from the fixings,
`W_{t_i} = (log(S_{t_i}/S_0) - (mu - sigma^2/2) t_i) / sigma`, and differentiate
each fixing, `dS_{t_i}/dsigma = S_{t_i} (W_{t_i} - sigma t_i)`. The average
inherits it linearly or multiplicatively depending on the averaging:

    arithmetic:  dA/dsigma = (1/n) sum_i S_{t_i} (W_{t_i} - sigma t_i),
    geometric:   dG/dsigma = G * (1/n) sum_i (W_{t_i} - sigma t_i),

the second because `log G` is the *mean of the logs*, so only the exponent's
derivative survives. Both payoffs are Lipschitz in the average and the average
is smooth in the path, so the pathwise interchange is legitimate (Glasserman
2003, section 7.2.2) -- which is exactly what a digital does not have.

*Likelihood ratio* is available for delta, and it is a one-line score because
`S_0` enters the joint density of the fixings through the **first transition
only**: `log S_{t_1} = log S_0 + (mu - sigma^2/2) t_1 + sigma sqrt(t_1) Z_1`, so

    delta_lr = e^{-rT} f(A) Z_1 / ( S_0 sigma sqrt(t_1) ).

That is an unbiased estimator of the same delta and it is measurably worse than
the pathwise one, because it throws away everything the other `n - 1`
increments know about the payoff and divides by `sqrt(t_1)` -- which is small
precisely when the schedule is dense. The full likelihood-ratio scores for
`sigma`, `r` and the roll are sums over all `n` increments and are deliberately
not implemented in this slice; those Greeks come from a bump.

*Bump* covers everything and is the default. Its theta is the **roll**: the
settlement date and every fixing shift back together, which is the only reading
of time decay an average admits and is the same convention
`qpl.engines.analytic.asian` uses, so the two are comparable.

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
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..analytic.asian import discrete_geometric_price
from ..base import GreeksResult, PriceResult
from .greeks import (
    BUMP,
    GREEK_NAMES,
    LIKELIHOOD_RATIO,
    PATHWISE,
    GreekEstimate,
    PathSample,
    Scenario,
    bump_estimates,
    bump_sizes,
    estimate_greek,
    greeks_result,
    normalise_greeks_estimator,
    one_sided_estimate,
    reduce_to_units,
)
from .pricers import MCConfig
from .processes import gbm_paths_from_normals
from .variance_reduction import (
    ANTITHETIC,
    CONTROL_VARIATE,
    STRATIFIED,
    TerminalSample,
    control_variate_coefficient,
    estimate_from_sample,
    normalise_variance_reduction,
    validate_sampler,
)

__all__ = [
    "asian_fixings_sample",
    "asian_terminal_sample",
    "greeks_asian",
    "price_asian",
]

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


def asian_fixings_sample(
    *,
    s0: float,
    mu: float,
    sigma: float,
    times: np.ndarray,
    n_paths: int,
    seed: int,
    methods: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    """`(fixings, n_normal_draws)` under the requested sampler.

    Split out of :func:`asian_terminal_sample` so the Greek estimators can read
    the *path* rather than the reduced payoff, and so a bumped scenario draws
    exactly the normals the base one did (common random numbers). The generator
    consumption is unchanged, so a Greek and a price at the same seed run on the
    same paths.
    """
    n_fixings = times.size
    rng = np.random.default_rng(seed)
    if ANTITHETIC in methods:
        n_pairs = n_paths // 2
        base = rng.normal(size=(n_pairs, n_fixings))
        z = np.concatenate([base, -base], axis=0)
        n_draws = n_pairs * n_fixings
    else:
        z = rng.normal(size=(n_paths, n_fixings))
        n_draws = n_paths * n_fixings
    return (
        gbm_paths_from_normals(z, s0=s0, mu=mu, sigma=sigma, times=times),
        n_draws,
    )


def _average(fixings: np.ndarray, averaging: str) -> np.ndarray:
    return (
        geometric_average(fixings)
        if averaging == "geometric"
        else arithmetic_average(fixings)
    )


def _asian_scenario(
    option: AsianOption,
    *,
    s0: float,
    r: float,
    q: float,
    sigma: float,
    times: np.ndarray,
    expiry: float,
    discount: float,
    n_paths: int,
    seed: int,
    methods: tuple[str, ...],
) -> Scenario:
    """One repriced Asian state: its value and the per-unit sample behind it.

    Reproduces `price_asian`'s arithmetic for the base state, so a difference of
    two scenario values is the same common-random-numbers bump a caller would
    build by repricing -- and keeps the sample, so the paired difference has a
    standard error.
    """
    fixings, _ = asian_fixings_sample(
        s0=s0, mu=r - q, sigma=sigma, times=times, n_paths=n_paths, seed=seed,
        methods=methods,
    )
    geometric = geometric_average(fixings)
    average = geometric if option.averaging == "geometric" else arithmetic_average(fixings)
    y = discount * np.asarray(
        asian_payoff(average, option.strike, option.kind), dtype=float
    )
    if CONTROL_VARIATE not in methods:
        units = reduce_to_units(y, methods=methods)
        return Scenario(value=float(np.mean(units)), units=units)

    if option.averaging == "arithmetic":
        x = discount * np.asarray(
            asian_payoff(geometric, option.strike, option.kind), dtype=float
        )
        x_mean = discrete_geometric_price(
            S=s0, K=option.strike, T=expiry, r=r, sigma=sigma,
            fixing_times=tuple(float(t) for t in times), q=q, kind=option.kind,
        )
    else:
        x = discount * fixings[:, -1]
        x_mean = discount * s0 * math.exp((r - q) * float(times[-1]))
    template = TerminalSample(
        y=reduce_to_units(y, methods=methods),
        x=reduce_to_units(x, methods=methods),
        x_mean=float(x_mean),
        stratum=None,
        n_strata=0,
        n_normal_draws=n_paths,
        n_paths=n_paths,
    )
    reduced = estimate_from_sample(template, methods=methods)
    beta = control_variate_coefficient(template.y, template.x)
    return Scenario(
        value=reduced.value, units=template.y - beta * (template.x - template.x_mean)
    )


def _asian_bump_estimates(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None,
    methods: tuple[str, ...],
    names: tuple[str, ...],
    meta: dict,
) -> dict[str, GreekEstimate]:
    """Common-random-numbers bumps for whichever Greeks the caller still needs.

    Only the scenarios the requested `names` use are simulated, so a pathwise
    result pays for three extra repricings (gamma, rho, theta) rather than
    eight.
    """
    s0 = market.spot
    t = option.expiry
    sigma = model.sigma
    times = np.asarray(option.fixing_times, dtype=float)
    steps = bump_sizes(bumps=bumps, s0=s0, sigma=sigma, t=t)
    # The roll cannot step past the first fixing: at `t_1 - dt <= 0` the
    # schedule is no longer a schedule. Halving the first fixing is the same
    # guard `bump_sizes` applies to the maturity.
    steps["time"] = min(steps["time"], 0.5 * float(times[0]))
    extra = {"ddof": 2} if CONTROL_VARIATE in methods else {}
    meta.setdefault("bumps", {}).update(steps)

    r, q, df = market.rate(t), market.dividend_yield(t), market.df_r(t)

    def _at(*, spot=None, vol=None, rate=None, div=None, grid=None, expiry=None,
            discount=None):
        return _asian_scenario(
            option,
            s0=s0 if spot is None else spot,
            r=r if rate is None else rate,
            q=q if div is None else div,
            sigma=sigma if vol is None else vol,
            times=times if grid is None else grid,
            expiry=t if expiry is None else expiry,
            discount=df if discount is None else discount,
            n_paths=cfg.n_paths,
            seed=cfg.seed,
            methods=methods,
        )

    base = _at()
    out: dict[str, GreekEstimate] = {}
    if "delta" in names or "gamma" in names:
        up, down = _at(spot=s0 + steps["spot"]), _at(spot=s0 - steps["spot"])
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
            up=_at(vol=sigma + steps["sigma"]),
            down=_at(vol=sigma - steps["sigma"]),
            step=steps["sigma"],
            order=1,
            **extra,
        )
    if "rho" in names:

        def _rate_shifted(rate: float) -> Scenario:
            mkt = Market(
                spot=s0,
                rate_curve=FlatRateCurve(rate, allow_negative=True),
                dividend_curve=FlatDividendCurve(q, allow_negative=True),
            )
            return _at(
                rate=mkt.rate(t), div=mkt.dividend_yield(t), discount=mkt.df_r(t)
            )

        out["rho"] = bump_estimates(
            base=base,
            up=_rate_shifted(r + steps["r"]),
            down=_rate_shifted(r - steps["r"]),
            step=steps["r"],
            order=1,
            **extra,
        )
    if "theta" in names:
        d_t = steps["time"]
        rolled = times - d_t
        t_rolled = t - d_t
        out["theta"] = one_sided_estimate(
            base=base,
            shifted=_at(
                grid=rolled,
                expiry=t_rolled,
                rate=market.rate(t_rolled),
                div=market.dividend_yield(t_rolled),
                discount=market.df_r(t_rolled),
            ),
            step=d_t,
            **extra,
        )
    return out


def greeks_asian(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Asian Greeks by bump, pathwise or likelihood ratio; the result mixes them.

    See the module docstring for the derivations and for which Greek each
    estimator covers: `"pathwise"` supplies delta and vega, `"likelihood_ratio"`
    supplies delta, and everything else falls back to a common-random-numbers
    bump -- recorded per Greek in `meta["estimator"]`, never silently.

    Theta is the **roll** derivative in every case: settlement and all fixings
    shift back together. `qpl.engines.analytic.asian` uses the same convention,
    so the geometric case can be checked against a closed form.

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps != 1`, an unknown estimator, `bumps` with a
        non-bump estimator, or `sigma = 0` for a sample-level estimator.
    NotSupportedError
        `variance_reduction` including `"stratified"`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps != 1:
        raise InvalidInputError(
            "n_steps must be 1 for an Asian option: the simulation grid is the "
            "option's fixing schedule"
        )
    estimator = normalise_greeks_estimator(cfg.greeks_estimator)
    methods = normalise_variance_reduction(cfg.variance_reduction)
    if STRATIFIED in methods:
        raise NotSupportedError(_STRATIFIED_REFUSAL)
    validate_sampler(methods=methods, n_paths=cfg.n_paths, n_steps=1, n_strata=cfg.n_strata)
    if estimator != BUMP and bumps is not None:
        raise InvalidInputError(
            f"bumps are meaningless for greeks_estimator '{estimator}' except "
            "for the Greeks it falls back to; pass greeks_estimator='bump' to "
            "size every bump, or none at all"
        )

    t = option.expiry
    sigma = model.sigma
    s0 = market.spot
    r, q, df = market.rate(t), market.dividend_yield(t), market.df_r(t)
    times = np.asarray(option.fixing_times, dtype=float)

    meta: dict = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "asian",
        "averaging": option.averaging,
        "n_fixings": option.n_fixings,
        "n_paths": cfg.n_paths,
        "seed": cfg.seed,
        "greeks_estimator": estimator,
        "variance_reduction": methods if methods else "none",
        "theta_convention": "roll: settlement and every fixing shift together",
    }

    if estimator == BUMP:
        estimates = _asian_bump_estimates(
            option, model, market, cfg=cfg, bumps=bumps, methods=methods,
            names=GREEK_NAMES, meta=meta,
        )
        if CONTROL_VARIATE in methods:
            meta["control_variate_greeks"] = GREEK_NAMES
        return greeks_result(estimates, meta)

    if sigma == 0.0:
        raise InvalidInputError(
            f"greeks_estimator '{estimator}' needs sigma > 0: the path is "
            "deterministic at sigma = 0, so the density has no score and the "
            "payoff's pathwise derivative is not integrable against it"
        )

    fixings, n_draws = asian_fixings_sample(
        s0=s0, mu=r - q, sigma=sigma, times=times, n_paths=cfg.n_paths,
        seed=cfg.seed, methods=methods,
    )
    meta["n_normal_draws"] = n_draws
    average = _average(fixings, option.averaging)
    payoff = np.asarray(asian_payoff(average, option.strike, option.kind), dtype=float)
    if option.kind == "call":
        payoff_derivative = np.asarray(average > option.strike, dtype=float)
    else:
        payoff_derivative = -np.asarray(average < option.strike, dtype=float)

    sample = PathSample(
        spots=average,
        z=np.zeros_like(average),
        stratum=None,
        n_strata=0,
        n_normal_draws=n_draws,
        n_paths=cfg.n_paths,
    )
    # Brownian path recovered from the fixings; see the module docstring.
    brownian = (
        np.log(fixings / s0) - (r - q - 0.5 * sigma * sigma) * times[None, :]
    ) / sigma

    estimates: dict[str, GreekEstimate] = {}
    if estimator == PATHWISE:
        d_sigma = brownian - sigma * times[None, :]
        if option.averaging == "geometric":
            average_dsigma = average * np.mean(d_sigma, axis=1)
        else:
            average_dsigma = np.mean(fixings * d_sigma, axis=1)
        estimates["delta"] = estimate_greek(
            df * payoff_derivative * average / s0,
            sample=sample, methods=methods, estimator=PATHWISE,
        )
        estimates["vega"] = estimate_greek(
            df * payoff_derivative * average_dsigma,
            sample=sample, methods=methods, estimator=PATHWISE,
        )
        remaining = ("gamma", "rho", "theta")
    else:
        t1 = float(times[0])
        z1 = brownian[:, 0] / math.sqrt(t1)
        estimates["delta"] = estimate_greek(
            df * payoff * z1 / (s0 * sigma * math.sqrt(t1)),
            sample=sample, methods=methods, estimator=LIKELIHOOD_RATIO,
        )
        remaining = ("gamma", "vega", "rho", "theta")

    estimates.update(
        _asian_bump_estimates(
            option, model, market, cfg=cfg, bumps=None, methods=methods,
            names=remaining, meta=meta,
        )
    )
    if CONTROL_VARIATE in methods:
        # The control is applied inside each bumped *price*, as Slice 7 does;
        # the sample-level estimators above are left uncontrolled, because the
        # exact mean of the geometric control's pathwise derivative is a second
        # closed form and is not in this slice.
        meta["control_variate_greeks"] = remaining
    return greeks_result(estimates, meta)
