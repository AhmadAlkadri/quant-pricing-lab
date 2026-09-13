from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...instruments.payoffs import call_payoff, put_payoff
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from ..registry import MethodSpec
from .greeks import (
    BUMP,
    CONTROL_VARIATE_GREEKS,
    GREEK_NAMES,
    LIKELIHOOD_RATIO,
    MIXED_PATHWISE_LR,
    PATHWISE,
    GreekEstimate,
    bump_estimates,
    control_samples,
    draw_path_sample,
    estimate_greek,
    greeks_result,
    likelihood_ratio_terminal_greeks,
    normalise_greeks_estimator,
    one_sided_estimate,
    pathwise_terminal_greeks,
    scenario,
)
from .processes import price_european_from_terminal, simulate_gbm_exact
from .variance_reduction import (
    CONTROL_VARIATE,
    STRATIFIED,
    normalise_variance_reduction,
    price_with_variance_reduction,
    validate_sampler,
)


@dataclass(frozen=True)
class MCConfig:
    """Monte Carlo configuration.

    Parameters
    ----------
    n_paths
        Number of simulated paths.
    n_steps
        Number of time steps per path. `n_steps=1` corresponds to terminal sampling.
    seed
        Seed for reproducible random number generation.
    variance_reduction
        Estimator selector: `"none"` (the default, and bit-for-bit what this
        engine returned before variance reduction existed), `"antithetic"`,
        `"control_variate"`, `"stratified"`, or a tuple combining a sampler
        with the control variate, e.g. `("antithetic", "control_variate")`.
        Validated at engine entry by
        `qpl.engines.mc.variance_reduction.normalise_variance_reduction`, which
        also states which combinations do not compose and why.
    n_strata
        Number of equal-probability strata, read only when
        `variance_reduction` includes `"stratified"`. `n_paths` must be a
        multiple of it, with at least two paths per stratum.
    greeks_estimator
        Which Greek estimator `greeks_european` and friends use: `"bump"` (the
        default, and bit-for-bit the central-difference common-random-numbers
        path this engine had before Slice 10), `"pathwise"`, or
        `"likelihood_ratio"`. Read by the Greeks entry points only; a price is
        unaffected. Derivations, and which estimator exists for which payoff:
        `qpl.engines.mc.greeks`.
    """
    n_paths: int = 50_000
    n_steps: int = 1
    seed: int = 123
    variance_reduction: str | tuple[str, ...] = "none"
    n_strata: int = 64
    greeks_estimator: Literal["bump", "pathwise", "likelihood_ratio"] = "bump"


def _validate_bumps(value: object) -> None:
    """Reject a `bumps` argument that is not a mapping of bump sizes."""
    if not isinstance(value, dict):
        raise InvalidInputError("bumps must be a dict of bump sizes")


MC_METHOD_SPEC = MethodSpec(
    method="mc",
    cfg_type=MCConfig,
    greeks_optional_kwargs={"bumps": _validate_bumps},
)
"""Keyword contract for `method="mc"`: a required `MCConfig`, plus an optional
`bumps` mapping accepted by `greeks` only."""


def price_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a European option with Monte Carlo simulation.

    Parameters
    ----------
    option
        European option (`call` or `put`).
    model
        Black-Scholes model.
    market
        Market object.
    cfg
        Monte Carlo settings.

    Returns
    -------
    PriceResult
        Monte Carlo price estimate, stderr, and metadata.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    methods = normalise_variance_reduction(cfg.variance_reduction)
    validate_sampler(
        methods=methods,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        n_strata=cfg.n_strata,
    )

    s0 = market.spot
    k = option.strike
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    sigma = model.sigma
    df_r = market.df_r(t)

    meta = {
        "method": "mc",
        "model": "BlackScholes",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
        "variance_reduction": methods if methods else "none",
        "n_normal_draws": cfg.n_paths * cfg.n_steps,
    }

    if t == 0.0:
        if option.kind == "call":
            value = max(s0 - k, 0.0)
        else:
            value = max(k - s0, 0.0)
        return PriceResult(value=float(value), stderr=0.0, meta=meta)

    if sigma == 0.0:
        forward = s0 * math.exp((r - q) * t)
        if option.kind == "call":
            value = df_r * max(forward - k, 0.0)
        else:
            value = df_r * max(k - forward, 0.0)
        return PriceResult(value=float(value), stderr=0.0, meta=meta)

    if methods:
        # Every reduced estimator is built on the terminal sample, so it shares
        # one implementation with the digital engine; `variance_reduction` owns
        # both the sampler and the standard error that belongs to it.
        reduced = price_with_variance_reduction(
            payoff=lambda s_t: (
                call_payoff(s_t, k) if option.kind == "call" else put_payoff(s_t, k)
            ),
            s0=s0,
            mu=r - q,
            sigma=sigma,
            t=t,
            discount_factor=df_r,
            n_paths=cfg.n_paths,
            n_steps=cfg.n_steps,
            seed=cfg.seed,
            methods=methods,
            n_strata=cfg.n_strata,
        )
        meta.update(reduced.meta)
        return PriceResult(value=reduced.value, stderr=reduced.stderr, meta=meta)

    paths = simulate_gbm_exact(
        s0=s0,
        mu=r - q,
        sigma=sigma,
        t=t,
        n_steps=cfg.n_steps,
        n_paths=cfg.n_paths,
        seed=cfg.seed,
    )
    estimate = price_european_from_terminal(
        paths[:, -1],
        strike=k,
        discount_factor=df_r,
        kind=option.kind,
    )

    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def _vanilla_payoff(option: EuropeanOption):
    """`f(S_T)` for the contract, as an array callable."""
    k = option.strike
    if option.kind == "call":
        return lambda s_t: call_payoff(s_t, k)
    return lambda s_t: put_payoff(s_t, k)


def _vanilla_payoff_derivative(option: EuropeanOption, spots):
    """`f'(S_T)`: the indicator of finishing in the money, signed by the kind.

    Defined everywhere except at `S_T = K`, which has probability zero under a
    continuous law -- that null set is exactly what the pathwise method needs
    and what a digital does not have (its derivative is zero on a set of *full*
    measure, which is a different and fatal statement).
    """
    import numpy as np

    if option.kind == "call":
        return np.asarray(spots > option.strike, dtype=float)
    return -np.asarray(spots < option.strike, dtype=float)


def _resolved_market(market: Market, t: float) -> tuple[float, float, float]:
    """`(r, q, df_r)` as the pricing engines read them off the curves."""
    return market.rate(t), market.dividend_yield(t), market.df_r(t)


def _bump_greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None,
    meta: dict,
) -> GreeksResult:
    """Central-difference common-random-numbers Greeks, with paired stderrs.

    The bump sizes, the ordering of the checks and the arithmetic that produces
    each *value* are the pre-slice ones: every scenario is priced as
    `price_european` prices it (same normals in the same order, same reduction,
    same `mean`), and each Greek is the same difference of scenario values. What
    is new is that the per-path sample behind each price is kept, so the paired
    difference can report a standard error. A difference of two independently
    reported standard errors would be wrong by orders of magnitude here, since
    common random numbers make the two legs almost perfectly correlated.

    Theta is a **backward** difference `(V(T - dt) - V(T)) / dt`, not a central
    one: that is what this engine did before this slice and the pinned values
    are reproduced rather than improved. It is first order in `dt` where the
    other four are second order, and `meta["fd"]["time"]` says so.
    """
    s0 = market.spot
    t = option.expiry
    sigma = model.sigma
    methods = meta["variance_reduction_methods"]
    payoff = _vanilla_payoff(option)

    def _bump(name: str, default: float) -> float:
        if bumps is None or name not in bumps:
            return default
        value = float(bumps[name])
        if value <= 0.0:
            raise InvalidInputError(f"{name} bump must be > 0")
        return value

    dS = _bump("spot", max(abs(s0) * 1e-4, 1e-6))
    if s0 <= dS:
        dS = 0.5 * s0
    dsigma = _bump("sigma", 1e-4)
    if sigma > 0.0 and sigma <= dsigma:
        dsigma = 0.5 * sigma
    dr = _bump("r", 1e-5)
    dt = _bump("time", min(1e-4, t / 2.0))
    if t <= dt:
        dt = t * 0.5

    meta["fd"] = "central"
    meta["fd_by_greek"] = {
        "delta": "central",
        "gamma": "central",
        "vega": "central",
        "rho": "central",
        "theta": "backward",
    }
    meta["bumps"] = {"spot": dS, "sigma": dsigma, "r": dr, "time": dt}

    r, q, df_r = _resolved_market(market, t)
    # The stratified sampler lays equal-size strata out in contiguous blocks
    # (`qpl.engines.mc.variance_reduction._stratified_normals`), so the labels
    # are reconstructible without carrying them back out of every scenario --
    # and a paired difference of two stratified samples is still stratified.
    stratum_kwargs: dict = {}
    if STRATIFIED in methods:
        import numpy as np

        stratum_kwargs = {
            "stratum": np.repeat(
                np.arange(cfg.n_strata), cfg.n_paths // cfg.n_strata
            ),
            "n_strata": cfg.n_strata,
        }
    if CONTROL_VARIATE in methods:
        # One degree of freedom for the mean, one for the fitted slope.
        stratum_kwargs["ddof"] = 2

    def _at(
        *, spot: float, vol: float, mu: float, expiry: float, discount: float
    ):
        return scenario(
            payoff=payoff,
            s0=spot,
            mu=mu,
            sigma=vol,
            t=expiry,
            discount_factor=discount,
            n_paths=cfg.n_paths,
            n_steps=cfg.n_steps,
            seed=cfg.seed,
            methods=methods,
            n_strata=cfg.n_strata,
        )

    base = _at(spot=s0, vol=sigma, mu=r - q, expiry=t, discount=df_r)
    up = _at(spot=s0 + dS, vol=sigma, mu=r - q, expiry=t, discount=df_r)
    down = _at(spot=s0 - dS, vol=sigma, mu=r - q, expiry=t, discount=df_r)

    estimates = {
        "delta": bump_estimates(
            base=base, up=up, down=down, step=dS, order=1, **stratum_kwargs
        ),
        "gamma": bump_estimates(
            base=base, up=up, down=down, step=dS, order=2, **stratum_kwargs
        ),
    }

    if sigma == 0.0:
        estimates["vega"] = GreekEstimate(value=0.0, stderr=0.0, estimator=BUMP)
    else:
        estimates["vega"] = bump_estimates(
            base=base,
            up=_at(spot=s0, vol=sigma + dsigma, mu=r - q, expiry=t, discount=df_r),
            down=_at(spot=s0, vol=sigma - dsigma, mu=r - q, expiry=t, discount=df_r),
            step=dsigma,
            order=1,
            **stratum_kwargs,
        )

    def _rate_shifted(rate: float):
        mkt = Market(
            spot=s0,
            rate_curve=FlatRateCurve(rate, allow_negative=True),
            dividend_curve=FlatDividendCurve(q, allow_negative=True),
        )
        r_s, q_s, df_s = _resolved_market(mkt, t)
        return _at(spot=s0, vol=sigma, mu=r_s - q_s, expiry=t, discount=df_s)

    estimates["rho"] = bump_estimates(
        base=base,
        up=_rate_shifted(r + dr),
        down=_rate_shifted(r - dr),
        step=dr,
        order=1,
        **stratum_kwargs,
    )

    r_b, q_b, df_b = _resolved_market(market, t - dt)
    estimates["theta"] = one_sided_estimate(
        base=base,
        shifted=_at(spot=s0, vol=sigma, mu=r_b - q_b, expiry=t - dt, discount=df_b),
        step=dt,
        **stratum_kwargs,
    )

    return greeks_result(estimates, meta)


def _terminal_law_greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    estimator: str,
    meta: dict,
) -> GreeksResult:
    """Pathwise or likelihood-ratio Greeks from one sample of the terminal law.

    One set of paths produces all five Greeks, which is the structural
    difference from the bump: no revaluation, no bump size, no bias. See
    `qpl.engines.mc.greeks` for the derivations; this function is the wiring.
    """
    t = option.expiry
    sigma = model.sigma
    s0 = market.spot
    r, q, df_r = _resolved_market(market, t)
    mu = r - q
    methods = meta["variance_reduction_methods"]

    sample = draw_path_sample(
        s0=s0,
        mu=mu,
        sigma=sigma,
        t=t,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
        methods=methods,
        n_strata=cfg.n_strata,
    )
    payoff = _vanilla_payoff(option)(sample.spots)

    if estimator == PATHWISE:
        values = pathwise_terminal_greeks(
            sample,
            payoff=payoff,
            payoff_derivative=_vanilla_payoff_derivative(option, sample.spots),
            s0=s0,
            mu=mu,
            sigma=sigma,
            t=t,
            r=r,
            discount_factor=df_r,
        )
        names = dict.fromkeys(GREEK_NAMES, PATHWISE)
        names["gamma"] = MIXED_PATHWISE_LR
    else:
        values = likelihood_ratio_terminal_greeks(
            sample,
            payoff=payoff,
            s0=s0,
            mu=mu,
            sigma=sigma,
            t=t,
            r=r,
            discount_factor=df_r,
        )
        names = dict.fromkeys(GREEK_NAMES, LIKELIHOOD_RATIO)

    control, control_mean = (None, None)
    if CONTROL_VARIATE in methods:
        control, control_mean = control_samples(
            sample,
            s0=s0,
            mu=mu,
            sigma=sigma,
            t=t,
            discount_factor=df_r,
            estimator=estimator,
        )
        meta["control_variate_greeks"] = CONTROL_VARIATE_GREEKS

    estimates = {
        name: estimate_greek(
            values[name],
            sample=sample,
            methods=methods,
            estimator=names[name],
            control=control if name in CONTROL_VARIATE_GREEKS else None,
            control_mean=control_mean if name in CONTROL_VARIATE_GREEKS else None,
        )
        for name in GREEK_NAMES
    }
    meta["n_normal_draws"] = sample.n_normal_draws
    return greeks_result(estimates, meta)


def greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """European Greeks by bump, pathwise or likelihood ratio.

    Which estimator runs is `cfg.greeks_estimator`; the default `"bump"` is the
    pre-slice central-difference common-random-numbers path and reproduces its
    numbers exactly. All three report a **per-Greek** standard error in
    `meta["stderr"]` and name the estimator that produced each Greek in
    `meta["estimator"]` -- they are not all the same even within one call, since
    a pathwise result carries a mixed LR-PW gamma (the payoff has no second
    derivative).

    Parameters
    ----------
    option, model, market
        As for :func:`price_european`.
    cfg
        Monte Carlo settings, including `greeks_estimator` and any variance
        reduction. Antithetic and stratified sampling reduce the Greek sample
        exactly as they reduce the price sample; the control variate is applied
        to delta only under the sample-level estimators (see
        `qpl.engines.mc.greeks`) and at the price level under `"bump"`.
    bumps
        Finite-difference steps keyed by `spot`, `sigma`, `r` and `time`.
        Meaningful for `greeks_estimator="bump"` only.

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps < 1`, a non-positive bump, an unknown
        `greeks_estimator`, `bumps` passed to an estimator that has no bump to
        size, or a pathwise/likelihood-ratio request at `T = 0` or `sigma = 0`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    estimator = normalise_greeks_estimator(cfg.greeks_estimator)
    methods = normalise_variance_reduction(cfg.variance_reduction)
    validate_sampler(
        methods=methods,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        n_strata=cfg.n_strata,
    )

    t = option.expiry
    sigma = model.sigma

    meta: dict = {
        "method": "mc",
        "model": "BlackScholes",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
        "greeks_estimator": estimator,
        # Every bumped revaluation uses the same `cfg`, hence the same seed,
        # hence the same normals (or the same stratified uniforms): common
        # random numbers survive variance reduction unchanged, and the reduced
        # estimator's lower noise is inherited by the difference quotient.
        # Measured in `tests/test_mc_variance_ratios.py`.
        "variance_reduction": methods if methods else "none",
        "variance_reduction_methods": methods,
    }

    if estimator == BUMP:
        if t == 0.0:
            meta["t0"] = True
            meta.pop("variance_reduction_methods")
            return GreeksResult(
                delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, meta=meta
            )
        result = _bump_greeks_european(
            option, model, market, cfg=cfg, bumps=bumps, meta=meta
        )
    else:
        if bumps is not None:
            raise InvalidInputError(
                f"bumps are meaningless for greeks_estimator '{estimator}': it "
                "differentiates the payoff or the density, not a repriced "
                "difference. Use greeks_estimator='bump' to size a bump."
            )
        if t == 0.0 or sigma == 0.0:
            raise InvalidInputError(
                f"greeks_estimator '{estimator}' needs T > 0 and sigma > 0: at "
                "either limit the terminal law is a point mass, its density has "
                "no score and the pathwise derivative of the payoff is not "
                "integrable against it. The analytic engine refuses the same "
                "two limits."
            )
        result = _terminal_law_greeks_european(
            option, model, market, cfg=cfg, estimator=estimator, meta=meta
        )

    assert result.meta is not None
    result.meta.pop("variance_reduction_methods", None)
    return result
