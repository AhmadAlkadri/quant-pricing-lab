"""Cash-or-nothing digitals by Monte Carlo: price only, and why only price.

The price is the easy half. A digital's payoff is an indicator, so the
estimator is a sample proportion: draw `N` terminal spots, count how many
finish in the money, discount. Nothing about the discontinuity troubles it --
the payoff is bounded, its variance is finite and known exactly
(`cash**2 p (1 - p)` for a Bernoulli), and the central limit theorem applies
as it does to a vanilla. The estimator is unbiased for **any** `N`, and the
standard error falls at `N**-1/2` with the constant `cash * e^{-rT} sqrt(p(1-p))`.
Measured in `tests/test_digital_mc.py`.

That is the point worth making about Monte Carlo here: unlike the lattice and
the grid, it does not care that the payoff jumps. A tree has to decide which
side of the strike each node is on; a finite-difference grid has to represent
a step function in a space of grid functions. A sample path either finishes
above the strike or it does not, and the estimator asks nothing more of the
payoff than that it be measurable and square-integrable.

**Greeks: two of the three estimators work, and the third returns zero.**

The pathwise method differentiates the payoff along the path,

    d/dS  cash * 1{S_T > K}  =  cash * delta_Dirac(S_T - K) * dS_T/dS,

which is not a function. What a program actually computes is the *almost
everywhere* derivative, which is **identically zero** -- the indicator is
locally constant at every point except the strike, and the strike is a null
event. So a pathwise digital delta is not merely inaccurate: it is exactly
`0.0` on every sample, at every path count, for ever. `greeks_digital` refuses
`greeks_estimator="pathwise"` and `tests/test_digital_mc.py` computes the zero
rather than taking the refusal on trust.

The finite-difference bump *is* available (Slice 10) and is the default, but it
is the estimator with the worst properties here. With common random numbers the
bumped and unbumped payoffs differ only on the paths that cross the strike, a
fraction `O(h)` of them, each contributing a full `cash`; so the difference
quotient has variance `O(cash**2 / (N h))` and its standard deviation **grows**
like `h**-1/2` as the bump shrinks, while the bias falls only like `h**2`.
There is therefore an optimal `h` -- the bias/variance trade-off is visible, not
avoidable -- and it is measured over seeds in
`tests/test_mc_greeks_variance.py`. The estimator is *inconsistent* in the
`h -> 0` limit at fixed `N`, which is what "this is not how you differentiate a
digital" means precisely.

The estimator that works is the likelihood-ratio (score function) one, which
differentiates the lognormal density instead of the payoff and therefore does
not care that the payoff is discontinuous (Glasserman, *Monte Carlo Methods in
Financial Engineering*, section 7.3). All five Greeks come from one sample of
the terminal law, all five are unbiased, and the coverage is measured in
`tests/test_digital_mc.py`. Its price is variance: the score is mean zero with
unbounded support, and multiplying a bounded indicator by it scales the noise
like `1/(S_0 sigma sqrt(T))` -- so the LR digital delta gets *worse* as maturity
shortens, which is measured at `T = 1` against `T = 0.05`.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import DigitalOption
from ...instruments.payoffs import digital_payoff
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .greeks import (
    BUMP,
    CONTROL_VARIATE_GREEKS,
    GREEK_NAMES,
    LIKELIHOOD_RATIO,
    PATHWISE,
    bump_greeks_terminal,
    control_samples,
    draw_path_sample,
    estimate_greek,
    greeks_result,
    likelihood_ratio_terminal_greeks,
    normalise_greeks_estimator,
)
from .pricers import MCConfig
from .processes import price_european_from_terminal, simulate_gbm_exact
from .variance_reduction import (
    CONTROL_VARIATE,
    normalise_variance_reduction,
    price_with_variance_reduction,
    validate_sampler,
)

__all__ = ["digital_payoff_derivative", "greeks_digital", "price_digital"]


def price_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a cash-or-nothing digital by Monte Carlo.

    Parameters
    ----------
    option, model, market
        As for `qpl.engines.mc.pricers.price_european`.
    cfg
        Monte Carlo settings. `n_steps = 1` is terminal sampling and is all a
        European digital needs; larger `n_steps` simulates the path and uses
        its endpoint, which is the same law and more work.

    Returns
    -------
    PriceResult
        Value and standard error, with `meta` reporting paths, steps and seed.
        Deterministic for a fixed seed.

    Raises
    ------
    InvalidInputError
        If `n_paths < 2` (the `ddof=1` standard error needs two samples) or
        `n_steps < 1`.
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
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    df_r = market.df_r(t)

    meta = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "digital",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
        "variance_reduction": methods if methods else "none",
        "n_normal_draws": cfg.n_paths * cfg.n_steps,
    }

    if t == 0.0:
        value = float(digital_payoff(s0, option.strike, option.cash, option.kind))
        return PriceResult(value=value, stderr=0.0, meta=meta)

    if model.sigma == 0.0:
        forward = s0 * math.exp((r - q) * t)
        value = df_r * float(
            digital_payoff(forward, option.strike, option.cash, option.kind)
        )
        return PriceResult(value=value, stderr=0.0, meta=meta)

    def _payoff(s_t: np.ndarray) -> np.ndarray:
        return np.asarray(
            digital_payoff(s_t, option.strike, option.cash, option.kind), dtype=float
        )

    if methods:
        reduced = price_with_variance_reduction(
            payoff=_payoff,
            s0=s0,
            mu=r - q,
            sigma=model.sigma,
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
        sigma=model.sigma,
        t=t,
        n_steps=cfg.n_steps,
        n_paths=cfg.n_paths,
        seed=cfg.seed,
    )
    estimate = price_european_from_terminal(
        paths[:, -1],
        strike=option.strike,
        discount_factor=df_r,
        kind=option.kind,
        payoff=_payoff,
    )
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def digital_payoff_derivative(
    s_t: np.ndarray, *, strike: float, cash: float, kind: str
) -> np.ndarray:
    """The a.e. derivative of a cash-or-nothing payoff: **exactly zero**.

    Written out, exported and tested rather than left implicit, because it is
    the whole reason the pathwise estimator is refused here. `cash * 1{x > K}`
    is locally constant at every `x != K`, so its derivative exists and is `0`
    on a set of *full* measure -- the opposite of a call payoff, whose
    derivative fails to exist on a set of *zero* measure. The pathwise
    interchange of derivative and expectation is legitimate in the second case
    and not in the first, and the failure is silent: it returns a number, and
    the number is zero.

    The arguments are accepted and ignored except for their shape. That is the
    point: no strike, no cash amount and no kind changes the answer.
    """
    del strike, cash, kind
    return np.zeros_like(np.asarray(s_t, dtype=float))


_PATHWISE_REFUSAL = (
    "greeks_estimator='pathwise' is not available for a cash-or-nothing "
    "digital: the payoff's derivative is a Dirac mass, so the almost-everywhere "
    "derivative a program can evaluate is identically zero and the pathwise "
    "delta is biased to exactly 0.0 -- not noisy, not inaccurate, zero, at "
    "every path count. See qpl.engines.mc.digital.digital_payoff_derivative, "
    "which computes it. Use greeks_estimator='likelihood_ratio', which "
    "differentiates the lognormal density instead of the payoff and is "
    "unbiased here (Glasserman 2003, section 7.3), or 'bump', whose standard "
    "deviation grows like h**-1/2 as the bump shrinks."
)

_DEGENERATE_REFUSAL = (
    "digital Greeks need T > 0 and sigma > 0: in either limit the price is a "
    "step function of the spot, so delta is a Dirac mass and gamma its "
    "derivative. The analytic digital engine refuses the same two limits."
)


def greeks_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Digital Greeks by likelihood ratio or by bump; pathwise is refused.

    `cfg.greeks_estimator` selects. `"likelihood_ratio"` is the estimator this
    payoff deserves and the one the Slice 6 refusal message pointed at;
    `"bump"` is available, is the default for consistency with every other
    instrument, and is measured to be the worse choice (module docstring);
    `"pathwise"` raises, with the reason and with a pointer to the function that
    computes the zero.

    Returns
    -------
    GreeksResult
        With `meta["stderr"]` and `meta["estimator"]` per Greek, as for the
        vanilla engine.

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps < 1`, an unknown estimator, `bumps` passed to a
        non-bump estimator, or `T = 0` / `sigma = 0`.
    NotSupportedError
        `greeks_estimator="pathwise"`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    estimator = normalise_greeks_estimator(cfg.greeks_estimator)
    if estimator == PATHWISE:
        raise NotSupportedError(_PATHWISE_REFUSAL)
    methods = normalise_variance_reduction(cfg.variance_reduction)
    validate_sampler(
        methods=methods,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        n_strata=cfg.n_strata,
    )

    t = option.expiry
    sigma = model.sigma
    if t == 0.0 or sigma == 0.0:
        raise InvalidInputError(_DEGENERATE_REFUSAL)

    s0 = market.spot
    r = market.rate(t)
    q = market.dividend_yield(t)
    df_r = market.df_r(t)

    def _payoff(s_t: np.ndarray) -> np.ndarray:
        return np.asarray(
            digital_payoff(s_t, option.strike, option.cash, option.kind), dtype=float
        )

    meta: dict = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "digital",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
        "greeks_estimator": estimator,
        "variance_reduction": methods if methods else "none",
    }

    if estimator == BUMP:
        meta["estimator_caveat"] = (
            "a common-random-numbers bump on a DISCONTINUOUS payoff: the two "
            "legs differ only on the O(h) fraction of paths that cross the "
            "strike, so the estimator's standard deviation grows like h**-1/2 "
            "while the bias falls like h**2, and at the small default steps "
            "for sigma, r and the roll the crossing event is so rare that the "
            "reported standard error describes the discount factor and not the "
            "estimator. Measured coverage of the nominal 95% interval over 40 "
            "seeds at 20k paths, ATM: delta 36/40, gamma 39/40 (with a "
            "standard deviation 4665x the Greek), vega 36/40, rho 26/40, theta "
            "0/40. Use greeks_estimator='likelihood_ratio'."
        )
        return bump_greeks_terminal(
            payoff=_payoff,
            market=market,
            sigma=sigma,
            expiry=t,
            cfg=cfg,
            bumps=bumps,
            methods=methods,
            meta=meta,
        )

    if bumps is not None:
        raise InvalidInputError(
            "bumps are meaningless for greeks_estimator 'likelihood_ratio': it "
            "differentiates the density, not a repriced difference"
        )

    sample = draw_path_sample(
        s0=s0,
        mu=r - q,
        sigma=sigma,
        t=t,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
        methods=methods,
        n_strata=cfg.n_strata,
    )
    values = likelihood_ratio_terminal_greeks(
        sample,
        payoff=_payoff(sample.spots),
        s0=s0,
        mu=r - q,
        sigma=sigma,
        t=t,
        r=r,
        discount_factor=df_r,
    )
    control, control_mean = (None, None)
    if CONTROL_VARIATE in methods:
        control, control_mean = control_samples(
            sample,
            s0=s0,
            mu=r - q,
            sigma=sigma,
            t=t,
            discount_factor=df_r,
            estimator=estimator,
        )
        meta["control_variate_greeks"] = CONTROL_VARIATE_GREEKS
    meta["n_normal_draws"] = sample.n_normal_draws
    estimates = {
        name: estimate_greek(
            values[name],
            sample=sample,
            methods=methods,
            estimator=LIKELIHOOD_RATIO,
            control=control if name in CONTROL_VARIATE_GREEKS else None,
            control_mean=control_mean if name in CONTROL_VARIATE_GREEKS else None,
        )
        for name in GREEK_NAMES
    }
    return greeks_result(estimates, meta)
