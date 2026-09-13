"""`method="fourier"`: European vanillas priced through a characteristic function.

This module owns the keyword contract (`FourierConfig`, `FOURIER_METHOD_SPEC`)
and the dispatcher entry points. The mathematics lives one module down, in
`cos`, and every one of those modules talks to the model only through
`qpl.engines.fourier.charfn.CharacteristicFunctionModel`.

What the method is for
----------------------
Every other engine in this package discretises the *dynamics*: a lattice, a
grid, a path. A transform method discretises the *law* instead, and it can do
that because the terminal law is the only thing a European payoff sees. Under
Black-Scholes that buys nothing an exact formula does not already give -- which
is the point of doing it here first. The Black-Scholes price is known to
machine precision, so the error of each transform method against it is the
method's *own* error with nothing else mixed in, and the convergence studies in
`tests/test_fourier_*.py` measure exactly that before the next slice points the
same machinery at a model with no closed form.

Degenerate inputs
-----------------
``T = 0`` and ``sigma = 0`` are **refused**, unlike the analytic, tree, grid and
simulation engines, which return the intrinsic or discounted-forward-intrinsic
value. In both limits the terminal law is a point mass: its density is a Dirac
delta, so there is no cosine series to truncate and no decay to integrate
against. Returning the limit anyway would mean the engine silently stopped
being a transform method at exactly the inputs where a caller might be checking
that it still is one. The refusal names the limit and points at `method="analytic"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ..base import GreeksResult, PriceResult
from ..registry import MethodSpec
from .charfn import CharacteristicFunctionModel, characteristic_function_model
from .cos import cos_price

__all__ = [
    "FOURIER_METHODS",
    "FOURIER_METHOD_SPEC",
    "FourierConfig",
    "FourierInputs",
    "fourier_inputs",
    "greeks_european",
    "price_european",
]

FOURIER_METHODS: tuple[str, ...] = ("cos",)
"""Transform methods this package implements, as data for validation."""

GREEKS_METHODS: tuple[str, ...] = ("cos",)
"""The subset whose Greeks are closed forms rather than bumps."""

VEGA_BUMP = 1e-5
"""Absolute volatility bump for the transform engines' vega.

Vega, theta and rho are *not* closed forms here. Delta and gamma are, because
the spot enters the COS sum only through the payoff coefficients (see
`qpl.engines.fourier.cos`); the other three enter through the characteristic
function itself, and differentiating that is a model-specific calculation this
package deliberately does not ask a `CharacteristicFunctionModel` for -- the
whole point of the interface is that Heston can satisfy it with two methods.

So those three are central differences of the transform price, and the bump is
chosen from a measured scan rather than picked. The transform price is exact to
about 1e-13, so the difference quotient has an ``O(h^2)`` truncation bias and a
round-off floor of about ``1e-13 / h``; the crossover is where the bump should
sit. Measured worst absolute residual against the closed-form Greeks over
{call, put} x {vanilla, digital} at `S = K = 100, r = 5%, q = 1%, sigma = 20%,
T = 1`:

    h       vega       theta      rho
    1e-02   2.04e-03   5.05e-05   8.59e-03
    1e-03   2.03e-05   5.05e-07   8.60e-05
    1e-04   2.03e-07   5.06e-09   8.60e-07
    1e-05   1.81e-09   2.75e-09   8.65e-09     <- chosen
    1e-06   4.13e-08   4.41e-08   6.25e-09
    1e-07   3.89e-07   2.65e-08   4.18e-08

The first four rows fall by exactly a factor of 100 per decade, which is the
``h^2`` bias; below 1e-05 the floor takes over and the error stops improving.
Note how much better behaved this is than the same bump on a lattice price
(`qpl.engines.tree.pricers.VEGA_BUMP` is a thousand times larger): there the
binding term is the tree's own oscillating price error, which does not cancel
between the two evaluations. Here the two evaluations are each exact.
"""

RATE_BUMP = 1e-5
"""Absolute rate bump for rho. Same scan, same reasoning, as `VEGA_BUMP`."""

TIME_BUMP = 1e-5
"""Absolute maturity bump for theta, in years.

Theta is ``-dV/dT`` at fixed spot, by a *central* difference at ``T +/- h``, so
the engine refuses ``T <= h`` rather than silently switching to a one-sided
quotient at short maturities.
"""


@dataclass(frozen=True)
class FourierConfig:
    """Configuration for `method="fourier"`.

    Parameters
    ----------
    method
        Which transform method to use. ``"cos"`` is the Fourier-cosine
        expansion of Fang and Oosterlee.
    n_terms
        Number of cosine terms ``N`` for ``method="cos"``. The error decays
        exponentially in ``N`` until it reaches the floating-point floor.
    truncation_l
        ``L`` in the COS truncation range ``[c1 - L w, c1 + L w]``. Fang and
        Oosterlee recommend 10 for Black-Scholes-like models; the cost of
        smaller values is measured in `tests/test_fourier_cos.py`.
    """

    method: Literal["cos"] = "cos"
    n_terms: int = 256
    truncation_l: float = 10.0

    def __post_init__(self) -> None:
        if self.method not in FOURIER_METHODS:
            raise InvalidInputError(f"method must be one of {FOURIER_METHODS}")
        if not isinstance(self.n_terms, int) or self.n_terms < 2:
            raise InvalidInputError("n_terms must be an integer >= 2")
        if not math.isfinite(self.truncation_l) or self.truncation_l <= 0.0:
            raise InvalidInputError("truncation_l must be finite and > 0")


FOURIER_METHOD_SPEC = MethodSpec(method="fourier", cfg_type=FourierConfig)
"""Keyword contract for `method="fourier"`: one `FourierConfig`, no extras."""


@dataclass(frozen=True)
class FourierInputs:
    """The scalars every transform engine here reads, gathered once."""

    cf: CharacteristicFunctionModel
    s0: float
    expiry: float
    rate: float
    dividend: float


def fourier_inputs(model: Any, market: Market, expiry: float) -> FourierInputs:
    """Resolve the model's transform and read the market at ``expiry``.

    Raises
    ------
    InvalidInputError
        If ``expiry <= 0``: the terminal law is a point mass (module docstring).
    NotSupportedError
        If the model provides no characteristic function.
    """
    if expiry <= 0.0:
        raise InvalidInputError(
            "Fourier methods require T > 0; at T = 0 the terminal law is a point "
            "mass with no density to transform. Use method='analytic'."
        )
    return FourierInputs(
        cf=characteristic_function_model(model),
        s0=market.spot,
        expiry=expiry,
        rate=market.rate(expiry),
        dividend=market.dividend_yield(expiry),
    )


def _cos_value(
    inputs: FourierInputs,
    cfg: FourierConfig,
    *,
    strike: float,
    kind: str,
    payoff: str,
    cash: float,
    s0: float | None = None,
    expiry: float | None = None,
    rate: float | None = None,
    cf: CharacteristicFunctionModel | None = None,
):
    return cos_price(
        cf if cf is not None else inputs.cf,
        s0=s0 if s0 is not None else inputs.s0,
        strike=strike,
        expiry=expiry if expiry is not None else inputs.expiry,
        rate=rate if rate is not None else inputs.rate,
        dividend=inputs.dividend,
        kind=kind,
        payoff=payoff,
        cash=cash,
        n_terms=cfg.n_terms,
        truncation_l=cfg.truncation_l,
    )


def transform_value(
    inputs: FourierInputs,
    cfg: FourierConfig,
    *,
    strike: float,
    kind: str,
    payoff: str = "vanilla",
    cash: float = 1.0,
    cf: CharacteristicFunctionModel | None = None,
    s0: float | None = None,
    expiry: float | None = None,
    rate: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Value one payoff by ``cfg.method``; return the value and its metadata."""
    if cfg.method == "cos":
        result = _cos_value(
            inputs,
            cfg,
            strike=strike,
            kind=kind,
            payoff=payoff,
            cash=cash,
            cf=cf,
            s0=s0,
            expiry=expiry,
            rate=rate,
        )
        meta = {
            "fourier_method": "cos",
            "n_terms": result.n_terms,
            "truncation_l": cfg.truncation_l,
            "truncation_range": (result.lower, result.upper),
        }
        return result.value, meta
    raise NotSupportedError(f"method '{cfg.method}' is not implemented")


def _meta(cfg: FourierConfig, extra: dict[str, Any], instrument: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "method": "fourier",
        "model": "BlackScholes",
        "instrument": instrument,
    }
    meta.update(extra)
    return meta


def price_european(
    option: EuropeanOption,
    model: Any,
    market: Market,
    *,
    cfg: FourierConfig,
) -> PriceResult:
    """Price a European vanilla through the model's characteristic function."""
    inputs = fourier_inputs(model, market, option.expiry)
    value, extra = transform_value(
        inputs, cfg, strike=option.strike, kind=option.kind
    )
    return PriceResult(value=value, meta=_meta(cfg, extra, "european"))


def _bumped_greeks(
    inputs: FourierInputs,
    cfg: FourierConfig,
    model: Any,
    market: Market,
    *,
    strike: float,
    kind: str,
    payoff: str,
    cash: float,
) -> tuple[float, float, float]:
    """vega, theta, rho by central differences of the transform price."""
    from dataclasses import replace

    def value_at(
        *,
        cf: CharacteristicFunctionModel | None = None,
        expiry: float | None = None,
        rate: float | None = None,
    ) -> float:
        return transform_value(
            inputs,
            cfg,
            strike=strike,
            kind=kind,
            payoff=payoff,
            cash=cash,
            cf=cf,
            expiry=expiry,
            rate=rate,
        )[0]

    if not hasattr(model, "sigma"):
        raise NotSupportedError(
            "vega needs a volatility parameter to bump; this model has none"
        )
    up = characteristic_function_model(replace(model, sigma=model.sigma + VEGA_BUMP))
    down = characteristic_function_model(replace(model, sigma=model.sigma - VEGA_BUMP))
    vega = (value_at(cf=up) - value_at(cf=down)) / (2.0 * VEGA_BUMP)

    if inputs.expiry <= TIME_BUMP:
        raise InvalidInputError(
            f"theta needs T > {TIME_BUMP}; the central difference would cross T = 0"
        )
    later = value_at(expiry=inputs.expiry + TIME_BUMP)
    earlier = value_at(expiry=inputs.expiry - TIME_BUMP)
    theta = -(later - earlier) / (2.0 * TIME_BUMP)

    rho = (
        value_at(rate=inputs.rate + RATE_BUMP) - value_at(rate=inputs.rate - RATE_BUMP)
    ) / (2.0 * RATE_BUMP)
    return vega, theta, rho


def _cos_greeks(
    inputs: FourierInputs,
    cfg: FourierConfig,
    model: Any,
    market: Market,
    *,
    strike: float,
    kind: str,
    payoff: str,
    cash: float,
    instrument: str,
) -> GreeksResult:
    if cfg.method not in GREEKS_METHODS:
        raise NotSupportedError(
            f"Greeks are not available for fourier method '{cfg.method}'; "
            f"use FourierConfig(method='cos'), whose delta and gamma are closed "
            "forms of the cosine expansion"
        )
    result = _cos_value(
        inputs, cfg, strike=strike, kind=kind, payoff=payoff, cash=cash
    )
    vega, theta, rho = _bumped_greeks(
        inputs, cfg, model, market, strike=strike, kind=kind, payoff=payoff, cash=cash
    )
    extra = {
        "fourier_method": "cos",
        "n_terms": result.n_terms,
        "truncation_l": cfg.truncation_l,
        "truncation_range": (result.lower, result.upper),
        "greeks": {"delta": "closed_form", "gamma": "closed_form", "other": "central_difference"},
    }
    return GreeksResult(
        delta=result.delta,
        gamma=result.gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        meta=_meta(cfg, extra, instrument),
    )


def greeks_european(
    option: EuropeanOption,
    model: Any,
    market: Market,
    *,
    cfg: FourierConfig,
) -> GreeksResult:
    """European vanilla Greeks: delta and gamma closed form, the rest bumped."""
    inputs = fourier_inputs(model, market, option.expiry)
    return _cos_greeks(
        inputs,
        cfg,
        model,
        market,
        strike=option.strike,
        kind=option.kind,
        payoff="vanilla",
        cash=1.0,
        instrument="european",
    )
