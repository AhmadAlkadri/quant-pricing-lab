"""Cash-or-nothing digitals through a characteristic function.

A digital is the transform methods' easiest instrument and the grid's hardest,
which is the comparison worth having. The finite-difference engine loses a full
order of convergence to the payoff's jump (Slice 6) because it represents the
*payoff* on a mesh. A transform method never represents the payoff on a mesh:
it expands the **density**, which is smooth, and pairs it with a payoff
coefficient that is an exact integral of an indicator against a cosine. So the
discontinuity costs nothing at all here, and the digital converges at least as
fast as the vanilla -- measured in `tests/test_fourier_cos.py`.

Which methods price one
-----------------------
`"cos"` (its own payoff coefficients) and `"gil_pelaez"` (the digital call *is*
the discounted exercise probability, so Gil-Pelaez's inversion gives it with no
extra derivation). `"carr_madan"` and `"lewis"` are refused: both transform the
*call price* rather than the law, so a digital would have to be recovered as
minus the strike-derivative of that transform, which is a different derivation
and not a configuration flag.
"""

from __future__ import annotations

from typing import Any

from ...exceptions import NotSupportedError
from ...instruments.options import DigitalOption
from ...market.market import Market
from ..base import GreeksResult, PriceResult
from .pricers import (
    FourierConfig,
    _cos_greeks,
    _meta,
    fourier_inputs,
    transform_value,
)

__all__ = ["DIGITAL_METHODS", "greeks_digital", "price_digital"]

DIGITAL_METHODS: tuple[str, ...] = ("cos", "gil_pelaez")
"""Transform methods with a digital payoff here (module docstring)."""


def _check_method(cfg: FourierConfig) -> None:
    if cfg.method not in DIGITAL_METHODS:
        raise NotSupportedError(
            f"fourier method '{cfg.method}' prices the call transform, not the "
            f"terminal law, so it has no cash-or-nothing payoff here; use one of "
            f"{DIGITAL_METHODS}"
        )


def price_digital(
    option: DigitalOption,
    model: Any,
    market: Market,
    *,
    cfg: FourierConfig,
) -> PriceResult:
    """Price a cash-or-nothing digital through the model's transform."""
    _check_method(cfg)
    inputs = fourier_inputs(model, market, option.expiry)
    value, extra = transform_value(
        inputs,
        cfg,
        strike=option.strike,
        kind=option.kind,
        payoff="digital",
        cash=option.cash,
    )
    return PriceResult(value=value, meta=_meta(cfg, extra, "digital", model))


def greeks_digital(
    option: DigitalOption,
    model: Any,
    market: Market,
    *,
    cfg: FourierConfig,
) -> GreeksResult:
    """Digital Greeks: delta and gamma closed form, vega/theta/rho bumped."""
    _check_method(cfg)
    inputs = fourier_inputs(model, market, option.expiry)
    return _cos_greeks(
        inputs,
        cfg,
        model,
        market,
        strike=option.strike,
        kind=option.kind,
        payoff="digital",
        cash=option.cash,
        instrument="digital",
    )
