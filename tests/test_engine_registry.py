"""The dispatcher's registry: lookup, keyword contracts, and error messages.

The messages asserted here are the ones the pre-registry `isinstance` ladder
produced. They are pinned because callers (and the existing engine tests) rely
on the distinction between `InvalidInputError` (the caller's keyword arguments
are wrong) and `NotSupportedError` (nothing is registered for this
combination), and on *which* of the two a given mistake produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.registry import (
    MethodSpec,
    known_methods,
    method_spec,
    register,
    resolve_greeks,
    resolve_price,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price


def _market() -> Market:
    return Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.0),
    )


def _option() -> EuropeanOption:
    return EuropeanOption(kind="call", strike=100.0, expiry=1.0)


def test_builtin_methods_are_registered_for_price_and_greeks() -> None:
    assert set(known_methods()) >= {"analytic", "mc", "pde", "tree"}

    option, model, market = _option(), BlackScholesModel(sigma=0.2), _market()
    for method in ("analytic", "mc", "pde", "tree"):
        assert callable(resolve_price(option, model, market, method))
    for method in ("analytic", "mc", "pde"):
        assert callable(resolve_greeks(option, model, market, method))


def test_unknown_method_is_not_supported() -> None:
    option, model, market = _option(), BlackScholesModel(sigma=0.2), _market()
    with pytest.raises(NotSupportedError, match="method 'lsm' is not supported"):
        price(option, model, market, method="lsm")  # type: ignore[arg-type]
    with pytest.raises(NotSupportedError, match="method 'lsm' is not supported"):
        greeks(option, model, market, method="lsm")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["analytic", "mc", "pde", "tree"])
def test_unregistered_instrument_model_or_market_is_not_supported(method: str) -> None:
    """All three slots are checked; none of them silently falls through."""
    option, model, market = _option(), BlackScholesModel(sigma=0.2), _market()
    kwargs: dict[str, object] = {}
    spec = method_spec(method)
    if spec.cfg_type is not None:
        kwargs["cfg"] = spec.cfg_type()

    message = "Unsupported instrument/model/market combination"
    with pytest.raises(NotSupportedError, match=message):
        price(object(), model, market, method=method, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(NotSupportedError, match=message):
        price(option, object(), market, method=method, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(NotSupportedError, match=message):
        price(option, model, object(), method=method, **kwargs)  # type: ignore[arg-type]


def test_keyword_contract_messages_are_preserved() -> None:
    option, model, market = _option(), BlackScholesModel(sigma=0.2), _market()

    with pytest.raises(InvalidInputError, match="Unexpected keyword arguments for method 'analytic'"):
        price(option, model, market, method="analytic", cfg=MCConfig())

    with pytest.raises(InvalidInputError, match="cfg is required for method 'pde'"):
        price(option, model, market, method="pde")

    with pytest.raises(InvalidInputError, match="cfg must be an instance of PDEConfig"):
        price(option, model, market, method="pde", cfg=MCConfig())

    with pytest.raises(InvalidInputError, match="Unexpected keyword arguments for method 'pde'"):
        price(option, model, market, method="pde", cfg=PDEConfig(), bumps={"spot": 1e-4})


def test_bumps_is_accepted_by_greeks_only_and_type_checked() -> None:
    """`bumps` is declared on the MC method spec, so only `greeks` takes it."""
    option, model, market = _option(), BlackScholesModel(sigma=0.2), _market()
    cfg = MCConfig(n_paths=64, n_steps=1, seed=1)

    with pytest.raises(InvalidInputError, match="Unexpected keyword arguments for method 'mc'"):
        price(option, model, market, method="mc", cfg=cfg, bumps={"spot": 1e-4})

    with pytest.raises(InvalidInputError, match="bumps must be a dict of bump sizes"):
        greeks(option, model, market, method="mc", cfg=cfg, bumps=1.0)

    # Omitted entirely: the engine's own default applies, no error.
    assert greeks(option, model, market, method="mc", cfg=cfg).meta is not None


def test_cfg_check_runs_before_the_combination_check() -> None:
    """Keyword validation is ordered ahead of engine lookup, as it always was.

    A caller who passes both a bad `cfg` and an unsupported instrument gets the
    `InvalidInputError`, not the `NotSupportedError`. Pinning the order keeps
    the error a caller sees first pointing at the mistake they can fix.
    """
    with pytest.raises(InvalidInputError, match="cfg is required for method 'mc'"):
        price(object(), object(), object(), method="mc")


def test_registration_rejects_conflicting_specs_and_duplicate_keys() -> None:
    @dataclass(frozen=True)
    class _ToyInstrument:
        pass

    @dataclass(frozen=True)
    class _ToyModel:
        pass

    spec = MethodSpec(method="_registry_test_toy")

    def _engine(instrument, model, market):  # pragma: no cover - never invoked
        raise AssertionError

    register(
        instrument_type=_ToyInstrument,
        model_type=_ToyModel,
        spec=spec,
        price=_engine,
    )

    with pytest.raises(InvalidInputError, match="already registered"):
        register(
            instrument_type=_ToyInstrument,
            model_type=_ToyModel,
            spec=spec,
            price=_engine,
        )

    with pytest.raises(InvalidInputError, match="different MethodSpec"):
        register(
            instrument_type=_ToyInstrument,
            model_type=_ToyModel,
            spec=MethodSpec(method="_registry_test_toy", cfg_type=MCConfig),
            price=_engine,
        )

    with pytest.raises(InvalidInputError, match="at least one of price="):
        register(
            instrument_type=_ToyInstrument,
            model_type=_ToyModel,
            spec=MethodSpec(method="_registry_test_toy_2"),
        )

    # Registered for price only: the Greeks table is genuinely separate.
    with pytest.raises(NotSupportedError):
        resolve_greeks(_ToyInstrument(), _ToyModel(), _market(), "_registry_test_toy")


def test_lookup_walks_the_mro_so_subclasses_still_resolve() -> None:
    """`isinstance` semantics are preserved by an explicit MRO walk."""

    class _RefinedOption(EuropeanOption):
        pass

    option = _RefinedOption(kind="call", strike=100.0, expiry=1.0)
    model, market = BlackScholesModel(sigma=0.2), _market()
    assert price(option, model, market, method="analytic").value == pytest.approx(
        price(_option(), model, market, method="analytic").value
    )
