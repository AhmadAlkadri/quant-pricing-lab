"""The characteristic-function interface every Fourier engine in this package reads.

Why this exists
---------------
The four methods in this package -- COS, Carr-Madan, Lewis and Gil-Pelaez --
differ in *which transform of the payoff* they pair with the law of the log
spot, and in nothing else. Each of them needs exactly two things from a model:

1. ``phi(u) = E^Q[ exp(i u ln(S_T / S_0)) ]`` for real or complex ``u``, and
2. enough of the cumulants of ``ln(S_T / S_0)`` to choose a truncation range or
   an integration reach.

That is the whole contract, and it is stated as a `Protocol` rather than as a
base class so a model satisfies it by having the two methods, not by inheriting
from anything. The next slice's Heston model implements the same two methods
and none of `cos`, `carr_madan`, `lewis` or `gil_pelaez` changes.

The log-return convention, stated once
--------------------------------------
`characteristic_function` is the transform of the **log return**
``Z = ln(S_T / S_0)``, not of ``ln S_T``. The spot is carried separately by
every pricer. Two reasons, both practical:

- the truncation range the COS method needs is a statement about ``Z`` (its
  cumulants do not involve ``S_0``), so keeping ``S_0`` out of ``phi`` keeps
  the range independent of the spot, which is what makes the COS delta and
  gamma closed forms rather than bumps (see `qpl.engines.fourier.cos`);
- a model that quotes its transform in forward terms needs no ``S_0`` at all.

The transform of the log spot is one multiplication away and each engine that
wants it does that multiplication explicitly:

    E[ exp(i u ln S_T) ] = exp(i u ln S_0) * phi(u).

``u`` may be complex. Carr-Madan evaluates ``phi`` at ``u - (alpha + 1) i``,
Lewis on the line ``Im(u) = -1/2``, and Gil-Pelaez's share-measure probability
at ``u - i``; a model whose transform is only valid for real ``u`` does not
satisfy this protocol.

Cumulants
---------
`log_return_cumulants` returns ``c1``, ``c2`` and ``c4`` of ``Z``: the mean,
the variance, and the fourth cumulant. They are used in two places:

- the COS truncation range `[a, b] = [c1 - L w, c1 + L w]` with
  ``w = sqrt(c2 + sqrt(c4))``, which is Fang and Oosterlee's rule;
- the default integration reach of the Carr-Madan direct-quadrature variant,
  which is a multiple of ``sqrt(c2)`` (the transform of a law with variance
  ``c2`` decays on the scale ``1 / sqrt(c2)``).

``c4`` is carried rather than dropped because it is zero for Black-Scholes and
is *not* zero for Heston, and the range rule is the place that difference shows
up first.

Sources for the interface shape (no formula below is quoted from them):
Fang and Oosterlee (2008), SIAM J. Sci. Comput. 31(2), 826-848, section 3 for
the cumulant-based range; Schmelzle (2010), "Option pricing formulae using
Fourier transform: theory and application", for the survey view of why every
one of these methods only ever touches a model through its transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...exceptions import NotSupportedError
from ...models.black_scholes import BlackScholesModel

__all__ = [
    "BlackScholesCharacteristicFunction",
    "CharacteristicFunctionModel",
    "LogReturnCumulants",
    "black_scholes_characteristic_function",
    "black_scholes_log_return_cumulants",
    "characteristic_function_model",
    "register_characteristic_function",
]


@dataclass(frozen=True)
class LogReturnCumulants:
    """First, second and fourth cumulants of ``ln(S_T / S_0)``.

    The third is not carried because no rule here reads it: Fang and
    Oosterlee's truncation range uses ``c1``, ``c2`` and ``c4`` only.
    """

    c1: float
    c2: float
    c4: float = 0.0

    def truncation_width(self) -> float:
        """``sqrt(c2 + sqrt(c4))``, the scale the COS range is measured in."""
        return float(np.sqrt(self.c2 + np.sqrt(self.c4)))


@runtime_checkable
class CharacteristicFunctionModel(Protocol):
    """What a model must provide for the Fourier engines to price under it."""

    def characteristic_function(
        self,
        u: np.ndarray,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> np.ndarray:
        """``E^Q[exp(i u ln(S_T / S_0))]`` at (possibly complex) ``u``."""
        ...

    def log_return_cumulants(
        self,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> LogReturnCumulants:
        """Cumulants of ``ln(S_T / S_0)`` under the same measure."""
        ...


def black_scholes_log_return_cumulants(
    expiry: float, *, rate: float, dividend: float, sigma: float
) -> LogReturnCumulants:
    """Cumulants of ``ln(S_T / S_0)`` under Black-Scholes.

    ``ln(S_T / S_0) = (r - q - sigma^2 / 2) T + sigma sqrt(T) Z`` with
    ``Z ~ N(0, 1)``, so the law is Gaussian and every cumulant past the second
    vanishes identically:

        c1 = (r - q - sigma^2 / 2) T,   c2 = sigma^2 T,   c4 = 0.

    The ``c4 = 0`` is the whole reason `LogReturnCumulants` carries a fourth
    cumulant at all: it is the slot Heston fills.
    """
    drift = (rate - dividend - 0.5 * sigma * sigma) * expiry
    return LogReturnCumulants(c1=float(drift), c2=float(sigma * sigma * expiry), c4=0.0)


def black_scholes_characteristic_function(
    u: np.ndarray,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    sigma: float,
) -> np.ndarray:
    """``E[exp(i u ln(S_T / S_0))]`` under Black-Scholes, for complex ``u``.

    Derivation
    ----------
    Under the risk-neutral measure ``ln(S_T / S_0) = m + s Z`` with
    ``m = (r - q - sigma^2 / 2) T``, ``s = sigma sqrt(T)`` and ``Z`` standard
    normal. The Gaussian moment generating function
    ``E[exp(w Z)] = exp(w^2 / 2)`` holds for every complex ``w`` by analytic
    continuation (both sides are entire and agree on the real line), so with
    ``w = i u s``

        E[exp(i u (m + s Z))] = exp(i u m) exp(-u^2 s^2 / 2)
                              = exp(i u (r - q - sigma^2/2) T - sigma^2 u^2 T / 2).

    That continuation is what makes the complex arguments the other three
    methods need legitimate here, and it is the step that stops being free for
    Heston -- where the same continuation runs into the branch cut of a complex
    square root, which is the next slice's problem, not this one's.

    Note the sign convention: ``exp(-u^2 s^2 / 2)`` with ``u`` **complex** is
    ``u^2``, not ``|u|^2``, so evaluating at ``u = v - (alpha + 1) i`` grows
    like ``exp(+(alpha + 1)^2 sigma^2 T / 2)``. That growth is exactly the
    amplification the Carr-Madan damping study measures.
    """
    u_c = np.asarray(u, dtype=complex)
    drift = (rate - dividend - 0.5 * sigma * sigma) * expiry
    return np.exp(1j * u_c * drift - 0.5 * sigma * sigma * u_c * u_c * expiry)


@dataclass(frozen=True)
class BlackScholesCharacteristicFunction:
    """`CharacteristicFunctionModel` adapter for `BlackScholesModel`."""

    sigma: float

    def characteristic_function(
        self,
        u: np.ndarray,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> np.ndarray:
        return black_scholes_characteristic_function(
            u, expiry, rate=rate, dividend=dividend, sigma=self.sigma
        )

    def log_return_cumulants(
        self,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> LogReturnCumulants:
        return black_scholes_log_return_cumulants(
            expiry, rate=rate, dividend=dividend, sigma=self.sigma
        )


_ADAPTERS: dict[type, Any] = {}


def register_characteristic_function(model_type: type, adapter: Any) -> None:
    """Bind ``model_type`` to a callable returning a `CharacteristicFunctionModel`.

    Only needed for a model that does *not* implement the protocol itself.
    A model that grows the two methods needs no entry here: the resolver below
    checks the object first and the table second.
    """
    _ADAPTERS[model_type] = adapter


register_characteristic_function(
    BlackScholesModel, lambda model: BlackScholesCharacteristicFunction(sigma=model.sigma)
)


def characteristic_function_model(model: Any) -> CharacteristicFunctionModel:
    """Return the transform interface for ``model``.

    Checks the object itself before the adapter table, so a future model that
    implements `characteristic_function` and `log_return_cumulants` directly is
    usable with no registration at all -- which is the intended route for
    Heston.

    Raises
    ------
    NotSupportedError
        If the model neither implements the protocol nor has an adapter.
    """
    if hasattr(model, "characteristic_function") and hasattr(model, "log_return_cumulants"):
        return model
    for model_type in type(model).__mro__:
        adapter = _ADAPTERS.get(model_type)
        if adapter is not None:
            return adapter(model)
    raise NotSupportedError(
        f"{type(model).__name__} provides no characteristic function; "
        "implement characteristic_function(u, expiry, *, rate, dividend) and "
        "log_return_cumulants(expiry, *, rate, dividend), or register an adapter"
    )
