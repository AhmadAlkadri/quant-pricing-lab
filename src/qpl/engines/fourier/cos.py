"""The COS method: a cosine expansion of the risk-neutral density.

Derivation
----------
Write ``Z = ln(S_T / S_0)`` and let ``f`` be its risk-neutral density. Pick a
finite range ``[a, b]`` that carries essentially all of the mass. On that range
``f`` has a cosine series

    f(z) ~ sum_{k >= 0}' A_k cos( k pi (z - a) / (b - a) ),

where the prime halves the ``k = 0`` term and

    A_k = (2 / (b - a)) * Integral_a^b f(z) cos( k pi (z - a) / (b - a) ) dz.

The step that makes the method is that the integral above is *almost* the
characteristic function. Writing ``u_k = k pi / (b - a)`` and extending the
integral to the whole line (the error of doing so is the range-truncation error
and nothing else),

    Integral_R f(z) cos(u_k (z - a)) dz = Re{ E[e^{i u_k Z}] e^{-i u_k a} },

so

    A_k ~= (2 / (b - a)) Re{ phi(u_k) e^{-i u_k a} },

which needs one evaluation of the model's transform per term and no density.

Now the price. For a payoff ``g`` paid at ``T``,

    V = e^{-rT} E[g(S_0 e^Z)]
      = e^{-rT} Integral_a^b g(S_0 e^z) f(z) dz + (truncation)
      ~= e^{-rT} sum_k' Re{ phi(u_k) e^{-i u_k a} } * V_k,

    V_k = (2 / (b - a)) Integral_a^b g(S_0 e^z) cos(u_k (z - a)) dz,

and ``V_k`` is a closed form for every payoff in this module. Two integrals do
all the work:

    chi_k(c, d) = Integral_c^d e^z cos(u_k (z - a)) dz
                = [ e^d (cos(u_k (d-a)) + u_k sin(u_k (d-a)))
                  - e^c (cos(u_k (c-a)) + u_k sin(u_k (c-a))) ] / (1 + u_k^2)

    psi_k(c, d) = Integral_c^d cos(u_k (z - a)) dz
                = [ sin(u_k (d-a)) - sin(u_k (c-a)) ] / u_k     (k >= 1)
                = d - c                                          (k = 0)

both obtained by the standard antiderivative of ``e^z cos(wz + p)``.

With ``z* = ln(K / S_0)`` clipped into ``[a, b]`` (the log return at which the
option goes in the money),

    call:         V_k = (2/(b-a)) [ S_0 chi_k(z*, b) - K psi_k(z*, b) ]
    put:          V_k = (2/(b-a)) [ K psi_k(a, z*) - S_0 chi_k(a, z*) ]
    digital call: V_k = (2/(b-a)) cash * psi_k(z*, b)
    digital put:  V_k = (2/(b-a)) cash * psi_k(a, z*)

Why the log *return* and not ``ln(S_T / K)``
--------------------------------------------
Fang and Oosterlee state the method in ``y = ln(S_T / K)``, where the range is
``[a, b] = [x + c1 - L w, x + c1 + L w]`` with ``x = ln(S_0 / K)``. That is the
same range as the one used here, shifted by ``x``: expanding in ``Z`` instead
moves ``x`` out of the range and into ``V_k``. Numerically the two are the same
method. The difference matters for one thing only, and it is the reason for the
choice: with the range independent of ``S_0``, the spot appears **only** in
``V_k``, so differentiating the price in ``S_0`` differentiates a closed form
and the COS delta and gamma are exact rather than bumped.

Greeks
------
Differentiate ``V_k`` in ``S_0``. For the call, ``z* = ln(K / S_0)`` moves with
the spot, but the integrand vanishes at ``z = z*`` (the payoff is zero exactly
at the boundary), so the boundary terms cancel and

    dV_k / dS_0 = (2 / (b - a)) chi_k(z*, b).

Differentiating once more, only the moving endpoint is left:

    d2V_k / dS_0^2 = (2 / (b - a)) * (K / S_0^2) * cos(u_k (z* - a)),

using ``d chi_k(c, d) / dc = -e^c cos(u_k (c - a))`` and ``dz*/dS_0 = -1/S_0``.
The put's derivatives are the same expressions with the opposite sign on the
first and the same sign on the second -- a put and a call differ by a forward,
whose gamma is zero and whose delta is ``e^{-qT}``, which the two expressions
reproduce. When ``ln(K / S_0)`` falls outside ``[a, b]`` the gamma term is zero:
the strike is outside the represented range, so nothing there responds to the
spot.

The digital's first derivative keeps a boundary term instead of losing one,
because its payoff does *not* vanish at ``z*``:

    dV_k / dS_0 = +/- (2 / (b - a)) cash cos(u_k (z* - a)) / S_0.

Truncation range
----------------
``[a, b] = [c1 - L w, c1 + L w]`` with ``w = sqrt(c2 + sqrt(c4))`` from the
model's cumulants. Fang and Oosterlee recommend ``L = 10`` for
Black-Scholes-like models; `tests/test_fourier_cos.py` measures what smaller
values cost.

Source: Fang and Oosterlee (2008), "A novel pricing method for European options
based on Fourier-cosine series expansions", SIAM J. Sci. Comput. 31(2),
826-848 -- sections 2 and 3 for the expansion and the range rule, section 4 for
the vanilla payoff coefficients. Everything above is re-derived here; no
expression, number or table is copied from that paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from .charfn import CharacteristicFunctionModel

__all__ = [
    "COS_PAYOFFS",
    "CosResult",
    "chi_coefficients",
    "cos_price",
    "cos_truncation_range",
    "psi_coefficients",
]

COS_PAYOFFS: tuple[str, ...] = ("vanilla", "digital")
"""Payoffs with a closed-form `V_k` in this module."""


@dataclass(frozen=True)
class CosResult:
    """A COS valuation and the two Greeks that come free with it.

    `delta` and `gamma` are exact derivatives of the truncated sum, not bumps:
    they differ from the analytic Greeks by the same truncation and range error
    the price carries, and by nothing else.
    """

    value: float
    delta: float
    gamma: float
    lower: float
    upper: float
    n_terms: int


def cos_truncation_range(
    cf: CharacteristicFunctionModel,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    truncation_l: float,
) -> tuple[float, float]:
    """``[c1 - L w, c1 + L w]`` from the model's cumulants of the log return."""
    cumulants = cf.log_return_cumulants(expiry, rate=rate, dividend=dividend)
    width = truncation_l * cumulants.truncation_width()
    if not math.isfinite(width) or width <= 0.0:
        raise InvalidInputError(
            "COS truncation range is degenerate: the log return has zero variance, "
            "so the terminal law is a point mass with no density to expand. This is "
            "what sigma = 0 looks like to a transform method; use method='analytic'."
        )
    return cumulants.c1 - width, cumulants.c1 + width


def chi_coefficients(k: np.ndarray, c: float, d: float, a: float, b: float) -> np.ndarray:
    """``Integral_c^d e^z cos(k pi (z - a) / (b - a)) dz`` (module docstring)."""
    w = k * math.pi / (b - a)
    arg_c = w * (c - a)
    arg_d = w * (d - a)
    return (
        np.exp(d) * (np.cos(arg_d) + w * np.sin(arg_d))
        - np.exp(c) * (np.cos(arg_c) + w * np.sin(arg_c))
    ) / (1.0 + w * w)


def psi_coefficients(k: np.ndarray, c: float, d: float, a: float, b: float) -> np.ndarray:
    """``Integral_c^d cos(k pi (z - a) / (b - a)) dz`` (module docstring)."""
    w = np.asarray(k, dtype=float) * math.pi / (b - a)
    zero = w == 0.0
    safe = np.where(zero, 1.0, w)
    return np.where(
        zero, d - c, (np.sin(safe * (d - a)) - np.sin(safe * (c - a))) / safe
    )


def _payoff_coefficients(
    k: np.ndarray,
    *,
    a: float,
    b: float,
    s0: float,
    strike: float,
    kind: str,
    payoff: str,
    cash: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``V_k`` and its first two derivatives in ``S_0`` (module docstring)."""
    scale = 2.0 / (b - a)
    boundary = math.log(strike / s0)
    inside = a < boundary < b
    clipped = min(max(boundary, a), b)
    cos_boundary = np.cos(k * math.pi * (clipped - a) / (b - a))

    if payoff == "vanilla":
        if kind == "call":
            chi = chi_coefficients(k, clipped, b, a, b)
            psi = psi_coefficients(k, clipped, b, a, b)
            value = scale * (s0 * chi - strike * psi)
            first = scale * chi
        else:
            chi = chi_coefficients(k, a, clipped, a, b)
            psi = psi_coefficients(k, a, clipped, a, b)
            value = scale * (strike * psi - s0 * chi)
            first = -scale * chi
        second = (
            scale * (strike / (s0 * s0)) * cos_boundary
            if inside
            else np.zeros_like(k, dtype=float)
        )
        return value, first, second

    if kind == "call":
        psi = psi_coefficients(k, clipped, b, a, b)
        value = scale * cash * psi
        first = scale * cash * cos_boundary / s0 if inside else np.zeros_like(k, dtype=float)
    else:
        psi = psi_coefficients(k, a, clipped, a, b)
        value = scale * cash * psi
        first = -scale * cash * cos_boundary / s0 if inside else np.zeros_like(k, dtype=float)
    # The digital's gamma is the derivative of a boundary term, so it carries a
    # factor 1/S_0^2 and the derivative of the cosine at the moving endpoint.
    if inside:
        w = k * math.pi / (b - a)
        sin_boundary = np.sin(k * math.pi * (clipped - a) / (b - a))
        sign = 1.0 if kind == "call" else -1.0
        second = sign * scale * cash * (w * sin_boundary - cos_boundary) / (s0 * s0)
    else:
        second = np.zeros_like(k, dtype=float)
    return value, first, second


def cos_price(
    cf: CharacteristicFunctionModel,
    *,
    s0: float,
    strike: float,
    expiry: float,
    rate: float,
    dividend: float,
    kind: str,
    payoff: str = "vanilla",
    cash: float = 1.0,
    n_terms: int,
    truncation_l: float,
) -> CosResult:
    """Price and differentiate a European payoff by the cosine expansion.

    Parameters
    ----------
    cf
        Anything satisfying `CharacteristicFunctionModel`.
    payoff
        ``"vanilla"`` for a call/put, ``"digital"`` for cash-or-nothing.
    n_terms
        Number of cosine terms ``N``. The error decays *exponentially* in ``N``
        for a smooth density and then stops at the floating-point floor; see
        `tests/test_fourier_cos.py` for the measured sequence.
    truncation_l
        ``L`` in the range rule.
    """
    if payoff not in COS_PAYOFFS:
        raise InvalidInputError(f"payoff must be one of {COS_PAYOFFS}")
    a, b = cos_truncation_range(
        cf, expiry, rate=rate, dividend=dividend, truncation_l=truncation_l
    )

    k = np.arange(n_terms, dtype=float)
    u = k * math.pi / (b - a)
    phi = np.asarray(cf.characteristic_function(u, expiry, rate=rate, dividend=dividend))
    weights = np.real(phi * np.exp(-1j * u * a))
    weights[0] *= 0.5

    value_k, first_k, second_k = _payoff_coefficients(
        k, a=a, b=b, s0=s0, strike=strike, kind=kind, payoff=payoff, cash=cash
    )
    discount = math.exp(-rate * expiry)
    return CosResult(
        value=discount * float(np.sum(weights * value_k)),
        delta=discount * float(np.sum(weights * first_k)),
        gamma=discount * float(np.sum(weights * second_k)),
        lower=a,
        upper=b,
        n_terms=n_terms,
    )
