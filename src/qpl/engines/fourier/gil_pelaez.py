"""Gil-Pelaez: the two probabilities a Black-Scholes price is made of.

The idea
--------
The other three methods here transform a *payoff*. This one transforms nothing:
it inverts the characteristic function to a distribution function directly, and
then reads the option price off the two probabilities that the Black-Scholes
formula was always written in terms of.

    C = S_0 e^{-qT} Pi_1 - K e^{-rT} Pi_2,
    Pi_2 = Q(S_T > K),
    Pi_1 = Q^S(S_T > K)   (the same event under the share measure),

which for Black-Scholes are `N(d1)` and `N(d2)`. The cash-or-nothing digital
call is `cash * e^{-rT} * Pi_2` with no further work, which is why this is the
method that prices a digital as naturally as a vanilla.

Gil-Pelaez's inversion
----------------------
For a random variable ``X`` with characteristic function ``phi``,

    Q(X > k) = 1/2 + (1 / pi) Integral_0^inf Re[ e^{-i u k} phi(u) / (i u) ] du.

The integrand's apparent singularity at ``u = 0`` is removable: writing
``Re[w / i] = Im[w]``, it is ``Im[e^{-i u k} phi(u)] / u``, and the numerator
vanishes linearly at the origin with slope ``c1 - k``, where ``c1`` is the first
cumulant. So the limit is finite and the integral converges. (QUADPACK's
infinite-interval rule never evaluates the endpoint, but the implementation
below returns the limit there anyway rather than a division by zero.)

The share measure
-----------------
``Pi_1`` is the same probability computed under the measure whose
Radon-Nikodym derivative against ``Q`` is ``S_T / E[S_T]``. Its characteristic
function is an argument shift of the original one:

    phi_1(u) = E[ (S_T / E[S_T]) e^{i u ln S_T} ] = phi(u - i) / phi(-i),

so no second model is needed, only one more evaluation of the same transform one
unit below the real axis. ``phi(-i) = E[S_T] = S_0 e^{(r-q)T}`` is the forward,
which is a cheap arithmetic check on any model that claims to satisfy the
`CharacteristicFunctionModel` protocol.

Source: Gil-Pelaez (1951), "Note on the inversion theorem", Biometrika 38(3-4),
481-482, for the inversion formula; the `Pi_1`/`Pi_2` decomposition in this form
is the one Heston (1993) uses. Both derivations above are written out
independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .charfn import CharacteristicFunctionModel

__all__ = ["GilPelaezProbabilities", "gil_pelaez_probabilities"]

_ZERO_SUBSTITUTE = 1e-100
"""Stand-in for ``u = 0`` in the integrand.

At this argument the numerator is exactly ``u (c1 - k)`` in double precision --
the higher-order terms underflow -- so the ratio evaluates to the removable
limit without a special case and without a division by zero. Cheaper and less
error-prone than carrying a per-model formula for the limit, and it is a value
the adaptive rule never actually asks for.
"""


@dataclass(frozen=True)
class GilPelaezProbabilities:
    """``Pi_1``, ``Pi_2`` and the quadrature's own error bound on each."""

    pi1: float
    pi2: float
    pi1_abserr: float
    pi2_abserr: float


def _tail_probability(
    transform,
    log_strike: float,
    *,
    limit: int,
    tolerance: float,
) -> tuple[float, float]:
    """``1/2 + (1/pi) Integral_0^inf Im[e^{-iuk} phi(u)] / u du``."""

    def integrand(u: float) -> float:
        arg = u if u != 0.0 else _ZERO_SUBSTITUTE
        value = np.exp(-1j * arg * log_strike) * transform(arg)
        return float(np.imag(value) / arg)

    integral, abserr = quad(
        integrand, 0.0, np.inf, limit=limit, epsabs=tolerance, epsrel=tolerance
    )
    return 0.5 + integral / math.pi, abserr / math.pi


def gil_pelaez_probabilities(
    cf: CharacteristicFunctionModel,
    *,
    s0: float,
    strike: float,
    expiry: float,
    rate: float,
    dividend: float,
    limit: int,
    tolerance: float,
) -> GilPelaezProbabilities:
    """``Q(S_T > K)`` and its share-measure twin, by Gil-Pelaez inversion."""
    log_spot = math.log(s0)
    log_strike = math.log(strike)
    forward = s0 * math.exp((rate - dividend) * expiry)

    def phi_spot(u: complex) -> complex:
        arg = np.array([u], dtype=complex)
        return complex(
            (
                np.exp(1j * arg * log_spot)
                * cf.characteristic_function(arg, expiry, rate=rate, dividend=dividend)
            )[0]
        )

    pi2, pi2_err = _tail_probability(
        phi_spot, log_strike, limit=limit, tolerance=tolerance
    )
    pi1, pi1_err = _tail_probability(
        lambda u: phi_spot(u - 1j) / forward,
        log_strike,
        limit=limit,
        tolerance=tolerance,
    )
    return GilPelaezProbabilities(
        pi1=pi1, pi2=pi2, pi1_abserr=pi1_err, pi2_abserr=pi2_err
    )
