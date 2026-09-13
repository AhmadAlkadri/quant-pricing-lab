"""Lewis: one integral on the strip Im(z) = 1/2, with no damping parameter.

The idea
--------
Carr-Madan makes the call price integrable by multiplying it by ``e^{alpha k}``
and then has to defend the choice of ``alpha``. Lewis's route removes the
choice: transform the *payoff* rather than the price, pair it with the
characteristic function through Parseval's relation, and then move the contour
to the one line where the resulting integrand is symmetric in a way that makes
the answer manifestly real. That line is ``Im(z) = 1/2`` and it is not a free
parameter -- it is the midpoint of the payoff transform's strip of analyticity
and the strike's own natural scale.

Derivation
----------
Work in ``x = ln S_T`` and let ``w(x) = (e^x - K)^+``. Its Fourier transform

    w_hat(z) = Integral_R e^{i z x} w(x) dx

converges for ``Im(z) > 1`` (the payoff grows like ``e^x``, so the damping has
to beat it), and there

    w_hat(z) = Integral_{ln K}^{inf} e^{i z x}(e^x - K) dx
             = -K^{iz+1} / (iz + 1) + K^{iz+1} / (iz)
             = K^{iz+1} / (i z (i z + 1))
             = -K^{iz+1} / (z^2 - i z).

Parseval, with ``phi_S(u) = E[e^{i u ln S_T}]``:

    E[w(X_T)] = (1 / 2 pi) Integral_{Im z = nu} w_hat(z) phi_S(-z) dz,  nu > 1.

Now push the contour down to ``nu = 1/2``. The integrand has poles at ``z = 0``
and ``z = i``; only ``z = i`` lies between the two lines. Its residue is

    Res_{z=i} [ -K^{iz+1} / (z(z - i)) ] phi_S(-z) = i * phi_S(-i) = i S_0 e^{(r-q)T},

and moving a contour down past a pole adds ``-2 pi i`` times it, so

    C = e^{-rT} E[w] = S_0 e^{-qT} + e^{-rT} (1/2 pi) Integral_{Im z = 1/2} ... dz.

Parametrise ``z = u + i/2`` with ``u`` real. Then ``K^{iz+1} = sqrt(K) e^{iu ln K}``
and ``z^2 - i z = z(z - i) = (u + i/2)(u - i/2) = u^2 + 1/4`` -- real, positive,
and never zero, which is the whole point of that line. The integrand at ``-u``
is the conjugate of the integrand at ``u``, so the two halves fold:

    C = S_0 e^{-qT}
        - (sqrt(K) e^{-rT} / pi) Integral_0^inf Re[ e^{-i u ln K} phi_S(u - i/2) ]
                                                 / (u^2 + 1/4) du.

One integral, one contour, no parameter to tune, and a denominator bounded below
by 1/4 so nothing can blow up at the origin. The price paid is that ``phi_S``
must be evaluated off the real axis at ``u - i/2``, which for a model with a
complex square root in its exponent is exactly where branch cuts start to matter
-- the reason this formula is the one the Heston literature reaches for, and the
reason it is worth having in place before Heston arrives.

Source: Lewis (2001), "A simple option formula for general jump-diffusions and
other exponential Levy processes". The derivation above is written out
independently; nothing is reproduced from that paper.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from .charfn import CharacteristicFunctionModel

__all__ = ["LEWIS_STRIP_OFFSET", "lewis_call"]

LEWIS_STRIP_OFFSET = 0.5
"""``Im(z)`` of the integration contour. Not tunable -- see the module docstring."""


def lewis_call(
    cf: CharacteristicFunctionModel,
    *,
    s0: float,
    strike: float,
    expiry: float,
    rate: float,
    dividend: float,
    limit: int,
    tolerance: float,
) -> tuple[float, float]:
    """Lewis's call price and the quadrature's own reported error bound.

    The integral is handed to `scipy.integrate.quad` on ``[0, inf)``, which uses
    QUADPACK's adaptive infinite-interval rule. The second return value is the
    bound that routine reports; `tests/test_fourier_lewis_gil_pelaez.py`
    measures how far that bound is from the error actually made.
    """
    log_strike = math.log(strike)
    log_spot = math.log(s0)

    def integrand(u: float) -> float:
        shifted = np.array([u - 1j * LEWIS_STRIP_OFFSET])
        phi_spot = np.exp(1j * shifted * log_spot) * cf.characteristic_function(
            shifted, expiry, rate=rate, dividend=dividend
        )
        value = np.exp(-1j * shifted.real * log_strike) * phi_spot / (u * u + 0.25)
        return float(np.real(value[0]))

    integral, abserr = quad(
        integrand, 0.0, np.inf, limit=limit, epsabs=tolerance, epsrel=tolerance
    )
    scale = math.sqrt(strike) * math.exp(-rate * expiry) / math.pi
    return s0 * math.exp(-dividend * expiry) - scale * integral, scale * abserr
