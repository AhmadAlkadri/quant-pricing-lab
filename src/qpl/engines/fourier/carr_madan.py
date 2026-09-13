"""Carr-Madan: damp the call price in log-strike, then transform.

The problem and the fix
-----------------------
The call price ``C_T(k)`` as a function of the log strike ``k = ln K`` is not
square integrable: as ``k -> -inf`` it tends to ``S_0 e^{-qT}``, not to zero, so
its Fourier transform does not exist in the ordinary sense. Carr and Madan's
move is to multiply by a decaying factor first. With ``alpha > 0``,

    c_T(k) = e^{alpha k} C_T(k)

does decay at both ends -- at ``-inf`` because ``e^{alpha k}`` does, at ``+inf``
because the call does, provided ``E[S_T^{alpha+1}]`` is finite -- and its
transform can be written in closed form in terms of the model's characteristic
function.

Derivation
----------
Write ``phi_S(u) = E[e^{i u ln S_T}]``, and let ``psi_T`` be the transform of
the damped price:

    psi_T(v) = Integral_R e^{i v k} c_T(k) dk.

Substituting ``C_T(k) = e^{-rT} E[(S_T - e^k)^+]`` and swapping the order of
integration (Fubini, justified by the damping),

    psi_T(v) = e^{-rT} E[ Integral_{-inf}^{ln S_T} e^{(alpha + i v) k} (S_T - e^k) dk ].

The inner integral is elementary. With ``w = alpha + i v`` and ``x = ln S_T``,

    Integral_{-inf}^{x} e^{w k}(e^x - e^k) dk = e^{(w+1)x} [ 1/w - 1/(w+1) ]
                                             = e^{(w+1)x} / (w (w + 1)),

so

    psi_T(v) = e^{-rT} E[ e^{(alpha + 1 + i v) ln S_T} ] / ( w (w + 1) )
             = e^{-rT} phi_S( v - (alpha + 1) i ) / ( alpha^2 + alpha - v^2 + i (2 alpha + 1) v ),

using ``w(w+1) = alpha^2 + alpha - v^2 + i(2 alpha + 1) v`` and
``E[e^{z ln S_T}] = phi_S(-i z)``. Inverting, and using that ``c_T`` is real so
``psi_T(-v) = conj(psi_T(v))``,

    C_T(k) = (e^{-alpha k} / pi) Integral_0^inf Re[ e^{-i v k} psi_T(v) ] dv.

Two ways to evaluate that integral, and this module implements both
--------------------------------------------------------------------
1. **FFT** (`carr_madan_fft`). Sample ``v_j = j eta`` and truncate the integral
   at ``N eta``. The trapezoidal sum then has the shape of a discrete Fourier
   transform if -- and only if -- the log strikes are placed on the reciprocal
   grid ``k_u = -b + u lambda`` with ``lambda = 2 pi / (N eta)`` and
   ``b = N lambda / 2``. One FFT produces ``N`` strikes at once. The price is
   that ``lambda`` is fixed by ``N`` and ``eta``: a *requested* strike almost
   never lands on the grid, and what a caller gets is an interpolation.
2. **Direct quadrature** (`carr_madan_quadrature`). Integrate the same
   integrand with one of the rules in `qpl.numerics.quadrature` at the strike
   that was actually asked for. No grid, no interpolation, one strike per call.

The two are worth having side by side precisely because the first is the famous
one and the second is the accurate one: see `tests/test_fourier_carr_madan.py`,
where the FFT's interpolation error at the recommended settings is 6.4e-04 and
the same integral by trapezoid at 512 nodes is at the floating-point floor.

Damping
-------
``alpha`` must be strictly positive, and ``E[S_T^{alpha + 1}]`` must be finite.
For Black-Scholes it is finite for every ``alpha``, which turns out to matter:
the usual warning that "large alpha breaks the method" is about models with
fat tails, and this package measures what actually goes wrong under
Black-Scholes instead of repeating it. See the alpha study in the tests.

Source: Carr and Madan (1999), "Option valuation using the fast Fourier
transform", Journal of Computational Finance 2(4), 61-73 -- the damped
transform, the FFT strike grid, and the Simpson weighting are theirs. The
derivation above is written out independently; no expression, number or table
is reproduced from that paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from ...numerics.quadrature import composite_simpson, composite_trapezoid, gauss_legendre
from .charfn import CharacteristicFunctionModel

__all__ = [
    "FFT_WEIGHTS",
    "ON_GRID_FRACTION",
    "QUADRATURE_RULES",
    "U_MAX_STANDARD_DEVIATIONS",
    "CarrMadanGrid",
    "carr_madan_fft",
    "carr_madan_quadrature",
    "damped_call_transform",
    "default_u_max",
]

QUADRATURE_RULES: tuple[str, ...] = ("trapezoid", "simpson", "gauss_legendre")
"""The rules from `qpl.numerics.quadrature` this engine can drive."""

FFT_WEIGHTS: tuple[str, ...] = ("simpson", "trapezoid")
"""Weightings for the FFT's sampled integral. Carr and Madan's is Simpson's."""

ON_GRID_FRACTION = 1e-09
"""How close to a node a requested log strike must be to count as on-grid.

A fraction of one log-strike spacing, not an absolute tolerance: a caller who
asks for `exp(node)` gets back a strike whose logarithm is the node to within
rounding, and that is the case the on-grid/off-grid study needs to identify."""

U_MAX_STANDARD_DEVIATIONS = 12.0
"""Default integration reach, in units of ``1 / sqrt(c2)``.

The integrand carries ``|phi(v)|``, which for a law of variance ``c2`` decays on
the scale ``1 / sqrt(c2)``; a Gaussian law is down by ``exp(-72) ~ 5e-32`` at
twelve of those. A *fixed* reach would be wrong by orders of magnitude between
a one-month 10% contract and a two-year 40% one, which is why the default is
derived from the model's own second cumulant rather than being a number.
"""


@dataclass(frozen=True)
class CarrMadanGrid:
    """One FFT's worth of call prices, on the log-strike grid the FFT forces."""

    log_strikes: np.ndarray
    values: np.ndarray
    eta: float
    spacing: float

    def reach(self) -> float:
        """``N eta``: where the ``v`` integral was truncated."""
        return self.eta * self.values.size

    def interpolate(self, log_strike: float) -> tuple[float, float, int]:
        """Linear interpolation in ``k``; returns the value, the weight, the node.

        The weight is the fractional position between the two bracketing nodes,
        so ``0`` (or ``1``) means the requested strike sits on a grid point and
        the value is the FFT's own output with no interpolation error at all.
        That distinction is the whole subject of the on-grid/off-grid study.
        """
        lower = self.log_strikes[0]
        if not lower <= log_strike <= self.log_strikes[-1]:
            raise InvalidInputError(
                f"log strike {log_strike:.6f} is outside the FFT grid "
                f"[{lower:.6f}, {self.log_strikes[-1]:.6f}]; the grid half-width is "
                f"pi / eta, so reduce eta"
            )
        index = min(int((log_strike - lower) / self.spacing), self.values.size - 2)
        weight = (log_strike - self.log_strikes[index]) / self.spacing
        value = (1.0 - weight) * self.values[index] + weight * self.values[index + 1]
        return float(value), float(weight), index


def default_u_max(
    cf: CharacteristicFunctionModel, expiry: float, *, rate: float, dividend: float
) -> float:
    """`U_MAX_STANDARD_DEVIATIONS / sqrt(c2)` from the model's cumulants."""
    cumulants = cf.log_return_cumulants(expiry, rate=rate, dividend=dividend)
    if cumulants.c2 <= 0.0 or not math.isfinite(cumulants.c2):
        raise InvalidInputError(
            "Carr-Madan needs a log return with positive variance; at sigma = 0 the "
            "terminal law is a point mass. Use method='analytic'."
        )
    return U_MAX_STANDARD_DEVIATIONS / math.sqrt(cumulants.c2)


def damped_call_transform(
    cf: CharacteristicFunctionModel,
    v: np.ndarray,
    *,
    s0: float,
    expiry: float,
    rate: float,
    dividend: float,
    alpha: float,
) -> np.ndarray:
    """``psi_T(v)``, the transform of the damped call price (module docstring)."""
    v_c = np.asarray(v, dtype=complex)
    shifted = v_c - (alpha + 1.0) * 1j
    phi_spot = np.exp(1j * shifted * math.log(s0)) * cf.characteristic_function(
        shifted, expiry, rate=rate, dividend=dividend
    )
    denominator = alpha * alpha + alpha - v_c * v_c + 1j * (2.0 * alpha + 1.0) * v_c
    return math.exp(-rate * expiry) * phi_spot / denominator


def carr_madan_fft(
    cf: CharacteristicFunctionModel,
    *,
    s0: float,
    expiry: float,
    rate: float,
    dividend: float,
    alpha: float,
    n_grid: int,
    eta: float,
    weights: str = "simpson",
) -> CarrMadanGrid:
    """One FFT: ``n_grid`` call prices on the reciprocal log-strike grid.

    ``weights`` selects Carr and Madan's own Simpson weighting or the plain
    trapezoid one. Simpson is the default because it is what the paper
    prescribes, and `tests/test_fourier_carr_madan.py` measures what it costs:
    on this integrand the trapezoid weighting is up to 8.4e+06 times more
    accurate at the same ``eta``, for a reason specific to the integrand and
    derived there.
    """
    if weights not in FFT_WEIGHTS:
        raise InvalidInputError(f"weights must be one of {FFT_WEIGHTS}")
    spacing = 2.0 * math.pi / (n_grid * eta)
    half_width = n_grid * spacing / 2.0
    j = np.arange(n_grid, dtype=float)
    v = eta * j

    rule_weights = np.ones(n_grid, dtype=float)
    if weights == "simpson":
        rule_weights[1:-1:2] = 4.0
        rule_weights[2:-1:2] = 2.0
        rule_weights /= 3.0
    else:
        rule_weights[0] = 0.5

    transform = damped_call_transform(
        cf, v, s0=s0, expiry=expiry, rate=rate, dividend=dividend, alpha=alpha
    )
    integrand = np.exp(1j * half_width * v) * transform * eta * rule_weights
    spectrum = np.real(np.fft.fft(integrand))

    log_strikes = -half_width + spacing * np.arange(n_grid, dtype=float)
    values = np.exp(-alpha * log_strikes) * spectrum / math.pi
    return CarrMadanGrid(
        log_strikes=log_strikes, values=values, eta=eta, spacing=spacing
    )


def carr_madan_quadrature(
    cf: CharacteristicFunctionModel,
    *,
    s0: float,
    strike: float,
    expiry: float,
    rate: float,
    dividend: float,
    alpha: float,
    rule: str,
    n_quad: int,
    u_max: float,
) -> float:
    """The same integral at one strike, by a rule from `qpl.numerics.quadrature`.

    This is where the Chapter 6 quadrature toolkit meets the pricing core: the
    integrand is a real, even, analytic function of ``v`` that decays like a
    Gaussian, and the rules behave on it nothing like they do on the smooth
    finite-interval integrands they were built against. The measurement is in
    `tests/test_fourier_carr_madan.py`.
    """
    log_strike = math.log(strike)

    def integrand(v: np.ndarray) -> np.ndarray:
        transform = damped_call_transform(
            cf, v, s0=s0, expiry=expiry, rate=rate, dividend=dividend, alpha=alpha
        )
        return np.real(np.exp(-1j * v * log_strike) * transform)

    if rule == "trapezoid":
        integral = composite_trapezoid(integrand, 0.0, u_max, n_quad)
    elif rule == "simpson":
        integral = composite_simpson(integrand, 0.0, u_max, n_quad)
    elif rule == "gauss_legendre":
        integral = gauss_legendre(integrand, 0.0, u_max, n_quad)
    else:  # pragma: no cover -- the config validates this
        raise InvalidInputError(f"rule must be one of {QUADRATURE_RULES}")

    return math.exp(-alpha * log_strike) * integral / math.pi
