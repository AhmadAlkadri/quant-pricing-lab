"""The Heston (1993) stochastic-volatility model, as a characteristic function.

This module ships one model and no pricer. `HestonModel` satisfies
`qpl.engines.fourier.charfn.CharacteristicFunctionModel` directly -- it has the
two methods that protocol names and needs no adapter -- so every transform
engine in `qpl.engines.fourier` prices under it with no pricer change at all.
That was the point of the Slice 14 interface and this module is the test of it.

The model
---------
Under the pricing measure, with `x_t = ln(S_t / S_0)`:

    dx_t = (r - q - v_t / 2) dt + sqrt(v_t) dW1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
    d<W1, W2>_t = rho dt

`v` is a square-root (CIR) diffusion, so `E[v_t] = theta + (v_0 - theta)
e^{-kappa t}` and the Feller condition `2 kappa theta >= xi^2` decides whether
the origin is attainable. `feller_number` here is `4 kappa theta / xi^2`, the
degrees of freedom of the transition law's noncentral chi-square, matching
`qpl.cases.sde_discretization.CIRSpec`; the condition is exactly
`feller_number >= 2`.

The affine solution
-------------------
The model is affine in `(x, v)`, so
`E[e^{i u x_T}] = exp(C(u, T) + D(u, T) v_0)`. Substituting that ansatz into
the backward equation for `f(t, x, v) = E[e^{i u x_T} | x_t, v_t]`,

    -f_tau + (r - q - v/2) f_x + (v/2) f_xx + kappa (theta - v) f_v
           + (xi^2 v / 2) f_vv + rho xi v f_xv = 0,      tau = T - t,

and collecting powers of `v` gives two ordinary differential equations with
`C(0) = D(0) = 0`:

    D' = (xi^2 / 2) D^2 + b D - c,     b = rho xi i u - kappa,
                                        c = (u^2 + i u) / 2
    C' = (r - q) i u + kappa theta D.

The first is a Riccati equation with constant coefficients. Its two stationary
roots are `D_pm = (-b +/- d) / xi^2` with

    d = sqrt(b^2 + 2 xi^2 c) = sqrt((kappa - rho xi i u)^2 + xi^2 (u^2 + i u)),

and the solution through `D(0) = 0` is

    D(tau) = D_- (1 - e^{-d tau}) / (1 - g e^{-d tau}),   g = D_- / D_+,

which integrates (`C' = (r - q) i u + kappa theta D`) to

    C(tau) = (r - q) i u tau
             + (kappa theta / xi^2) [ (kappa - rho xi i u - d) tau
                                      - 2 ln( (1 - g e^{-d tau}) / (1 - g) ) ].

Writing `beta = kappa - rho xi i u = -b`, `D_- = (beta - d) / xi^2` and
`g = (beta - d) / (beta + d)`.

The little Heston trap: which `d`, and which `g`
------------------------------------------------
`d` is a complex square root, so the formula above is two formulas: one for
each sign. Heston's own paper writes the equivalent expression with the *other*
sign, `+d`, and with `g` replaced by `1/g`, and the two are algebraically the
same function -- but only one of them is the same function *numerically*, and
that is the whole finding of Albrecher, Mayer, Schoutens and Tistaert (2007),
"The little Heston trap", Wilmott Magazine, section 3.

Take `d` on the **principal branch**, so `Re(d) >= 0`. Then

- `|g| <= 1`, because `|beta - d| <= |beta + d|` whenever `Re(beta) >= 0` and
  `Re(d) >= 0` (the point `-d` is at least as far from `beta` reflected as `+d`
  is; `Re(beta) = kappa + rho xi Im(u) > 0` on the strips these engines use);
- `|e^{-d tau}| = e^{-Re(d) tau} <= 1`;
- so `|g e^{-d tau}| <= 1` and `1 - g e^{-d tau}` never reaches the negative
  real axis by winding around the origin: its argument stays in `(-pi/2, pi/2)`
  and the principal branch of `ln` is the continuous one.

With the other sign the exponential is `e^{+d tau}`, `|1/g| >= 1`, and
`1 - e^{d tau} / g` *does* wind. Each winding adds `2 pi i` to the logarithm,
and `C` multiplies that by `-2 kappa theta / xi^2`, so the characteristic
function is multiplied by

    exp(-4 pi i kappa theta / xi^2)

at every crossing. Lord and Kahl (2010), "Complex logarithms in Heston-like
models", Mathematical Finance 20(4), section 2 and Theorem 3.1, make that
rotation count the object of study and prove that the `Re(d) >= 0` choice is
the one whose logarithm never leaves the principal branch for Heston. This
module implements that choice; the other one is implemented too, as
`characteristic_function_original_branch`, purely as a diagnostic, and
`tests/test_heston_fourier.py` measures what it costs.

One consequence of the factor above is worth stating before the tests find it:
the jump is **invisible whenever `2 kappa theta / xi^2` is an integer**, since
then `exp(-4 pi i kappa theta / xi^2) = 1`. That is not a safety property, it
is a coincidence, and it happens to hold exactly for the Alan Lewis reference
parameter set used throughout this repository's Heston tests.

Cumulants
---------
The COS truncation range needs `c1` and `c2` of `x_T = ln(S_T / S_0)`. Both are
derived here from the model rather than from the transform, which keeps them
independent evidence for the transform rather than a restatement of it.

Write `I = int_0^T v_s ds` and `M = int_0^T sqrt(v_s) dW1_s`, so that
`x_T = (r - q) T - I/2 + M`. Integrating the variance dynamics,

    v_t = theta + (v_0 - theta) e^{-kappa t}
               + xi int_0^t e^{-kappa (t - s)} sqrt(v_s) dW2_s,

and swapping the order of integration in `I = int_0^T v_t dt`,

    I - E[I] = (xi / kappa) int_0^T (1 - e^{-kappa (T - s)}) sqrt(v_s) dW2_s.

With `m(s) = E[v_s] = theta + (v_0 - theta) e^{-kappa s}` and
`g(s) = 1 - e^{-kappa (T - s)}`, the Ito isometry gives

    E[I]      = int_0^T m(s) ds                     =: vbar
    Var(I)    = (xi^2 / kappa^2) int_0^T g(s)^2 m(s) ds
    E[I M]    = (xi rho / kappa) int_0^T g(s) m(s) ds
    Var(M)    = E[I]

(`M` is a martingale with `E[M^2] = E[I]`, and `dW1 = rho dW2 + sqrt(1 - rho^2)
dW_perp` kills the orthogonal part of the covariance). Hence

    c1 = (r - q) T - vbar / 2
    c2 = Var(-I/2 + M) = vbar - (xi rho / kappa) J1 + (xi^2 / (4 kappa^2)) J2

with `J1 = int_0^T g m` and `J2 = int_0^T g^2 m` evaluated in closed form in
`heston_log_return_cumulants` below. Note what is *not* in any denominator:
`xi` appears only in numerators, so `c1` and `c2` degrade continuously to the
Black-Scholes values as `xi -> 0`. That is the direct repair of the Slice 14
oracle finding, where QuantLib's `COSHestonEngine` diverges in exactly that
limit because its range formulas carry the vol-of-vol below the line.

`c4` is reported as **zero**, which is what Fang and Oosterlee (2008, appendix)
do for Heston, and it is not a harmless simplification: the fourth cumulant of
this law is large -- larger than `c2` -- for Feller-violating parameters, and
the COS range `[c1 - L w, c1 + L w]` with `w = sqrt(c2 + sqrt(c4))` is then too
narrow at the default `L = 10`. The error that produces is a *range* error: it
is flat in the term count `N`, and it is repaired by raising `truncation_l`, by
the measured factor `sqrt(1 + sqrt(c4) / c2)`. `tests/test_heston_fourier.py`
measures all of that, `numerical_log_return_cumulant` computes the `c4` the
statement refers to, and `HESTON_TRUNCATION_L_FELLER_VIOLATED` records the
value that works.

Sources (nothing below is quoted or transcribed from any of them; the algebra
above is re-derived here):

- Heston (1993), "A closed-form solution for options with stochastic
  volatility", Review of Financial Studies 6(2), 327-343 -- the model and the
  original branch choice.
- Albrecher, Mayer, Schoutens and Tistaert (2007), "The little Heston trap",
  Wilmott Magazine (Jan 2007), 83-92, section 3 -- the stable form and the
  `|g| <= 1` argument.
- Lord and Kahl (2010), "Complex logarithms in Heston-like models",
  Mathematical Finance 20(4), 671-694, section 2 / Theorem 3.1 -- the rotation
  count and the proof that the principal branch suffices for Heston.
- Gatheral (2006), "The Volatility Surface", chapter 2 -- the same transform in
  his notation, and the smile properties measured in
  `tests/test_heston_smile.py`.
- Fang and Oosterlee (2008), SIAM J. Sci. Comput. 31(2), 826-848, appendix --
  the rule that the COS range is built from `c1`, `c2` and `c4`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import InvalidInputError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..engines.fourier.charfn import LogReturnCumulants

__all__ = [
    "HESTON_TRUNCATION_L_FELLER_VIOLATED",
    "HestonModel",
    "heston_characteristic_function",
    "heston_characteristic_function_original_branch",
    "heston_log_return_cumulants",
    "numerical_log_return_cumulant",
]

HESTON_TRUNCATION_L_FELLER_VIOLATED = 30.0
"""`FourierConfig.truncation_l` that the COS method needs when `c4 = 0` lies.

Measured, not chosen. At `v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1.0,
rho = -0.9` (the repository's own Feller-violating CIR parameters, Feller
number 0.08) and `T = 1`, the ATM call error against QuantLib is flat in `N`
at 9.1e-04 for `L = 10`, 1.6e-04 for `L = 12`, 4.6e-06 for `L = 16`,
1.3e-07 for `L = 20` and 2.3e-11 for `L = 30`. The predicted repair factor is
`sqrt(1 + sqrt(c4)/c2) = 2.80`, i.e. `L = 28`, and 30 is the first tested value
past it. See `tests/test_heston_fourier.py`."""


def _variance_integrals(
    expiry: float, *, v0: float, kappa: float, theta: float
) -> tuple[float, float, float]:
    """`(vbar, J1, J2)` -- the three deterministic integrals the cumulants need.

    With `m(s) = theta + (v0 - theta) e^{-kappa s}`,
    `g(s) = 1 - e^{-kappa (T - s)}` and `E = e^{-kappa T}`:

        vbar = int_0^T m      = theta T + (v0 - theta) (1 - E) / kappa
        J1   = int_0^T g m    = T (A - B E) + (B - A) (1 - E) / kappa
        J2   = int_0^T g^2 m  = T (A - 2 B E)
                                + (B (1 + E) - 2 A) (1 - E) / kappa
                                + A (1 - E^2) / (2 kappa)

    with `A = theta`, `B = v0 - theta`. Both `J` expressions come from
    expanding `g` and `g^2` in `e^{kappa s}` and using `E e^{kappa T} = 1`;
    the derivation is in this module's docstring.
    """
    a = kappa
    e = math.exp(-a * expiry)
    big_a = theta
    big_b = v0 - theta
    vbar = theta * expiry + big_b * (1.0 - e) / a
    j1 = expiry * (big_a - big_b * e) + (big_b - big_a) * (1.0 - e) / a
    j2 = (
        expiry * (big_a - 2.0 * big_b * e)
        + (big_b * (1.0 + e) - 2.0 * big_a) * (1.0 - e) / a
        + big_a * (1.0 - e * e) / (2.0 * a)
    )
    return vbar, j1, j2


def heston_log_return_cumulants(
    expiry: float,
    *,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> tuple[float, float]:
    """`(c1, c2)` of `ln(S_T / S_0)` under Heston (module docstring).

    Returns a plain tuple rather than a `LogReturnCumulants` so that this
    module imports nothing from `qpl.engines`; `HestonModel` wraps it.
    """
    vbar, j1, j2 = _variance_integrals(expiry, v0=v0, kappa=kappa, theta=theta)
    c1 = (rate - dividend) * expiry - 0.5 * vbar
    c2 = vbar - (xi * rho / kappa) * j1 + (xi * xi / (4.0 * kappa * kappa)) * j2
    return float(c1), float(c2)


def _deterministic_variance_transform(
    u: np.ndarray,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
) -> np.ndarray:
    """The `xi = 0` limit, in closed form rather than as a `0/0`.

    With no volatility of volatility the variance is deterministic,
    `v_t = theta + (v_0 - theta) e^{-kappa t}`, so `ln(S_T/S_0)` is Gaussian
    with variance `V = int_0^T v_t dt` and mean `(r - q) T - V/2`:

        phi(u) = exp( i u ((r - q) T - V / 2) - u^2 V / 2 )
               = exp( i u (r - q) T - (u^2 + i u) V / 2 ).

    Taking `xi -> 0` in the general formula reproduces exactly this (the
    module docstring's `D_-` tends to `-(u^2 + i u) / (2 kappa)` and the log
    term to `theta D_- (e^{-kappa T} - 1)`), so the branch below is the limit
    and not a separate model. In particular at `v0 = theta` it is the
    Black-Scholes transform with `sigma^2 = theta`, which is what
    `tests/test_heston_fourier.py` checks as an algebraic identity.
    """
    u_c = np.asarray(u, dtype=complex)
    variance = _variance_integrals(expiry, v0=v0, kappa=kappa, theta=theta)[0]
    return np.exp(
        1j * u_c * (rate - dividend) * expiry
        - 0.5 * (u_c * u_c + 1j * u_c) * variance
    )


def _complex_log1p(z: np.ndarray) -> np.ndarray:
    """`ln(1 + z)` for complex `z`, accurate when `|z|` is small.

    `np.log(1 + z)` loses every significant digit once `|z|` falls below the
    double-precision epsilon, and numpy's own `log1p` is real-accurate but not
    complex-accurate (`np.log1p(1e-18 + 1e-18j)` returns a zero real part).
    Splitting into modulus and argument fixes both halves:

        ln(1 + z) = (1/2) ln|1 + z|^2 + i arg(1 + z)
                  = (1/2) ln1p(2 Re z + |z|^2) + i atan2(Im z, 1 + Re z),

    where the real `log1p` and `atan2` are each accurate near their own
    singular points. Needed because `heston_characteristic_function` divides a
    logarithm by `xi^2`; see the note there.
    """
    z_c = np.asarray(z, dtype=complex)
    re = z_c.real
    im = z_c.imag
    return 0.5 * np.log1p(2.0 * re + re * re + im * im) + 1j * np.arctan2(
        im, 1.0 + re
    )


def _riccati_pieces(
    u: np.ndarray, *, kappa: float, xi: float, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(beta, d)` with `d` on the principal branch, so `Re(d) >= 0`."""
    u_c = np.asarray(u, dtype=complex)
    beta = kappa - rho * xi * 1j * u_c
    d = np.sqrt(beta * beta + xi * xi * (u_c * u_c + 1j * u_c))
    return beta, d


def heston_characteristic_function(
    u: np.ndarray,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> np.ndarray:
    """`E[exp(i u ln(S_T / S_0))]` in the Albrecher / Lord-Kahl stable form.

    `u` may be complex: Carr-Madan evaluates at `u - (alpha + 1) i`, Lewis on
    `Im(u) = -1/2` and Gil-Pelaez's share-measure probability at `u - i`.

    `xi = 0` is accepted and returns the deterministic-variance limit
    (`_deterministic_variance_transform`); `HestonModel` itself refuses it,
    because a model whose vol-of-vol is zero is Black-Scholes with a
    time-dependent variance and should be built as such. The branch exists so
    that the limit can be tested as an identity on the *formula*.
    """
    if xi == 0.0:
        return _deterministic_variance_transform(
            u, expiry, rate=rate, dividend=dividend, v0=v0, kappa=kappa, theta=theta
        )
    u_c = np.asarray(u, dtype=complex)
    beta, d = _riccati_pieces(u_c, kappa=kappa, xi=xi, rho=rho)
    # `beta - d` and `g` are written through `beta^2 - d^2 = -xi^2 (u^2 + i u)`
    # rather than as literal subtractions, and the logarithm through `log1p`.
    # Both rewrites exist for the same reason: the textbook transcription
    # multiplies a bracket that is `O(xi^2)` by `kappa theta / xi^2`, and the
    # subtraction `beta - d` loses `eps |beta|` absolutely, so the transform
    # carries an error of order `eps kappa^2 theta T / xi^2`. Measured on
    # `v0 = theta = 0.04, kappa = 1, rho = 0, T = 1` as
    # `max_u |phi_xi(u) - phi_0(u)|`, the literal form reads
    #
    #     xi     1e-03     1e-04     1e-05     1e-06     1e-07     1e-08
    #     err   1.15e-06  1.18e-08  2.15e-07  1.63e-05  1.72e-03  1.35e-01
    #
    # -- a minimum near `xi = 1e-04` and then divergence, which is the same
    # *shape* as Slice 14's QuantLib `COSHestonEngine` finding with a different
    # cause (there the truncation range, here the affine coefficients). The
    # form below has no `1 / xi^2` in it at all and the same column falls
    # monotonically to zero; `tests/test_heston_model.py` asserts the monotone
    # version down to `xi = 1e-12`.
    reciprocal = 1.0 / (beta + d)
    minus_root = -(u_c * u_c + 1j * u_c) * reciprocal  # = (beta - d) / xi^2
    g = (xi * xi) * minus_root * reciprocal  # = (beta - d) / (beta + d)
    decay = np.exp(-d * expiry)
    log_ratio = (_complex_log1p(-g * decay) - _complex_log1p(-g)) / (xi * xi)
    c_term = 1j * u_c * (rate - dividend) * expiry + kappa * theta * (
        minus_root * expiry - 2.0 * log_ratio
    )
    d_term = minus_root * (1.0 - decay) / (1.0 - g * decay)
    return np.exp(c_term + d_term * v0)


def heston_characteristic_function_original_branch(
    u: np.ndarray,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> np.ndarray:
    """**Diagnostic only.** The unstable branch choice; never used to price.

    This is the same algebra with `+d` in place of `-d` and `1/g` in place of
    `g` -- the form in Heston's own paper. It is mathematically identical to
    `heston_characteristic_function` and numerically is not: the logarithm's
    argument winds around the origin, each winding multiplies the result by
    `exp(-4 pi i kappa theta / xi^2)`, and the result is a discontinuous
    "characteristic function" whose integral is a wrong price. It is exported
    so that the failure can be *measured* rather than asserted, which is what
    `tests/test_heston_fourier.py` does.

    It also overflows for large `|u|` (`e^{+d T}` rather than `e^{-d T}`),
    which is a second, independent reason the form is unusable and is why the
    diagnostic tests keep `|u|` bounded.
    """
    if xi == 0.0:
        raise InvalidInputError("the original-branch diagnostic needs xi > 0")
    u_c = np.asarray(u, dtype=complex)
    beta, d = _riccati_pieces(u_c, kappa=kappa, xi=xi, rho=rho)
    numerator = beta + d
    # At `u = 0` the *other* root is the removable one: `beta - d = 0`, so this
    # form divides by zero at the one argument where `phi` is exactly 1. That
    # is the first symptom of picking the wrong root and it is patched rather
    # than reported, so that the branch-cut failure measured downstream is the
    # only failure in the comparison.
    zero = u_c == 0.0
    denominator = np.where(zero, 1.0, beta - d)
    g = numerator / denominator
    growth = np.exp(d * expiry)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c_term = 1j * u_c * (rate - dividend) * expiry + (
            kappa * theta / (xi * xi)
        ) * (numerator * expiry - 2.0 * np.log((1.0 - g * growth) / (1.0 - g)))
        d_term = (numerator / (xi * xi)) * (1.0 - growth) / (1.0 - g * growth)
        return np.where(zero, 1.0 + 0.0j, np.exp(c_term + d_term * v0))


def numerical_log_return_cumulant(
    cf,
    order: int,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    step: float,
) -> float:
    """**Diagnostic only.** The `order`-th cumulant by finite differences of `ln phi`.

    `c_n = i^{-n} d^n/du^n ln phi(u)` at `u = 0`, evaluated by the central
    difference of that order on the grid `u = 0, +/- step, ...`. Orders 1, 2
    and 4 are implemented, which is what the COS range rule reads.

    This exists for one purpose: to *measure* the `c4` that
    `HestonModel.log_return_cumulants` reports as zero, so that the cost of
    Fang and Oosterlee's simplification can be stated as a number instead of as
    a caveat. It is not used to price, and it is not a closed form -- the
    difference carries an `O(step^2)` truncation error and an
    `O(eps / step^order)` round-off floor, so `step` matters and the tests
    check the answer at several values of it.
    """
    stencils = {
        1: (np.array([-1.0, 1.0]), np.array([-0.5, 0.5])),
        2: (np.array([-1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0])),
        4: (
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.array([1.0, -4.0, 6.0, -4.0, 1.0]),
        ),
    }
    if order not in stencils:
        raise InvalidInputError(f"order must be one of {tuple(stencils)}")
    offsets, weights = stencils[order]
    values = np.log(
        np.asarray(
            cf.characteristic_function(
                offsets * step, expiry, rate=rate, dividend=dividend
            )
        )
    )
    derivative = np.sum(weights * values) / step**order
    return float((derivative / 1j**order).real)


@dataclass(frozen=True)
class HestonModel:
    """Heston stochastic volatility, as the Fourier engines see it.

    Parameters
    ----------
    v0
        Initial instantaneous variance, `>= 0` (a variance, not a volatility).
    kappa
        Mean-reversion speed of the variance, `> 0`.
    theta
        Long-run variance, `> 0`.
    xi
        Volatility of volatility, `> 0`. Zero is refused: the resulting model
        has a deterministic variance and is Black-Scholes with a time-dependent
        volatility, which this package represents as `BlackScholesModel`. The
        *limit* is still available on the free function
        `heston_characteristic_function`, which is how the tests check it.
    rho
        Correlation between the two Brownian motions, strictly inside
        `(-1, 1)`. The endpoints are refused because `rho = +/-1` makes the two
        drivers the same process, which is a degenerate model rather than a
        parameter value.

    Notes
    -----
    This class carries **no** pricing method. It is registered in
    `qpl.pricing` for `method="fourier"` only, and the four analytic, lattice,
    grid and simulation methods refuse it with a message naming that route.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        for name in ("v0", "kappa", "theta", "xi", "rho"):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise InvalidInputError(f"{name} must be finite")
        if self.v0 < 0.0:
            raise InvalidInputError("v0 must be >= 0")
        if self.kappa <= 0.0:
            raise InvalidInputError("kappa must be > 0")
        if self.theta <= 0.0:
            raise InvalidInputError("theta must be > 0")
        if self.xi <= 0.0:
            raise InvalidInputError("xi must be > 0")
        if not -1.0 < self.rho < 1.0:
            raise InvalidInputError("rho must satisfy -1 < rho < 1")

    @property
    def feller_number(self) -> float:
        """`4 kappa theta / xi^2`, the variance transition's ncx2 degrees of freedom.

        The Feller condition `2 kappa theta >= xi^2` is exactly `>= 2` in these
        units, matching `qpl.cases.sde_discretization.CIRSpec.feller_number`
        so that the two Feller regimes mean the same thing in both places.
        """
        return 4.0 * self.kappa * self.theta / (self.xi * self.xi)

    @property
    def feller_satisfied(self) -> bool:
        """Whether the variance process cannot reach zero."""
        return self.feller_number >= 2.0

    def moment_discriminant(self, moment: float) -> float:
        """`Delta(w)`, whose sign decides whether `E[S_T^w]` explodes at all.

        Substituting `u = -i w` into `d^2` (so that
        `phi(-i w) = E[(S_T/S_0)^w]`) gives

            Delta(w) = (kappa - rho xi w)^2 + xi^2 w (1 - w)
                     = xi^2 (rho^2 - 1) w^2 + (xi^2 - 2 kappa rho xi) w
                       + kappa^2,

        a downward parabola in `w` because `rho^2 < 1`. Where `Delta >= 0` the
        root `d` is real, `|g e^{-d tau}| < 1` for every `tau`, and the moment
        is finite for **all** maturities. Where `Delta < 0` the root is
        imaginary, the denominator `1 - g e^{-d tau}` rotates on the unit
        circle and reaches zero at a finite time: that is the moment explosion
        of Andersen and Piterbarg (2007), "Moment explosions in stochastic
        volatility models", Finance and Stochastics 11, section 3, re-derived
        here from this module's own `d` and `g`.
        """
        w = float(moment)
        return (self.kappa - self.rho * self.xi * w) ** 2 + self.xi * self.xi * w * (
            1.0 - w
        )

    def moment_explosion_time(self, moment: float) -> float:
        """`T*(w) = sup{T : E[S_T^w] < infinity}`; `inf` when `Delta(w) >= 0`.

        With `Delta(w) < 0` write `d = i D`, `D = sqrt(-Delta) > 0`, and
        `beta = kappa - rho xi w`. Then `g = (beta - i D)/(beta + i D)` has
        modulus one and equals `e^{-2 i psi}` with `psi = atan2(D, beta) in
        (0, pi)`, so

            1 - g e^{-d tau} = 1 - e^{-i (2 psi + D tau)}

        first vanishes at `D tau = 2 pi - 2 psi`, i.e.

            T*(w) = 2 (pi - psi) / D.
        """
        discriminant = self.moment_discriminant(moment)
        if discriminant >= 0.0:
            return math.inf
        magnitude = math.sqrt(-discriminant)
        beta = self.kappa - self.rho * self.xi * float(moment)
        return 2.0 * (math.pi - math.atan2(magnitude, beta)) / magnitude

    def critical_moment(self, expiry: float) -> float:
        """The largest `w` with `E[S_T^w] < infinity` at `T = expiry`.

        `T*` is `inf` on `[w_-, w_+]` (the roots of `Delta`) and decreases
        strictly from `inf` to `0` as `w` grows past `w_+`, so the answer is
        the unique root of `T*(w) = expiry` above `w_+`, found by bisection.

        Carr-Madan's damped transform needs `E[S_T^{alpha + 1}]` to be finite,
        so `critical_moment(T) - 1` is the largest usable `alpha`, and
        `tests/test_heston_fourier.py` measures where the method actually
        breaks against it.
        """
        if expiry <= 0.0:
            raise InvalidInputError("expiry must be > 0")
        quadratic_a = self.xi * self.xi * (self.rho * self.rho - 1.0)
        quadratic_b = self.xi * self.xi - 2.0 * self.kappa * self.rho * self.xi
        upper_root = (
            -quadratic_b
            - math.sqrt(quadratic_b * quadratic_b - 4.0 * quadratic_a * self.kappa**2)
        ) / (2.0 * quadratic_a)
        low = upper_root
        high = upper_root + 1.0
        while self.moment_explosion_time(high) > expiry:
            high += high - low
            if high > 1e12:  # pragma: no cover - unreachable for finite xi
                raise InvalidInputError("critical moment did not bracket")
        for _ in range(200):
            mid = 0.5 * (low + high)
            if self.moment_explosion_time(mid) > expiry:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    @property
    def spot_volatility(self) -> float:
        """`sqrt(v0)`: the instantaneous volatility, the unit vega is taken in."""
        return math.sqrt(self.v0)

    def with_volatility_bump(self, bump: float) -> HestonModel:
        """Move the *spot volatility* by `bump`, i.e. `v0 -> (sqrt(v0) + bump)^2`.

        Read by `qpl.engines.fourier.pricers` to produce a vega. Bumping `v0`
        directly would give `dV/dv0`, which is a derivative with respect to a
        variance and is not comparable with the Black-Scholes vega the rest of
        this package reports; bumping `sqrt(v0)` gives `dV/d sqrt(v0)`, which
        is. The metadata of a Heston `GreeksResult` says which one it is.
        """
        bumped = self.spot_volatility + bump
        if bumped < 0.0:
            raise InvalidInputError(
                f"volatility bump {bump} takes sqrt(v0) below zero"
            )
        return replace(self, v0=bumped * bumped)

    def characteristic_function(
        self,
        u: np.ndarray,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> np.ndarray:
        """`E[exp(i u ln(S_T / S_0))]`, stable form (module docstring)."""
        return heston_characteristic_function(
            u,
            expiry,
            rate=rate,
            dividend=dividend,
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )

    def characteristic_function_original_branch(
        self,
        u: np.ndarray,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> np.ndarray:
        """**Diagnostic only.** The unstable branch; see the free function."""
        return heston_characteristic_function_original_branch(
            u,
            expiry,
            rate=rate,
            dividend=dividend,
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )

    def log_return_cumulants(
        self,
        expiry: float,
        *,
        rate: float,
        dividend: float,
    ) -> LogReturnCumulants:
        """`c1`, `c2` in closed form and `c4 = 0` (module docstring).

        The import is deferred on purpose: `qpl.engines.fourier.charfn` imports
        `qpl.models.black_scholes`, so a module-level import here would close
        an import cycle through `qpl.models.__init__`.
        """
        from ..engines.fourier.charfn import LogReturnCumulants

        c1, c2 = heston_log_return_cumulants(
            expiry,
            rate=rate,
            dividend=dividend,
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )
        return LogReturnCumulants(c1=c1, c2=c2, c4=0.0)
