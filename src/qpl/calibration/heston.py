"""Calibrating Heston to a set of European quotes, and what the fit is worth.

This module ships one solver and one honest caveat. The solver is
`calibrate_heston`: a bounded Levenberg-Marquardt / trust-region least squares
(`scipy.optimize.least_squares`) over the five Heston parameters, pricing every
quote by the Slice 14 COS expansion and differentiating it by the **analytic
gradient of the characteristic function**. The caveat is that a calibration
which finds parameters is not evidence that the parameters are identified, and
this module is built so that the evidence for the second statement comes back
with the answer rather than having to be asked for: `CalibrationResult` carries
the Jacobian at the optimum, its singular values, the Gauss-Newton parameter
covariance and the condition number, because those are what say whether the
number in `model.kappa` means anything.

The objective
-------------
With parameters `p = (v0, kappa, theta, xi, rho)` and quotes `i = 1..n`,

    minimise  F(p) = (1/2) sum_i r_i(p)^2,
    r_i(p)   = w_i (m_i(p) - q_i)

where `m_i` and `q_i` are both prices (`objective="price"`) or both
Black-Scholes implied volatilities (`objective="implied_vol"`). Gatheral (2006)
chapter 3 argues for the implied-volatility objective, or equivalently for a
vega-weighted price objective, on the ground that a price objective is
dominated by the long-dated at-the-money cells where a price is large and a
volatility error is small; `vega_weights` builds the second of those and
`tests/test_heston_calibration.py` measures what each buys on the same noisy
data.

`objective="implied_vol"` costs one Brent inversion per quote per residual
evaluation on top of the transform price. That is measured, not feared: at the
grids used here it roughly doubles the cost of a residual evaluation and the
analytic Jacobian absorbs the difference, because `d sigma_i / dp` is
`(d V_i / dp) / vega_BS(sigma_i)` -- one division, no second inversion.

The pricer, and the setting a calibrator can get wrong
------------------------------------------------------
Every quote is priced by `qpl.engines.fourier.cos`, vectorised over the strikes
of one maturity: the characteristic function is evaluated once per maturity on
the `N`-point frequency grid and reused for every strike, which is what makes
a 5-strike by 3-maturity calibration take milliseconds rather than seconds.
COS is used rather than Lewis or Gil-Pelaez for that reason alone -- measured
here, 15 quotes cost **1.2 ms** by COS and **85 ms** by adaptive-quadrature
Lewis, a factor of 70, and a calibration makes hundreds of those calls.

Slice 15 warned that COS "needs a setting chosen per parameter set", and it is
right, but the warning turns out to be about the *payoff leg* first and the
range second.

The leg. A COS call's payoff coefficient integrates `e^z` over `[z*, b]`, so it
carries `e^b`; the cosine sum then has to cancel that against a price of order
`S_0`, and its round-off grows exponentially with the range's upper end. A COS
put's coefficient integrates over `[a, z*]`, so it carries `e^{z*} = K / S_0`
and nothing else -- but it is exposed to the **truncated left tail**, which for
`rho < 0` is where the missing mass is. Neither leg is uniformly better and
`b` is the variable that decides: see `COS_PUT_LEG_THRESHOLD` for the crossover
table and the rule. Picking the leg by `b` is also what makes the residual
function **total**: the put leg's coefficients cannot overflow, so a search in
the far corner of the parameter box gets a wrong price rather than `-3.2e+43`.
Measured over all fourteen starts of `qpl.cases.heston_calibration` and 200
uniform draws inside `DEFAULT_BOUNDS`, every price and every gradient entry is
finite and inside the no-arbitrage strip.

The range. With the leg chosen that way the round-off side of the failure is
gone and only the truncation side is left. Worst absolute error against a Lewis
integral of the same transform, over strikes 80-120 and maturities 0.25-2 on
the reference set (Feller number 4) and on the repository's Feller-violating
set (0.08):

    L,  N        Feller-satisfying      Feller-violating
    10,  256          4.92e-12               1.43e-02
    16, 1024          3.70e-11               3.53e-04
    24, 2048          3.23e-11               1.22e-06
    28, 4096          2.05e-12               7.74e-08
    32, 8192          8.16e-12               4.88e-09
    40,16384          2.13e-11               1.86e-11

The Feller-satisfying column is now flat at round-off instead of turning round
and diverging (before the leg rule it read 4.92e-12, 9.50e-10, 8.00e-08,
8.00e-08, 8.00e-08, 1.20e-07 *with* a range clip in place, and 5.17e-07 /
1.26e-05 / 2.30e-04 / 4.07e-02 without one). The Feller-violating column falls
monotonically, because `c4 = 0` makes the default range too narrow there and
the only repair is a wider one -- which is what
`HESTON_TRUNCATION_L_FELLER_VIOLATED` records.

So a single setting would now work: `L = 40, N = 16384` is 2.1e-11 on both. It
costs **64x** the term count of `L = 10, N = 256`, and `L = 28, N = 4096`
already costs 6.5x (9.1 ms against 1.4 ms for 30 quotes). `default_cos_settings`
is therefore a two-branch rule keyed on the Feller number, justified by run
time rather than by accuracy, and `calibrate_heston` resolves it **once**, from
the initial guess, holding it fixed for the whole solve: re-resolving per
evaluation would make the residual discontinuous exactly where a search crosses
`4 kappa theta = 2 xi^2`, and a discontinuous residual is not something a
Gauss-Newton step or a finite-difference Jacobian can be asked to handle. Pass
`cos=` to override.

The analytic Jacobian
---------------------
`heston_charfn_gradient` returns `phi(u)` together with
`d phi / d(v0, kappa, theta, xi, rho)` at the same `u`, from the closed-form
affine solution rather than by differencing. Writing the stable form used in
`qpl.models.heston`,

    phi = exp(C + D v0),   A = u^2 + i u,   beta = kappa - rho xi i u,
    d   = sqrt(beta^2 + xi^2 A),   s = beta + d,
    m   = -A / s        (= D_-, the Riccati root),
    g   = xi^2 m / s    (= (beta - d) / (beta + d)),   E = exp(-d T),
    Lh  = [ln(1 - g E) - ln(1 - g)] / xi^2,
    C   = i u (r - q) T + kappa theta (m T - 2 Lh),
    D   = m (1 - E) / (1 - g E),

only `beta` and `d` depend on `kappa`, `xi`, `rho`, and neither depends on
`v0` or `theta`. So two of the five derivatives are free:

    d phi / d v0    = phi D
    d phi / d theta = phi kappa (m T - 2 Lh)

and the other three follow from `d beta/dp` and `d d/dp = (beta d beta/dp +
xi A d xi/dp) / d` by the chain rule through `m`, `g`, `E`, `Lh`, `C` and `D`,
each written out in the code below. Source for the strategy and for the claim
that this is what makes a Heston calibration fast: Cui, del Bano Rollin and
Germano (2017), "Full and fast calibration of the Heston stochastic volatility
model", European Journal of Operational Research 263(2), 625-638, section 3.
Their paper differentiates their own rearrangement of the transform; the five
derivatives below are derived here for the rearrangement this repository
actually uses (theirs has no `g` at all), and are checked against central
differences rather than against their formulas.

The price gradient is then the same cosine sum with `phi` replaced by
`d phi / dp`, holding the truncation range `[a, b]` fixed. The range does move
with the parameters -- it is built from `c1` and `c2` -- but a COS price is
invariant to the range up to its truncation error, so the omitted term is the
derivative of an error measured at 5e-12. That is a claim, and it is measured:
against central differences **of the full pricer, range motion included**, the
analytic price gradient agrees to 1e-08 relative or better, which is the
difference quotient's own floor.

What is deliberately not here
-----------------------------
- No local-volatility, SABR or Bates calibration, and no market-data loading:
  the data strand is frozen (ADR-0004, D9) and the quotes are plain data.
- No penalty term, no regularisation, no Tikhonov prior. A regularised fit
  hides the flat direction that this slice exists to measure; the bounds are
  the only constraint and a Feller-violating optimum is **reported** through
  `CalibrationResult.feller_satisfied`, never forbidden.
- No global optimiser. `n_starts` is multi-start local search -- the cheapest
  thing that answers "did this land in the global basin", and the measured
  answer (`tests/test_heston_calibration.py`) is that from 12 spread starts a
  minority reach the best objective, so the multi-start is not a formality.

Sources (nothing below is quoted or transcribed; the algebra is re-derived
here):

- Gatheral (2006), "The Volatility Surface", chapter 3 -- calibration in
  practice, the objective choice, and the kappa/xi flat direction.
- Cui, del Bano Rollin and Germano (2017), EJOR 263(2), 625-638, section 3 --
  the analytic gradient strategy and the ill-conditioning along kappa-xi.
- Mikhailov and Nogel (2003), Wilmott Magazine (July), 74-79 -- Heston
  calibration as a least-squares problem and the role of the initial guess.
- Levenberg (1944), Quart. Appl. Math. 2, 164-168; Marquardt (1963), SIAM J.
  Appl. Math. 11(2), 431-441 -- the damped Gauss-Newton step, reached here
  through `scipy.optimize.least_squares`.
- Nocedal and Wright (2006), "Numerical Optimization" 2nd ed., chapter 10 --
  the Gauss-Newton covariance `sigma^2 (J^T J)^{-1}` and what it assumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..engines.analytic.black_scholes import implied_volatility
from ..engines.fourier.cos import chi_coefficients, psi_coefficients
from ..exceptions import InvalidInputError
from ..instruments.options import EuropeanOption
from ..market.market import Market
from ..models.black_scholes import bs_price
from ..models.heston import HestonModel, heston_log_return_cumulants

__all__ = [
    "CALIBRATION_METHODS",
    "COS_PUT_LEG_THRESHOLD",
    "DEFAULT_BOUNDS",
    "DOMAIN_PENALTY",
    "FELLER_SATISFIED_COS",
    "FELLER_VIOLATED_COS",
    "HESTON_PARAMETERS",
    "IMPLIED_VOL_BRACKET",
    "MULTISTART_SEED",
    "OBJECTIVES",
    "RANK_TOLERANCE",
    "VEGA_FLOOR",
    "CalibrationResult",
    "CosSettings",
    "OptionQuote",
    "StartSummary",
    "calibrate_heston",
    "cos_call_prices",
    "default_cos_settings",
    "heston_charfn_gradient",
    "heston_quote_values",
    "parameter_covariance",
    "residual_jacobian",
    "vega_weights",
]

HESTON_PARAMETERS: tuple[str, ...] = ("v0", "kappa", "theta", "xi", "rho")
"""Parameter order used by every array in this module, including the Jacobian's
columns and the covariance matrix's rows."""

OBJECTIVES: tuple[str, ...] = ("price", "implied_vol")
CALIBRATION_METHODS: tuple[str, ...] = ("trf", "lm")

RANK_TOLERANCE = float(np.finfo(float).eps)
"""Relative singular-value floor below which a Jacobian is called rank
deficient (`s_min <= RANK_TOLERANCE * max(n, p) * s_max`), the standard
relative rank test. Exposed because it is the line between "no error bar
exists" and "the error bar is enormous", and those are different findings."""

MULTISTART_SEED = 20260913
"""Seed for the multi-start draw, so `n_starts` is reproducible run to run."""


# --------------------------------------------------------------------------
# The COS settings, and the two-sided failure they have to straddle.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CosSettings:
    """`(n_terms, truncation_l)` for the COS expansion, held fixed per solve."""

    n_terms: int = 256
    truncation_l: float = 10.0

    def __post_init__(self) -> None:
        if self.n_terms < 2:
            raise InvalidInputError("n_terms must be >= 2")
        if not math.isfinite(self.truncation_l) or self.truncation_l <= 0.0:
            raise InvalidInputError("truncation_l must be finite and > 0")


FELLER_SATISFIED_COS = CosSettings(n_terms=256, truncation_l=10.0)
"""The package defaults, machine-accurate (4.9e-12) where `4 kappa theta >= 2 xi^2`."""

FELLER_VIOLATED_COS = CosSettings(n_terms=4096, truncation_l=28.0)
"""`HESTON_TRUNCATION_L_FELLER_VIOLATED` with the term count its decay exponent
(`N^2 / L^2`) needs. Worth 1.5e-07 on the Feller-violating set where the
defaults are worth 1.4e-02, and worth only 1.3e-05 on the Feller-satisfying
set where the defaults are at 4.9e-12 -- which is the round-off half of the
module docstring's table and the reason this is a branch and not a default."""


COS_PUT_LEG_THRESHOLD = 7.5
"""Range endpoint `b` above which the call is priced through the **put** leg.

Slice 15 measured the COS call/put asymmetry twice and got opposite answers,
and both are right, because the two legs fail for different reasons.

- The **call**'s payoff coefficient integrates `e^z` over `[z*, b]`, so it
  carries `e^b`. The cosine sum then has to cancel `e^b` against a price of
  order `S_0`, and its round-off grows exponentially with the range endpoint.
  No value of `L` repairs that -- widening the range to cover a fat tail makes
  it worse, which is the oracle finding in
  `tests/oracle/test_heston_vs_quantlib.py` (error rising monotonically through
  7.2e+00, 1.3e+01, 4.9e+01, 4.1e+02, 1.6e+05, 9.4e+08, 4.4e+15 as `L` goes 8
  to 50 on the Feller-violating set at `T = 10`).
- The **put**'s coefficient integrates `e^z` over `[a, z*]`, so it carries
  `e^{z*} = K / S_0` and nothing else, and is immune to that. What it is not
  immune to is the **truncated left tail**: with `rho < 0` the density is
  left-skewed, the mass the range misses sits below `a`, and that is the put's
  end. That is the finding in `tests/test_heston_fourier.py`, where at the
  package defaults the call is at 1.4e-12 and the put at 2.70e-08.

So the leg to use depends on the regime, and `b` is the variable that decides
it. Measured, worst absolute error over strikes 80-120 against a Lewis integral
of the same transform, at this module's own settings:

    b       set / T          call direct    put then parity
     1.73   reference 0.25     2.2e-13          3.6e-07
     2.95   violating 0.25     2.1e-12          1.7e-10
     4.56   reference 1        1.5e-12          2.7e-08
     6.69   violating 1        1.5e-10          4.8e-08
     8.49   reference 3        1.3e-10          1.6e-11
    12.80   reference 10       2.1e-08          2.1e-14
    14.13   violating 3        2.9e-05          1.9e-07
    16.00   violating 10       7.8e+01          1.2e-03

The crossover is between 6.69 and 8.49 and it is sharp; `7.5` is the midpoint
of that gap, and any threshold inside it picks the better leg in all eight
cells. The worst error over the eight drops from 7.8e+01 (call only) and
3.6e-07 (put only) to 1.2e-03, and over the six cells with `T <= 3` -- the
region a calibration actually visits -- from 2.9e-05 and 3.6e-07 to 1.9e-07.

This is the one place this module does something `qpl.engines.fourier.cos`
deliberately does not. Slice 15 recorded the put-then-parity route as a recipe
and refused to put it in the engine, because doing so would destroy the one
test where put-call parity is a real check on the COS coefficients rather than
an identity of the implementation. That argument is about the engine's test
coverage and does not apply to a residual function, so the recipe is applied
here and the engine is left alone."""


def default_cos_settings(parameters) -> CosSettings:
    """Pick the COS settings from the Feller number of one parameter vector.

    `4 kappa theta / xi^2 >= 2` selects `FELLER_SATISFIED_COS`, otherwise
    `FELLER_VIOLATED_COS`. Called **once** per solve by `calibrate_heston`, on
    the initial guess: see the module docstring for why the alternative (per
    evaluation) is worse than the error it would fix.
    """
    _v0, kappa, theta, xi, _rho = (float(x) for x in parameters)
    if xi <= 0.0 or kappa <= 0.0 or theta <= 0.0:
        return FELLER_VIOLATED_COS
    feller = 4.0 * kappa * theta / (xi * xi)
    return FELLER_SATISFIED_COS if feller >= 2.0 else FELLER_VIOLATED_COS


# --------------------------------------------------------------------------
# The transform and its gradient.
# --------------------------------------------------------------------------


def _complex_log1p(z: np.ndarray) -> np.ndarray:
    """`ln(1 + z)` for complex `z` (`qpl.models.heston._complex_log1p`).

    Duplicated rather than imported because it is four lines and importing a
    private name across packages is a worse dependency than a copy whose two
    versions are asserted equal in `tests/test_heston_calibration.py`.
    """
    re = z.real
    im = z.imag
    return 0.5 * np.log1p(2.0 * re + re * re + im * im) + 1j * np.arctan2(im, 1.0 + re)


def heston_charfn_gradient(
    u,
    expiry: float,
    *,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`(phi, dphi)` with `dphi[j]` the derivative in `HESTON_PARAMETERS[j]`.

    `phi` reproduces `qpl.models.heston.heston_characteristic_function` to
    round-off (asserted to 1e-15 in the tests); it is recomputed here rather
    than called, because every intermediate (`beta`, `d`, `m`, `g`, `E`, the
    log term) is needed again by the derivatives and evaluating the transform
    twice would double the cost of the one thing this module does most.

    The `xi = 0` limit is **not** handled: `calibrate_heston` keeps `xi` above
    a positive bound, and a deterministic-variance Heston is Black-Scholes
    (`qpl.models.heston` says the same thing about `HestonModel` itself).
    """
    u_c = np.asarray(u, dtype=complex)
    t = float(expiry)
    iu = 1j * u_c
    a_coef = u_c * u_c + iu
    beta = kappa - rho * xi * iu
    d = np.sqrt(beta * beta + xi * xi * a_coef)
    s = beta + d
    m = -a_coef / s
    g = (xi * xi) * m / s
    decay = np.exp(-d * t)
    log_hat = (_complex_log1p(-g * decay) - _complex_log1p(-g)) / (xi * xi)
    c_fn = iu * (rate - dividend) * t + kappa * theta * (m * t - 2.0 * log_hat)
    d_fn = m * (1.0 - decay) / (1.0 - g * decay)
    phi = np.exp(c_fn + d_fn * v0)

    grad = np.empty((5, u_c.size), dtype=complex)
    # v0 enters only through `D v0`; theta only through the factor
    # `kappa theta` multiplying a bracket that does not contain it.
    grad[0] = phi * d_fn
    grad[2] = phi * kappa * (m * t - 2.0 * log_hat)

    d_beta = {"kappa": np.ones_like(u_c), "xi": -rho * iu, "rho": -xi * iu}
    d_d = {
        "kappa": beta / d,
        "xi": (beta * (-rho * iu) + xi * a_coef) / d,
        "rho": beta * (-xi * iu) / d,
    }
    for index, name in ((1, "kappa"), (3, "xi"), (4, "rho")):
        db = d_beta[name]
        dd = d_d[name]
        ds = db + dd
        dm = a_coef * ds / (s * s)
        # d g = 2 (d dbeta - beta dd) / s^2, from g = (beta - d)/(beta + d).
        dg = 2.0 * (d * db - beta * dd) / (s * s)
        d_decay = -t * decay * dd
        d_log = -(dg * decay + g * d_decay) / (1.0 - g * decay) + dg / (1.0 - g)
        if name == "xi":
            # d/dxi of (L / xi^2) picks up the explicit xi^{-2}.
            d_log_hat = d_log / (xi * xi) - 2.0 * log_hat / xi
        else:
            d_log_hat = d_log / (xi * xi)
        if name == "kappa":
            d_c = theta * (m * t - 2.0 * log_hat) + kappa * theta * (
                dm * t - 2.0 * d_log_hat
            )
        else:
            d_c = kappa * theta * (dm * t - 2.0 * d_log_hat)
        numerator = -d_decay * (1.0 - g * decay) + (1.0 - decay) * (
            dg * decay + g * d_decay
        )
        d_d_fn = dm * (1.0 - decay) / (1.0 - g * decay) + m * numerator / (
            (1.0 - g * decay) ** 2
        )
        grad[index] = phi * (d_c + v0 * d_d_fn)
    return phi, grad


def cos_call_prices(
    parameters,
    *,
    s0: float,
    strikes,
    expiry: float,
    rate: float,
    dividend: float,
    settings: CosSettings,
    gradient: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """COS **call** prices at one maturity, vectorised over strikes.

    Returns `(values, gradients)` with `gradients` of shape `(5, n_strikes)`
    when `gradient=True` and `None` otherwise. The characteristic function is
    evaluated once on the frequency grid and reused for every strike, which is
    the whole reason a calibration is cheap.

    The call is always what comes back, but which cosine sum produces it is
    decided by `COS_PUT_LEG_THRESHOLD`: see that constant for the measurement.
    """
    v0, kappa, theta, xi, rho = (float(x) for x in parameters)
    c1, c2 = heston_log_return_cumulants(
        expiry,
        rate=rate,
        dividend=dividend,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )
    width = settings.truncation_l * math.sqrt(abs(c2))
    if not math.isfinite(width) or width <= 0.0:
        raise InvalidInputError(
            "COS truncation range is degenerate: the log return has zero variance"
        )
    a = c1 - width
    b = c1 + width
    k = np.arange(settings.n_terms, dtype=float)
    u = k * math.pi / (b - a)
    phi, dphi = heston_charfn_gradient(
        u,
        expiry,
        rate=rate,
        dividend=dividend,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )
    shift = np.exp(-1j * u * a)
    weights = np.real(phi * shift)
    weights[0] *= 0.5
    if gradient:
        d_weights = np.real(dphi * shift[None, :])
        d_weights[:, 0] *= 0.5
    discount = math.exp(-rate * expiry)
    scale = 2.0 / (b - a)

    use_put_leg = b >= COS_PUT_LEG_THRESHOLD
    spot_leg = s0 * math.exp(-dividend * expiry)

    strike_grid = np.asarray(strikes, dtype=float)
    values = np.empty(strike_grid.size)
    gradients = np.zeros((5, strike_grid.size)) if gradient else None
    for j, strike in enumerate(strike_grid):
        boundary = min(max(math.log(strike / s0), a), b)
        if use_put_leg:
            chi = chi_coefficients(k, a, boundary, a, b)
            psi = psi_coefficients(k, a, boundary, a, b)
            payoff_k = scale * (strike * psi - s0 * chi)
        else:
            chi = chi_coefficients(k, boundary, b, a, b)
            psi = psi_coefficients(k, boundary, b, a, b)
            payoff_k = scale * (s0 * chi - strike * psi)
        value = discount * float(np.sum(weights * payoff_k))
        if use_put_leg:
            # C = P + S e^{-qT} - K e^{-rT}. Both corrections are free of the
            # Heston parameters, so the gradient is the put's unchanged.
            value = value + spot_leg - strike * discount
        values[j] = value
        if gradient:
            gradients[:, j] = discount * (d_weights @ payoff_k)
    return values, gradients


# --------------------------------------------------------------------------
# Quotes.
# --------------------------------------------------------------------------

QUOTE_KINDS: tuple[str, ...] = ("call", "put")
QUOTE_VALUE_TYPES: tuple[str, ...] = ("implied_vol", "price")


@dataclass(frozen=True)
class OptionQuote:
    """One European quote: a strike, a maturity, a kind, and a number.

    `value_type="implied_vol"` (the default) means `value` is a Black-Scholes
    implied volatility; `"price"` means it is a premium in the market's
    currency. Both are converted once at setup into *both* representations, so
    the objective can be evaluated in either space without the caller having to
    restate the quote.

    `weight` multiplies this quote's residual. It is a per-quote weight and not
    a per-parameter one; `vega_weights` builds the Gatheral weighting from a
    list of quotes.
    """

    strike: float
    expiry: float
    kind: str
    value: float
    value_type: str = "implied_vol"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.strike) or self.strike <= 0.0:
            raise InvalidInputError("strike must be finite and > 0")
        if not math.isfinite(self.expiry) or self.expiry <= 0.0:
            raise InvalidInputError(
                "expiry must be finite and > 0: a transform price needs a "
                "terminal law with a density, and at T = 0 it is a point mass"
            )
        if self.kind not in QUOTE_KINDS:
            raise InvalidInputError(f"kind must be one of {QUOTE_KINDS}")
        if self.value_type not in QUOTE_VALUE_TYPES:
            raise InvalidInputError(f"value_type must be one of {QUOTE_VALUE_TYPES}")
        if not math.isfinite(self.value) or self.value <= 0.0:
            raise InvalidInputError("value must be finite and > 0")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise InvalidInputError("weight must be finite and > 0")


def _forward_legs(market: Market, strike: float, expiry: float) -> tuple[float, float]:
    """`(S e^{-qT}, K e^{-rT})`, the two legs of put-call parity."""
    return (
        market.spot * math.exp(-market.dividend_yield(expiry) * expiry),
        strike * math.exp(-market.rate(expiry) * expiry),
    )


IMPLIED_VOL_BRACKET = (1e-06, 5.0)
"""The bracket `implied_volatility` searches, restated here so a price can be
clamped into the interval it can actually invert."""

VEGA_FLOOR = 1e-06
"""Smallest vega used in the implied-volatility chain rule.

`d sigma_i / dp = (d V_i / dp) / vega_i` is exact wherever `vega_i > 0`, and a
search that has wandered to parameters where a cell prices at its intrinsic
value has `vega_i` underflowing to zero there. The floor keeps the Jacobian
finite so the optimiser can step back out. It is a guard on the *path*, not on
the answer: `tests/test_heston_calibration.py` asserts it does not bind at any
converged fit reported here."""


def _implied_vol_or_bound(price: float, quote: OptionQuote, market: Market) -> float:
    """Invert to a Black-Scholes volatility, clamping into the invertible range.

    A search that has wandered somewhere silly can produce a price outside the
    no-arbitrage bounds -- or inside them but outside `[BS(sigma_lo),
    BS(sigma_hi)]` -- where `implied_volatility` correctly refuses to bracket.
    Refusing here would abort the solve rather than push it back, so the price
    is clamped a hair inside the interval the inverter can reach and the result
    is a finite number the optimiser can descend from. On a converged fit the
    clamp never binds, which is asserted in the tests.
    """
    option = EuropeanOption(kind=quote.kind, strike=quote.strike, expiry=quote.expiry)
    lo_vol, hi_vol = IMPLIED_VOL_BRACKET
    rate = market.rate(quote.expiry)
    dividend = market.dividend_yield(quote.expiry)
    low = float(
        bs_price(
            S=market.spot, K=quote.strike, T=quote.expiry, r=rate,
            sigma=lo_vol, q=dividend, kind=quote.kind,
        )
    )
    high = float(
        bs_price(
            S=market.spot, K=quote.strike, T=quote.expiry, r=rate,
            sigma=hi_vol, q=dividend, kind=quote.kind,
        )
    )
    span = high - low
    clamped = min(max(price, low + 1e-09 * span), high - 1e-09 * span)
    return implied_volatility(clamped, option, market, lower=lo_vol, upper=hi_vol)


def heston_quote_values(
    model: HestonModel | Any,
    market: Market,
    quotes,
    *,
    settings: CosSettings | None = None,
) -> np.ndarray:
    """Transform prices of `quotes` under `model`, in quote order.

    A convenience over `cos_call_prices` for callers who want the fitted
    model's prices without re-deriving the maturity grouping or the parity
    step. `settings` defaults to `default_cos_settings` on the model's own
    parameters.
    """
    parameters = _parameters_of(model)
    resolved = settings or default_cos_settings(parameters)
    quote_list = list(quotes)
    groups = _group_by_expiry(quote_list)
    values = np.empty(len(quote_list))
    for expiry, index, strikes in groups:
        calls, _ = cos_call_prices(
            parameters,
            s0=market.spot,
            strikes=strikes,
            expiry=expiry,
            rate=market.rate(expiry),
            dividend=market.dividend_yield(expiry),
            settings=resolved,
        )
        for position, quote_index in enumerate(index):
            quote = quote_list[quote_index]
            value = calls[position]
            if quote.kind == "put":
                spot_leg, strike_leg = _forward_legs(market, quote.strike, expiry)
                value = value - spot_leg + strike_leg
            values[quote_index] = value
    return values


def _parameters_of(model) -> tuple[float, ...]:
    return tuple(float(getattr(model, name)) for name in HESTON_PARAMETERS)


def _group_by_expiry(quotes) -> list[tuple[float, list[int], np.ndarray]]:
    """`[(expiry, [quote indices], strikes)]`, one entry per distinct maturity.

    The grouping is the optimisation: one characteristic-function evaluation
    per maturity instead of one per quote.
    """
    order: dict[float, list[int]] = {}
    for index, quote in enumerate(quotes):
        order.setdefault(quote.expiry, []).append(index)
    return [
        (expiry, index, np.array([quotes[i].strike for i in index], dtype=float))
        for expiry, index in sorted(order.items())
    ]


def vega_weights(quotes, market: Market) -> np.ndarray:
    """`1 / vega` at each quote's own implied volatility (Gatheral chapter 3).

    A price residual divided by vega *is* an implied-volatility residual to
    first order, so this is the cheap version of `objective="implied_vol"`:
    the same weighting of the cells, with no Brent inversion per evaluation.
    Where the two differ is second order in the residual, and the tests measure
    that difference rather than assuming it away.
    """
    weights = np.empty(len(quotes))
    for index, quote in enumerate(quotes):
        rate = market.rate(quote.expiry)
        dividend = market.dividend_yield(quote.expiry)
        vol = (
            quote.value
            if quote.value_type == "implied_vol"
            else _implied_vol_or_bound(quote.value, quote, market)
        )
        weights[index] = 1.0 / _bs_vega(
            s0=market.spot,
            strike=quote.strike,
            expiry=quote.expiry,
            rate=rate,
            dividend=dividend,
            sigma=vol,
        )
    return weights


def _bs_vega(*, s0: float, strike: float, expiry: float, rate: float,
             dividend: float, sigma: float) -> float:
    """`S e^{-qT} n(d1) sqrt(T)`, the same for a call and a put."""
    sqrt_t = math.sqrt(expiry)
    vol = max(float(sigma), IMPLIED_VOL_BRACKET[0])
    d1 = (
        math.log(s0 / strike) + (rate - dividend + 0.5 * vol * vol) * expiry
    ) / (vol * sqrt_t)
    vega = (
        s0
        * math.exp(-dividend * expiry)
        * math.exp(-0.5 * d1 * d1)
        / math.sqrt(2.0 * math.pi)
        * sqrt_t
    )
    return max(vega, VEGA_FLOOR)


# --------------------------------------------------------------------------
# Bounds and the domain.
# --------------------------------------------------------------------------

DEFAULT_BOUNDS: tuple[tuple[float, ...], tuple[float, ...]] = (
    (1e-06, 1e-03, 1e-06, 1e-03, -0.999),
    (1.0, 20.0, 1.0, 5.0, 0.999),
)
"""`(lower, upper)` in `HESTON_PARAMETERS` order.

These keep `v0, kappa, theta, xi > 0` and `|rho| < 1` -- the conditions
`HestonModel` itself enforces -- and nothing else. In particular they do **not**
impose the Feller condition: a Feller-violating optimum is a real feature of
equity smiles, it is reported through `CalibrationResult.feller_satisfied`, and
forbidding it would be an unstated prior rather than a constraint."""

_DOMAIN_FLOOR = 1e-08
_RHO_CEILING = 1.0 - 1e-08
DOMAIN_PENALTY = 1.0e03
"""Residual added per unit of excursion outside the Heston domain.

Only `method="lm"` can reach it: `least_squares(method='lm')` takes no bounds,
so an unconstrained run is free to step to a negative `kappa` or `theta`, or a
`|rho| > 1`, where the transform is not a characteristic function of anything.
The residual function stays *total* by pricing at the clamped parameters and
adding `DOMAIN_PENALTY * excursion` to every component -- continuous, zero
inside the domain, and with a gradient that points back in.

It is a guard on the evaluation, not a soft constraint on the answer, and the
difference is measured: from `(0.04, 0.05, 0.25, 3.0, -0.95)` the
unconstrained route still converges (by its own `xtol`) to `theta = -1.70`, so
`CalibrationResult.model` comes back as `None` and `in_domain` as `False`.
The bounded run from the same start recovers the truth. `tests/test_heston_calibration.py`
pins both."""


def _clamp_and_violation(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Parameters clamped into the evaluation domain, and how far outside."""
    clamped = np.array(x, dtype=float)
    violation = 0.0
    for index in (0, 1, 2, 3):
        if clamped[index] < _DOMAIN_FLOOR:
            violation += _DOMAIN_FLOOR - clamped[index]
            clamped[index] = _DOMAIN_FLOOR
    if abs(clamped[4]) > _RHO_CEILING:
        violation += abs(clamped[4]) - _RHO_CEILING
        clamped[4] = math.copysign(_RHO_CEILING, clamped[4])
    return clamped, violation


def _in_domain(x) -> bool:
    v0, kappa, theta, xi, rho = (float(v) for v in x)
    return (
        v0 > 0.0
        and kappa > 0.0
        and theta > 0.0
        and xi > 0.0
        and abs(rho) < 1.0
        and all(math.isfinite(v) for v in (v0, kappa, theta, xi, rho))
    )


# --------------------------------------------------------------------------
# The residual problem.
# --------------------------------------------------------------------------


class _Problem:
    """Residuals, Jacobian and the bookkeeping both need, for one quote set.

    Not exported. It exists so that `calibrate_heston` and its multi-start
    loop share one cached setup (the maturity grouping, the target prices and
    volatilities, the weights) instead of rebuilding it per start.
    """

    def __init__(self, quotes, market: Market, *, objective: str,
                 weights, settings: CosSettings) -> None:
        self.quotes = list(quotes)
        self.market = market
        self.objective = objective
        self.settings = settings
        self.groups = _group_by_expiry(self.quotes)
        self.n_evals = 0
        self.n_jac_evals = 0

        self.target_price = np.empty(len(self.quotes))
        self.target_vol = np.empty(len(self.quotes))
        for index, quote in enumerate(self.quotes):
            rate = market.rate(quote.expiry)
            dividend = market.dividend_yield(quote.expiry)
            if quote.value_type == "implied_vol":
                self.target_vol[index] = quote.value
                self.target_price[index] = float(
                    bs_price(
                        S=market.spot,
                        K=quote.strike,
                        T=quote.expiry,
                        r=rate,
                        sigma=quote.value,
                        q=dividend,
                        kind=quote.kind,
                    )
                )
            else:
                self.target_price[index] = quote.value
                self.target_vol[index] = _implied_vol_or_bound(
                    quote.value, quote, market
                )

        quote_weights = np.array([q.weight for q in self.quotes], dtype=float)
        if weights is None:
            self.weights = quote_weights
        else:
            supplied = np.asarray(weights, dtype=float)
            if supplied.shape != (len(self.quotes),):
                raise InvalidInputError(
                    f"weights must have shape ({len(self.quotes)},)"
                )
            if not np.all(np.isfinite(supplied)) or np.any(supplied <= 0.0):
                raise InvalidInputError("weights must be finite and > 0")
            self.weights = supplied * quote_weights

    # -- pricing ---------------------------------------------------------
    def prices(self, parameters, *, gradient: bool = False):
        n = len(self.quotes)
        values = np.empty(n)
        gradients = np.zeros((5, n)) if gradient else None
        for expiry, index, strikes in self.groups:
            calls, call_grad = cos_call_prices(
                parameters,
                s0=self.market.spot,
                strikes=strikes,
                expiry=expiry,
                rate=self.market.rate(expiry),
                dividend=self.market.dividend_yield(expiry),
                settings=self.settings,
                gradient=gradient,
            )
            for position, quote_index in enumerate(index):
                quote = self.quotes[quote_index]
                value = calls[position]
                if quote.kind == "put":
                    spot_leg, strike_leg = _forward_legs(
                        self.market, quote.strike, expiry
                    )
                    value = value - spot_leg + strike_leg
                values[quote_index] = value
                if gradient:
                    # Parity's two legs are parameter-free, so a put and a
                    # call at the same cell have the *same* gradient.
                    gradients[:, quote_index] = call_grad[:, position]
        return values, gradients

    def implied_vols(self, prices) -> np.ndarray:
        return np.array(
            [
                _implied_vol_or_bound(float(prices[i]), quote, self.market)
                for i, quote in enumerate(self.quotes)
            ]
        )

    # -- least-squares interface -----------------------------------------
    def residuals(self, x) -> np.ndarray:
        self.n_evals += 1
        clamped, violation = _clamp_and_violation(np.asarray(x, dtype=float))
        prices, _ = self.prices(clamped)
        if self.objective == "price":
            base = self.weights * (prices - self.target_price)
        else:
            base = self.weights * (self.implied_vols(prices) - self.target_vol)
        if violation > 0.0:
            base = base + DOMAIN_PENALTY * violation
        return base

    def jacobian(self, x) -> np.ndarray:
        self.n_jac_evals += 1
        clamped, _ = _clamp_and_violation(np.asarray(x, dtype=float))
        prices, gradients = self.prices(clamped, gradient=True)
        if self.objective == "price":
            return (self.weights[None, :] * gradients).T
        vols = self.implied_vols(prices)
        vega = np.array(
            [
                _bs_vega(
                    s0=self.market.spot,
                    strike=quote.strike,
                    expiry=quote.expiry,
                    rate=self.market.rate(quote.expiry),
                    dividend=self.market.dividend_yield(quote.expiry),
                    sigma=vols[i],
                )
                for i, quote in enumerate(self.quotes)
            ]
        )
        return (self.weights[None, :] * gradients / vega[None, :]).T


# --------------------------------------------------------------------------
# Diagnostics.
# --------------------------------------------------------------------------


def residual_jacobian(
    quotes,
    market: Market,
    parameters,
    *,
    objective: Literal["price", "implied_vol"] = "price",
    weights=None,
    cos: CosSettings | None = None,
):
    """`(residuals, jacobian)` at one parameter vector, without running a solve.

    The same objects `calibrate_heston` reports at its optimum, evaluated
    anywhere. This is what makes the identifiability claims checkable at the
    **true** parameters rather than only at a fitted point: a condition number
    measured where the answer is known is a statement about the experiment
    design (which strikes, which maturities), while one measured at a fitted
    point also carries the fit's own error.
    """
    if objective not in OBJECTIVES:
        raise InvalidInputError(f"objective must be one of {OBJECTIVES}")
    x = np.asarray(parameters, dtype=float)
    if x.shape != (5,):
        raise InvalidInputError(
            f"parameters must be 5 numbers in the order {HESTON_PARAMETERS}"
        )
    settings = cos if cos is not None else default_cos_settings(x)
    problem = _Problem(
        list(quotes), market, objective=objective, weights=weights, settings=settings
    )
    return problem.residuals(x), problem.jacobian(x)


def parameter_covariance(jacobian, residuals):
    """Gauss-Newton covariance, its condition number and its singular values.

    Returns `(covariance, condition_number, singular_values, standard_errors)`.

    With `J` the residual Jacobian at the optimum and `r` the residuals,

        sigma^2 = r^T r / (n - p),      cov = sigma^2 (J^T J)^{-1}

    (Nocedal and Wright (2006) chapter 10). That is the linearised, large-
    sample covariance of the estimate under independent errors of equal
    variance in the **residual's own units**: it is a statement about the
    curvature of the objective, not a proof that the errors are independent or
    Gaussian. `tests/test_heston_calibration.py` checks it the only way it can
    be checked -- by resampling the noise and comparing the spread of the
    fitted parameters against these numbers.

    `condition_number` is `s_max / s_min` of `J`, so the covariance's own
    condition number is its square. A **numerically** rank-deficient `J` --
    `s_min <= RANK_TOLERANCE * max(n, p) * s_max`, the usual relative rank
    test -- returns infinities rather than a pseudo-inverse: a flat direction
    is the finding, and hiding it behind a pseudo-inverse would report a finite
    error bar for a parameter the data does not determine. Note that the
    single-maturity Jacobian measured in `tests/test_heston_calibration.py` is
    *not* rank deficient by this test (its condition number is 6.7e+07, twelve
    decimal orders inside the floor); it is ill-conditioned, which is a
    quantitative statement and the one this slice is about.
    """
    j = np.asarray(jacobian, dtype=float)
    r = np.asarray(residuals, dtype=float)
    n_obs, n_par = j.shape
    singular_values = np.linalg.svd(j, compute_uv=False)
    smallest = float(singular_values[-1])
    largest = float(singular_values[0])
    rank_floor = RANK_TOLERANCE * max(n_obs, n_par) * largest
    deficient = smallest <= rank_floor
    condition = math.inf if deficient else largest / smallest
    dof = n_obs - n_par
    variance = float(r @ r) / dof if dof > 0 else math.nan
    if deficient:
        covariance = np.full((n_par, n_par), math.inf)
        return covariance, condition, singular_values, np.full(n_par, math.inf)
    gram_inverse = np.linalg.pinv(j.T @ j, rcond=0.0)
    covariance = variance * gram_inverse
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return covariance, condition, singular_values, standard_errors


# --------------------------------------------------------------------------
# Results.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StartSummary:
    """One local solve inside a multi-start run."""

    initial: tuple[float, ...]
    parameters: tuple[float, ...]
    objective: float
    success: bool
    n_iter: int
    n_evals: int
    message: str


@dataclass(frozen=True)
class CalibrationResult:
    """A fitted model and everything needed to decide whether to believe it."""

    model: HestonModel | None
    parameters: tuple[float, ...]
    objective: float
    objective_kind: str
    residuals: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray
    condition_number: float
    singular_values: np.ndarray
    standard_errors: np.ndarray
    n_iter: int
    n_evals: int
    success: bool
    message: str
    feller_number: float
    feller_satisfied: bool
    in_domain: bool
    cos_settings: CosSettings
    jac_mode: str
    method: str
    starts: tuple[StartSummary, ...] = field(default_factory=tuple)

    @property
    def rmse(self) -> float:
        """Root-mean-square residual in the objective's own units."""
        return float(math.sqrt(2.0 * self.objective / self.residuals.size))

    def parameter(self, name: str) -> float:
        return self.parameters[HESTON_PARAMETERS.index(name)]

    def standard_error(self, name: str) -> float:
        return float(self.standard_errors[HESTON_PARAMETERS.index(name)])


# --------------------------------------------------------------------------
# The solver.
# --------------------------------------------------------------------------


def _multistart_points(initial, bounds, n_starts: int, seed: int):
    """`initial` first, then `n_starts - 1` uniform draws inside the bounds.

    Uniform rather than latin-hypercube or Sobol because the measured failure
    rate is what this reports, and a low-discrepancy sequence would make the
    number a property of the sequence. Reproducible at a fixed `seed`.
    """
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    rng = np.random.default_rng(seed)
    points = [np.asarray(initial, dtype=float)]
    for _ in range(max(0, n_starts - 1)):
        points.append(lower + rng.random(5) * (upper - lower))
    return points


def calibrate_heston(
    quotes,
    market: Market,
    *,
    initial,
    objective: Literal["price", "implied_vol"] = "price",
    weights=None,
    bounds=None,
    method: Literal["trf", "lm"] = "trf",
    jac: Literal["analytic", "2-point", "3-point"] = "analytic",
    n_starts: int = 1,
    starts=None,
    cos: CosSettings | None = None,
    seed: int = MULTISTART_SEED,
    max_nfev: int | None = None,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    gtol: float = 1e-12,
) -> CalibrationResult:
    """Fit `HestonModel` to `quotes` and report how well the fit is determined.

    Parameters
    ----------
    quotes
        A sequence of `OptionQuote`. Strikes may repeat across maturities and
        maturities across strikes; nothing is assumed about the grid's shape.
    initial
        Starting parameters in `HESTON_PARAMETERS` order.
    objective
        `"price"` or `"implied_vol"`. See the module docstring; `vega_weights`
        gives the third option (`"price"` weighted by `1/vega`).
    bounds
        `(lower, upper)` tuples in parameter order, defaulting to
        `DEFAULT_BOUNDS`. Must be `None` for `method="lm"`, which is the
        unconstrained route and refuses bounds rather than ignoring them.
    method
        `"trf"` (bounded trust-region) or `"lm"` (Levenberg-Marquardt,
        unconstrained). Both are `scipy.optimize.least_squares`.
    jac
        `"analytic"` uses `heston_charfn_gradient`; `"2-point"` and
        `"3-point"` hand the choice to scipy's differencing. The analytic
        Jacobian is the default because it is both faster and more accurate,
        and the tests measure both claims.
    n_starts, starts, seed
        Multi-start. `starts` is an explicit sequence of starting vectors and
        takes precedence; `n_starts > 1` draws `n_starts - 1` uniform points
        inside the bounds after `initial`. The returned result is the start
        with the lowest objective, and `result.starts` carries all of them.

    Returns
    -------
    CalibrationResult
        `model` is `None` when the winning parameters are outside the Heston
        domain (only reachable with `method="lm"`), in which case
        `in_domain` is `False` and `parameters` still carries the answer.

    Raises
    ------
    InvalidInputError
        Empty quote list, unknown `objective`/`method`/`jac`, bounds that do
        not bracket `initial`, or `bounds` given with `method="lm"`.
    """
    from scipy.optimize import least_squares

    quote_list = list(quotes)
    if not quote_list:
        raise InvalidInputError("quotes must be non-empty")
    if any(not isinstance(q, OptionQuote) for q in quote_list):
        raise InvalidInputError("quotes must be OptionQuote instances")
    if objective not in OBJECTIVES:
        raise InvalidInputError(f"objective must be one of {OBJECTIVES}")
    if method not in CALIBRATION_METHODS:
        raise InvalidInputError(f"method must be one of {CALIBRATION_METHODS}")
    if jac not in ("analytic", "2-point", "3-point"):
        raise InvalidInputError("jac must be 'analytic', '2-point' or '3-point'")

    x0 = np.asarray(initial, dtype=float)
    if x0.shape != (5,) or not np.all(np.isfinite(x0)):
        raise InvalidInputError(
            f"initial must be 5 finite numbers in the order {HESTON_PARAMETERS}"
        )
    if method == "lm":
        if bounds is not None:
            raise InvalidInputError(
                "method='lm' is the unconstrained route and takes no bounds; "
                "use method='trf' for a bounded solve"
            )
        resolved_bounds = None
    else:
        resolved_bounds = DEFAULT_BOUNDS if bounds is None else bounds
        lower = np.asarray(resolved_bounds[0], dtype=float)
        upper = np.asarray(resolved_bounds[1], dtype=float)
        if lower.shape != (5,) or upper.shape != (5,):
            raise InvalidInputError("bounds must be two sequences of 5 numbers")
        if np.any(lower >= upper):
            raise InvalidInputError("every lower bound must be below its upper bound")
        if np.any(x0 < lower) or np.any(x0 > upper):
            raise InvalidInputError("initial must lie inside bounds")

    settings = cos if cos is not None else default_cos_settings(x0)
    problem = _Problem(
        quote_list, market, objective=objective, weights=weights, settings=settings
    )

    if starts is not None:
        points = [np.asarray(s, dtype=float) for s in starts]
        if not points:
            raise InvalidInputError("starts must be non-empty when given")
    elif n_starts < 1:
        raise InvalidInputError("n_starts must be >= 1")
    elif n_starts == 1:
        points = [x0]
    else:
        if resolved_bounds is None:
            raise InvalidInputError(
                "n_starts > 1 needs bounds to draw from; use method='trf'"
            )
        points = _multistart_points(x0, resolved_bounds, n_starts, seed)

    kwargs: dict[str, Any] = {
        "xtol": xtol,
        "ftol": ftol,
        "gtol": gtol,
        "method": method,
    }
    if resolved_bounds is not None:
        kwargs["bounds"] = resolved_bounds
    if max_nfev is not None:
        kwargs["max_nfev"] = max_nfev
    jac_argument = problem.jacobian if jac == "analytic" else jac

    summaries: list[StartSummary] = []
    best = None
    for point in points:
        problem.n_evals = 0
        problem.n_jac_evals = 0
        solution = least_squares(problem.residuals, point, jac=jac_argument, **kwargs)
        summaries.append(
            StartSummary(
                initial=tuple(float(v) for v in point),
                parameters=tuple(float(v) for v in solution.x),
                objective=float(solution.cost),
                success=bool(solution.success),
                n_iter=int(getattr(solution, "njev", 0) or 0),
                n_evals=int(solution.nfev),
                message=str(solution.message),
            )
        )
        if best is None or solution.cost < best.cost:
            best = solution

    assert best is not None  # noqa: S101 -- points is non-empty by construction
    x = np.asarray(best.x, dtype=float)
    residuals = problem.residuals(x)
    jacobian = problem.jacobian(x)
    covariance, condition, singular_values, standard_errors = parameter_covariance(
        jacobian, residuals
    )
    v0, kappa, theta, xi, rho = (float(v) for v in x)
    in_domain = _in_domain(x)
    feller_number = 4.0 * kappa * theta / (xi * xi) if xi != 0.0 else math.inf
    return CalibrationResult(
        model=HestonModel(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        if in_domain
        else None,
        parameters=(v0, kappa, theta, xi, rho),
        objective=float(best.cost),
        objective_kind=objective,
        residuals=residuals,
        jacobian=jacobian,
        covariance=covariance,
        condition_number=condition,
        singular_values=singular_values,
        standard_errors=standard_errors,
        n_iter=int(getattr(best, "njev", 0) or 0),
        n_evals=int(best.nfev),
        success=bool(best.success),
        message=str(best.message),
        feller_number=feller_number,
        feller_satisfied=feller_number >= 2.0,
        in_domain=in_domain,
        cos_settings=settings,
        jac_mode=jac,
        method=method,
        starts=tuple(summaries),
    )
