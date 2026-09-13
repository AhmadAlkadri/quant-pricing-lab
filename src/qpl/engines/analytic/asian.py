"""Asian options on a discretely monitored average, in closed form and not.

This module contains one exact result, one exact expectation, and one
explicitly-labelled approximation. Keeping them in the same file and *not*
registering the approximation as `method="analytic"` is the point: the registry
must never hand back an approximation as if it were a closed form.

Derivation: the discrete geometric average is exactly lognormal
--------------------------------------------------------------
Under Black-Scholes with continuous dividend yield `q`, write `mu = r - q` and
let the monitoring times be `0 < t_1 < ... < t_n <= T`. Then

    S_{t_i} = S_0 exp( (mu - sigma^2/2) t_i + sigma W_{t_i} ),

with `W` a standard Brownian motion under the pricing measure. The geometric
average `G = (prod_i S_{t_i})^{1/n}` therefore has

    log G = log S_0 + (mu - sigma^2/2) * tbar + (sigma / n) * sum_i W_{t_i},
    tbar  = (1/n) sum_i t_i.

The last term is a linear combination of jointly Gaussian variables, so `log G`
is Gaussian -- which is the whole reason the geometric average has a closed form
and the arithmetic one does not. Its mean is

    m = log S_0 + (mu - sigma^2/2) * tbar,

since `E[W_{t_i}] = 0`, and its variance follows from `Cov(W_s, W_t) = min(s,t)`:

    v = (sigma^2 / n^2) * sum_i sum_j min(t_i, t_j).

That double sum collapses. For sorted times, fixing `i` and splitting the inner
sum at `j = i` gives `sum_j min(t_i, t_j) = sum_{j<i} t_j + (n - i + 1) t_i`;
summing over `i` and collecting the coefficient of each `t_i` leaves

    v = (sigma^2 / n^2) * sum_{i=1}^{n} (2(n - i) + 1) * t_i,                (*)

i.e. the earliest fixing is weighted `2n - 1` and the last one `1`. Equation (*)
is what `geometric_average_log_moments` evaluates, and it is `O(n)` work rather
than the `O(n^2)` of the double sum.

With `log G ~ N(m, v)`, `F = E[G] = exp(m + v/2)` and the discounted
expectation of `max(G - K, 0)` is the Black (1976) formula in `F` with total
variance `v`:

    C = e^{-rT} ( F N(d1) - K N(d2) ),   d1 = (log(F/K) + v/2)/sqrt(v),
    P = e^{-rT} ( K N(-d2) - F N(-d1) ), d2 = d1 - sqrt(v).

Note that `T` (the settlement date) enters only through the discount factor,
while `m` and `v` depend only on the fixing times. An average that stops before
settlement is priced correctly by construction, not by a special case.

Continuous limit. With `t_i = iT/n` the two moments are

    tbar = (T/2)(1 + 1/n),     v = (sigma^2 T / 3)(1 + 3/(2n) + 1/(2n^2)),

both of which approach the continuous-averaging values `T/2` and
`sigma^2 T / 3` with an error of order `1/n`. The `sigma^2 T / 3` is the
familiar Kemna-Vorst variance (Kemna & Vorst 1990, *Journal of Banking and
Finance* 14, 113-129, who give the continuous geometric-average closed form and
propose it as a control variate for the arithmetic average). So the discrete
formula converges to the continuous one at **order 1 in the number of
fixings**, and the order is measured rather than asserted in
`tests/test_asian_analytic.py`.

The expected arithmetic average
-------------------------------
Also exact, and it is what makes Asian put-call parity checkable:

    E[A] = (1/n) sum_i E[S_{t_i}] = (S_0 / n) sum_i e^{mu t_i},

a sum of forwards. Since `max(A - K, 0) - max(K - A, 0) = A - K` identically,

    C_arith - P_arith = e^{-rT} ( E[A] - K ),

and the same statement with `E[G] = exp(m + v/2)` holds for the geometric pair.
Both are exact identities, not approximations, and hold *pathwise* in a Monte
Carlo sample that uses the same paths for both legs.

Turnbull-Wakeman: an approximation, labelled as one
---------------------------------------------------
The arithmetic average of lognormals is not lognormal, and no closed form for
its distribution is known in elementary functions. The standard cheap answer is
to match the first two moments of `A` to a lognormal and price that instead
(Turnbull & Wakeman 1991, *Journal of Financial and Quantitative Analysis*
26(3), 377-389; Levy 1992, *Journal of International Money and Finance* 11,
474-491, gives an equivalent construction). The two moments are exact:

    M1 = E[A]  = (S_0/n) sum_i e^{mu t_i},
    M2 = E[A^2] = (S_0^2/n^2) sum_i sum_j exp( mu (t_i + t_j)
                                               + sigma^2 min(t_i, t_j) ),

the second because `E[S_u S_w] = S_0^2 exp(mu(u+w) + sigma^2 min(u,w))` for
geometric Brownian motion. Matching a lognormal with the same two moments fixes
its total variance at `v_A = log(M2 / M1^2)` and its mean at `M1`, and the price
is Black (1976) again with `F = M1` and variance `v_A`.

What this is *not*: a bound, an asymptotic expansion, or a closed form. It is a
two-moment fit whose error has no rigorous control, and it is registered
nowhere -- `method="analytic"` on an arithmetic Asian raises `NotSupportedError`
naming this function and Monte Carlo. The measured gap against a
control-variate Monte Carlo run is reported in
`tests/cases/test_asian_black_scholes_cases.py` and
`docs/notes/asian_options_control_variate.md`; it is recorded with its sign, not
asserted to be zero.

Greeks of the geometric Asian, and what "theta" means for an average
--------------------------------------------------------------------
With `F = E[G] = exp(m + v/2)` and total variance `v`, the price is Black (1976)
and every Greek is a chain rule through `(F, v)` plus the discount factor. Two
facts do the work. First, `F` is **linear in the spot**: `F = S * exp((mu -
sigma^2/2) tbar + v/2)`, so `dF/dS = F/S` and `F/S` is a constant. Second, the
Black identity `F phi(d1) = K phi(d2)` holds for these `d1`, `d2` exactly as it
does for a vanilla, so in every derivative the two `phi` terms collapse into
one multiple of `d(sqrt(v))`. Writing `D = e^{-rT}`, `sd = sqrt(v)`:

    delta = D N(d1) F / S
    gamma = D (F / S) phi(d1) / (S sd)
    vega  = D [ N(d1) dF/dsigma + F phi(d1) d(sd)/dsigma ],
            dF/dsigma = F (-sigma tbar + v / sigma),   d(sd)/dsigma = sd / sigma
    rho   = D F tbar N(d1) - T V

`rho` uses `dm/dr = tbar` (the drift enters `m` through `mu = r - q` and
nothing else) and the discount factor's own `-T V`.

**Theta needs a convention, and it is not `-dV/dT`.** The settlement date `T`
enters this price *only* through `e^{-rT}`, because `m` and `v` depend on the
fixing times alone. So `-dV/dT = -r V`, which is a true derivative and is not
time decay: it is what the contract loses by being paid later while the average
is unchanged. Time decay is the derivative along the **roll**, where calendar
time advances by `s` and the settlement date and *every* fixing date come
closer together: `T -> T - s` and `t_i -> t_i - s`. Along that path

    dtbar/ds = -1,     dv/ds = -(sigma^2 / n^2) sum_i (2(n - i) + 1) = -sigma^2,

the second because those weights sum to exactly `n^2` -- so `dF/ds = -mu F` and
`d(sd)/ds = -sigma^2 / (2 sd)`, giving

    theta = r V - D [ mu F N(d1) + F phi(d1) sigma^2 / (2 sd) ].

That convention is not a choice made to be tidy: at `n = 1`, where the
geometric Asian **is** a European vanilla, it reproduces the Black-Scholes
theta term for term, and `tests/test_asian_analytic.py` asserts all five Greeks
against `qpl.engines.analytic.black_scholes` at that point to 1e-12. A
`-dV/dT` theta would not. It is also the convention a Monte Carlo bump can
reproduce, by shifting the whole schedule, which is how the MC Asian engine
estimates theta.

Puts come from put-call parity, which is exact here: `V_put = V_call - D (F - K)`,
so every Greek of the put is the call's minus the same derivative of the
forward contract `D (F - K)`. That is one line per Greek and no second
derivation to get wrong.

Provenance
----------
Every formula above was re-derived here from the lognormal transition density;
nothing is transcribed from the cited papers, and the only published *numbers*
used anywhere in this slice are the three benchmark values cited in
`qpl.cases.asian_black_scholes`. Fusai & Roncoroni, *Implementing Models in
Quantitative Finance*, chapter 15, is used as a map of the problem and not as a
source of formulas; its Laplace-transform and PDE routes to the arithmetic
average are a later slice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import AsianOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .black_scholes import _norm_cdf, _norm_pdf

__all__ = [
    "arithmetic_average_moments",
    "discrete_geometric_greeks",
    "discrete_geometric_price",
    "expected_arithmetic_average",
    "expected_geometric_average",
    "geometric_average_log_moments",
    "greeks_asian",
    "price_asian",
    "turnbull_wakeman_price",
]


def _clean_times(fixing_times: Sequence[float]) -> np.ndarray:
    """Fixing times as a validated float array (sorted, positive, distinct)."""
    times = np.asarray(fixing_times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise InvalidInputError("fixing_times must be a non-empty 1D sequence")
    if not np.all(np.isfinite(times)):
        raise InvalidInputError("fixing_times must be finite")
    if np.any(times <= 0.0):
        raise InvalidInputError("fixing_times must all be > 0")
    if np.any(np.diff(times) <= 0.0):
        raise InvalidInputError("fixing_times must be strictly increasing")
    return times


def _black_76(
    *, forward: float, strike: float, variance: float, discount: float, kind: str
) -> float:
    """`discount * (F N(d1) - K N(d2))` for a call, its mirror for a put.

    `variance` is the *total* variance of the log of the underlying average
    over the life of the contract, not an annualised one: the discrete
    geometric average and the moment-matched arithmetic one each supply their
    own, and neither is `sigma**2 T`.
    """
    if variance <= 0.0:
        intrinsic = forward - strike if kind == "call" else strike - forward
        return discount * max(intrinsic, 0.0)
    sd = math.sqrt(variance)
    d1 = (math.log(forward / strike) + 0.5 * variance) / sd
    d2 = d1 - sd
    if kind == "call":
        return discount * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return discount * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def geometric_average_log_moments(
    *, S: float, mu: float, sigma: float, fixing_times: Sequence[float]
) -> tuple[float, float]:
    """`(m, v)`: the mean and variance of `log G`, exactly.

    See the module docstring for the derivation; `v` uses the collapsed form
    `(sigma^2/n^2) sum_i (2(n-i)+1) t_i` with `i` counted from 1 over sorted
    times, which is `O(n)` rather than the `O(n^2)` double sum.
    """
    if S <= 0.0:
        raise InvalidInputError("S must be > 0")
    if sigma < 0.0:
        raise InvalidInputError("sigma must be >= 0")
    times = _clean_times(fixing_times)
    n = times.size
    weights = 2.0 * np.arange(n - 1, -1, -1, dtype=float) + 1.0
    m = math.log(S) + (mu - 0.5 * sigma * sigma) * float(times.mean())
    v = sigma * sigma * float(np.dot(weights, times)) / (n * n)
    return m, v


def expected_geometric_average(
    *, S: float, mu: float, sigma: float, fixing_times: Sequence[float]
) -> float:
    """`E[G] = exp(m + v/2)`, exactly."""
    m, v = geometric_average_log_moments(
        S=S, mu=mu, sigma=sigma, fixing_times=fixing_times
    )
    return math.exp(m + 0.5 * v)


def expected_arithmetic_average(
    *, S: float, mu: float, fixing_times: Sequence[float]
) -> float:
    """`E[A] = (S/n) sum_i e^{mu t_i}`, exactly.

    Independent of `sigma`: the average of forwards does not care about the
    volatility, which is exactly why Asian put-call parity does not either.
    """
    if S <= 0.0:
        raise InvalidInputError("S must be > 0")
    times = _clean_times(fixing_times)
    return float(S * np.mean(np.exp(mu * times)))


def arithmetic_average_moments(
    *, S: float, mu: float, sigma: float, fixing_times: Sequence[float]
) -> tuple[float, float]:
    """`(E[A], E[A^2])` for the arithmetic average, both exact.

    The second moment is the `O(n^2)` double sum
    `(S^2/n^2) sum_i sum_j exp(mu(t_i+t_j) + sigma^2 min(t_i,t_j))`, evaluated
    as one outer product. At the fixing counts this package uses (10 to a few
    hundred) that is cheaper than any restructuring would be, and it is the
    form the derivation is written in, so there is nothing to get wrong
    separately.
    """
    if S <= 0.0:
        raise InvalidInputError("S must be > 0")
    if sigma < 0.0:
        raise InvalidInputError("sigma must be >= 0")
    times = _clean_times(fixing_times)
    n = times.size
    m1 = float(S * np.mean(np.exp(mu * times)))
    pair_sum = times[:, None] + times[None, :]
    pair_min = np.minimum(times[:, None], times[None, :])
    m2 = float(
        S * S * np.sum(np.exp(mu * pair_sum + sigma * sigma * pair_min)) / (n * n)
    )
    return m1, m2


def discrete_geometric_price(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    fixing_times: Sequence[float],
    q: float = 0.0,
    kind: str = "call",
) -> float:
    """Exact price of a fixed-strike **geometric**-average Asian option.

    Closed form, not an approximation: `log G` is exactly normal with the
    moments derived in the module docstring, so the price is Black (1976) with
    `F = E[G]` and total variance `v`. `T` enters only through `e^{-rT}`.

    Raises
    ------
    InvalidInputError
        On a non-positive spot or strike, a negative `sigma`, a fixing time
        outside `(0, T]`, or a non-increasing fixing schedule.
    """
    if K <= 0.0:
        raise InvalidInputError("K must be > 0")
    if T < 0.0:
        raise InvalidInputError("T must be >= 0")
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    times = _clean_times(fixing_times)
    if float(times[-1]) > T:
        raise InvalidInputError("fixing_times must all be <= T")

    m, v = geometric_average_log_moments(
        S=S, mu=r - q, sigma=sigma, fixing_times=times
    )
    return _black_76(
        forward=math.exp(m + 0.5 * v),
        strike=K,
        variance=v,
        discount=math.exp(-r * T),
        kind=kind_l,
    )


def discrete_geometric_greeks(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    fixing_times: Sequence[float],
    q: float = 0.0,
    kind: str = "call",
) -> dict[str, float]:
    """Exact Greeks of a fixed-strike **geometric**-average Asian option.

    Returns a dict keyed by `delta`, `gamma`, `vega`, `theta`, `rho` (and
    `value`, since every one of them is derived alongside it and a caller
    checking `rho = D F tbar N(d1) - T V` needs `V`). The derivations, and the
    roll convention `theta` uses, are in the module docstring.

    Raises
    ------
    InvalidInputError
        On `T <= 0` or `sigma <= 0`. Both make the average deterministic, so
        gamma is a point mass; the vanilla engine refuses the same two limits
        and `discrete_geometric_price` still returns the correct price there.
    """
    if K <= 0.0:
        raise InvalidInputError("K must be > 0")
    if T <= 0.0:
        raise InvalidInputError("T must be > 0 for Greeks")
    if sigma <= 0.0:
        raise InvalidInputError("sigma must be > 0 for Greeks")
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    times = _clean_times(fixing_times)
    if float(times[-1]) > T:
        raise InvalidInputError("fixing_times must all be <= T")

    mu = r - q
    m, v = geometric_average_log_moments(S=S, mu=mu, sigma=sigma, fixing_times=times)
    sd = math.sqrt(v)
    forward = math.exp(m + 0.5 * v)
    discount = math.exp(-r * T)
    tbar = float(times.mean())

    d1 = (math.log(forward / K) + 0.5 * v) / sd
    d2 = d1 - sd
    nd1 = _norm_cdf(d1)
    pdf_d1 = _norm_pdf(d1)

    value = discount * (forward * nd1 - K * _norm_cdf(d2))
    df_dsigma = forward * (-sigma * tbar + v / sigma)

    delta = discount * nd1 * forward / S
    gamma = discount * (forward / S) * pdf_d1 / (S * sd)
    vega = discount * (nd1 * df_dsigma + forward * pdf_d1 * sd / sigma)
    rho = discount * forward * tbar * nd1 - T * value
    theta = r * value - discount * (
        mu * forward * nd1 + forward * pdf_d1 * sigma * sigma / (2.0 * sd)
    )

    if kind_l == "put":
        # Put-call parity, exactly: V_put = V_call - D (F - K). Every Greek is
        # the call's minus the same derivative of that forward contract.
        forward_value = discount * (forward - K)
        value = value - forward_value
        delta = delta - discount * forward / S
        rho = rho - (discount * forward * tbar - T * forward_value)
        theta = theta - (r * forward_value - discount * mu * forward)
        vega = vega - discount * df_dsigma
        # gamma is unchanged: the forward contract is linear in the spot.

    return {
        "value": value,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def turnbull_wakeman_price(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    fixing_times: Sequence[float],
    q: float = 0.0,
    kind: str = "call",
) -> float:
    """**Approximate** price of an arithmetic-average Asian by moment matching.

    Turnbull & Wakeman (1991); Levy (1992) gives an equivalent construction.
    The first two moments of the arithmetic average are exact (see
    `arithmetic_average_moments`); the approximation is the assumption that a
    lognormal with those two moments prices the option, which it does not.

    This function is **not** registered with the engine registry and
    `method="analytic"` will not return it. Ask for it by name, and read the
    measured gap against a control-variate Monte Carlo run before using the
    number for anything: `docs/notes/asian_options_control_variate.md`.

    The approximation is at its best for low volatility, short maturities and
    dense fixing schedules -- exactly where the arithmetic average is closest to
    lognormal -- and degrades as `sigma^2 T` grows, where the true distribution
    of `A` is visibly less skewed than the fitted lognormal.
    """
    if K <= 0.0:
        raise InvalidInputError("K must be > 0")
    if T < 0.0:
        raise InvalidInputError("T must be >= 0")
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    times = _clean_times(fixing_times)
    if float(times[-1]) > T:
        raise InvalidInputError("fixing_times must all be <= T")

    m1, m2 = arithmetic_average_moments(
        S=S, mu=r - q, sigma=sigma, fixing_times=times
    )
    # `m2 / m1**2 >= 1` always (Jensen); it is exactly 1 at sigma = 0, where the
    # fitted variance is 0 and `_black_76` falls through to the intrinsic value
    # of the forward average.
    variance = math.log(m2 / (m1 * m1)) if m2 > m1 * m1 else 0.0
    return _black_76(
        forward=m1,
        strike=K,
        variance=variance,
        discount=math.exp(-r * T),
        kind=kind_l,
    )


_ARITHMETIC_REFUSAL = (
    "There is no closed form for a fixed-strike arithmetic-average Asian "
    "option under Black-Scholes: the sum of lognormals is not lognormal and "
    "its law has no elementary density. Use method='mc' (with "
    "MCConfig(variance_reduction='control_variate'), which uses the geometric "
    "average and its exact closed form as the control), or call one of the "
    "explicitly-approximate formulas by name: "
    "qpl.engines.analytic.asian.turnbull_wakeman_price (Turnbull & Wakeman "
    "1991 / Levy 1992 two-moment lognormal fit). Those are deliberately not "
    "registered as method='analytic', so that the dispatcher never returns an "
    "approximation as if it were exact. The Laplace-transform and PDE routes "
    "(Fusai & Roncoroni, chapter 15) are a later slice."
)


_ARITHMETIC_GREEKS_REFUSAL = (
    "Greeks are not available in closed form for a fixed-strike "
    "arithmetic-average Asian option, for the same reason the price is not: "
    "the sum of lognormals has no elementary density. The moment-matched "
    "approximations (qpl.engines.analytic.asian.turnbull_wakeman_price) are "
    "differentiable, but a derivative of an approximation whose error has no "
    "rigorous control has no rigorous control either, and method='analytic' "
    "must not return one. Use method='mc' with "
    "MCConfig(greeks_estimator='pathwise') -- the arithmetic Asian payoff is "
    "Lipschitz in the average and the average is smooth in the path, so the "
    "pathwise estimator is unbiased there. The geometric average IS available "
    "in closed form from this engine."
)


def price_asian(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
) -> PriceResult:
    """Price a **geometric**-average Asian option in closed form.

    Raises
    ------
    NotSupportedError
        If `option.averaging == "arithmetic"`. The message names the
        approximations and the Monte Carlo route; see `_ARITHMETIC_REFUSAL`.
    """
    if option.averaging != "geometric":
        raise NotSupportedError(_ARITHMETIC_REFUSAL)

    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    value = discrete_geometric_price(
        S=market.spot,
        K=option.strike,
        T=t,
        r=r,
        sigma=model.sigma,
        fixing_times=option.fixing_times,
        q=q,
        kind=option.kind,
    )
    m, v = geometric_average_log_moments(
        S=market.spot, mu=r - q, sigma=model.sigma, fixing_times=option.fixing_times
    )
    return PriceResult(
        value=value,
        meta={
            "method": "analytic",
            "model": "BlackScholes",
            "instrument": "asian",
            "averaging": "geometric",
            "n_fixings": option.n_fixings,
            "log_average_mean": m,
            "log_average_variance": v,
            "expected_average": math.exp(m + 0.5 * v),
        },
    )


def greeks_asian(
    option: AsianOption,
    model: BlackScholesModel,
    market: Market,
) -> GreeksResult:
    """Exact Greeks of a **geometric**-average Asian option.

    Derivations in the module docstring. `theta` is the derivative along the
    roll (settlement and every fixing shifted together), which is the only
    reading of "time decay" an average admits and is the one that reduces to
    the Black-Scholes theta when the schedule has a single fixing at expiry.

    Raises
    ------
    NotSupportedError
        If `option.averaging == "arithmetic"`. The moment-matched
        approximations are differentiable, but differentiating an approximation
        whose error has no control produces a Greek whose error has no control
        either, and the registry would be handing it back as `method="analytic"`.
        Use `method="mc"` with `MCConfig(greeks_estimator="pathwise")`, which is
        well defined for an arithmetic average and is measured against a bump
        and against this closed form in `tests/test_asian_mc_greeks.py`.
    InvalidInputError
        At `T = 0` or `sigma = 0`.
    """
    if option.averaging != "geometric":
        raise NotSupportedError(_ARITHMETIC_GREEKS_REFUSAL)

    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    values = discrete_geometric_greeks(
        S=market.spot,
        K=option.strike,
        T=t,
        r=r,
        sigma=model.sigma,
        fixing_times=option.fixing_times,
        q=q,
        kind=option.kind,
    )
    return GreeksResult(
        delta=values["delta"],
        gamma=values["gamma"],
        vega=values["vega"],
        theta=values["theta"],
        rho=values["rho"],
        meta={
            "method": "analytic",
            "model": "BlackScholes",
            "instrument": "asian",
            "averaging": "geometric",
            "n_fixings": option.n_fixings,
            "value": values["value"],
            "theta_convention": "roll: settlement and every fixing shift together",
        },
    )
