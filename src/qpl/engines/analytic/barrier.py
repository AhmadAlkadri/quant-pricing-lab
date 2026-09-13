"""Single-barrier options in closed form, continuous monitoring.

Derivation
----------
Write the log-spot as an arithmetic Brownian motion. Under the risk-neutral
measure with `b = r - q`,

    S_t = S exp(sigma X_t),   X_t = nu t + W_t,   nu = (b - sigma^2/2) / sigma,

so a *down* barrier at `H` is the level `a = log(H / S) / sigma < 0` for the
drifted Brownian motion `X`, and the option's value needs the joint law of the
terminal value `X_T` and the running minimum `m_T = min_{0<=t<=T} X_t`.

For a **driftless** Brownian motion the reflection principle gives that joint
law directly: a path that ends at `x` having touched `a <= 0` is in bijection
with a path ending at `2a - x` (reflect the post-touch segment in the level
`a`), and every path ending below `a` has touched it. Hence

    P(m_T <= a, W_T >= x) = P(W_T >= 2a - x)     for a <= 0, x >= a,

and differentiating in `x` gives the density of `(W_T, m_T)`. With a drift,
Girsanov replaces the reflected density by the same reflected density times
`exp(2 nu a)`: changing the measure that turns `nu t + W_t` into a driftless
motion contributes `exp(nu X_T - nu^2 T / 2)`, and on the reflected path
`X_T` is replaced by `2a - X_T`, so the two Radon-Nikodym factors differ by
exactly `exp(2 nu a)`. This is the joint law of the terminal value and the
running extremum in Shreve, *Stochastic Calculus for Finance II*, chapter 7
(reflection equality, the density of the maximum to date, and the
corresponding results under a drift); it is re-derived above rather than
quoted.

Every single-barrier value is then an integral of a vanilla payoff against
that law, and each such integral is the difference of two normal tails: one
from the ordinary terminal density and one from the reflected term, the latter
carrying the factor `(H/S)^{2 mu}` or `(H/S)^{2(mu+1)}` that `exp(2 nu a)`
produces once the payoff's `S_T` is absorbed. Collecting them gives six
building blocks -- the standard `A`-`F` of the barrier literature -- from which
all eight types are assembled by addition.

The building blocks
-------------------
With `v = sigma sqrt(T)`, `mu = (b - sigma^2/2) / sigma^2` and
`lambda = sqrt(mu^2 + 2 r / sigma^2)`, and with `phi = +1` for a call and `-1`
for a put, `eta = +1` for a down barrier and `-1` for an up one:

    x1 = log(S/X)/v + (1 + mu) v          y1 = log(H^2/(S X))/v + (1 + mu) v
    x2 = log(S/H)/v + (1 + mu) v          y2 = log(H/S)/v + (1 + mu) v
    z  = log(H/S)/v + lambda v

    A = phi S e^{(b-r)T} N(phi x1) - phi X e^{-rT} N(phi (x1 - v))
    B = phi S e^{(b-r)T} N(phi x2) - phi X e^{-rT} N(phi (x2 - v))
    C = phi S e^{(b-r)T} (H/S)^{2(mu+1)} N(eta y1)
        - phi X e^{-rT} (H/S)^{2mu} N(eta (y1 - v))
    D = phi S e^{(b-r)T} (H/S)^{2(mu+1)} N(eta y2)
        - phi X e^{-rT} (H/S)^{2mu} N(eta (y2 - v))
    E = K e^{-rT} [ N(eta (x2 - v)) - (H/S)^{2mu} N(eta (y2 - v)) ]
    F = K [ (H/S)^{mu+lambda} N(eta z) + (H/S)^{mu-lambda} N(eta z - 2 eta lambda v) ]

`A` is the plain Black-Scholes value (the terminal density with no reflected
term). `B` is the same integral truncated at the barrier instead of at the
strike. `C` and `D` are their reflected counterparts and are where the joint
law enters. `E` is a rebate paid **at expiry** if the barrier was never
touched, and `F` is a rebate paid **at the touch time**: `F` is the Laplace
transform of the first-passage time at the rate `r`, which is why it carries
`lambda` rather than `mu + 1` and why it is the only block with no `e^{-rT}`
in front of it.

Two sanity properties are worth naming, because the tests check them and
because they catch the two sign errors this construction invites.

First, `F -> K` as `H -> S`: `z -> lambda v`, `(H/S)^k -> 1`, and the two tails
sum to `N(lambda v) + N(-lambda v) = 1`, so the first-passage claim is worth
its face value when the barrier is already underfoot.

Second, **every knock-out assembly collapses to `F` alone at `H = S`** -- the
option part is worth nothing and only the rebate survives -- and it does so
exactly, with no limit taken. Measured worst residual over three points and
four knock-out types: 1.42e-14. The route differs by kind, which is the part
worth writing down because it is not visible from the table. For a *call* with
a down barrier, `y1 = x1` and `y2 = x2` at `H = S`, and with `phi = eta = +1`
the pairs `A - C` and `B - D` vanish block by block. For a *put* they do not:
`phi = -1` puts `N(-x1)` in `A` against `N(+x1)` in `C`, so instead
`A + C = -S e^{(b-r)T} + X e^{-rT} = B + D`, and the assembly
`A - B + C - D` vanishes as a difference of two *sums*. Same zero, different
cancellation -- and a test that only checked the call would pass with a sign
error in the put's `C`.

Assembly
--------
Which blocks appear depends on the type *and* on whether the strike is above
or below the barrier, because `B` and `D` truncate at the barrier and the
truncation is vacuous on one side. For the eight types (`X` the strike):

    down-and-in  call:  X > H: C + E              X < H: A - B + D + E
    up-and-in    call:  X > H: A + E              X < H: B - C + D + E
    down-and-in  put :  X > H: B - C + D + E      X < H: A + E
    up-and-in    put :  X > H: A - B + D + E      X < H: C + E
    down-and-out call:  X > H: A - C + F          X < H: B - D + F
    up-and-out   call:  X > H: F                  X < H: A - B + C - D + F
    down-and-out put :  X > H: A - B + C - D + F  X < H: F
    up-and-out   put :  X > H: B - D + F          X < H: A - C + F

The formulas are those of Reiner, E. and Rubinstein, M. (1991), "Breaking down
the barriers", *Risk* 4(8), 28-35, who give all eight single-barrier types with
rebates; the down-and-out call alone is Merton, R.C. (1973), "Theory of
rational option pricing", *Bell Journal of Economics and Management Science*
4(1), 141-183, section 8. Haug, E.G. (2007), *The Complete Guide to Option
Pricing Formulas*, 2nd ed., section 4.17, is the standard tabulation of the
same eight and is the source of the three published values used as fixtures in
`qpl.cases.barrier_black_scholes` (cited there). Derman, E., Kani, I., Ergener,
D. and Bardhan, I. (1995), "Enhanced numerical methods for options with
barriers", *Risk* 8(6), is cited for the lattice side of the problem and is not
used here. Nothing above is copied from any of them: the blocks are written out
in this module's own notation and the assembly table was checked against
`tests/oracle/test_barrier_vs_quantlib.py`, not read off a page.

In-out parity, and what the rebate does to it
---------------------------------------------
With `rebate = 0` the blocks `E` and `F` vanish and the assembly table
collapses to an exact identity, one addition at a time: on `X > H`,
`(C) + (A - C) = A`; on `X < H`, `(A - B + D) + (B - D) = A`. So

    knock-in + knock-out = vanilla,

exactly, for all four in/out pairs, at any parameters -- and the residual is a
round-off budget rather than a model error. With a rebate it is **false**, and
not by a small amount: the two rebate legs are paid on complementary events at
different times (`E` at expiry if never touched, `F` at the touch time), so
`in + out = vanilla + E + F` and the sum exceeds the vanilla by the value of a
claim that pays the rebate either way.

Degenerate limits
-----------------
`T = 0` and `sigma = 0` make the path deterministic and are handled in closed
form rather than by the formulas above (`v = 0` divides by zero). At
`sigma = 0` the spot follows `S e^{b t}`, which is monotone, so the barrier is
touched at most once and the touch time solves `S e^{b t} = H` exactly. Every
engine in this package is required to return that same value, which is what
makes it a statement about the model rather than about a discretisation.
"""

from __future__ import annotations

import math

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import BARRIER_TYPES, BarrierOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel, bs_price
from ..base import GreeksResult, PriceResult
from .black_scholes import _norm_cdf

__all__ = [
    "BGK_BETA",
    "BarrierBlocks",
    "barrier_blocks",
    "barrier_price",
    "bgk_continuity_corrected_price",
    "greeks_barrier",
    "price_barrier",
    "shifted_barrier",
]

BGK_BETA = 0.5825971579390107
"""`-zeta(1/2) / sqrt(2 pi)`, the Broadie-Glasserman-Kou continuity-correction
constant.

The number is the Riemann zeta function at `1/2`, `-1.4603545088095868`,
divided by `sqrt(2 pi)`. It is *not* fitted: it comes from the asymptotic
expected overshoot of a random walk past a level, which is what a discretely
monitored barrier gets wrong relative to a continuous one. See
:func:`bgk_continuity_corrected_price`.
"""

_DISCRETE_REFUSAL = (
    "method='analytic' prices the CONTINUOUSLY monitored barrier: the "
    "Reiner-Rubinstein formulas integrate against the joint law of the "
    "terminal value and the running extremum of the whole path, and a "
    "discretely monitored contract is a different (more valuable, for a "
    "knock-out) contract worth O(1/sqrt(m)) more. Use method='mc' with the "
    "monitoring schedule, or "
    "qpl.engines.analytic.barrier.bgk_continuity_corrected_price(...) for the "
    "Broadie-Glasserman-Kou barrier-shift APPROXIMATION of the discrete price, "
    "which is deliberately not registered as an engine."
)


class BarrierBlocks(tuple):
    """The six building blocks `(A, B, C, D, E, F)` of the module docstring.

    A named tuple subclass rather than a dataclass so that a test can unpack
    it positionally in the order the literature uses; the attribute names exist
    so that an assembly line reads as the table above rather than as indices.
    """

    __slots__ = ()

    def __new__(cls, a: float, b: float, c: float, d: float, e: float, f: float):
        return super().__new__(cls, (a, b, c, d, e, f))

    @property
    def A(self) -> float:  # noqa: N802 - the literature's name for the block
        return self[0]

    @property
    def B(self) -> float:  # noqa: N802
        return self[1]

    @property
    def C(self) -> float:  # noqa: N802
        return self[2]

    @property
    def D(self) -> float:  # noqa: N802
        return self[3]

    @property
    def E(self) -> float:  # noqa: N802
        return self[4]

    @property
    def F(self) -> float:  # noqa: N802
        return self[5]


def barrier_blocks(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    H: float,
    rebate: float,
    phi: float,
    eta: float,
) -> BarrierBlocks:
    """The six blocks `A`-`F`; see the module docstring for the derivation.

    Exposed so that a test can check the two structural properties directly
    (`F -> K` as `H -> S`, and `A == C`, `B == D` at `H == S`) instead of
    inferring them from an assembled price, where a sign error in two blocks
    can cancel.
    """
    v = sigma * math.sqrt(T)
    b = r - q
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = math.sqrt(mu * mu + 2.0 * r / (sigma * sigma))

    x1 = math.log(S / K) / v + (1.0 + mu) * v
    x2 = math.log(S / H) / v + (1.0 + mu) * v
    y1 = math.log(H * H / (S * K)) / v + (1.0 + mu) * v
    y2 = math.log(H / S) / v + (1.0 + mu) * v
    z = math.log(H / S) / v + lam * v

    hs = H / S
    carry = math.exp((b - r) * T)
    disc = math.exp(-r * T)

    a_block = phi * S * carry * _norm_cdf(phi * x1) - phi * K * disc * _norm_cdf(
        phi * (x1 - v)
    )
    b_block = phi * S * carry * _norm_cdf(phi * x2) - phi * K * disc * _norm_cdf(
        phi * (x2 - v)
    )
    c_block = phi * S * carry * hs ** (2.0 * (mu + 1.0)) * _norm_cdf(
        eta * y1
    ) - phi * K * disc * hs ** (2.0 * mu) * _norm_cdf(eta * (y1 - v))
    d_block = phi * S * carry * hs ** (2.0 * (mu + 1.0)) * _norm_cdf(
        eta * y2
    ) - phi * K * disc * hs ** (2.0 * mu) * _norm_cdf(eta * (y2 - v))
    e_block = rebate * disc * (
        _norm_cdf(eta * (x2 - v)) - hs ** (2.0 * mu) * _norm_cdf(eta * (y2 - v))
    )
    f_block = rebate * (
        hs ** (mu + lam) * _norm_cdf(eta * z)
        + hs ** (mu - lam) * _norm_cdf(eta * z - 2.0 * eta * lam * v)
    )
    return BarrierBlocks(a_block, b_block, c_block, d_block, e_block, f_block)


def _assemble(blocks: BarrierBlocks, *, kind: str, barrier_type: str, strike_above: bool) -> float:
    """The assembly table of the module docstring, and nothing else."""
    a, b, c, d, e, f = blocks
    down = barrier_type.startswith("down")
    out = barrier_type.endswith("out")
    call = kind == "call"

    if not out:
        if call and down:
            return (c + e) if strike_above else (a - b + d + e)
        if call:
            return (a + e) if strike_above else (b - c + d + e)
        if down:
            return (b - c + d + e) if strike_above else (a + e)
        return (a - b + d + e) if strike_above else (c + e)

    if call and down:
        return (a - c + f) if strike_above else (b - d + f)
    if call:
        return f if strike_above else (a - b + c - d + f)
    if down:
        return (a - b + c - d + f) if strike_above else f
    return (b - d + f) if strike_above else (a - c + f)


def _vanilla(*, S: float, K: float, T: float, r: float, sigma: float, q: float, kind: str) -> float:
    return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)


def _deterministic_value(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    H: float,
    rebate: float,
    barrier_type: str,
    kind: str,
) -> float:
    """Price at `T = 0` or at `sigma = 0`, where the path is deterministic.

    At `sigma = 0` the spot is `S e^{(r - q) t}`, which is monotone in `t`, so
    the barrier is touched at most once and the touch time is the exact
    solution of `S e^{(r - q) t} = H` clipped to `[0, T]`. Every engine here
    reproduces this value, because it is a consequence of the model and not of
    a discretisation.
    """
    down = barrier_type.startswith("down")
    out = barrier_type.endswith("out")
    touched_now = (S <= H) if down else (S >= H)

    if T == 0.0:
        if out:
            return rebate if touched_now else _intrinsic(S, K, kind)
        return _intrinsic(S, K, kind) if touched_now else rebate

    b = r - q
    forward = S * math.exp(b * T)
    extreme = min(S, forward) if down else max(S, forward)
    touched = (extreme <= H) if down else (extreme >= H)

    if touched_now:
        touch_time = 0.0
    elif touched:
        # b != 0 whenever the path moves at all, and it must have moved to get
        # here: `touched_now` is False and `touched` is True.
        touch_time = min(max(math.log(H / S) / b, 0.0), T)
    else:
        touch_time = math.inf

    disc_t = math.exp(-r * T)
    if out:
        if touched:
            return rebate * math.exp(-r * touch_time)
        return disc_t * _intrinsic(forward, K, kind)
    if touched:
        return disc_t * _intrinsic(forward, K, kind)
    return rebate * disc_t


def barrier_price(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    H: float,
    rebate: float = 0.0,
    barrier_type: str = "down-and-out",
    kind: str = "call",
) -> float:
    """Closed-form continuously monitored single-barrier value.

    Parameters mirror `qpl.engines.analytic.digital.digital_price`: `H` is the
    barrier level, `rebate` the cash amount (paid at the touch time for a
    knock-out and at expiry for a knock-in), and `barrier_type` one of
    `qpl.instruments.options.BARRIER_TYPES`.

    A spot already at or beyond the barrier settles the contract at inception
    and is handled before the formulas: a knock-out is dead and worth `rebate`
    (paid now, undiscounted), a knock-in is alive and worth the vanilla. The
    formulas themselves assume the barrier has not been crossed and would
    return a meaningless number there -- `(H/S)^{2mu}` is perfectly finite on
    the wrong side of the barrier, which is exactly why this has to be a guard
    and not a limit.
    """
    if T < 0:
        raise InvalidInputError("T must be >= 0")
    if sigma < 0:
        raise InvalidInputError("sigma must be >= 0")
    if K <= 0:
        raise InvalidInputError("K must be > 0")
    if S <= 0:
        raise InvalidInputError("S must be > 0")
    if H <= 0:
        raise InvalidInputError("H must be > 0")
    if rebate < 0:
        raise InvalidInputError("rebate must be >= 0")
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    barrier_type_l = barrier_type.lower()
    if barrier_type_l not in BARRIER_TYPES:
        raise InvalidInputError(
            "barrier_type must be one of " + ", ".join(repr(n) for n in BARRIER_TYPES)
        )

    down = barrier_type_l.startswith("down")
    out = barrier_type_l.endswith("out")
    if (S <= H) if down else (S >= H):
        if out:
            return float(rebate)
        return _vanilla(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind_l)

    if T == 0.0 or sigma == 0.0:
        return _deterministic_value(
            S=S, K=K, T=T, r=r, sigma=sigma, q=q, H=H, rebate=rebate,
            barrier_type=barrier_type_l, kind=kind_l,
        )

    blocks = barrier_blocks(
        S=S, K=K, T=T, r=r, sigma=sigma, q=q, H=H, rebate=rebate,
        phi=1.0 if kind_l == "call" else -1.0,
        eta=1.0 if down else -1.0,
    )
    return float(
        _assemble(blocks, kind=kind_l, barrier_type=barrier_type_l, strike_above=K > H)
    )


def bgk_continuity_corrected_price(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    H: float,
    rebate: float = 0.0,
    barrier_type: str = "down-and-out",
    kind: str = "call",
    n_monitoring: int,
) -> float:
    """Broadie-Glasserman-Kou approximation of the **discretely** monitored price.

    A knock-out monitored on `m` equally spaced dates is worth more than the
    continuous contract, because the path can dip past the barrier and come
    back between observations. Broadie, Glasserman and Kou (1997), "A
    continuity correction for discrete barrier options", *Mathematical Finance*
    7(4), 325-348, show that the whole `O(1/sqrt(m))` difference is captured by
    moving the barrier away from the spot by one asymptotic overshoot:

        V_m(H) = V(H e^{+beta sigma sqrt(dt)})  for an **up** barrier,
        V_m(H) = V(H e^{-beta sigma sqrt(dt)})  for a **down** barrier,

    with `dt = T / m`, `V` the continuous closed form, and
    `beta = -zeta(1/2) / sqrt(2 pi) ~ 0.5826` (:data:`BGK_BETA`) the expected
    overshoot of a random walk past a level. The error of the corrected value
    is `o(1/sqrt(m))`, so it is a genuinely higher-order approximation rather
    than a rescaling of the same error; what order it actually reaches here is
    measured in `docs/notes/barrier_options_monitoring_bias.md` rather than
    assumed.

    This is **an approximation and is deliberately not registered** as an
    engine: `qpl.pricing.price` never returns an approximate value under a
    method name that otherwise means "exact". Same treatment as
    `qpl.engines.analytic.asian.turnbull_wakeman_price`.

    The correction is derived for a *uniform* monitoring schedule, so
    `n_monitoring` is the count and `dt = T / m`; a non-uniform schedule has no
    single `dt` and is not covered.

    Raises
    ------
    InvalidInputError
        `n_monitoring < 1`, or any input the closed form itself refuses.
    """
    if n_monitoring < 1:
        raise InvalidInputError("n_monitoring must be >= 1")
    if T < 0:
        raise InvalidInputError("T must be >= 0")
    barrier_type_l = barrier_type.lower()
    if barrier_type_l not in BARRIER_TYPES:
        raise InvalidInputError(
            "barrier_type must be one of " + ", ".join(repr(n) for n in BARRIER_TYPES)
        )
    shift = shifted_barrier(
        H=H, sigma=sigma, dt=T / n_monitoring, down=barrier_type_l.startswith("down"),
        toward_spot=False,
    )
    return barrier_price(
        S=S, K=K, T=T, r=r, sigma=sigma, q=q, H=shift, rebate=rebate,
        barrier_type=barrier_type_l, kind=kind,
    )


def shifted_barrier(*, H: float, sigma: float, dt: float, down: bool, toward_spot: bool) -> float:
    """The BGK-shifted barrier `H exp(+- beta sigma sqrt(dt))`.

    Two callers want opposite signs and confusing them is the easy mistake, so
    the direction is a named argument rather than a sign the caller supplies:

    - `toward_spot=False` moves the barrier **away** from the spot (down for a
      down barrier, up for an up one). That is the shift that turns the
      continuous closed form into an approximation of the *discrete* price
      (:func:`bgk_continuity_corrected_price`).
    - `toward_spot=True` moves it **toward** the spot. That is the shift a
      discretely monitored simulation applies to its own barrier so that its
      estimate approximates the *continuous* price -- the same relation read
      backwards (`qpl.engines.mc.barrier`).
    """
    if dt < 0.0:
        raise InvalidInputError("dt must be >= 0")
    sign = 1.0 if (down == toward_spot) else -1.0
    return H * math.exp(sign * BGK_BETA * sigma * math.sqrt(dt))


def price_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
) -> PriceResult:
    """Price a continuously monitored single-barrier option in closed form.

    Raises
    ------
    NotSupportedError
        If `option.monitoring` is a discrete schedule. The message names
        `method="mc"` and :func:`bgk_continuity_corrected_price`.
    """
    if not option.is_continuous:
        raise NotSupportedError(_DISCRETE_REFUSAL)

    t = option.expiry
    s = market.spot
    value = barrier_price(
        S=s,
        K=option.strike,
        T=t,
        r=market.rate(t),
        sigma=model.sigma,
        q=market.dividend_yield(t),
        H=option.barrier,
        rebate=option.rebate,
        barrier_type=option.barrier_type,
        kind=option.kind,
    )
    meta: dict[str, object] = {
        "method": "analytic",
        "model": "BlackScholes",
        "instrument": "barrier",
        "barrier_type": option.barrier_type,
        "monitoring": "continuous",
        "barrier": option.barrier,
        "rebate": option.rebate,
        "touched_at_inception": option.is_touched(s),
    }
    return PriceResult(value=value, meta=meta)


GREEK_BUMPS: dict[str, float] = {
    "spot": 1e-3,
    "sigma": 1e-4,
    "r": 1e-4,
    "time": 1e-4,
}
"""Central-difference steps for :func:`greeks_barrier`, in absolute units.

The closed form is smooth in every argument *away from the barrier*, so a
central difference is `O(h^2)` there and the steps are chosen small enough for
the truncation term to sit below the round-off floor of a difference of two
`O(10)` prices. Measured worst **relative** residual against the vanilla
engine's closed-form Greeks over four points, taken in the `H -> 0` limit where
a down-and-out call *is* the vanilla: 1.948e-10 (delta), 1.440e-06 (gamma),
2.011e-08 (vega), 4.291e-09 (theta) and 2.286e-08 (rho). `spot=1e-3` is stated
absolutely because the engine has no scale to normalise by; the reference spots
here are `O(100)`, so it is a 1e-05 relative move.

Gamma is the loose one and unavoidably so: a second difference divides by
`h^2 = 1e-06`, so it starts several decimal orders behind delta before anything
about a barrier is involved.
"""


def greeks_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
) -> GreeksResult:
    """Barrier Greeks by central differences of the closed form.

    All five come from bump-and-revalue on :func:`barrier_price` with the steps
    in :data:`GREEK_BUMPS`; theta is `-dV/dT`, the roll of the whole contract
    (barrier and all) toward expiry, which is the only reading a contract whose
    monitoring window shrinks admits.

    **There are no analytic barrier Greeks in this slice**, and that is a
    deliberate scope line rather than an oversight: the closed form is a sum of
    six blocks each of which differentiates into several terms, the derivative
    of `F` in particular involving `dlambda/dsigma`, and a hand-derived version
    would need its own convergence evidence before it could be trusted over a
    difference of a formula that is already checked to 1e-13 against QuantLib.
    What is *measured* here is the difference quotient, and what it is measured
    against is the vanilla closed form in the limit where a barrier degenerates
    into one.

    Where the derivative actually stops existing -- not where the slice expected
    ----------------------------------------------------------------------------
    The slice this engine was written for said to "document the blow-up of
    gamma as `S -> H`". **There is no such blow-up, and the measurement says so
    plainly.** On the reference down-and-out call (`K = 100`, `H = 95`,
    `T = 0.5`, `r = 8%`, `q = 4%`, `sigma = 25%`, zero rebate), sweeping the
    spot down to the barrier gives

        S        110      105      100       96      95.5     95.1    95.001
        value    13.2972  8.9034   4.5126   0.9206  0.4618   0.0926  0.00093
        delta     0.8843   0.8754   0.8850   0.9150  0.9205   0.9253  0.9265
        gamma    +0.0029  +0.0003  -0.0046  -0.0107 -0.0116  -0.0123 -0.0125

    -- a value vanishing **linearly** in `S - H` with delta and gamma both
    converging to finite limits (0.9265 and -0.0125). The continuously
    monitored value is smooth at the barrier for any `T > 0`, because the
    reflected terms are analytic there; the only thing the barrier does to the
    shape is flip the **sign** of gamma. A knock-out call near its barrier is
    *concave* (gamma -0.0125 against the vanilla's +0.0217 at the same point),
    since the value is being pinned toward the rebate from one side rather than
    bending away from the strike.

    The real singularity is at the **corner** `(S = H, t = T)`, and it is
    reached by letting `T -> 0` at a spot near the barrier -- and only when the
    terminal payoff is discontinuous across `H`, i.e. when
    `vanilla(H) != rebate`. Measured on a down-and-out call with `K = 90 < H`
    (so the payoff jumps by `H - K = 5` across the barrier at expiry) at
    `S = 96`:

        T         0.5      0.1      0.02     0.005    0.001
        delta     1.376    1.586    2.150    2.980    3.185
        gamma    -0.018   -0.029   -0.123   -0.721   -3.842

    and at `S = 95.5`, gamma -0.0186, -0.0252, -0.0791, -0.4380, -3.7884 on the
    same `T` ladder. That is the blow-up, in the variable it actually happens
    in. With `K = 100 > H` the same contract's payoff is continuous across the
    barrier (both sides are zero) and no such growth appears at all.

    The engine does not refuse any of those points, and the estimator's own
    limit is reported rather than guarded. A central difference with
    `h = 1e-3` straddles the barrier once `|S - H| < h`, at which point the
    bumped price jumps to the rebate and the quotient reports the jump; that is
    a real estimator meeting a real contract, and hiding it behind a guard
    would replace a visibly wrong number with an invisible one. What the caller
    gets instead is `meta["distance_to_barrier_in_bumps"]`, `|S - H| / h` in
    units of the spot step, which is below 1 exactly when the difference
    quotient has stopped being a derivative.

    Raises
    ------
    NotSupportedError
        If `option.monitoring` is a discrete schedule.
    InvalidInputError
        At `T = 0` or `sigma = 0`, where the value is a step function of the
        spot (the barrier is either touched on the deterministic path or it is
        not) and its derivative is a Dirac mass -- the same two limits the
        vanilla and digital analytic engines refuse.
    """
    if not option.is_continuous:
        raise NotSupportedError(_DISCRETE_REFUSAL)
    t = option.expiry
    if t <= 0.0:
        raise InvalidInputError("T must be > 0 for Greeks")
    if model.sigma <= 0.0:
        raise InvalidInputError("sigma must be > 0 for Greeks")

    s = market.spot
    r = market.rate(t)
    q = market.dividend_yield(t)
    sigma = model.sigma

    def value(
        *, spot: float | None = None, vol: float | None = None,
        rate: float | None = None, expiry: float | None = None,
    ) -> float:
        return barrier_price(
            S=s if spot is None else spot,
            K=option.strike,
            T=t if expiry is None else expiry,
            r=r if rate is None else rate,
            sigma=sigma if vol is None else vol,
            q=q,
            H=option.barrier,
            rebate=option.rebate,
            barrier_type=option.barrier_type,
            kind=option.kind,
        )

    h_s = GREEK_BUMPS["spot"]
    h_v = GREEK_BUMPS["sigma"]
    h_r = GREEK_BUMPS["r"]
    h_t = min(GREEK_BUMPS["time"], 0.5 * t)

    base = value()
    up, down = value(spot=s + h_s), value(spot=s - h_s)
    delta = (up - down) / (2.0 * h_s)
    gamma = (up - 2.0 * base + down) / (h_s * h_s)
    vega = (value(vol=sigma + h_v) - value(vol=max(sigma - h_v, 0.0))) / (
        (sigma + h_v) - max(sigma - h_v, 0.0)
    )
    rho = (value(rate=r + h_r) - value(rate=r - h_r)) / (2.0 * h_r)
    theta = -(value(expiry=t + h_t) - value(expiry=t - h_t)) / (2.0 * h_t)

    meta: dict[str, object] = {
        "method": "analytic",
        "model": "BlackScholes",
        "instrument": "barrier",
        "barrier_type": option.barrier_type,
        "monitoring": "continuous",
        "fd": "central",
        "bumps": {"spot": h_s, "sigma": h_v, "r": h_r, "time": h_t},
        "theta_convention": "roll: -dV/dT with the barrier window shrinking",
        "distance_to_barrier_in_bumps": abs(s - option.barrier) / h_s,
    }
    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        meta=meta,
    )
