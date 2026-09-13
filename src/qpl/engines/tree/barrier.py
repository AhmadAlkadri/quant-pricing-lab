"""Single barriers on a recombining binomial lattice, and the Boyle-Lau sawtooth.

What the lattice does to a barrier
----------------------------------
The backward induction is the vanilla one with one line added: after each
roll-back step, every node whose spot is on the dead side of the barrier is
overwritten -- by the rebate for a knock-out, by zero for the survival claim
that prices a knock-in's rebate. That is the whole engine. The payoff is
otherwise untouched and the lattice is the shared `lattice_parameters`, so
CRR and Leisen-Reimer differ here only in `u`, `d` and `p`.

The consequence is the entire subject. A lattice cannot knock out *at* the
barrier; it knocks out at the first node beyond it. On a CRR lattice the node
spots form a single geometric ladder `S u^i`, `u = exp(sigma sqrt(dt))`, so the
tree's *effective* barrier for a down type is

    H_eff = S exp( -ceil(lambda) sigma sqrt(dt) ) <= H,
    lambda = log(S / H) / (sigma sqrt(dt)) = lambda_0 sqrt(n),
    lambda_0 = log(S / H) / (sigma sqrt(T)),

which sits **below** the contractual barrier by a factor that depends on the
fractional part of `lambda`. Since `lambda` grows like `sqrt(n)`, that
fractional part cycles as `n` increases, and so does the error: a sawtooth in
`n` rather than the parity oscillation a vanilla shows. This is the effect
Boyle and Lau (1994), "Bumping up against the barrier with the binomial
method", *Journal of Derivatives* 1(4), 6-14, identify; the quantitative
picture below is measured in this repository and no number is taken from them.

The Boyle-Lau step counts
-------------------------
The remedy is to choose `n` so that `lambda` is (just below) an integer, which
places a layer of nodes essentially on the barrier and makes `H_eff` almost
exactly `H`. Setting `lambda_0 sqrt(n) = k` and rounding down,

    n_k = floor( k^2 / lambda_0^2 ) = floor( k^2 sigma^2 T / log(S/H)^2 ),

which is :func:`boyle_lau_steps`. Along that subsequence the error should be
smooth and convergent; whether it *is*, and at what order, is measured in
`tests/test_barrier_tree_convergence.py` rather than asserted here.

Leisen-Reimer is included and is **not** expected to help, for a reason worth
stating in advance: its construction places the *strike* where the binomial and
normal tails already agree, and says nothing at all about a second level. Its
lattice is not even spot-centred (`u d != 1`), so the node spots are not a
single geometric ladder and there is no `n` that puts a layer on the barrier by
construction. Whether the sawtooth is smaller, larger or the same size is an
empirical question, and the test answers it.

Continuous monitoring only
--------------------------
This engine approximates the **continuously** monitored contract: it tests the
barrier at every time level, which is as close to continuous as a lattice with
`n` steps can get. A discretely monitored contract has its own observation
dates and is refused here, naming `method="mc"` -- knocking out only at the
levels that coincide with monitoring dates is a real construction, but it needs
`n` to be a multiple of `m` *and* a barrier-aligned `n`, which are two
conditions on one integer, and it is not in this slice.

References for the two lattice schemes are in `qpl.engines.tree.lattice`. The
closed forms this engine is measured against are in
`qpl.engines.analytic.barrier`. Derman, Kani, Ergener and Bardhan (1995),
"Enhanced numerical methods for options with barriers", *Risk* 8(6), give the
interpolation remedy that is the alternative to choosing `n`; it is cited here
and deliberately not implemented, so that the slice measures one remedy
properly instead of two badly.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import BarrierOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..analytic.barrier import barrier_price
from ..base import GreeksResult, PriceResult
from .lattice import BinomialLattice, crr_spot_level, lattice_parameters
from .pricers import TreeConfig, _meta, _validate

__all__ = [
    "barrier_layer_index",
    "boyle_lau_steps",
    "greeks_barrier",
    "price_barrier",
]

_DISCRETE_REFUSAL = (
    "method='tree' approximates the CONTINUOUSLY monitored barrier: it tests "
    "the barrier at every one of its n time levels, which is the lattice's "
    "nearest thing to continuous monitoring. A contract with its own "
    "monitoring dates is priced by method='mc', which simulates exactly those "
    "dates. Knocking out only at the levels that coincide with the monitoring "
    "schedule is a real construction, but it needs n to be a multiple of m and "
    "barrier-aligned at once; it is not in this slice."
)

_GREEKS_REFUSAL = (
    "Tree Greeks are not available for a barrier option in this slice. The "
    "lattice estimators read levels 1 and 2 and divide by node spacings, so "
    "they amplify the price's Boyle-Lau sawtooth by 1/sqrt(dt) in delta and "
    "1/dt in gamma -- on a quantity that is already the slice's headline "
    "negative finding. Reporting them would be reporting the sawtooth with a "
    "Greek's name on it. Use method='analytic', whose Greeks are central "
    "differences of the closed form."
)


def barrier_layer_index(
    *, spot: float, barrier: float, sigma: float, expiry: float, n_steps: int
) -> float:
    """`lambda = log(S/H) / (sigma sqrt(dt))`: where the barrier falls, in layers.

    Positive for a down barrier and negative for an up one. Its **fractional
    part** is what drives the sawtooth: the lattice knocks out at
    `ceil(|lambda|)` layers from the root, so an integer `lambda` puts nodes on
    the barrier and a `lambda` just above an integer puts the effective barrier
    a whole layer past it.

    Exact for CRR, where `u d = 1` makes the node spots one geometric ladder.
    On a Leisen-Reimer lattice it is a *characteristic scale* rather than an
    index -- the spots are `S u^j d^{L-j}` with `u d != 1`, so different time
    levels straddle the barrier differently and no single index exists. It is
    still reported, because it is the quantity the CRR sawtooth is periodic in
    and the comparison is the point.
    """
    if sigma <= 0.0 or expiry <= 0.0 or n_steps < 1:
        raise InvalidInputError("sigma, expiry and n_steps must be positive")
    return math.log(spot / barrier) / (sigma * math.sqrt(expiry / n_steps))


def boyle_lau_steps(
    layer: int, *, spot: float, barrier: float, sigma: float, expiry: float
) -> int:
    """The largest `n` that puts layer `layer` at or just below the barrier.

    `n_k = floor(k^2 sigma^2 T / log(S/H)^2)`; see the module docstring. The
    floor is what makes the effective barrier land *at or beyond* the
    contractual one rather than short of it, so the lattice never prices a
    barrier that is closer to the spot than the contract's.

    Raises
    ------
    InvalidInputError
        If `layer < 1`, if the spot sits on the barrier (no ladder), or if the
        resulting `n` would be below 1.
    """
    if layer < 1:
        raise InvalidInputError("layer must be >= 1")
    if spot <= 0.0 or barrier <= 0.0 or spot == barrier:
        raise InvalidInputError("spot and barrier must be positive and distinct")
    if sigma <= 0.0 or expiry <= 0.0:
        raise InvalidInputError("sigma and expiry must be > 0")
    lambda_0 = abs(math.log(spot / barrier)) / (sigma * math.sqrt(expiry))
    n = math.floor(layer * layer / (lambda_0 * lambda_0))
    if n < 1:
        raise InvalidInputError(
            f"layer {layer} gives n_steps = {n}: the barrier is too far from "
            "the spot for this layer index; use a larger layer"
        )
    return n


def _rollback_with_barrier(
    *,
    lattice: BinomialLattice,
    spot: float,
    terminal: np.ndarray,
    dead_value: float,
    is_down: bool,
    barrier: float,
) -> tuple[float, float | None]:
    """Roll `terminal` back to the root, overwriting dead nodes at every level.

    Returns `(root value, extreme knocked node spot)`. The second is the
    lattice's **effective barrier**: the node furthest from the spot on the live
    side that is still counted as dead, i.e. what the tree actually priced
    instead of `barrier`. It is `None` when no node was ever knocked, which
    happens on a coarse lattice whose reach does not span the barrier and is a
    fact the caller should see rather than a case to hide.
    """
    # A level-`L` spot is `spot * d**L * (u/d)**j`, so one `O(n)` power array
    # plus a scalar per level lays out every level. Going through
    # `crr_spot_level` instead would rebuild two power arrays at every level
    # and triple the cost of the roll-back for no gain.
    n = lattice.n_steps
    up, down_move = lattice.up, lattice.down
    ratio = np.power(up / down_move, np.arange(n + 1, dtype=float))

    values = terminal
    effective: float | None = None
    disc, p = lattice.discount, lattice.p

    def level_spots(level: int) -> np.ndarray:
        if level == n:
            # The terminal level is laid out exactly as the vanilla engine lays
            # it out, so that a barrier no node can reach reproduces the vanilla
            # lattice price bit for bit. The two formulas agree in exact
            # arithmetic and differ in the last bits, which is invisible in the
            # mask but visible in the payoff.
            return crr_spot_level(spot=spot, up=up, down=down_move, level=n)
        return spot * (down_move**level) * ratio[: level + 1]

    def apply(level: int, values: np.ndarray) -> np.ndarray:
        nonlocal effective
        spots = level_spots(level)
        dead = spots <= barrier if is_down else spots >= barrier
        if dead.any():
            extreme = float(spots[dead].max() if is_down else spots[dead].min())
            if effective is None:
                effective = extreme
            elif (extreme > effective) if is_down else (extreme < effective):
                effective = extreme
            values = np.where(dead, dead_value, values)
        return values

    values = apply(n, values)
    for level in range(n - 1, 0, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        values = apply(level, values)
    values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
    return float(values[0]), effective


def _terminal_payoff(spots: np.ndarray, option: BarrierOption) -> np.ndarray:
    if option.kind == "call":
        return np.maximum(spots - option.strike, 0.0)
    return np.maximum(option.strike - spots, 0.0)


def _lattice(
    option: BarrierOption, model: BlackScholesModel, market: Market, cfg: TreeConfig
) -> BinomialLattice:
    t = option.expiry
    return lattice_parameters(
        scheme=cfg.scheme,
        spot=market.spot,
        strike=option.strike,
        sigma=model.sigma,
        expiry=t,
        rate=market.rate(t),
        dividend_yield=market.dividend_yield(t),
        n_steps=cfg.n_steps,
    )


def price_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> PriceResult:
    """Price a continuously monitored barrier by backward induction on a lattice.

    A knock-out is rolled back with every dead node overwritten by the rebate.
    A knock-in is assembled on the **same lattice** from three roll-backs, so
    that in-out parity holds on the lattice exactly and not merely in the limit:

        knock-in = vanilla - knock-out(rebate = 0) + rebate * survival,

    where `survival` is the discounted lattice probability of never reaching the
    barrier, computed as a roll-back of the terminal payoff `1` with dead nodes
    set to `0`. The first two terms are the parity; the third is the knock-in
    rebate, which is paid at expiry exactly when the barrier was never touched.

    Returns
    -------
    PriceResult
        Value, with `meta` carrying the lattice parameters plus
        `barrier_layer_index` (`log(S/H) / (sigma sqrt(dt))`),
        `effective_barrier` (the node the lattice actually knocked at) and
        `effective_barrier_ratio` (`H_eff / H`, which is `1` exactly when a
        layer lands on the barrier and is what the sawtooth is periodic in).

    Raises
    ------
    NotSupportedError
        Discrete monitoring; the message names `method="mc"`.
    InvalidInputError
        As `qpl.engines.tree.price_european`.
    """
    if not option.is_continuous:
        raise NotSupportedError(_DISCRETE_REFUSAL)
    _validate(cfg, min_steps=1)

    t = option.expiry
    s0 = market.spot
    meta: dict[str, object]

    if option.is_touched(s0):
        value = (
            float(option.rebate)
            if option.is_knock_out
            else barrier_price(
                S=s0, K=option.strike, T=t, r=market.rate(t), sigma=model.sigma,
                q=market.dividend_yield(t), H=option.barrier, rebate=option.rebate,
                barrier_type=option.barrier_type, kind=option.kind,
            )
        )
        meta = _meta(None, cfg, degenerate="touched_at_inception")
        meta["instrument"] = "barrier"
        meta["barrier_type"] = option.barrier_type
        meta["touched_at_inception"] = True
        return PriceResult(value=value, meta=meta)

    if t == 0.0 or model.sigma == 0.0:
        # A fact about the model rather than about the lattice: delegate to the
        # closed form rather than keeping a second copy of the deterministic
        # first-passage argument.
        value = barrier_price(
            S=s0, K=option.strike, T=t, r=market.rate(t), sigma=model.sigma,
            q=market.dividend_yield(t), H=option.barrier, rebate=option.rebate,
            barrier_type=option.barrier_type, kind=option.kind,
        )
        degenerate = "expiry" if t == 0.0 else "zero_vol"
        lattice = None if t == 0.0 else _lattice(option, model, market, cfg)
        meta = _meta(lattice, cfg, degenerate=degenerate)
        meta["instrument"] = "barrier"
        meta["barrier_type"] = option.barrier_type
        meta["touched_at_inception"] = False
        return PriceResult(value=float(value), meta=meta)

    lattice = _lattice(option, model, market, cfg)
    n = lattice.n_steps
    terminal_spots = crr_spot_level(
        spot=s0, up=lattice.up, down=lattice.down, level=n
    )
    payoff = _terminal_payoff(terminal_spots, option)

    knock_out, effective = _rollback_with_barrier(
        lattice=lattice, spot=s0, terminal=payoff.copy(),
        dead_value=option.rebate, is_down=option.is_down, barrier=option.barrier,
    )

    if option.is_knock_out:
        value = knock_out
    else:
        # In-out parity on this lattice, plus the knock-in rebate leg.
        bare_out, _ = _rollback_with_barrier(
            lattice=lattice, spot=s0, terminal=payoff.copy(), dead_value=0.0,
            is_down=option.is_down, barrier=option.barrier,
        )
        vanilla = _vanilla_rollback(lattice, payoff.copy())
        value = vanilla - bare_out
        if option.rebate > 0.0:
            survival, _ = _rollback_with_barrier(
                lattice=lattice, spot=s0,
                terminal=np.ones_like(terminal_spots), dead_value=0.0,
                is_down=option.is_down, barrier=option.barrier,
            )
            value += option.rebate * survival

    meta = _meta(lattice, cfg, degenerate=None)
    meta["instrument"] = "barrier"
    meta["barrier_type"] = option.barrier_type
    meta["barrier"] = option.barrier
    meta["rebate"] = option.rebate
    meta["monitoring"] = "continuous"
    meta["touched_at_inception"] = False
    meta["barrier_layer_index"] = barrier_layer_index(
        spot=s0, barrier=option.barrier, sigma=model.sigma, expiry=t, n_steps=n
    )
    meta["effective_barrier"] = effective
    meta["effective_barrier_ratio"] = (
        None if effective is None else effective / option.barrier
    )
    return PriceResult(value=float(value), meta=meta)


def _vanilla_rollback(lattice: BinomialLattice, values: np.ndarray) -> float:
    """Plain backward induction, no barrier: the parity leg of a knock-in."""
    disc, p = lattice.discount, lattice.p
    for _ in range(lattice.n_steps):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
    return float(values[0])


def greeks_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> GreeksResult:
    """Always raises `NotSupportedError`; see :data:`_GREEKS_REFUSAL`.

    Registered rather than omitted so the caller gets a reason instead of the
    generic unsupported-combination message.
    """
    raise NotSupportedError(_GREEKS_REFUSAL)
