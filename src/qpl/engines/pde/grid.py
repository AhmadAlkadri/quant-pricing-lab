"""Spot grids for the finite-difference engines: uniform and `sinh`.

Everything in this module is about *where the nodes go*. The time march, the
payoffs and the boundary data live in `qpl.engines.pde.pricers`; what changes
here is the spatial discretisation those read, and the three-point stencil that
turns a list of node positions into the Black-Scholes operator.

The non-uniform three-point stencil
-----------------------------------
Let `S_{i-1} < S_i < S_{i+1}` be three consecutive nodes and write

    h- = S_i - S_{i-1},      h+ = S_{i+1} - S_i.

Expand a smooth `V` about `S_i`:

    V_{i+1} = V_i + h+ V' + h+^2/2 V'' + h+^3/6 V''' + O(h^4),
    V_{i-1} = V_i - h- V' + h-^2/2 V'' - h-^3/6 V''' + O(h^4).

Eliminating `V''` between the two (multiply the first by `h-^2`, the second by
`h+^2`, subtract) gives the first derivative, and eliminating `V'` (multiply by
`h-` and `h+` respectively and add) gives the second:

    V'(S_i)  ~  -h+ / (h- (h- + h+)) * V_{i-1}
                + (h+ - h-) / (h- h+) * V_i
                + h- / (h+ (h- + h+)) * V_{i+1},

    V''(S_i) ~   2 / (h- (h- + h+)) * V_{i-1}
                - 2 / (h- h+)       * V_i
                + 2 / (h+ (h- + h+)) * V_{i+1}.

At `h- = h+ = h` these collapse to the textbook central differences
`(V_{i+1} - V_{i-1}) / (2h)` and `(V_{i+1} - 2 V_i + V_{i-1}) / h^2`.

**Order, stated honestly.** Carrying the next term in the expansions gives

    V'  : local truncation error  -(h+ h-)/6 V''' + O(h^3),
    V'' : local truncation error  (h+ - h-)/3 V''' + O(h^2).

So the first-derivative stencil is second order on any grid, while the
second-derivative stencil is **first order pointwise** unless `h+ = h-`: its
leading error is proportional to the *difference* of the neighbouring spacings.
That is the whole subtlety of non-uniform meshes, and it is why a naive reading
predicts a first-order scheme.

The global error is nonetheless second order when the grid is the image of a
uniform grid under a smooth, strictly increasing map `S = g(xi)`, `xi_i = i dxi`:
then

    h+ - h- = g(xi_{i+1}) - 2 g(xi_i) + g(xi_{i-1}) = g''(xi_i) dxi^2 + O(dxi^4),
    h+ + h- = 2 g'(xi_i) dxi + O(dxi^3),

so `(h+ - h-) / (h+ + h-) = O(dxi)` and the offending term is
`O(dxi) * O(h) = O(h^2)` after all -- the pointwise first-order term is
*multiplied by a spacing that is itself shrinking*. Equivalently: the scheme is
second order in the transformed variable, in which the grid is uniform. This is
the standard argument (Duffy (2006), *Finite Difference Methods in Financial
Engineering*, chapter on non-uniform meshes, states the consistency order of
this stencil; Tavella and Randall (2000), *Pricing Financial Instruments: The
Finite Difference Method*, chapter 5, is the coordinate-transformation view).
It is a statement about grids from smooth maps, not about arbitrary node lists,
and it is **measured** in `tests/test_pde_nonuniform_grid.py` rather than taken
on trust.

The `sinh` mesh
---------------
The mesh is defined by its *density*: nodes are uniform in a coordinate `xi`
whose derivative is large near the points that matter. Take

    xi(S) = sum_j asinh( (S - c_j) / alpha ),       alpha > 0,

so that

    dxi/dS = sum_j 1 / sqrt( alpha^2 + (S - c_j)^2 ),

a sum of bumps of height `1/alpha` and width `alpha` centred on the critical
points `c_j`. Nodes cluster where `dxi/dS` is large, i.e. within about `alpha`
of each `c_j`, and thin out like `1/|S - c_j|` away from them. With a single
critical point the map inverts in closed form to

    S(xi) = c + alpha sinh(xi),

which is the mesh of In 't Hout and Foulon (2010), "ADI finite difference
schemes for option pricing in the Heston model with correlation", *IJNAM* 7(2),
303-320, section 3 -- cited here for the mesh formula alone, since this
repository has no ADI scheme and no Heston model. With several critical points
there is no closed-form inverse and the nodes are obtained by bisection on the
monotone `xi(S)`; the anchor nodes are then overwritten with their exact values,
so "the barrier is on a node" is exact and not merely converged.

`alpha` is supplied as the dimensionless `PDEConfig.concentration`, multiplied
by the nominal domain length: `alpha = concentration * (s_max - s_min)`. Large
`concentration` makes `asinh` nearly linear and the grid nearly uniform; small
`concentration` piles nodes onto the critical points and starves the tails.

Anchors: points that must be nodes, and the strike that must not
----------------------------------------------------------------
Two different placement requirements meet on the same grid.

*The barrier must be a node.* A knock-out's boundary condition is imposed at
`S = H`; if `H` falls between two nodes, the scheme imposes it at whichever node
it is applied to, which is a *different contract* -- the barrier is displaced by
`O(ds)` and the price error is first order (Zvan, Vetzal and Forsyth (2000),
"PDE methods for pricing barrier options", *Journal of Economic Dynamics and
Control* 24, 1563-1590). `node_points` is the list of such points; each becomes
a segment boundary of the `xi` grid and is written back exactly.

*The strike must not be a node.* The payoff kink at `K` costs the scheme a
power of `ds` when it lands on a node and does not when it lands at a cell
midpoint (Slice 0, `docs/notes/pde_strike_alignment.md`). So the spacing of the
segment containing the strike is nudged until `K` is the arithmetic midpoint of
its cell -- exactly as the uniform grid nudges `ds` -- which is possible
whenever that segment's far end is free to move (`upper is None`, i.e. `s_max`
absorbs the adjustment). On a uniform grid the `xi` map is the identity and the
half-integer rule already gives the midpoint exactly; on a `sinh` grid the
half-integer rule in `xi` misses the `S`-midpoint by `O(h^2)`, so the spacing is
refined by bisection until the `S`-midpoint condition holds to round-off.

When both ends of the strike's segment are pinned -- an up-and-out barrier, say,
whose domain is `[0, H]` with both ends fixed -- the midpoint cannot be imposed
and the realised offset is reported in `SpotGrid.meta["strike_cell_offset"]`
rather than silently accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...exceptions import InvalidInputError

__all__ = [
    "GRID_KINDS",
    "SINH",
    "UNIFORM",
    "SpotGrid",
    "build_spot_grid",
    "delta_gamma_nodes",
    "inverse_sinh_coordinate",
    "operator_coefficients",
    "sinh_coordinate",
    "stencil_weights",
]

UNIFORM = "uniform"
SINH = "sinh"
GRID_KINDS: tuple[str, ...] = (UNIFORM, SINH)
"""The two spatial meshes, as data, for validation and messages."""

_INVERSE_BISECTIONS = 100
"""Bisection steps used to invert `xi(S)`.

The bracket is the whole domain, so 100 halvings take it below `1e-30` of its
width -- far past double precision -- and the loop is vectorised over the whole
grid, so the cost is 100 passes over an array of a few thousand doubles. A
Newton iteration would be faster and would need a safeguard anyway; bisection
is chosen because it is unconditionally convergent on a monotone function and
because a grid that depends on an iteration count is a reproducibility hazard.
"""

_MIDPOINT_BISECTIONS = 80
"""Bisection steps used to place the strike at the midpoint of its `sinh` cell."""


def sinh_coordinate(
    s: np.ndarray | float, points: tuple[float, ...], alpha: float
) -> np.ndarray:
    """`xi(S) = sum_j asinh((S - c_j) / alpha)`: the mesh density's antiderivative.

    Strictly increasing in `S` for any `alpha > 0` and any set of points, since
    every term is. See the module docstring for why this is the mesh generator.
    """
    values = np.asarray(s, dtype=float)
    out = np.zeros_like(values, dtype=float)
    for c in points:
        out = out + np.arcsinh((values - c) / alpha)
    return out


def inverse_sinh_coordinate(
    xi: np.ndarray | float,
    points: tuple[float, ...],
    alpha: float,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Invert :func:`sinh_coordinate` on `[lower, upper]` by vectorised bisection.

    `xi` must lie within `[xi(lower), xi(upper)]`; values outside are clamped to
    the bracket, which is what the endpoints of a grid do anyway.
    """
    target = np.asarray(xi, dtype=float)
    lo = np.full(target.shape, float(lower))
    hi = np.full(target.shape, float(upper))
    for _ in range(_INVERSE_BISECTIONS):
        mid = 0.5 * (lo + hi)
        too_low = sinh_coordinate(mid, points, alpha) < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class SpotGrid:
    """A spot discretisation: the node positions and everything derived from them.

    Parameters
    ----------
    s
        Ascending node positions, `n_s + 1` of them. Read-only.
    kind
        `"uniform"` or `"sinh"`.
    uniform
        Whether the spacing is constant. `True` only for `kind="uniform"`; it is
        a separate field because it is what the operator and the Greek stencils
        branch on, and because a `sinh` grid with a huge `concentration` is
        *nearly* uniform without being uniform.
    ds
        The spacing on a uniform grid. On a `sinh` grid this is the **mean**
        spacing, reported so that `meta` carries one comparable length scale;
        nothing computes with it there.
    meta
        Grid description for `PriceResult.meta`.
    """

    s: np.ndarray
    kind: str
    uniform: bool
    ds: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_s(self) -> int:
        """Number of intervals."""
        return int(self.s.size) - 1

    @property
    def s_min(self) -> float:
        return float(self.s[0])

    @property
    def s_max(self) -> float:
        return float(self.s[-1])

    @property
    def h_minus(self) -> np.ndarray:
        """`S_i - S_{i-1}` at the interior nodes `i = 1 .. n_s - 1`."""
        return self.s[1:-1] - self.s[:-2]

    @property
    def h_plus(self) -> np.ndarray:
        """`S_{i+1} - S_i` at the interior nodes `i = 1 .. n_s - 1`."""
        return self.s[2:] - self.s[1:-1]

    @property
    def cell_faces(self) -> np.ndarray:
        """Midpoints between consecutive nodes, `n_s` of them.

        The cell of node `i` is `[faces[i-1], faces[i]]`; the two end nodes'
        cells are completed by reflecting the single face they have.
        """
        return 0.5 * (self.s[:-1] + self.s[1:])

    def index_of(self, level: float) -> int:
        """Index of the node equal to `level`, or `-1` if there is none.

        Exact equality, deliberately: the anchors are written back exactly, so
        a tolerance here would hide a grid that missed its anchor.
        """
        where = np.flatnonzero(self.s == level)
        return int(where[0]) if where.size else -1


def stencil_weights(
    grid: SpotGrid,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`((w1-, w1_0, w1+), (w2-, w2_0, w2+))` at the interior nodes.

    The three-point weights derived in the module docstring, for the first and
    second derivatives respectively. Valid for any ascending node list; the
    caller decides whether the grid they came from earns second-order accuracy.
    """
    hm = grid.h_minus
    hp = grid.h_plus
    hs = hm + hp
    first = (
        -hp / (hm * hs),
        (hp - hm) / (hm * hp),
        hm / (hp * hs),
    )
    second = (
        2.0 / (hm * hs),
        -2.0 / (hm * hp),
        2.0 / (hp * hs),
    )
    return first, second


def operator_coefficients(
    grid: SpotGrid, sigma: float, r: float, q: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub/diagonal/super coefficients `(a, b, c)` of the Black-Scholes operator.

        L V = 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V,
        (L V)_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1}.

    On a uniform grid this evaluates the **same expressions in the same order**
    as the pre-Slice-13 engine, which is what keeps every existing PDE output
    bit-for-bit; `tests/test_pde_nonuniform_grid.py` asserts that with `==`
    against the coefficients written out in the test. On a non-uniform grid it
    uses the three-point weights of :func:`stencil_weights`.
    """
    s_inner = grid.s[1:-1]
    if grid.uniform:
        ds = grid.ds
        diffusion = 0.5 * sigma * sigma * (s_inner**2) / (ds * ds)
        drift = (r - q) * s_inner / (2.0 * ds)
        a = diffusion - drift
        b = -(sigma * sigma) * (s_inner**2) / (ds * ds) - r
        c = diffusion + drift
        return a, b, c

    (w1m, w1_0, w1p), (w2m, w2_0, w2p) = stencil_weights(grid)
    diffusion = 0.5 * sigma * sigma * (s_inner**2)
    drift = (r - q) * s_inner
    a = diffusion * w2m + drift * w1m
    b = diffusion * w2_0 + drift * w1_0 - r
    c = diffusion * w2p + drift * w1p
    return a, b, c


def delta_gamma_nodes(grid: SpotGrid, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of `v` at the interior nodes.

    Index `j` of the returned arrays is grid node `j + 1`. Uniform grids take
    the pre-Slice-13 central differences unchanged, so grid Greeks are
    bit-for-bit what they were; non-uniform grids take the three-point weights.
    """
    if grid.uniform:
        ds = grid.ds
        delta = (v[2:] - v[:-2]) / (2.0 * ds)
        gamma = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (ds * ds)
        return delta, gamma

    (w1m, w1_0, w1p), (w2m, w2_0, w2p) = stencil_weights(grid)
    delta = w1m * v[:-2] + w1_0 * v[1:-1] + w1p * v[2:]
    gamma = w2m * v[:-2] + w2_0 * v[1:-1] + w2p * v[2:]
    return delta, gamma


def _legacy_uniform_grid(strike: float, spot: float, cfg: Any) -> SpotGrid:
    """`[0, s_max]` uniform, with the Slice 0 strike alignment. Unchanged.

    Byte-for-byte the arithmetic of the pre-Slice-13 `_build_grid`, kept in one
    place so that the default configuration cannot drift: every expression, and
    the order in which they are evaluated, is the original.
    """
    s_max = cfg.s_max if cfg.s_max is not None else cfg.s_max_multiplier * spot
    ds = s_max / cfg.n_s

    if cfg.strike_alignment == "midpoint":
        # Nearest half-integer node position for the strike.
        j = max(round(strike / ds - 0.5), 0)
        ds = strike / (j + 0.5)
        s_max = ds * cfg.n_s

    if not (0.0 < strike < s_max):
        raise InvalidInputError(
            f"strike {strike} must lie strictly inside the spot grid (0, {s_max})"
        )

    if cfg.strike_alignment == "midpoint":
        s_grid = ds * np.arange(cfg.n_s + 1, dtype=float)
    else:
        s_grid = np.linspace(0.0, s_max, cfg.n_s + 1)

    s_grid.setflags(write=False)
    return SpotGrid(
        s=s_grid,
        kind=UNIFORM,
        uniform=True,
        ds=float(ds),
        meta={
            "grid": UNIFORM,
            "s_min": 0.0,
            "s_max": float(s_max),
            "ds": float(ds),
            "ds_min": float(ds),
            "ds_max": float(ds),
            "strike_alignment": cfg.strike_alignment,
        },
    )


def _allocate(n_s: int, lengths: list[float]) -> list[int]:
    """Split `n_s` intervals between segments in proportion to their lengths.

    Largest-remainder apportionment with a floor of one interval per segment,
    so the counts sum to `n_s` exactly and are a deterministic function of the
    inputs. Proportional allocation is what keeps the ratio of neighbouring
    spacings across a segment boundary `1 + O(1/n)`, which is what keeps
    `h+ - h-` there at `O(h^2)` and the scheme at order two (module docstring).
    """
    k = len(lengths)
    if n_s < k:
        raise InvalidInputError(
            f"n_s={n_s} is too small for {k} grid segments"
        )
    total = sum(lengths)
    raw = [n_s * length / total for length in lengths]
    counts = [max(1, math.floor(x)) for x in raw]
    remaining = n_s - sum(counts)
    order = sorted(range(k), key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
    position = 0
    while remaining > 0:
        counts[order[position % k]] += 1
        remaining -= 1
        position += 1
    while remaining < 0:
        biggest = max(range(k), key=lambda i: counts[i])
        counts[biggest] -= 1
        remaining += 1
    return counts


def build_spot_grid(
    *,
    strike: float,
    spot: float,
    cfg: Any,
    lower: float = 0.0,
    upper: float | None = None,
    node_points: tuple[float, ...] = (),
    concentration_points: tuple[float, ...] = (),
    require_strike_inside: bool = True,
) -> SpotGrid:
    """Build the spot grid this configuration and this contract ask for.

    Parameters
    ----------
    strike, spot
        The contract's strike and the market's spot. `spot` sets the nominal
        `s_max` when `cfg.s_max` is `None`; `strike` is the point that is placed
        at a cell midpoint when `cfg.strike_alignment == "midpoint"`.
    cfg
        A `PDEConfig`. Read: `n_s`, `s_max`, `s_max_multiplier`,
        `strike_alignment`, `grid`, `concentration`, `grid_points`.
    lower, upper
        Domain. `upper=None` means "free": `s_max` floats to absorb the
        alignment nudge, exactly as it does on the legacy uniform grid. A
        knock-out truncates the domain at its barrier, which is how the barrier
        becomes a node with no arithmetic at all.
    node_points
        Interior points that must be nodes exactly (a barrier inside the
        domain, when the contract is monitored discretely and the dead region
        is therefore not absorbing).
    concentration_points
        Where a `sinh` grid piles its nodes, when `cfg.grid_points` is `None`.
        Ignored entirely on a uniform grid.
    require_strike_inside
        Whether the strike must lie inside the domain. True for every
        vanilla-shaped problem, where a strike outside the grid means the
        payoff was discretised wrongly. **False for a knock-out whose domain is
        truncated at its barrier**: a down-and-out call with `K < H` has no kink
        on `[H, s_max]` at all -- its payoff there is `S - K` everywhere -- and
        an up-and-out put with `K > H` is the mirror. Those are real contracts
        (two of the three published Haug rows are one of them) and refusing them
        would be refusing the easy case.

    Returns
    -------
    SpotGrid
    """
    n_s = cfg.n_s
    kind = cfg.grid
    if kind not in GRID_KINDS:
        raise InvalidInputError(
            "grid must be one of " + ", ".join(repr(k) for k in GRID_KINDS)
        )

    legacy = (
        kind == UNIFORM
        and lower == 0.0
        and upper is None
        and not node_points
    )
    if legacy:
        return _legacy_uniform_grid(strike, spot, cfg)

    nominal_upper = (
        upper
        if upper is not None
        else (cfg.s_max if cfg.s_max is not None else cfg.s_max_multiplier * spot)
    )
    if not (lower < nominal_upper):
        raise InvalidInputError(
            f"spot grid domain ({lower}, {nominal_upper}) is empty"
        )
    for point in node_points:
        if not (lower < point < nominal_upper):
            raise InvalidInputError(
                f"node point {point} must lie strictly inside the domain "
                f"({lower}, {nominal_upper})"
            )

    # Inversion bracket: generous above the nominal top, because the midpoint
    # nudge can push the last node past it.
    upper_bracket = lower + 4.0 * (nominal_upper - lower)

    if kind == SINH:
        points = tuple(
            float(p)
            for p in (
                cfg.grid_points
                if cfg.grid_points is not None
                else concentration_points
            )
        )
        if not points:
            points = (float(strike),)
        alpha = float(cfg.concentration) * (nominal_upper - lower)
        if not (math.isfinite(alpha) and alpha > 0.0):
            raise InvalidInputError("concentration must be finite and > 0")

        def coord(values: np.ndarray | float) -> np.ndarray:
            return sinh_coordinate(values, points, alpha)

        def invert(values: np.ndarray | float) -> np.ndarray:
            return inverse_sinh_coordinate(
                values, points, alpha, lower=lower, upper=upper_bracket
            )
    else:
        points = ()
        alpha = math.nan

        def coord(values: np.ndarray | float) -> np.ndarray:
            return np.asarray(values, dtype=float)

        def invert(values: np.ndarray | float) -> np.ndarray:
            return np.asarray(values, dtype=float)

    anchors = [float(lower), *sorted(float(p) for p in node_points), float(nominal_upper)]
    anchor_xi = [float(coord(a)) for a in anchors]
    lengths = [anchor_xi[i + 1] - anchor_xi[i] for i in range(len(anchors) - 1)]
    counts = _allocate(n_s, lengths)

    align = cfg.strike_alignment == "midpoint"
    strike_xi = float(coord(float(strike)))
    last = len(counts) - 1
    aligned_here = (
        align
        and upper is None
        and anchor_xi[last] < strike_xi < anchor_xi[last + 1]
    )

    nodes_xi: list[np.ndarray] = []
    exact: dict[int, float] = {}
    offset = 0
    dxi_by_segment: list[float] = []
    for index, count in enumerate(counts):
        a_xi = anchor_xi[index]
        if index == last and aligned_here:
            dxi = _aligned_spacing(
                a_xi=a_xi,
                b_xi=anchor_xi[index + 1],
                strike_xi=strike_xi,
                strike=float(strike),
                count=count,
                invert=invert,
                identity=(kind == UNIFORM),
            )
        else:
            dxi = (anchor_xi[index + 1] - a_xi) / count
        dxi_by_segment.append(dxi)
        segment = a_xi + dxi * np.arange(count + 1, dtype=float)
        nodes_xi.append(segment[:-1] if index < last else segment)
        exact[offset] = anchors[index]
        offset += count
    # The top node is the one place the alignment nudge is allowed to move the
    # domain, exactly as `s_max` floats on the legacy uniform grid.
    exact[n_s] = (
        float(invert(nodes_xi[-1][-1])) if aligned_here else anchors[-1]
    )

    xi_grid = np.concatenate(nodes_xi)
    s_grid = np.asarray(invert(xi_grid), dtype=float)
    for index, value in exact.items():
        s_grid[index] = value

    if not np.all(np.diff(s_grid) > 0.0):
        raise InvalidInputError(
            "spot grid is not strictly increasing: reduce n_s or raise "
            "PDEConfig.concentration"
        )
    strike_inside = bool(s_grid[0] <= strike <= s_grid[-1])
    if require_strike_inside and not strike_inside:
        raise InvalidInputError(
            f"strike {strike} must lie inside the spot grid "
            f"({s_grid[0]}, {s_grid[-1]})"
        )

    spacings = np.diff(s_grid)
    if strike_inside:
        cell = int(np.searchsorted(s_grid, strike, side="right")) - 1
        cell = min(max(cell, 0), n_s - 1)
        strike_offset = abs(
            0.5 * (s_grid[cell] + s_grid[cell + 1]) - strike
        ) / spacings[cell]
    else:
        # No kink inside the domain, so there is nothing to align and nothing
        # to report an offset for.
        strike_offset = math.nan

    # A single-segment uniform grid is uniform by construction, and is treated
    # as such so that its operator and its Greek stencils are the legacy ones.
    # Two segments of different lengths are piecewise uniform, i.e. genuinely
    # non-uniform at the junction, and go through the three-point weights.
    is_uniform = kind == UNIFORM and len(counts) == 1
    ds = dxi_by_segment[0] if is_uniform else float(spacings.mean())

    s_grid.setflags(write=False)
    meta: dict[str, Any] = {
        "grid": kind,
        "s_min": float(s_grid[0]),
        "s_max": float(s_grid[-1]),
        "ds": float(ds),
        "ds_min": float(spacings.min()),
        "ds_max": float(spacings.max()),
        "strike_alignment": cfg.strike_alignment,
        "strike_cell_offset": float(strike_offset),
        "strike_in_domain": strike_inside,
        "grid_node_points": tuple(float(p) for p in node_points),
        "grid_segments": tuple(counts),
    }
    if kind == SINH:
        meta["concentration"] = float(cfg.concentration)
        meta["concentration_alpha"] = float(alpha)
        meta["concentration_points"] = points
    return SpotGrid(
        s=s_grid,
        kind=kind,
        uniform=is_uniform,
        ds=float(ds),
        meta=meta,
    )


def _aligned_spacing(
    *,
    a_xi: float,
    b_xi: float,
    strike_xi: float,
    strike: float,
    count: int,
    invert: Any,
    identity: bool,
) -> float:
    """Spacing that puts the strike at the midpoint of its cell.

    The half-integer rule first: with `j = round((xi_K - a) / dxi0 - 1/2)`,
    demanding `xi_K = a + (j + 1/2) dxi` fixes `dxi`. On a uniform grid the
    coordinate is the identity and that *is* the `S`-midpoint condition, giving
    the Slice 0 alignment unchanged. On a `sinh` grid the two differ by
    `O(h^2)`, so the spacing is then refined by bisection until the arithmetic
    midpoint of the straddling cell equals the strike to round-off -- the cell
    index `j` is held fixed while the spacing moves, which is what makes the
    residual monotone in `dxi` and the bracket safe.
    """
    dxi0 = (b_xi - a_xi) / count
    j = max(round((strike_xi - a_xi) / dxi0 - 0.5), 0)
    dxi = (strike_xi - a_xi) / (j + 0.5)
    if identity or j + 1 > count:
        return dxi

    def residual(step: float) -> float:
        left = float(invert(a_xi + j * step))
        right = float(invert(a_xi + (j + 1) * step))
        return 0.5 * (left + right) - strike

    lo, hi = 0.9 * dxi, 1.1 * dxi
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo > 0.0 or f_hi < 0.0:
        # The midpoint is not bracketed by +-10% of the half-integer spacing,
        # which means the mesh is far too distorted for the cell index to be
        # stable. Report the half-integer spacing and let the caller's meta
        # carry the realised offset.
        return dxi
    for _ in range(_MIDPOINT_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
