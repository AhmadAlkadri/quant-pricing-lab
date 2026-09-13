"""Recombining binomial lattices, shared by every tree engine.

One place computes the per-step parameters and one place lays out the spot
nodes, so the European tree pricer in this package and the American-put
dynamic program in `qpl.engines.dp` cannot drift apart in their notion of what
"the tree" is. Two parameterisations live here, selected by
`TreeConfig.scheme`; everything downstream -- payoff, backward induction,
Bellman step, lattice Greeks -- reads only ``up``, ``down``, ``p``,
``discount`` and so is scheme-agnostic.

Cox-Ross-Rubinstein (`scheme="crr"`)
------------------------------------
Derived in `docs/notes/crr_tree_convergence.md`; the scheme is due to Cox,
Ross and Rubinstein (1979), "Option pricing: a simplified approach", Journal
of Financial Economics 7, 229-263.

- One step of length ``dt = T / n`` multiplies the spot by ``u`` or by ``d``.
- CRR picks the symmetric choice ``u = exp(sigma sqrt(dt))``, ``d = 1 / u``,
  which makes the lattice recombine and centres the log-spot grid on
  ``log(S0)``.
- Requiring the one-step expected return to equal the risk-neutral growth
  ``exp((r - q) dt)`` fixes the probability
  ``p = (exp((r - q) dt) - d) / (u - d)``.
- ``p`` lies in ``[0, 1]`` -- i.e. the one-step model admits no arbitrage --
  exactly when ``d <= exp((r - q) dt) <= u``. For fixed ``sigma > 0`` that
  holds for all small enough ``dt``; it can fail on a coarse tree with a large
  drift, and this module raises rather than returning a signed "probability".

Leisen-Reimer (`scheme="leisen-reimer"`)
----------------------------------------
Derived in `docs/notes/leisen_reimer.md`; the scheme is due to Leisen and
Reimer (1996), "Binomial models for option valuation -- examining and
improving convergence", Applied Mathematical Finance 3(4), 319-346. The
formulas below are restated in this module's own words from that construction;
no text, table or code is reproduced from the paper.

CRR fixes the *lattice* and solves for the probability. Leisen-Reimer does the
opposite: it fixes the two probabilities that the Black-Scholes formula
already names and solves for the lattice.

- The Black-Scholes price is ``S e^{-qT} N(d1) - K e^{-rT} N(d2)`` with
  ``d1 = (log(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T))`` and
  ``d2 = d1 - sigma sqrt(T)``. On an ``n``-step binomial tree the same price
  is ``S e^{-qT} Phi' - K e^{-rT} Phi``, where ``Phi`` is the binomial
  probability of finishing in the money and ``Phi'`` the same probability
  under the share measure. The tree's error is therefore, term by term, the
  error of two binomial tail probabilities against two normal ones.
- Leisen and Reimer choose the per-step probabilities so that those two tails
  are *already* right: pick ``p`` with ``P(Bin(n, p) > n/2) = N(d2)`` and
  ``p'`` with ``P(Bin(n, p') > n/2) = N(d1)``, to the accuracy of a normal
  approximation of the binomial that is itself high order. That inversion is
  the Peizer-Pratt formula, `peizer_pratt_inversion` below.
- The multipliers then follow from the two constraints that make the tree
  consistent: the share measure must be the ``p'``-measure
  (``u p / e^{(r-q) dt} = p'``) and the tree must reprice the one-step forward
  (``p u + (1 - p) d = e^{(r-q) dt}``). Solving,

      u = e^{(r-q) dt} p' / p,      d = (e^{(r-q) dt} - p u) / (1 - p).

- Consequences worth stating, because they are what the downstream code has
  to cope with: ``u d != 1``, so the lattice is **not** centred on the initial
  spot (`spot_centred` is `False`), and the strike enters the geometry, so a
  call and a put on different strikes no longer share a lattice.
- The construction needs an **odd** ``n``: ``P(Bin(n, p) > n/2)`` is the
  probability of strictly more than ``n/2`` up moves, which counts a whole
  number of outcomes only when ``n`` is odd. Even ``n`` is rejected rather
  than silently rounded (`qpl.engines.tree.pricers.TreeConfig`).
- No-arbitrage is automatic here rather than a condition to check: ``d1 > d2``
  and the inversion is increasing, so ``p' > p``, which is exactly
  ``d < e^{(r-q) dt} < u``.

Rates and dividend yields are read from `Market` once, at the option's expiry,
and held constant across the lattice: these engines assume flat curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...exceptions import InvalidInputError

__all__ = [
    "BinomialLattice",
    "Scheme",
    "build_recombining_spot_tree",
    "crr_parameters",
    "crr_spot_level",
    "lattice_parameters",
    "leisen_reimer_parameters",
    "peizer_pratt_inversion",
]

Scheme = Literal["crr", "leisen-reimer"]
"""The lattice parameterisations this package knows how to build."""

SCHEMES: tuple[str, ...] = ("crr", "leisen-reimer")
"""`Scheme` as data, for validation and error messages."""

_P_TOLERANCE = 1e-12
"""Slack on the no-arbitrage bound, absorbing round-off in `exp`/`sqrt`."""


@dataclass(frozen=True)
class BinomialLattice:
    """Per-step parameters of a recombining binomial lattice.

    Scheme-agnostic on purpose: everything downstream consumes these numbers
    and nothing else, so adding a parameterisation means adding a constructor
    here rather than a branch in a pricer.

    Parameters
    ----------
    n_steps
        Number of time steps.
    dt
        Step length in years, ``expiry / n_steps``.
    up, down
        Multiplicative up and down moves.
    p
        Risk-neutral probability of an up move.
    growth
        ``exp((r - q) dt)``, the one-step risk-neutral growth factor. Every
        scheme here satisfies ``p * up + (1 - p) * down == growth`` to
        round-off, which is what makes put-call parity exact on the lattice.
    discount
        ``exp(-r dt)``, the one-step discount factor.
    degenerate
        ``True`` when ``sigma == 0``: both branches then equal ``growth`` and
        ``p`` is an arbitrary ``0.5``, since the two successors coincide.
    spot_centred
        ``True`` when ``up * down == 1`` by construction, so that an even time
        level has the initial spot exactly at its centre. This holds for CRR
        and **not** for Leisen-Reimer. It is a field rather than a recomputed
        test because the lattice Greeks need to know it exactly: on a CRR
        lattice ``S0 u d`` differs from ``S0`` only by round-off, and applying
        the off-centre correction there would perturb the CRR Greeks in the
        last bits for no gain.
    """

    n_steps: int
    dt: float
    up: float
    down: float
    p: float
    growth: float
    discount: float
    degenerate: bool
    spot_centred: bool


def crr_parameters(
    *,
    sigma: float,
    expiry: float,
    rate: float,
    dividend_yield: float,
    n_steps: int,
) -> BinomialLattice:
    """Build the CRR per-step parameters for a flat-curve Black-Scholes model.

    Parameters
    ----------
    sigma
        Volatility. ``0.0`` is allowed and produces a degenerate lattice whose
        two branches coincide with the risk-neutral growth factor.
    expiry
        Time to maturity in years, ``> 0``.
    rate, dividend_yield
        Continuously compounded flat rate and dividend yield.
    n_steps
        Number of time steps, ``>= 1``.

    Returns
    -------
    BinomialLattice

    Raises
    ------
    InvalidInputError
        If the inputs are out of range, or if the no-arbitrage condition
        ``d <= exp((r - q) dt) <= u`` fails, which would make ``p`` fall
        outside ``[0, 1]``.
    """
    if not math.isfinite(sigma) or sigma < 0.0:
        raise InvalidInputError("sigma must be finite and >= 0")
    if not math.isfinite(expiry) or expiry <= 0.0:
        raise InvalidInputError("expiry must be finite and > 0")
    if n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")

    dt = expiry / n_steps
    growth = math.exp((rate - dividend_yield) * dt)
    discount = math.exp(-rate * dt)

    if sigma == 0.0:
        return BinomialLattice(
            n_steps=n_steps,
            dt=dt,
            up=growth,
            down=growth,
            p=0.5,
            growth=growth,
            discount=discount,
            degenerate=True,
            spot_centred=False,
        )

    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    p = (growth - down) / (up - down)

    if p < -_P_TOLERANCE or p > 1.0 + _P_TOLERANCE:
        raise InvalidInputError(
            "No-arbitrage condition violated: risk-neutral probability p must be in [0, 1]"
        )
    p = min(1.0, max(0.0, p))

    return BinomialLattice(
        n_steps=n_steps,
        dt=dt,
        up=up,
        down=down,
        p=p,
        growth=growth,
        discount=discount,
        degenerate=False,
        spot_centred=True,
    )


def peizer_pratt_inversion(z: float, n_steps: int) -> float:
    """Peizer-Pratt inversion (method 2) of the binomial tail at ``z``.

    Returns the probability ``p`` for which the binomial tail
    ``P(Bin(n, p) > n / 2)`` equals the standard normal tail ``N(z)``, to the
    accuracy of Peizer and Pratt's normal approximation of the binomial. This
    is the inversion Leisen and Reimer (1996) recommend and the one used
    throughout this package; the other variant they discuss ("method 1")
    drops the ``0.1 / (n + 1)`` term below and converges more slowly.

    Restated in this module's own words, the approximation is

        p = 1/2 + sign(z) * sqrt( 1/4 - 1/4 exp( -(n + 1/6) * t**2 ) ),
        t = z / (n + 1/3 + 0.1 / (n + 1)).

    Two properties the callers rely on, both checked in
    `tests/test_tree_leisen_reimer.py`:

    - It is antisymmetric about ``1/2``: ``h(-z) = 1 - h(z)``, exactly, because
      ``z`` enters only through ``t**2`` and a sign. This is what makes the
      Leisen-Reimer tree symmetric at the money forward.
    - It is strictly increasing in ``z``, which is what makes ``p' > p``
      (since ``d1 > d2``) and hence makes the resulting lattice arbitrage-free
      without a separate check.

    Source: Leisen and Reimer (1996), "Binomial models for option valuation --
    examining and improving convergence", Applied Mathematical Finance 3(4),
    319-346, who take the inversion from Peizer and Pratt (1968), "A normal
    approximation for binomial, F, beta, and other common, related tail
    probabilities, I", Journal of the American Statistical Association 63(324),
    1416-1456. The formula is restated here, not copied.

    Parameters
    ----------
    z
        The normal quantile to match (``d1`` or ``d2`` in the option case).
    n_steps
        Number of time steps. Must be odd and ``>= 1``.

    Returns
    -------
    float
        A probability in ``(0, 1)``; exactly ``0.5`` at ``z = 0``.

    Raises
    ------
    InvalidInputError
        If ``z`` is not finite, or ``n_steps`` is not an odd positive integer.
    """
    if not math.isfinite(z):
        raise InvalidInputError("z must be finite")
    if n_steps < 1 or n_steps % 2 == 0:
        raise InvalidInputError("n_steps must be an odd positive integer")

    t = z / (n_steps + 1.0 / 3.0 + 0.1 / (n_steps + 1.0))
    inner = 0.25 - 0.25 * math.exp(-(n_steps + 1.0 / 6.0) * t * t)
    # `inner` is non-negative analytically; clip the round-off that can push it
    # to -1e-17 when `z` is a few ULP from zero.
    root = math.sqrt(max(inner, 0.0))
    if z == 0.0:
        return 0.5
    return 0.5 + (root if z > 0.0 else -root)


def leisen_reimer_parameters(
    *,
    spot: float,
    strike: float,
    sigma: float,
    expiry: float,
    rate: float,
    dividend_yield: float,
    n_steps: int,
) -> BinomialLattice:
    """Build the Leisen-Reimer per-step parameters (module docstring for the
    derivation).

    Unlike `crr_parameters`, this construction depends on ``spot`` and
    ``strike``: the tree is built around the Black-Scholes ``d1``/``d2`` of the
    option being priced, so two options with different strikes get different
    lattices. That is the whole mechanism -- the terminal grid is arranged so
    that the strike falls where the binomial and normal tails already agree,
    instead of wherever the parity of ``n`` happens to put it.

    Parameters
    ----------
    spot
        Initial spot, ``> 0``.
    strike
        Option strike, ``> 0``.
    sigma
        Volatility, ``> 0``. ``sigma = 0`` has no Leisen-Reimer construction
        (``d1`` and ``d2`` are not defined); the callers handle that degenerate
        case in closed form before reaching this function, and it is rejected
        here rather than returning a collapsed lattice.
    expiry
        Time to maturity in years, ``> 0``.
    rate, dividend_yield
        Continuously compounded flat rate and dividend yield.
    n_steps
        Number of time steps. Must be **odd** and ``>= 1``; see the module
        docstring for why even ``n`` has no construction.

    Returns
    -------
    BinomialLattice
        With ``spot_centred=False``: ``up * down`` equals
        ``growth**2 p'(1 - p') / (p (1 - p))``, which is ``1`` only in the
        degenerate case ``r = q`` at the money forward.

    Raises
    ------
    InvalidInputError
        If any input is out of range, or ``n_steps`` is even.
    """
    if not math.isfinite(spot) or spot <= 0.0:
        raise InvalidInputError("spot must be finite and > 0")
    if not math.isfinite(strike) or strike <= 0.0:
        raise InvalidInputError("strike must be finite and > 0")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise InvalidInputError("sigma must be finite and > 0 for the Leisen-Reimer scheme")
    if not math.isfinite(expiry) or expiry <= 0.0:
        raise InvalidInputError("expiry must be finite and > 0")
    if n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    if n_steps % 2 == 0:
        raise InvalidInputError(
            "the Leisen-Reimer construction requires an odd n_steps"
        )

    dt = expiry / n_steps
    growth = math.exp((rate - dividend_yield) * dt)
    discount = math.exp(-rate * dt)

    vol = sigma * math.sqrt(expiry)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * expiry) / vol
    d2 = d1 - vol

    p = peizer_pratt_inversion(d2, n_steps)
    p_share = peizer_pratt_inversion(d1, n_steps)

    up = growth * p_share / p
    # Solved from `p up + (1 - p) down == growth` rather than from the
    # equivalent `growth (1 - p') / (1 - p)`: written this way the forward is
    # repriced to round-off by construction, which is what put-call parity on
    # the lattice depends on.
    down = (growth - p * up) / (1.0 - p)

    return BinomialLattice(
        n_steps=n_steps,
        dt=dt,
        up=up,
        down=down,
        p=p,
        growth=growth,
        discount=discount,
        degenerate=False,
        spot_centred=False,
    )


def lattice_parameters(
    *,
    scheme: Scheme,
    spot: float,
    strike: float,
    sigma: float,
    expiry: float,
    rate: float,
    dividend_yield: float,
    n_steps: int,
) -> BinomialLattice:
    """Build the per-step parameters for the requested `scheme`.

    The single place in the package that knows which parameterisation a
    `TreeConfig` names. Callers pass every input any scheme might need; CRR
    ignores `spot` and `strike`, which is the price of keeping the branch here
    rather than in each pricer.

    ``sigma == 0`` is degenerate for every scheme -- both branches collapse
    onto the forward -- so it is routed to `crr_parameters` regardless of
    `scheme`, and the resulting lattice is identical whatever was asked for.
    The pricers do not use it for a value (they have a closed form for
    ``sigma = 0``); they use it so that the reported metadata describes a real
    lattice.

    Raises
    ------
    InvalidInputError
        As `crr_parameters` / `leisen_reimer_parameters`, plus an unknown
        `scheme`.
    """
    if scheme == "crr" or sigma == 0.0:
        return crr_parameters(
            sigma=sigma,
            expiry=expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            n_steps=n_steps,
        )
    if scheme == "leisen-reimer":
        return leisen_reimer_parameters(
            spot=spot,
            strike=strike,
            sigma=sigma,
            expiry=expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            n_steps=n_steps,
        )
    raise InvalidInputError(f"unknown lattice scheme {scheme!r}")


def crr_spot_level(*, spot: float, up: float, down: float, level: int) -> np.ndarray:
    """Spot values at time level ``level``, indexed by number of up moves.

    Element ``j`` of the returned array is ``spot * up**j * down**(level - j)``,
    so the array is sorted ascending whenever ``down < up``. Parameterised by
    ``(up, down)`` and nothing else, so it lays out a Leisen-Reimer level as
    readily as a CRR one; only a CRR level has the initial spot exactly at its
    centre.
    """
    if not math.isfinite(spot) or spot <= 0.0:
        raise InvalidInputError("spot must be finite and > 0")
    if not math.isfinite(up) or up <= 0.0:
        raise InvalidInputError("up must be finite and > 0")
    if not math.isfinite(down) or down <= 0.0:
        raise InvalidInputError("down must be finite and > 0")
    if level < 0:
        raise InvalidInputError("level must be >= 0")

    up_counts = np.arange(level + 1, dtype=float)
    down_counts = level - up_counts
    return spot * (up**up_counts) * (down**down_counts)


def build_recombining_spot_tree(
    *, spot: float, up: float, down: float, n_steps: int
) -> list[np.ndarray]:
    """Build a recombining binomial spot lattice, level by level.

    Parameters
    ----------
    spot
        Initial spot value.
    up
        Up multiplier per time step.
    down
        Down multiplier per time step.
    n_steps
        Number of time steps.

    Returns
    -------
    list[np.ndarray]
        Lattice levels where level `j` has shape `(j + 1,)`.
    """
    if n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    return [
        crr_spot_level(spot=spot, up=up, down=down, level=j) for j in range(n_steps + 1)
    ]
