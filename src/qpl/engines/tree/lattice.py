"""The recombining Cox-Ross-Rubinstein lattice, shared by every tree engine.

One place computes the CRR parameters and one place lays out the spot nodes,
so the European tree pricer in this package and the American-put dynamic
program in `qpl.engines.dp` cannot drift apart in their notion of what "the
CRR tree" is.

Construction (derived in `docs/notes/crr_tree_convergence.md`; the scheme is
due to Cox, Ross and Rubinstein (1979), "Option pricing: a simplified
approach", Journal of Financial Economics 7, 229-263):

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

Rates and dividend yields are read from `Market` once, at the option's expiry,
and held constant across the lattice: these engines assume flat curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError

__all__ = [
    "CRRLattice",
    "build_recombining_spot_tree",
    "crr_parameters",
    "crr_spot_level",
]

_P_TOLERANCE = 1e-12
"""Slack on the no-arbitrage bound, absorbing round-off in `exp`/`sqrt`."""


@dataclass(frozen=True)
class CRRLattice:
    """Per-step parameters of a CRR binomial lattice.

    Parameters
    ----------
    n_steps
        Number of time steps.
    dt
        Step length in years, ``expiry / n_steps``.
    up, down
        Multiplicative up and down moves, ``d = 1 / u`` in the non-degenerate
        case.
    p
        Risk-neutral probability of an up move.
    growth
        ``exp((r - q) dt)``, the one-step risk-neutral growth factor.
    discount
        ``exp(-r dt)``, the one-step discount factor.
    degenerate
        ``True`` when ``sigma == 0``: both branches then equal ``growth`` and
        ``p`` is an arbitrary ``0.5``, since the two successors coincide.
    """

    n_steps: int
    dt: float
    up: float
    down: float
    p: float
    growth: float
    discount: float
    degenerate: bool


def crr_parameters(
    *,
    sigma: float,
    expiry: float,
    rate: float,
    dividend_yield: float,
    n_steps: int,
) -> CRRLattice:
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
    CRRLattice

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
        return CRRLattice(
            n_steps=n_steps,
            dt=dt,
            up=growth,
            down=growth,
            p=0.5,
            growth=growth,
            discount=discount,
            degenerate=True,
        )

    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    p = (growth - down) / (up - down)

    if p < -_P_TOLERANCE or p > 1.0 + _P_TOLERANCE:
        raise InvalidInputError(
            "No-arbitrage condition violated: risk-neutral probability p must be in [0, 1]"
        )
    p = min(1.0, max(0.0, p))

    return CRRLattice(
        n_steps=n_steps,
        dt=dt,
        up=up,
        down=down,
        p=p,
        growth=growth,
        discount=discount,
        degenerate=False,
    )


def crr_spot_level(*, spot: float, up: float, down: float, level: int) -> np.ndarray:
    """Spot values at time level ``level``, indexed by number of up moves.

    Element ``j`` of the returned array is ``spot * up**j * down**(level - j)``,
    so the array is sorted ascending whenever ``down < up``.
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
