"""Vanilla option instruments, split by *exercise style*.

Exercise style is a property of the instrument, not of the engine: whether the
holder may exercise before expiry changes what is being priced, not how it is
computed. Encoding it as the instrument's type is what lets the engine
registry (`.agents/brain/adr/0005-engine-registry.md`) answer "can this engine
price this?" by lookup rather than by an ``if instrument.american:`` branch
inside each engine. The analytic, Monte Carlo and PDE engines register for
`EuropeanOption` only, so asking them for an `AmericanOption` raises
`NotSupportedError` through the ordinary dispatcher path; the tree engine
registers for both.

`AmericanOption` is deliberately *not* a subclass of `EuropeanOption`. Registry
lookup walks the MRO, so a subclass would silently resolve to the European
engines and be priced with the wrong exercise rule. The shared validation lives
in a common base, `VanillaOption`, which no engine registers for.

`DigitalOption` (Slice 6) is not in that hierarchy at all. It shares three
fields with a vanilla option and validates them the same way -- through the
module-level `_validated_kind` helper, so there is one copy of the rules and
one copy of the messages -- but its payoff is a step function rather than a
hockey stick, so nothing that prices a vanilla can price it. Making it a
subclass of `VanillaOption` would be harmless today (no engine registers for
that base) and a trap tomorrow, the moment anything does; the shared *code* is
a function call, not an inheritance edge.

`AsianOption` (Slice 8) is outside the hierarchy for a third reason again: its
payoff is not a function of the terminal spot at all. It reads the path at a
list of fixing times, so an engine that only knows how to produce `S_T` cannot
price it even in principle. Same treatment as the digital: the kind/strike/
expiry rules come from `_validated_kind`, and the fixing schedule is validated
here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from ..exceptions import InvalidInputError

__all__ = [
    "AmericanOption",
    "AsianOption",
    "DigitalOption",
    "EuropeanOption",
    "VanillaOption",
    "uniform_fixing_times",
]


def _validated_kind(kind: str, strike: float, expiry: float) -> str:
    """Check the fields every option instrument here shares; return the kind.

    The order of the three checks and the exact messages are pinned by
    `tests/test_pricing_analytic.py` and are what `VanillaOption` raised
    before this was a function; `DigitalOption` reuses it rather than
    inheriting from `VanillaOption`.
    """
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    if strike <= 0:
        raise InvalidInputError("strike must be > 0")
    if expiry < 0:
        raise InvalidInputError("expiry must be >= 0")
    return kind_l


@dataclass(frozen=True)
class VanillaOption:
    """Common data and validation for a vanilla call or put.

    This base carries no exercise style of its own and is not registered with
    any engine; instantiate `EuropeanOption` or `AmericanOption` instead.

    Parameters
    ----------
    kind
        Option type: `"call"` or `"put"`. Normalised to lower case.
    strike
        Positive strike price.
    expiry
        Time to expiry in years (`>= 0`).
    """

    kind: Literal["call", "put"]
    strike: float
    expiry: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _validated_kind(self.kind, self.strike, self.expiry)
        )


@dataclass(frozen=True)
class EuropeanOption(VanillaOption):
    """European vanilla option: exercisable only at `expiry`."""


@dataclass(frozen=True)
class AmericanOption(VanillaOption):
    """American vanilla option: exercisable at any time up to `expiry`.

    Only the tree engine prices this; `method="analytic"`, `"mc"` and `"pde"`
    raise `NotSupportedError`, since none of them implements an early-exercise
    rule.
    """


@dataclass(frozen=True)
class DigitalOption:
    """European **cash-or-nothing** digital option.

    Pays a fixed amount `cash` at `expiry` if the option finishes in the
    money, and nothing otherwise:

        call:  cash * 1{S_T > K},        put:  cash * 1{S_T < K}.

    The two indicators are both *strict*, so at `S_T == K` neither pays. That
    event has probability zero under the model, which is why the convention is
    free to be symmetric between the call and the put; it matters only at a
    node or sample that lands exactly on the strike. QuantLib's
    `CashOrNothingPayoff` uses the same strict pair, which keeps
    `tests/oracle/test_digital_vs_quantlib.py` comparing the same contract.

    Why this is a separate type rather than a `VanillaOption` subclass: the
    payoff is a **jump**, not a kink. Every engine in this package that prices
    a vanilla assumes a continuous terminal condition somewhere -- the tree
    reads a payoff that is Lipschitz in the node spacing, the finite-difference
    grid samples a function whose one-sided limits agree, the Monte Carlo
    estimator has a payoff with bounded variation in the sample. None of that
    survives a step function, and each engine needs its own registered entry
    (see `.agents/brain/adr/0005-engine-registry.md`), which is exactly what
    keying the registry on the instrument type buys.

    Parameters
    ----------
    kind
        `"call"` (pays above the strike) or `"put"` (pays below). Normalised
        to lower case.
    strike
        Positive strike price: the level at which the payoff jumps.
    expiry
        Time to expiry in years (`>= 0`).
    cash
        Positive cash amount paid when in the money. The payoff and every
        price here are linear in `cash`, so it is a pure scaling; it is a
        field rather than a caller-side multiplication because the engines
        report prices and Greeks, and a Greek that silently assumed
        `cash = 1` would be wrong by that factor.
    """

    kind: Literal["call", "put"]
    strike: float
    expiry: float
    cash: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _validated_kind(self.kind, self.strike, self.expiry)
        )
        if not math.isfinite(self.cash) or self.cash <= 0:
            raise InvalidInputError("cash must be finite and > 0")


@dataclass(frozen=True)
class AsianOption:
    """Fixed-strike **Asian** option on a discretely monitored average.

    The payoff replaces the terminal spot of a vanilla by an average of the
    spot over a fixed list of monitoring (fixing) dates:

        call:  max(A - K, 0),        put:  max(K - A, 0),

    with `A` either the arithmetic mean `(1/n) sum_i S_{t_i}` or the geometric
    mean `(prod_i S_{t_i})^{1/n}` of the spot at the `n` fixing times, and the
    payoff settled at `expiry`.

    Why this is its own instrument type, and not a `VanillaOption` subclass:
    the payoff is not a function of `S_T`. Every engine registered for a
    European vanilla reduces the model to the terminal distribution -- the
    lattice reads the final level, the finite-difference grid solves a
    one-dimensional problem in `S` at `t = 0`, the Monte Carlo engine samples
    `S_T` in one exact step. An average over the path is a genuinely different
    state, so the registry (`.agents/brain/adr/0005-engine-registry.md`) has to
    refuse those engines by lookup rather than by a guard inside each of them.

    Parameters
    ----------
    kind
        `"call"` or `"put"`. Normalised to lower case.
    strike
        Positive fixed strike. (A *floating*-strike Asian, where the average
        replaces the strike instead of the spot, is a different contract and is
        deliberately not modelled here.)
    expiry
        Settlement time in years. Must be `> 0`, because a non-empty fixing
        schedule inside `(0, expiry]` cannot exist otherwise.
    fixing_times
        Monitoring times in years, **strictly increasing**, every one in
        `(0, expiry]`. The last fixing may equal `expiry` and usually does; it
        is not required to, because an average that stops before settlement is
        a real contract (the "averaging-out period ends early" convention) and
        the pricing formulas here handle it -- the discount runs to `expiry`
        while the average is over the fixings.
    averaging
        `"arithmetic"` or `"geometric"`.

    Notes
    -----
    `fixing_times` is stored as a `tuple`, so the instrument stays hashable and
    comparable like every other frozen dataclass here; a list or any iterable is
    accepted and converted at construction.

    The strict-increase rule is not pedantry. A repeated fixing time is a
    legitimate contract (a date that counts twice) but it makes the covariance
    matrix of the log-spots singular in a way the closed forms below would
    silently absorb, and it makes "the fixing grid" and "the simulation time
    grid" two different objects. Repeats are refused rather than handled.
    """

    kind: Literal["call", "put"]
    strike: float
    expiry: float
    fixing_times: tuple[float, ...] = ()
    averaging: Literal["arithmetic", "geometric"] = "arithmetic"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _validated_kind(self.kind, self.strike, self.expiry)
        )
        averaging = str(self.averaging).lower()
        if averaging not in {"arithmetic", "geometric"}:
            raise InvalidInputError("averaging must be 'arithmetic' or 'geometric'")
        object.__setattr__(self, "averaging", averaging)

        try:
            times = tuple(float(t) for t in self.fixing_times)
        except TypeError as exc:
            raise InvalidInputError("fixing_times must be an iterable of floats") from exc
        if not times:
            raise InvalidInputError("fixing_times must contain at least one time")
        if any(not math.isfinite(t) for t in times):
            raise InvalidInputError("fixing_times must be finite")
        if any(t <= 0.0 for t in times):
            raise InvalidInputError("fixing_times must all be > 0")
        if any(b <= a for a, b in pairwise(times)):
            raise InvalidInputError("fixing_times must be strictly increasing")
        if times[-1] > self.expiry:
            raise InvalidInputError("fixing_times must all be <= expiry")
        object.__setattr__(self, "fixing_times", times)

    @property
    def n_fixings(self) -> int:
        """Number of monitoring dates."""
        return len(self.fixing_times)


def uniform_fixing_times(expiry: float, n_fixings: int) -> tuple[float, ...]:
    """`n` equally spaced fixings ending exactly at `expiry`: `t_i = i T / n`.

    This is the convention every published Asian benchmark in this repository
    uses (see `qpl.cases.asian_black_scholes`), and the reason it is a function
    rather than a comprehension is floating point: writing `(i + 1) * T / n` for
    `i = n - 1` does **not** in general return `T` -- at `T = 90/365` and
    `n = 2560` it returns a value one ulp above it, which `AsianOption` then
    correctly rejects as a fixing after expiry. `numpy.linspace` guarantees the
    endpoint exactly, so the schedule this builds always satisfies
    `t_n == expiry`.

    Raises
    ------
    InvalidInputError
        If `expiry <= 0` or `n_fixings < 1`.
    """
    if not math.isfinite(expiry) or expiry <= 0.0:
        raise InvalidInputError("expiry must be finite and > 0")
    if n_fixings < 1:
        raise InvalidInputError("n_fixings must be >= 1")
    import numpy as np

    return tuple(float(t) for t in np.linspace(expiry / n_fixings, expiry, n_fixings))
