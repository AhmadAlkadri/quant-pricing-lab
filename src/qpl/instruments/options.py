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

`BarrierOption` (Slice 12) is outside it for a fourth reason: its payoff is a
function of the terminal spot *and* of whether an extreme of the path ever
reached a level. That makes it the first instrument here whose contract carries
a **monitoring convention** -- continuous, or a list of dates -- because the two
are different contracts with different prices, and the gap between them is the
quantity the slice exists to measure. Encoding the convention on the instrument
rather than on the engine is what lets `method="analytic"` refuse a discretely
monitored barrier by inspecting the contract, instead of silently pricing the
continuous one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from ..exceptions import InvalidInputError

__all__ = [
    "BARRIER_TYPES",
    "AmericanOption",
    "AsianOption",
    "BarrierOption",
    "DigitalOption",
    "EuropeanOption",
    "VanillaOption",
    "uniform_fixing_times",
    "uniform_monitoring_times",
]

BARRIER_TYPES: tuple[str, ...] = (
    "down-and-out",
    "down-and-in",
    "up-and-out",
    "up-and-in",
)
"""The four single-barrier types, as data, for validation and error messages."""


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


def _validated_schedule(
    times: object, expiry: float, name: str
) -> tuple[float, ...]:
    """Normalise and check a monitoring/fixing schedule; return it as a tuple.

    Shared by `AsianOption.fixing_times` and `BarrierOption.monitoring` because
    the rules are genuinely the same rules -- non-empty, finite, strictly
    increasing, every time in `(0, expiry]` -- and the messages are part of the
    public behaviour (pinned in `tests/test_asian_mc.py` and
    `tests/test_barrier_analytic.py`). `name` is threaded through so each
    instrument's message names its own field.

    The strict-increase rule is not pedantry in either case. For an average, a
    repeated fixing makes the log-spot covariance singular in a way the closed
    forms would silently absorb; for a barrier, a repeated observation date is
    simply the same observation twice and makes the schedule and the simulation
    grid two different objects. Both are refused rather than handled.
    """
    try:
        schedule = tuple(float(t) for t in times)  # type: ignore[union-attr]
    except TypeError as exc:
        raise InvalidInputError(f"{name} must be an iterable of floats") from exc
    if not schedule:
        raise InvalidInputError(f"{name} must contain at least one time")
    if any(not math.isfinite(t) for t in schedule):
        raise InvalidInputError(f"{name} must be finite")
    if any(t <= 0.0 for t in schedule):
        raise InvalidInputError(f"{name} must all be > 0")
    if any(b <= a for a, b in pairwise(schedule)):
        raise InvalidInputError(f"{name} must be strictly increasing")
    if schedule[-1] > expiry:
        raise InvalidInputError(f"{name} must all be <= expiry")
    return schedule


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

    Priced by `method="tree"` (Bellman maximum at every lattice node),
    `method="pde"` (the linear complementarity problem solved by projected
    SOR) and `method="mc"` (least-squares Monte Carlo). `method="analytic"`
    raises `NotSupportedError`; there is no closed form.

    The Monte Carlo engine is the one exception to "the instrument says what is
    priced": simulation cannot exercise continuously, so it prices a **Bermudan**
    option on `MCConfig.exercise_dates` equally spaced dates and says so in
    `meta["exercise_style"]`. See `qpl.engines.mc.american`.
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

        object.__setattr__(
            self,
            "fixing_times",
            _validated_schedule(self.fixing_times, self.expiry, "fixing_times"),
        )

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


@dataclass(frozen=True)
class BarrierOption:
    """Single-barrier knock-in / knock-out call or put, with a cash rebate.

    The payoff of a vanilla is modified by whether the underlying ever reached
    a level `H` during the option's life:

        knock-out:  vanilla(S_T) if the barrier was never touched,
                    otherwise the `rebate`, paid at the touch time;
        knock-in:   vanilla(S_T) if the barrier *was* touched,
                    otherwise the `rebate`, paid at expiry.

    The two rebate conventions differ on purpose and are the ones the
    Reiner-Rubinstein closed forms assume (the `E` and `F` building blocks in
    `qpl.engines.analytic.barrier`): a knock-out rebate compensates the holder
    at the moment the contract dies, while a knock-in rebate is only known to
    be due at expiry, when it is finally certain the option never came alive.

    Why this is its own instrument type, and not a `VanillaOption` subclass:
    the payoff depends on the running extreme of the path as well as on `S_T`.
    That is the same reason `AsianOption` is outside the hierarchy, but the
    consequence is different -- an average is a smooth functional of the path,
    whereas an extreme crossing a level is an indicator, so a barrier is
    path-dependent *and* discontinuous at once. The registry
    (`.agents/brain/adr/0005-engine-registry.md`) refuses the vanilla engines
    by lookup rather than by a guard inside each of them.

    Parameters
    ----------
    kind
        `"call"` or `"put"`. Normalised to lower case.
    strike
        Positive strike of the underlying vanilla payoff.
    expiry
        Time to expiry in years. Must be `>= 0`.
    barrier
        The level `H`, `> 0`. Its position relative to the spot is **not**
        checked here, because the spot lives in `Market` and an instrument in
        this package never reads a market. It is resolved at pricing time
        instead: see :meth:`is_touched`.
    barrier_type
        One of `BARRIER_TYPES`: `"down-and-out"`, `"down-and-in"`,
        `"up-and-out"`, `"up-and-in"`. Normalised to lower case.
    rebate
        Non-negative cash amount paid when the option fails, under the
        convention above. `0.0` (the default) is the case in which knock-in
        plus knock-out equals the vanilla exactly.
    monitoring
        `"continuous"` (the default), or a tuple of monitoring times in years.
        A discrete schedule follows the same rules as `AsianOption`'s fixings:
        non-empty, finite, strictly increasing, every time in `(0, expiry]`.

        This is a **contract term**, not an engine setting, because the two
        conventions price differently -- by `O(1/sqrt(m))` in the number of
        monitoring dates, which is the quantity Slice 12 measures. Keeping it
        here is what lets `method="analytic"` refuse a discretely monitored
        barrier (the Reiner-Rubinstein forms are continuous-monitoring
        formulas) rather than quietly answer a different question.

    Notes
    -----
    A `monitoring` schedule need not contain `expiry`: a barrier whose last
    observation is before settlement is a real contract. The engines that
    simulate the schedule append `expiry` to the *simulation* grid, because the
    terminal payoff is read at `expiry` whatever the last observation was.
    """

    kind: Literal["call", "put"]
    strike: float
    expiry: float
    barrier: float = 0.0
    barrier_type: Literal[
        "down-and-out", "down-and-in", "up-and-out", "up-and-in"
    ] = "down-and-out"
    rebate: float = 0.0
    monitoring: Literal["continuous"] | tuple[float, ...] = "continuous"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _validated_kind(self.kind, self.strike, self.expiry)
        )
        if not math.isfinite(self.barrier) or self.barrier <= 0.0:
            raise InvalidInputError("barrier must be finite and > 0")
        barrier_type = str(self.barrier_type).lower()
        if barrier_type not in BARRIER_TYPES:
            raise InvalidInputError(
                "barrier_type must be one of "
                + ", ".join(repr(name) for name in BARRIER_TYPES)
            )
        object.__setattr__(self, "barrier_type", barrier_type)
        if not math.isfinite(self.rebate) or self.rebate < 0.0:
            raise InvalidInputError("rebate must be finite and >= 0")

        monitoring = self.monitoring
        if isinstance(monitoring, str):
            if monitoring.lower() != "continuous":
                raise InvalidInputError(
                    "monitoring must be 'continuous' or a tuple of monitoring times"
                )
            object.__setattr__(self, "monitoring", "continuous")
            return
        object.__setattr__(
            self, "monitoring", _validated_schedule(monitoring, self.expiry, "monitoring")
        )

    @property
    def is_down(self) -> bool:
        """`True` for `"down-and-out"` and `"down-and-in"`."""
        return self.barrier_type.startswith("down")

    @property
    def is_knock_out(self) -> bool:
        """`True` for `"down-and-out"` and `"up-and-out"`."""
        return self.barrier_type.endswith("out")

    @property
    def is_continuous(self) -> bool:
        """`True` when the barrier is monitored continuously."""
        return self.monitoring == "continuous"

    @property
    def monitoring_times(self) -> tuple[float, ...]:
        """The monitoring schedule, or `()` under continuous monitoring."""
        return () if self.is_continuous else self.monitoring  # type: ignore[return-value]

    @property
    def n_monitoring(self) -> int:
        """Number of monitoring dates, or `0` under continuous monitoring."""
        return len(self.monitoring_times)

    def is_touched(self, spot: float) -> bool:
        """Has the barrier already been reached at a spot of `spot`?

        `spot <= barrier` for a down type and `spot >= barrier` for an up one:
        *weak* inequalities, because touching the level is what the contract
        says, and a spot sitting exactly on the barrier has touched it. Under
        continuous monitoring the answer at inception settles the contract
        immediately -- a knock-out is dead and worth its rebate, a knock-in is
        alive and worth the vanilla -- and every engine here is required to
        report that same value, which is what makes it a fact about the
        contract rather than about a discretisation.

        Under *discrete* monitoring the inception spot is not an observation
        date, so this is the wrong question to ask of the schedule; it is still
        the right question for the closed forms, which assume the barrier has
        not been crossed.
        """
        return spot <= self.barrier if self.is_down else spot >= self.barrier


def uniform_monitoring_times(expiry: float, n_monitoring: int) -> tuple[float, ...]:
    """`m` equally spaced barrier observations ending exactly at `expiry`.

    Identical construction to :func:`uniform_fixing_times` -- and identical for
    the same floating-point reason -- but named separately because the two are
    different contract features and a barrier schedule is not an averaging
    schedule. `t_i = i T / m`, with `t_m == expiry` exactly.
    """
    return uniform_fixing_times(expiry, n_monitoring)
