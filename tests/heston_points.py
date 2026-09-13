"""Shared Heston parameter sets for the Slice 15 test modules.

Not a test module (pytest does not collect it), and the counterpart of
`tests/fourier_points.py`: these are *coordinates*, not claims. The claims,
with their evidence classes, live in `qpl.cases.heston`.

Four parameter sets, each carried for a reason:

`LEWIS`
    The set behind the six reference values in `qpl.cases.heston`
    (`S = 100, r = 1%, q = 2%, v0 = 0.04, kappa = 4, theta = 0.25, xi = 1,
    rho = -0.5, T = 1`), attributed by the QuantLib test suite to Alan Lewis'
    Wilmott-forum posting. Two things about it are worth knowing before it is
    used as *the* Heston test point, and both are measured in
    `tests/test_heston_fourier.py`:

    - its Feller number is `4 kappa theta / xi^2 = 4.0`, so the Feller
      condition **is satisfied** (`2 kappa theta = 2.0` against `xi^2 = 1.0`);
    - `2 kappa theta / xi^2 = 2` is an **integer**, which is exactly the
      condition under which the little Heston trap's branch jump multiplies
      the transform by `exp(-4 pi i kappa theta / xi^2) = 1` and is therefore
      invisible. The Lewis set cannot demonstrate the trap at any maturity.

`FELLER_VIOLATED`
    `v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1.0` -- the variance
    parameters of `qpl.cases.sde_discretization.CIR_FELLER_VIOLATED`, so the
    two Feller regimes mean the same thing in the transform slice as in the
    simulation slice, with `rho = -0.9` added. Feller number 0.08.

`TRAP`
    `LEWIS` with `theta = 0.3125` and nothing else changed. `d` and `g` do not
    depend on `theta`, so this set has **exactly the same** branch windings, at
    exactly the same `u`, as `LEWIS`; only the multiplier
    `2 kappa theta / xi^2` changes, from 2 to 2.5. That makes the trap a
    controlled experiment with one changed parameter rather than a comparison of two
    unrelated models. Feller number 5.0, so the trap is not confounded with
    the Feller condition.

`SMALL_XI`
    `v0 = theta = 0.04, xi = 1e-06`. The corner Slice 14's QuantLib oracle
    pointed at: `COSHestonEngine` *diverges* as the vol-of-vol goes to zero
    because its truncation range carries `xi` in denominators. This package's
    cumulants do not, and this set is where that is checked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.heston import HestonModel

__all__ = [
    "EXPIRIES",
    "FELLER_VIOLATED",
    "KINDS",
    "LEWIS",
    "LEWIS_STRIKES",
    "POINTS",
    "SMALL_XI",
    "TRAP",
    "HestonPoint",
]


@dataclass(frozen=True)
class HestonPoint:
    """One Heston specification, as scalars."""

    name: str
    spot: float
    rate: float
    dividend: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    expiry: float = 1.0

    def model(self) -> HestonModel:
        return HestonModel(
            v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho
        )

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def instrument(self, kind: str, strike: float, payoff: str = "vanilla"):
        if payoff == "digital":
            return DigitalOption(
                kind=kind, strike=strike, expiry=self.expiry, cash=1.0
            )
        return EuropeanOption(kind=kind, strike=strike, expiry=self.expiry)

    def at(self, **changes) -> HestonPoint:
        """The same point with fields replaced (`expiry`, `rho`, ...)."""
        return replace(self, **changes)

    @property
    def log_multiplier(self) -> float:
        """`2 kappa theta / xi^2`: what a branch jump multiplies `ln phi` by."""
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)


LEWIS = HestonPoint(
    name="lewis",
    spot=100.0,
    rate=0.01,
    dividend=0.02,
    v0=0.04,
    kappa=4.0,
    theta=0.25,
    xi=1.0,
    rho=-0.5,
)

FELLER_VIOLATED = HestonPoint(
    name="feller_violated",
    spot=100.0,
    rate=0.01,
    dividend=0.02,
    v0=0.04,
    kappa=0.5,
    theta=0.04,
    xi=1.0,
    rho=-0.9,
)

TRAP = LEWIS.at(name="trap", theta=0.3125)

SMALL_XI = HestonPoint(
    name="small_xi",
    spot=100.0,
    rate=0.05,
    dividend=0.01,
    v0=0.04,
    kappa=1.0,
    theta=0.04,
    xi=1e-06,
    rho=0.0,
)

POINTS: tuple[HestonPoint, ...] = (LEWIS, FELLER_VIOLATED, TRAP, SMALL_XI)

LEWIS_STRIKES: tuple[float, ...] = (80.0, 100.0, 120.0)
"""The three strikes of the published reference table."""

EXPIRIES: tuple[float, ...] = (0.25, 1.0, 5.0)
"""The maturities the smile term structure is measured at."""

KINDS = ("call", "put")
