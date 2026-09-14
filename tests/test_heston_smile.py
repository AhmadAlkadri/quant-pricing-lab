"""What the Heston smile actually looks like, measured rather than asserted.

The point of quoting a Heston price as a Black-Scholes implied volatility is
that the model's three shape parameters then act on one axis, and each acts on
a different feature of it. Five claims, all measured here through
`qpl.engines.fourier.smile`, which is nothing but `qpl.pricing.price` with
`method="fourier"` fed into the Brent inverter this package has had since
before the curriculum started.

(a) **The inverter inverts.** A Black-Scholes model priced by the transform and
    then inverted returns its own `sigma` to 2e-09 over three volatilities and
    three maturities. EXACT_IDENTITY (a round trip), and the test that would
    catch a broken inversion before any Heston claim is read.

(b) **`rho` is the sign of the skew.** At `rho = -0.5` the implied volatility is
    strictly decreasing in strike over `K = 70 ... 140` at every maturity
    tested, and the at-the-forward skew is `-0.2235 / -0.0943 / -0.0231` at
    `T = 0.25 / 1 / 5`. At `rho = +0.5` the skew is `+0.2248 / +0.0972 /
    +0.0245` -- the same magnitudes with the sign reversed. At `rho = 0` it is
    **exactly** zero, because the smile is then symmetric in log-moneyness.
    CLOSED_FORM for the sign, measurement for the size.

(c) **The smile flattens with maturity.** `|skew|` falls monotonically:
    0.2235, 0.1508, 0.0943, 0.0537, 0.0231, 0.0118 at
    `T = 0.25, 0.5, 1, 2, 5, 10`. Fitted decay exponent 0.80 in `T`.
    CONVERGENCE_ORDER (of a decay rate, not of a numerical scheme).

(d) **The at-the-forward implied variance runs from `v0` to `theta`.** With
    `v0 = 0.04` and `theta = 0.25` it climbs monotonically from 0.0434 at
    `T = 0.01` to 0.2326 at `T = 30`; with the two swapped it falls
    monotonically from 0.2448 to 0.0387. Both ends and the monotonicity are
    asserted. CLOSED_FORM for the limits.

(e) **A long-dated smile needs a narrower COS range, not a wider one.** The
    package default `L = 10` is off by 3.6e-02 at `T = 50` while `L = 8` is at
    4.8e-05 and `L = 6` at 3.9e-03 -- the round-off floor that grows with the
    range (Slice 14 measured the same effect under Black-Scholes) meeting a
    payoff coefficient carrying `e^b` with `b ~ L sqrt(theta T)`.
    NEGATIVE_FINDING, and the reason every long-maturity study in this file
    passes an explicit `truncation_l`.

Source for the qualitative shape statements: Gatheral (2006), "The Volatility
Surface", chapter 2. Nothing is quoted; every number above is measured by this
repository.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from heston_points import LEWIS

from qpl.engines.fourier import (
    FourierConfig,
    atm_implied_variance,
    implied_vol,
    implied_vol_surface,
    smile_skew,
)
from qpl.exceptions import InvalidInputError
from qpl.models import BlackScholesModel, HestonModel
from qpl.validation import EvidenceClass, fit_convergence_order

MARKET = LEWIS.market()

LONG_DATED = FourierConfig(truncation_l=6.0, n_terms=512)
"""COS settings for the maturities past a few years; see claim (e).

`L = 6` rather than the default 10, because the payoff coefficient's `e^b`
grows like `exp(L sqrt(c2))` and `c2 ~ theta T`. Measured error against a
Lewis integral at the forward strike: `T = 30` gives -9.5e-01 at `L = 4`,
-7.2e-05 at 6, +4.2e-07 at 8 and +8.4e-05 at 10; `T = 100` gives -3.0e-01 at
6, +2.0e-03 at 8 and **-1.2e+02** at 10."""

SMILE_STRIKES = (70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0)
SKEW_MATURITIES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def _heston(rho: float, *, v0: float = 0.04, theta: float = 0.25) -> HestonModel:
    return HestonModel(v0=v0, kappa=4.0, theta=theta, xi=1.0, rho=rho)


# --------------------------------------------------------------------------
# (a) The round trip.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", (0.10, 0.20, 0.35))
@pytest.mark.parametrize("expiry", (0.25, 1.0, 5.0))
def test_a_black_scholes_price_inverts_to_its_own_volatility(
    sigma: float, expiry: float
) -> None:
    """Evidence class: EXACT_IDENTITY (a round trip through two engines).

    The transform price and the inverter are different code paths that must
    compose to the identity. Worst measured residual 1.9e-09, set by the
    inverter's own `tol = 1e-07` on the price rather than by the transform.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    recovered = implied_vol(
        BlackScholesModel(sigma=sigma), MARKET, strike=95.0, expiry=expiry
    )
    assert recovered == pytest.approx(sigma, abs=1e-08)


# --------------------------------------------------------------------------
# (b) rho sets the sign of the skew.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("expiry", (0.25, 1.0, 5.0))
def test_negative_correlation_makes_the_smile_decreasing_in_strike(
    expiry: float,
) -> None:
    """The headline shape claim, over seven strikes rather than two.

    `rho < 0` means a falling spot comes with rising volatility, which fattens
    the left tail of the log-return law and lifts low-strike implied vols. The
    smile is strictly monotone decreasing over `K = 70 ... 140` at every
    maturity tested.
    """
    smile = [
        implied_vol(_heston(-0.5), MARKET, strike=strike, expiry=expiry)
        for strike in SMILE_STRIKES
    ]
    assert all(later < earlier for earlier, later in pairwise(smile))


@pytest.mark.parametrize(
    ("rho", "expected"),
    ((-0.5, (-0.2235, -0.0943, -0.0231)), (0.5, (0.2248, 0.0972, 0.0245))),
    ids=("rho_negative", "rho_positive"),
)
def test_the_skew_changes_sign_with_rho(rho: float, expected) -> None:
    """The sign flips and the magnitude barely moves; both are measured.

    Skew at the forward, `d sigma_imp / d ln(K/F)`, at `T = 0.25 / 1 / 5`:

        rho = -0.5   -0.2235   -0.0943   -0.0231
        rho = +0.5   +0.2248   +0.0972   +0.0245

    The two rows are not exact mirrors -- the drift's `-v/2` term is not
    symmetric in `rho` -- but they are within 6% of each other at every
    maturity, which is what "rho sets the sign" means quantitatively.
    """
    model = _heston(rho)
    for expiry, target in zip((0.25, 1.0, 5.0), expected, strict=True):
        measured = smile_skew(model, MARKET, expiry)
        assert measured == pytest.approx(target, abs=5e-04)
        assert math.copysign(1.0, measured) == math.copysign(1.0, rho)


@pytest.mark.parametrize("expiry", (0.25, 1.0, 5.0))
def test_zero_correlation_gives_an_exactly_symmetric_smile(expiry: float) -> None:
    """CLOSED_FORM. At `rho = 0` the transform is even in `u` up to the drift.

    The at-the-forward skew is then zero identically, not approximately: the
    two inversions at `F e^{+/- h}` return the same volatility to the
    inverter's tolerance, so the difference quotient is 0.0. The smile is still
    a smile -- 0.3691 / 0.3317 / 0.3666 at `K = 70 / 100 / 140`, `T = 0.25` --
    it is just symmetric, which is the point: `xi` makes curvature and `rho`
    makes slope.
    """
    assert smile_skew(_heston(0.0), MARKET, expiry) == pytest.approx(0.0, abs=1e-07)
    smile = [
        implied_vol(_heston(0.0), MARKET, strike=strike, expiry=expiry)
        for strike in (70.0, 100.0, 140.0)
    ]
    assert smile[0] > smile[1] < smile[2]


# --------------------------------------------------------------------------
# (c) The smile flattens with maturity.
# --------------------------------------------------------------------------


def test_the_skew_magnitude_falls_monotonically_with_maturity() -> None:
    """CONVERGENCE_ORDER, applied to a decay rate rather than to a scheme.

        T       0.25     0.5      1        2        5        10
        skew  -0.2235  -0.1508  -0.0943  -0.0537  -0.0231  -0.0118

    Strictly decreasing in magnitude, and a power-law fit in `T` gives exponent
    **0.804** -- close to, but measurably not, the `1/T` a large-maturity
    expansion would give, because `kappa T` is only 1 at the short end of this
    ladder. The log-space residual is **0.097**, which is a tenth of a decade
    and says the same thing: over this range the decay is not a clean power
    law, and the exponent is a summary of it rather than a theorem about it.
    """
    model = _heston(-0.5)
    skews = np.array(
        [abs(smile_skew(model, MARKET, expiry, cfg=LONG_DATED)) for expiry in SKEW_MATURITIES]
    )
    assert all(later < earlier for earlier, later in pairwise(skews))
    fit = fit_convergence_order(np.array(SKEW_MATURITIES), skews)
    assert fit.order == pytest.approx(-0.804, abs=0.02)
    assert 0.05 < fit.residual < 0.15


@pytest.mark.parametrize("expiry", (0.25, 1.0, 5.0))
def test_the_smile_is_flatter_at_longer_maturity_across_the_whole_strike_range(
    expiry: float,
) -> None:
    """The same statement without differentiating: the range of the smile falls.

    Measured `max - min` over `K = 70 ... 140`: 0.1175 at `T = 0.25`, 0.0635 at
    `T = 1`, 0.0160 at `T = 5`.
    """
    spreads = {}
    for maturity in (0.25, 1.0, 5.0):
        smile = [
            implied_vol(_heston(-0.5), MARKET, strike=strike, expiry=maturity)
            for strike in SMILE_STRIKES
        ]
        spreads[maturity] = max(smile) - min(smile)
    assert spreads[0.25] > spreads[1.0] > spreads[5.0]
    assert spreads[expiry] > 0.0


# --------------------------------------------------------------------------
# (d) The term structure of at-the-forward implied variance.
# --------------------------------------------------------------------------

TERM_MATURITIES = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


@pytest.mark.parametrize(
    ("v0", "theta", "increasing"),
    ((0.04, 0.25, True), (0.25, 0.04, False)),
    ids=("v0_below_theta", "v0_above_theta"),
)
def test_at_the_forward_implied_variance_runs_from_v0_to_theta(
    v0: float, theta: float, increasing: bool
) -> None:
    """CLOSED_FORM for both limits, measurement for the path between them.

    `sigma_imp(F, T)^2 T` is an average of the instantaneous variance over
    `[0, T]` to leading order, and `E[v_t] = theta + (v0 - theta) e^{-kappa t}`
    runs from `v0` to `theta`. So the implied variance must start near `v0` and
    end near `theta`, monotonically, in whichever direction the two are
    ordered. Measured:

        v0 = 0.04, theta = 0.25   0.0434 (T=0.01) ... 0.2326 (T=30)
        v0 = 0.25, theta = 0.04   0.2448 (T=0.01) ... 0.0387 (T=30)

    Neither end lands exactly on its limit, and the way it misses is the
    finding. The start is inside 0.004 of `v0` in both directions. The long end
    **undershoots `theta` in both directions**: 0.2326 against 0.25 when
    climbing and 0.0387 against 0.04 when falling. That asymmetry is not a
    convergence failure -- the sequence is still monotone at `T = 30` -- it is
    the smile's own curvature, which pulls the at-the-forward implied variance
    below the average variance whichever side the average approaches from.
    """
    model = _heston(-0.5, v0=v0, theta=theta)
    variances = [
        atm_implied_variance(model, MARKET, expiry, cfg=LONG_DATED)
        for expiry in TERM_MATURITIES
    ]
    direction = 1.0 if increasing else -1.0
    assert all(
        direction * (later - earlier) > 0.0
        for earlier, later in pairwise(variances)
    )
    assert abs(variances[0] - v0) < 0.006
    # Approaches theta, and from below in both directions (docstring).
    assert abs(variances[-1] - theta) < 0.02
    assert variances[-1] < theta
    assert abs(variances[-1] - theta) < abs(variances[0] - theta)


# --------------------------------------------------------------------------
# (e) Where the default COS range stops working, and the surface helper.
# --------------------------------------------------------------------------


def test_a_long_dated_smile_needs_a_narrower_range_than_the_default() -> None:
    """NEGATIVE_FINDING, and the reason `LONG_DATED` exists.

    The COS call's payoff coefficient carries `e^b` with `b = c1 + L sqrt(c2)`
    and `c2 ~ theta T`, so widening the range costs precision exponentially in
    `L sqrt(T)`. At `T = 50` the package default `L = 10` is 3.6e-02 off a
    Lewis integral of the same transform while `L = 8` is 4.8e-05; at
    `T = 100` the default is off by **1.2e+02** on a price of 13.3.

    This is the Black-Scholes round-off floor of Slice 14 (2.7e-14 / 5.9e-13 /
    5.5e-11 at `L = 10 / 20 / 40`) with `sqrt(theta T)` in place of
    `sigma sqrt(T)`, so a long maturity moves along the same axis a large `L`
    does.
    """
    from qpl.engines.fourier.lewis import lewis_call
    from qpl.instruments.options import EuropeanOption
    from qpl.pricing import price

    model = _heston(-0.5)
    expiry = 100.0
    forward = LEWIS.spot * math.exp((LEWIS.rate - LEWIS.dividend) * expiry)
    reference, _ = lewis_call(
        model,
        s0=LEWIS.spot,
        strike=forward,
        expiry=expiry,
        rate=LEWIS.rate,
        dividend=LEWIS.dividend,
        limit=800,
        tolerance=1e-13,
    )
    option = EuropeanOption(kind="call", strike=forward, expiry=expiry)

    def cos_at(truncation_l: float) -> float:
        return price(
            option,
            model,
            MARKET,
            method="fourier",
            cfg=FourierConfig(truncation_l=truncation_l, n_terms=512),
        ).value

    assert abs(cos_at(10.0) - reference) > 1.0
    assert abs(cos_at(8.0) - reference) < 1e-02


def test_the_surface_helper_returns_expiries_by_strikes() -> None:
    model = _heston(-0.5)
    strikes = (80.0, 100.0, 120.0)
    expiries = (0.25, 1.0, 5.0)
    surface = implied_vol_surface(model, MARKET, strikes, expiries)
    assert surface.shape == (len(expiries), len(strikes))
    for row_index, expiry in enumerate(expiries):
        # Each row is a smile at one maturity: decreasing in strike.
        assert all(
            later < earlier for earlier, later in pairwise(surface[row_index])
        )
        for column_index, strike in enumerate(strikes):
            assert surface[row_index, column_index] == pytest.approx(
                implied_vol(model, MARKET, strike=strike, expiry=expiry)
            )
    # Each column is a term structure: rising toward sqrt(theta) = 0.5.
    for column in surface.T:
        assert all(later > earlier for earlier, later in pairwise(column))
        assert column[-1] < math.sqrt(0.25)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"strikes": [[100.0]], "expiries": [1.0]}, "one-dimensional"),
        ({"strikes": [], "expiries": [1.0]}, "non-empty"),
    ),
)
def test_surface_input_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        implied_vol_surface(_heston(-0.5), MARKET, **kwargs)


def test_skew_refuses_a_non_positive_bump() -> None:
    with pytest.raises(InvalidInputError, match="bump must be > 0"):
        smile_skew(_heston(-0.5), MARKET, 1.0, bump=0.0)
