"""Heston prices through the Slice 14 transform engines, and what each costs.

Six claims.

(a) **The six published values.** `S = 100, r = 1%, q = 2%, v0 = 0.04,
    kappa = 4, theta = 0.25, xi = 1, rho = -0.5, T = 1`, strikes 80/100/120,
    calls and puts, reproduced to better than 1e-04 by all four transform
    methods. PUBLISHED_BENCHMARK. The residuals are 1.3e-05 to 4.5e-05 and are
    **the rounding of the published four-decimal figures**, not method error:
    the four methods agree with each other to 1e-09, which is three decimal
    orders inside the published resolution.

(b) **Put-call parity, and a new failure of it.** Under Black-Scholes the COS
    parity residual was 2.4e-14 (Slice 14). Under Heston at the same settings
    it is **2.70e-08**, and the cause is measured: the COS *call* is at
    1.4e-12 while the COS *put* is at 2.70e-08, the same number at every
    strike. It is the truncated left tail -- with `rho < 0` the log-return
    density is left-skewed, and the put's payoff coefficient integrates from
    the range's lower end while the call's integrates to its upper end.
    EXACT_IDENTITY for the relation, NEGATIVE_FINDING for the size.

(c) **The Black-Scholes limit of the price.** Order 1.99 in `xi` at `rho = 0`
    and 1.00 at `rho = -0.5`, with the `rho != 0` error changing **sign**
    across the strike and vanishing near the money -- where the fitted order is
    0.32 with a log-space residual of 0.33, i.e. not a power law at all.
    CONVERGENCE_ORDER, plus one NEGATIVE_FINDING about where an order may not
    be fitted.

(d) **The little Heston trap.** The original branch choice, implemented as a
    labelled diagnostic, is right to 1.6e-08 relative at `T = 1` and wrong by
    **86%** at `T = 2` on a parameter set differing from the Lewis one in
    `theta` alone. On the Lewis set itself it is right at every maturity out to
    30 years, because `2 kappa theta / xi^2 = 2` is an integer and the branch
    jump multiplies the transform by `exp(-4 pi i kappa theta / xi^2) = 1`.
    NEGATIVE_FINDING, and it contradicts the slice statement, which proposed
    demonstrating the trap on the Lewis parameters.

(e) **The COS truncation range, and what `c4 = 0` costs.** On the Lewis set the
    default `L = 10` is at 1.4e-12 and flat in `N` from `N = 256`. On a
    Feller-violating set it is flat in `N` at **9.1e-04** -- a range error, not
    a series error, exactly the diagnosis Slice 14's QuantLib `COSHestonEngine`
    finding predicted -- and raising `L` by the derived factor
    `sqrt(1 + sqrt(c4)/c2) = 2.80` takes it to 1.2e-10. CONVERGENCE_ORDER and
    NEGATIVE_FINDING.

(f) **Carr-Madan's damping is now bounded, and by the right bound.** Under
    Black-Scholes `E[S^{alpha+1}]` is finite for every `alpha` and the
    integrability constraint never binds (Slice 14). Under Heston it binds:
    the critical moment at `T = 1` on the Lewis set is `w = 11.6905`, i.e.
    `alpha_max = 10.6905`, and the measured error is 3.1e-08 at `alpha = 10.60`
    and 2.7e+02 at `alpha = 10.65`. The method survives while the explosion
    time exceeds the maturity by about 2% and fails once the pole is within 1%
    of the contour. PUBLISHED_BENCHMARK for the bound's provenance (Andersen
    and Piterbarg 2007), CONVERGENCE_ORDER-free measurement for the onset.

Sources: Heston (1993); Albrecher et al. (2007) section 3; Lord and Kahl (2010)
section 2; Fang and Oosterlee (2008) appendix; Andersen and Piterbarg (2007)
section 3; the six reference values are attributed by the QuantLib test suite
to Alan Lewis' Wilmott-forum posting. Every other number is measured here.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from heston_points import FELLER_VIOLATED, LEWIS, LEWIS_STRIKES, SMALL_XI, TRAP

from qpl.engines.fourier import FourierConfig
from qpl.engines.fourier.lewis import lewis_call
from qpl.exceptions import NotSupportedError
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models import BlackScholesModel, HestonModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


def _price(point, kind: str, strike: float, payoff: str = "vanilla", **cfg) -> float:
    return price(
        point.instrument(kind, strike, payoff),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(**cfg),
    ).value


def _reference_call(point, strike: float, expiry: float | None = None) -> float:
    """A Lewis integral of the same transform, used as the convergence reference.

    This is **not** independent evidence about the model: it reads the same
    characteristic function the COS method reads. What differs is the payoff
    transform and the quadrature -- Lewis integrates along a contour with
    adaptive QUADPACK and has no truncation range at all -- so it is exactly
    the right reference for a claim about the COS *range*, and exactly the
    wrong one for a claim about the transform. The published values above and
    `tests/oracle/test_heston_vs_quantlib.py` carry the latter.
    """
    value, _ = lewis_call(
        point.model(),
        s0=point.spot,
        strike=strike,
        expiry=point.expiry if expiry is None else expiry,
        rate=point.rate,
        dividend=point.dividend,
        limit=800,
        tolerance=1e-13,
    )
    return value


@dataclass(frozen=True)
class _OriginalBranch:
    """A `CharacteristicFunctionModel` built on the **unstable** branch choice.

    Same cumulants, same everything, except that `characteristic_function`
    delegates to `HestonModel.characteristic_function_original_branch`. That
    isolates the branch choice as the only difference between the two COS
    prices compared below.
    """

    model: HestonModel

    def characteristic_function(self, u, expiry, *, rate, dividend):
        return self.model.characteristic_function_original_branch(
            u, expiry, rate=rate, dividend=dividend
        )

    def log_return_cumulants(self, expiry, *, rate, dividend):
        return self.model.log_return_cumulants(expiry, rate=rate, dividend=dividend)


# --------------------------------------------------------------------------
# (a) The published reference values.
# --------------------------------------------------------------------------

PUBLISHED: dict[tuple[float, str], float] = {
    (80.0, "put"): 7.9589,
    (80.0, "call"): 26.7748,
    (100.0, "put"): 17.0553,
    (100.0, "call"): 16.0702,
    (120.0, "put"): 29.8110,
    (120.0, "call"): 9.0249,
}
"""Six values at the Lewis parameter set, quoted to four decimals.

Source: the QuantLib test suite's Heston cases, which attribute them to Alan
Lewis' posting on the Wilmott forums. Used as fixtures with that citation and
nothing else taken; the pricing code here is derived independently."""

PUBLISHED_TOLERANCE = 1e-04
"""Half a unit in the last published digit is 5e-05; worst measured residual is
4.51e-05, so this tolerance is the resolution of the source rather than a
statement about the methods. What the methods are actually worth is the
cross-method spread below."""

TRANSFORM_METHODS = ("cos", "lewis", "gil_pelaez", "carr_madan")
CROSS_METHOD_TOLERANCE = 5e-09
"""Worst measured spread across the four methods over the six cells: 2.70e-08
including COS (whose put carries the left-tail truncation measured in (b)) and
**1.04e-09** excluding it. The bound below is applied to the three methods that
share a contour integral; COS is compared against them separately."""


def _cfg_for(method: str) -> dict[str, Any]:
    if method == "carr_madan":
        return {"method": method, "carr_madan_transform": "quadrature"}
    return {"method": method}


@pytest.mark.parametrize(
    ("strike", "kind"), sorted(PUBLISHED), ids=[f"K{int(k)}_{s}" for k, s in sorted(PUBLISHED)]
)
@pytest.mark.parametrize("method", TRANSFORM_METHODS)
def test_the_six_published_values(strike: float, kind: str, method: str) -> None:
    """Evidence class: PUBLISHED_BENCHMARK."""
    assert EvidenceClass.PUBLISHED_BENCHMARK is EvidenceClass.PUBLISHED_BENCHMARK
    computed = _price(LEWIS, kind, strike, **_cfg_for(method))
    assert computed == pytest.approx(PUBLISHED[(strike, kind)], abs=PUBLISHED_TOLERANCE)


@pytest.mark.parametrize(
    ("strike", "kind"), sorted(PUBLISHED), ids=[f"K{int(k)}_{s}" for k, s in sorted(PUBLISHED)]
)
def test_the_contour_methods_agree_far_inside_the_published_resolution(
    strike: float, kind: str
) -> None:
    """The published table's four decimals are the binding constraint, not us.

    Lewis, Gil-Pelaez and Carr-Madan reach the same number three decimal orders
    below the resolution of the source, which is what says the 1.3e-05 to
    4.5e-05 residuals above are the table's rounding and not a method error.
    """
    values = [
        _price(LEWIS, kind, strike, **_cfg_for(method))
        for method in ("lewis", "gil_pelaez", "carr_madan")
    ]
    assert max(values) - min(values) < CROSS_METHOD_TOLERANCE


# --------------------------------------------------------------------------
# (b) Parity, and the left tail.
# --------------------------------------------------------------------------

COS_CALL_TOLERANCE = 1e-11
COS_PUT_TOLERANCE = 1e-07
"""Two tolerances, deliberately four orders apart, because the two errors are
four orders apart and pretending otherwise would hide the finding. Measured at
the package defaults (`L = 10`, `N = 256`) over the three strikes: call
9.97e-13 to 1.41e-12, put 2.70e-08 at every strike."""


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
def test_the_cos_call_is_four_orders_better_than_the_cos_put(strike: float) -> None:
    """NEGATIVE_FINDING: the truncated left tail, priced.

    Both are the same cosine sum against the same coefficients; they differ in
    which end of `[a, b]` the payoff coefficient integrates from. With
    `rho = -0.5` the log-return density is left-skewed, so the mass the range
    misses is at the lower end -- which is the put's end. The put's error is
    the *same number* at all three strikes (2.70e-08), which is what identifies
    it as missing mass rather than as anything strike-dependent.
    """
    call_reference = _reference_call(LEWIS, strike)
    put_reference = (
        call_reference
        - LEWIS.spot * math.exp(-LEWIS.dividend * LEWIS.expiry)
        + strike * math.exp(-LEWIS.rate * LEWIS.expiry)
    )
    call_error = abs(_price(LEWIS, "call", strike) - call_reference)
    put_error = abs(_price(LEWIS, "put", strike) - put_reference)
    assert call_error < COS_CALL_TOLERANCE
    assert put_error < COS_PUT_TOLERANCE
    assert put_error > 1000.0 * call_error


PARITY_TOLERANCE = 1e-07


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
def test_put_call_parity(strike: float) -> None:
    """EXACT_IDENTITY for the relation; the residual is the range error.

    Parity is a real check for COS alone -- the other three methods build their
    put from the call by parity, so for them it is an identity of the
    implementation. The residual here is 2.70e-08 and is exactly the put's own
    truncation error above, with the call's 1e-12 invisible next to it.
    """
    call = _price(LEWIS, "call", strike)
    put = _price(LEWIS, "put", strike)
    forward_leg = LEWIS.spot * math.exp(-LEWIS.dividend * LEWIS.expiry)
    strike_leg = strike * math.exp(-LEWIS.rate * LEWIS.expiry)
    assert (call - put) - (forward_leg - strike_leg) == pytest.approx(
        0.0, abs=PARITY_TOLERANCE
    )


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
def test_a_wider_range_repairs_parity(strike: float) -> None:
    """CONVERGENCE_ORDER-free but measured: the residual is set by `L`, not `N`.

        L         8         10        12        14
        parity  2.28e-06  2.70e-08  3.13e-10  1.34e-11

    A factor of about 85 per two units of `L`, and identical at all three
    strikes. That is the shape of a tail-mass term, not of a series term.
    """
    residuals = []
    for truncation_l in (8.0, 10.0, 12.0, 14.0):
        call = _price(LEWIS, "call", strike, truncation_l=truncation_l)
        put = _price(LEWIS, "put", strike, truncation_l=truncation_l)
        residuals.append(
            abs(
                (call - put)
                - (
                    LEWIS.spot * math.exp(-LEWIS.dividend * LEWIS.expiry)
                    - strike * math.exp(-LEWIS.rate * LEWIS.expiry)
                )
            )
        )
    assert all(b < a for a, b in pairwise(residuals))
    assert residuals[0] > 1e-06
    assert residuals[-1] < 1e-10


# --------------------------------------------------------------------------
# (c) The Black-Scholes limit of the price.
# --------------------------------------------------------------------------

LIMIT_XI = (0.1, 0.05, 0.025, 0.0125, 0.00625)
LIMIT_SPEC = {"v0": 0.04, "kappa": 1.0, "theta": 0.04}
LIMIT_MARKET = Market(
    spot=100.0, rate_curve=FlatRateCurve(0.05), dividend_curve=FlatDividendCurve(0.01)
)


def _limit_errors(rho: float, strike: float) -> np.ndarray:
    from qpl.instruments.options import EuropeanOption

    option = EuropeanOption(kind="call", strike=strike, expiry=1.0)
    reference = price(
        option, BlackScholesModel(sigma=math.sqrt(LIMIT_SPEC["theta"])), LIMIT_MARKET
    ).value
    return np.array(
        [
            price(
                option,
                HestonModel(**LIMIT_SPEC, xi=xi, rho=rho),
                LIMIT_MARKET,
                method="fourier",
                cfg=FourierConfig(),
            ).value
            - reference
            for xi in LIMIT_XI
        ]
    )


@pytest.mark.parametrize("strike", (80.0, 100.0, 120.0))
def test_uncorrelated_heston_approaches_black_scholes_at_order_two(
    strike: float,
) -> None:
    """CONVERGENCE_ORDER. `rho = 0` removes the linear term, so the order is 2.

    Measured orders 1.996 / 1.997 / 1.991 at `K = 80 / 100 / 120`, every
    log-space residual below 0.006.
    """
    errors = np.abs(_limit_errors(0.0, strike))
    fit = fit_convergence_order(np.array(LIMIT_XI), errors)
    assert fit.order == pytest.approx(2.0, abs=0.05)
    assert fit.residual < 0.01


@pytest.mark.parametrize("strike", (80.0, 90.0, 120.0))
def test_correlated_heston_approaches_black_scholes_at_order_one(
    strike: float,
) -> None:
    """CONVERGENCE_ORDER, and the slice statement's guess of 2 is wrong here.

    The leading correction is the `rho xi` covariance term, which is linear in
    `xi`. Measured orders 0.999 / 0.951 / 1.030 at `K = 80 / 90 / 120`.
    """
    errors = np.abs(_limit_errors(-0.5, strike))
    fit = fit_convergence_order(np.array(LIMIT_XI), errors)
    assert fit.order == pytest.approx(1.0, abs=0.06)
    assert fit.residual < 0.03


def test_the_leading_correction_changes_sign_and_no_order_may_be_fitted_there() -> None:
    """NEGATIVE_FINDING: an order fitted at the money would be meaningless.

    The `rho xi` term is odd in log-moneyness, so it is positive below the
    forward and negative above it and passes through zero in between. Measured
    signed error at `xi = 0.1`: `+1.62e-01` at `K = 80`, `+1.39e-01` at
    `K = 90`, `-4.36e-03` at `K = 100`, `-1.75e-01` at `K = 110`. The fit at
    `K = 100` reports order 0.32 with a log-space residual of 0.33 -- which is
    the point: the residual is the thing that says so.
    """
    signs = {strike: _limit_errors(-0.5, strike)[0] for strike in (80.0, 90.0, 110.0)}
    assert signs[80.0] > 0.0
    assert signs[90.0] > 0.0
    assert signs[110.0] < 0.0

    atm = np.abs(_limit_errors(-0.5, 100.0))
    fit = fit_convergence_order(np.array(LIMIT_XI), atm)
    assert fit.residual > 0.1
    assert fit.order < 0.6


def test_the_black_scholes_limit_survives_a_vol_of_vol_of_one_millionth() -> None:
    """The Slice 14 hand-off, answered.

    QuantLib's `COSHestonEngine` diverges as the vol-of-vol goes to zero and no
    term count helps, because its truncation range is built from cumulant
    formulas with `xi` in denominators. This engine's range comes from `c1` and
    `c2` that carry `xi` only in numerators, so at `xi = 1e-06` it reproduces
    the Black-Scholes price to better than 1e-09 at the package defaults.
    """
    from qpl.instruments.options import EuropeanOption

    option = EuropeanOption(kind="call", strike=100.0, expiry=SMALL_XI.expiry)
    reference = price(
        option, BlackScholesModel(sigma=math.sqrt(SMALL_XI.v0)), SMALL_XI.market()
    ).value
    computed = _price(SMALL_XI, "call", 100.0)
    assert computed == pytest.approx(reference, abs=1e-09)


# --------------------------------------------------------------------------
# (d) The little Heston trap.
# --------------------------------------------------------------------------

TRAP_COS = {"n_terms": 256, "truncation_l": 8.0}
"""`L = 8` rather than the default 10 because the comparison runs out to
`T = 30`, where the default range is wide enough that the stable COS price
itself picks up a 4.1e-05 round-off error from `e^b` in the payoff
coefficients. At `L = 8` the stable price is within 1.2e-06 of the Lewis
integral at every maturity tested, so the difference measured below is the
branch choice and nothing else."""


def _trap_pair(point, expiry: float) -> tuple[float, float]:
    from qpl.engines.fourier import cos_price

    model = point.model()
    common = {
        "s0": point.spot,
        "strike": 100.0,
        "expiry": expiry,
        "rate": point.rate,
        "dividend": point.dividend,
        "kind": "call",
        **TRAP_COS,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        unstable = cos_price(_OriginalBranch(model), **common).value
    return cos_price(model, **common).value, unstable


TRAP_BREAKDOWN = (
    (1.0, 1e-07),
    (1.1, 1e-04),
    (1.2, 1e-02),
    (2.0, 1.0),
)
"""`(maturity, upper bound on the relative error)` on the `TRAP` set. Measured
relative `|original - stable| / stable`: 1.61e-08, 8.14e-05, 2.19e-03,
8.57e-01 -- so the original branch loses its first significant figure between
`T = 1` and `T = 1.2`, and by `T = 2` the price is wrong by 86%."""


@pytest.mark.parametrize(("expiry", "bound"), TRAP_BREAKDOWN)
def test_the_original_branch_breaks_down_with_maturity(
    expiry: float, bound: float
) -> None:
    """NEGATIVE_FINDING, pinned with the maturity and the error.

    The two prices come from the same COS sum, the same range, the same term
    count and the same model; the *only* difference is which square root and
    which `g` the transform uses. The stable one agrees with an independent
    contour integral throughout; the unstable one does not.
    """
    stable, unstable = _trap_pair(TRAP, expiry)
    assert stable == pytest.approx(
        _reference_call(TRAP, 100.0, expiry), rel=1e-06
    )
    relative = abs(unstable - stable) / stable
    assert relative < bound
    if expiry >= 1.2:
        assert relative > 1e-03


def test_the_trap_is_invisible_on_the_lewis_set_at_every_maturity() -> None:
    """NEGATIVE_FINDING, and it contradicts the slice statement directly.

    The slice proposed demonstrating the trap "at T = 10 or 30 with the Lewis
    parameters". It cannot be done. A branch jump adds `2 pi i` to the
    logarithm, and `C` multiplies that by `-2 kappa theta / xi^2`, so the
    transform is multiplied by `exp(-4 pi i kappa theta / xi^2)`. On the Lewis
    set `2 kappa theta / xi^2 = 2` exactly, so that factor is 1 and the jump
    cancels. Measured relative difference between the two branches at
    `T = 1 / 2 / 5 / 10 / 30`: all below 5e-07, against 0.86 at `T = 2` on
    `TRAP`, which differs from `LEWIS` in `theta` alone (and therefore has the
    same `d`, the same `g` and the same windings at the same `u`).
    """
    assert LEWIS.log_multiplier == pytest.approx(2.0)
    assert TRAP.log_multiplier == pytest.approx(2.5)
    assert (LEWIS.kappa, LEWIS.xi, LEWIS.rho, LEWIS.v0) == (
        TRAP.kappa,
        TRAP.xi,
        TRAP.rho,
        TRAP.v0,
    )
    for expiry in (1.0, 2.0, 5.0, 10.0, 30.0):
        stable, unstable = _trap_pair(LEWIS, expiry)
        assert abs(unstable - stable) / stable < 5e-07, expiry


def test_the_two_branches_agree_where_the_logarithm_has_not_yet_wound() -> None:
    """The positive half of the trap: it is the same algebra until it is not.

    On the `TRAP` set at `T = 1` the first winding sits at `u = 19.4`, which is
    far enough out that `|phi|` there is below 1e-12 and the price barely
    notices. The location of that first crossing scales as `1/T` (measured
    `u * T` = 19.4, 6.3, 5.6, 5.2, 5.1 at `T = 1, 2, 3, 5, 10`), which is why
    maturity is the axis the failure moves along.
    """
    model = TRAP.model()
    grid = np.linspace(1e-06, 20.0, 4001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for expiry, expected_break in ((1.0, 19.4), (2.0, 3.16), (10.0, 0.509)):
            stable = model.characteristic_function(
                grid, expiry, rate=TRAP.rate, dividend=TRAP.dividend
            )
            unstable = model.characteristic_function_original_branch(
                grid, expiry, rate=TRAP.rate, dividend=TRAP.dividend
            )
            gap = np.abs(stable - unstable)
            first = grid[np.argmax(gap > 1e-08)]
            assert first == pytest.approx(expected_break, rel=0.02), expiry


# --------------------------------------------------------------------------
# (e) The COS truncation range.
# --------------------------------------------------------------------------

RANGE_TERMS = (512, 1024, 2048, 4096, 8192)


def _cos_errors_vs_terms(point, truncation_l: float) -> list[float]:
    reference = _reference_call(point, 100.0)
    return [
        abs(_price(point, "call", 100.0, n_terms=n, truncation_l=truncation_l) - reference)
        for n in RANGE_TERMS
    ]


def test_cos_is_at_its_floor_by_256_terms_when_feller_holds() -> None:
    """CONVERGENCE_ORDER, of a kind: there is nothing left to converge.

    On the Lewis set the error at `L = 10` is 1.41e-12 at `N = 256` and the
    **same double** at 512, 1024, 2048 and 4096 -- the Gaussian decay in `N`
    measured in Slice 14 has already reached the floating-point floor.
    """
    errors = _cos_errors_vs_terms(LEWIS, 10.0)
    assert errors[0] < 1e-11
    assert max(errors) - min(errors) < 1e-15


FELLER_VIOLATED_RANGE_LEVELS = (
    (10.0, 5e-04, 2e-03),
    (14.0, 1e-05, 1e-04),
    (20.0, 5e-08, 5e-07),
    (28.0, 1e-11, 1e-09),
)
"""`(L, lower, upper)` brackets for the flat-in-`N` error on the
Feller-violating set. Measured plateau (`N = 8192`): 9.131e-04, 2.749e-05,
1.284e-07, 1.222e-10. The term count needed to *reach* each plateau grows with
`L`, because the decay exponent is `N^2 / L^2`: `L = 10` is flat from
`N = 1024` and `L = 28` only from `N = 4096`."""


@pytest.mark.parametrize(
    ("truncation_l", "lower", "upper"),
    FELLER_VIOLATED_RANGE_LEVELS,
    ids=[f"L{int(row[0])}" for row in FELLER_VIOLATED_RANGE_LEVELS],
)
def test_a_feller_violating_law_makes_the_default_range_a_lie(
    truncation_l: float, lower: float, upper: float
) -> None:
    """NEGATIVE_FINDING, and the diagnosis Slice 14 asked for.

    The error is **flat in `N`** at every `L`: at `L = 10` it sits at 9.13e-04
    from `N = 1024` onward and no term count moves it. A COS error that does
    not respond to `N` is a range error, which is the exact signature Slice 14
    measured on QuantLib's `COSHestonEngine`. The cause here is known and
    quantified: `log_return_cumulants` reports `c4 = 0`, and on this parameter
    set `sqrt(c4)/c2 = 6.85`, so the range rule's `w = sqrt(c2 + sqrt(c4))` is
    too small by `sqrt(1 + 6.85) = 2.80`. Multiplying `L` by that factor --
    `L = 28` -- takes the plateau from 9.13e-04 to 1.22e-10.
    """
    errors = _cos_errors_vs_terms(FELLER_VIOLATED, truncation_l)
    plateau = errors[-1]
    assert lower < plateau < upper
    # Flat: the last two term counts agree far better than the error itself.
    assert abs(errors[-1] - errors[-2]) < 0.1 * plateau


def test_the_derived_truncation_l_is_the_one_the_package_publishes() -> None:
    from qpl.models.heston import HESTON_TRUNCATION_L_FELLER_VIOLATED

    assert HESTON_TRUNCATION_L_FELLER_VIOLATED == 28.0
    assert not FELLER_VIOLATED.model().feller_satisfied


# --------------------------------------------------------------------------
# (f) Carr-Madan and the moment explosion.
# --------------------------------------------------------------------------

ALPHA_SAFE = (0.5, 1.5, 3.0, 5.0, 8.0, 10.0, 10.6)
ALPHA_BROKEN = (10.65, 11.0, 12.0)


def _carr_madan(strike: float, alpha: float) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _price(
            LEWIS,
            "call",
            strike,
            method="carr_madan",
            carr_madan_transform="quadrature",
            alpha=alpha,
            n_quad=4096,
        )


def test_the_critical_moment_is_where_carr_madan_stops_working() -> None:
    """The bound and the measurement, in one test.

    `alpha_max(T) = w_c(T) - 1` where `w_c` solves `T*(w) = T`, derived in
    `qpl.models.heston` from this repository's own `d` and `g` and matching the
    criterion of Andersen and Piterbarg (2007) section 3. At the Lewis set and
    `T = 1`, `w_c = 11.6905` and `alpha_max = 10.6905`. Measured error against
    a Lewis integral:

        alpha    0.5      1.5      3.0      5.0      8.0     10.0     10.6
        err    1.6e-10  1.0e-09  1.2e-09  6.0e-09  2.3e-08  1.7e-08  3.1e-08
        alpha   10.65    11.0     12.0
        err    2.7e+02  6.1e+03  6.1e+03

    So the method is exact while the pole of `1 - g e^{-dT}` stays off the
    contour and fails within about 1% of it -- 10.60 has `T*(w) = 1.0233` and
    10.65 has `T*(w) = 1.0103`. Under Black-Scholes (Slice 14) no `alpha`
    failed for this reason at all, because `E[S^{alpha+1}]` is finite there for
    every `alpha`; this is the constraint the earlier slice predicted and could
    not produce.
    """
    model = LEWIS.model()
    critical = model.critical_moment(LEWIS.expiry)
    assert critical == pytest.approx(11.6905, abs=1e-03)
    assert math.isinf(model.moment_explosion_time(critical - 3.0))
    assert model.moment_explosion_time(critical) == pytest.approx(
        LEWIS.expiry, rel=1e-06
    )

    reference = _reference_call(LEWIS, 100.0)
    for alpha in ALPHA_SAFE:
        assert alpha + 1.0 < critical
        assert abs(_carr_madan(100.0, alpha) - reference) < 1e-07, alpha
    for alpha in ALPHA_BROKEN:
        assert abs(_carr_madan(100.0, alpha) - reference) > 1.0, alpha


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
def test_the_damping_failure_is_the_moment_bound_and_not_the_moneyness(
    strike: float,
) -> None:
    """Which is the contrast with Slice 14, where the failure *was* moneyness.

    Under Black-Scholes the large-`alpha` failure was catastrophic
    cancellation, bounded by `(S/K)^alpha ...`, so it arrived at a different
    `alpha` for every strike. Under Heston the binding constraint arrives at
    the *same* `alpha` at all three strikes, because the moment explosion is a
    property of the law and not of the contract.
    """
    reference = _reference_call(LEWIS, strike)
    assert abs(_carr_madan(strike, 10.6) - reference) < 1e-06
    assert abs(_carr_madan(strike, 10.65) - reference) > 1.0


# --------------------------------------------------------------------------
# Digitals and Greeks under Heston.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
@pytest.mark.parametrize("kind", ("call", "put"))
def test_the_digital_agrees_between_cos_and_gil_pelaez(
    strike: float, kind: str
) -> None:
    """CLOSED_FORM-free cross-method check; worst measured 8.9e-16.

    Two different derivations of the same number: the COS digital is an exact
    integral of an indicator against a cosine, and the Gil-Pelaez digital *is*
    the discounted exercise probability. Nothing about the model is shared
    beyond `phi` itself.
    """
    cos = _price(LEWIS, kind, strike, payoff="digital")
    gil_pelaez = _price(LEWIS, kind, strike, payoff="digital", method="gil_pelaez")
    assert cos == pytest.approx(gil_pelaez, abs=1e-14)


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
def test_digital_static_replication(strike: float) -> None:
    """EXACT_IDENTITY: a digital call plus a digital put is one unit of cash."""
    total = _price(LEWIS, "call", strike, payoff="digital") + _price(
        LEWIS, "put", strike, payoff="digital"
    )
    assert total == pytest.approx(math.exp(-LEWIS.rate * LEWIS.expiry), abs=1e-14)


GREEK_SPOT_BUMP = 0.01


@pytest.mark.parametrize("strike", LEWIS_STRIKES)
@pytest.mark.parametrize("kind", ("call", "put"))
def test_cos_delta_and_gamma_are_the_derivatives_they_claim_to_be(
    strike: float, kind: str
) -> None:
    """CLOSED_FORM. The COS Greeks are exact derivatives of the same sum.

    Checked against central differences of the COS price in the spot, which is
    the only reference available under a model with no closed-form Greeks. The
    difference quotient is the approximation here, not the Greek: at
    `h = 0.01` the worst residual is 3.4e-09 (delta) and 6.1e-09 (gamma), and
    at `h = 0.05` the delta residual grows to 8.6e-08 -- the quotient's own
    `O(h^2)` term, moving as `h^2` should.
    """

    def at(spot: float) -> float:
        market = Market(
            spot=spot,
            rate_curve=FlatRateCurve(LEWIS.rate),
            dividend_curve=FlatDividendCurve(LEWIS.dividend),
        )
        return price(
            LEWIS.instrument(kind, strike),
            LEWIS.model(),
            market,
            method="fourier",
            cfg=FourierConfig(),
        ).value

    computed = greeks(
        LEWIS.instrument(kind, strike),
        LEWIS.model(),
        LEWIS.market(),
        method="fourier",
        cfg=FourierConfig(),
    )
    up = at(LEWIS.spot + GREEK_SPOT_BUMP)
    down = at(LEWIS.spot - GREEK_SPOT_BUMP)
    here = at(LEWIS.spot)
    assert computed.delta == pytest.approx(
        (up - down) / (2.0 * GREEK_SPOT_BUMP), abs=1e-08
    )
    assert computed.gamma == pytest.approx(
        (up - 2.0 * here + down) / GREEK_SPOT_BUMP**2, abs=1e-08
    )


def test_the_heston_vega_is_taken_in_the_spot_volatility() -> None:
    """The metadata says which vega it is, and the number is finite and positive.

    Black-Scholes has one volatility; Heston has `v0`, `theta` and `xi`. What
    this package reports is `dV / d sqrt(v0)` -- the derivative in the
    instantaneous volatility, which is the quantity that plays the role
    Black-Scholes' `sigma` plays -- and the `GreeksResult` metadata names it
    rather than leaving a caller to guess.
    """
    result = greeks(
        LEWIS.instrument("call", 100.0),
        LEWIS.model(),
        LEWIS.market(),
        method="fourier",
        cfg=FourierConfig(),
    )
    assert result.meta["model"] == "HestonModel"
    assert result.meta["vega_units"] == "d/d_sqrt(v0)"
    assert result.vega > 0.0
    assert math.isfinite(result.theta)
    assert math.isfinite(result.rho)


def test_greeks_are_refused_for_the_methods_that_have_none() -> None:
    for method in ("lewis", "gil_pelaez", "carr_madan"):
        with pytest.raises(NotSupportedError, match="method='cos'"):
            greeks(
                LEWIS.instrument("call", 100.0),
                LEWIS.model(),
                LEWIS.market(),
                method="fourier",
                cfg=FourierConfig(method=method),
            )


@pytest.mark.parametrize("method", ("analytic", "mc", "pde", "tree"))
def test_the_other_four_methods_refuse_heston_and_name_the_route(method: str) -> None:
    """Registered-and-raising, so the message says *why* and what to use.

    An unregistered key would report "Unsupported instrument/model/market
    combination", which says nothing about the transform route that does work
    or about the slice that will open the one the caller asked for.
    """
    from qpl.engines.mc.pricers import MCConfig
    from qpl.engines.pde.pricers import PDEConfig
    from qpl.engines.tree.pricers import TreeConfig

    kwargs: dict[str, Any] = {
        "analytic": {},
        "mc": {"cfg": MCConfig()},
        "pde": {"cfg": PDEConfig()},
        "tree": {"cfg": TreeConfig()},
    }[method]
    with pytest.raises(NotSupportedError, match="method='fourier'"):
        price(
            LEWIS.instrument("call", 100.0),
            LEWIS.model(),
            LEWIS.market(),
            method=method,
            **kwargs,
        )
    assert "Heston" in _refusal_message(method, kwargs)


def _refusal_message(method: str, kwargs: dict[str, Any]) -> str:
    try:
        price(
            LEWIS.instrument("call", 100.0),
            LEWIS.model(),
            LEWIS.market(),
            method=method,
            **kwargs,
        )
    except NotSupportedError as error:  # pragma: no branch - always raised
        return str(error)
    raise AssertionError("expected a refusal")  # pragma: no cover
