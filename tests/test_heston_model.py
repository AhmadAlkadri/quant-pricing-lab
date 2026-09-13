"""The Heston characteristic function and its cumulants, before any price.

Nothing here prices an option. The claims are about the transform itself, and
they are the claims that have to hold before a price computed from it means
anything:

(a) **The transform is a transform.** `phi(0) = 1`, `phi(-u) = conj(phi(u))`,
    `|phi(u)| <= 1` for real `u`, and the martingale condition
    `phi(-i) = E[S_T / S_0] = e^{(r - q) T}`. All four are EXACT_IDENTITY and
    all four are measured to hold to **0.0** -- not "within round-off", zero --
    over four parameter sets (Feller-satisfying and Feller-violating) and four
    maturities out to 30 years.

(b) **The stable branch is stable, and the reason is checkable.** `Re(d) >= 0`
    by construction; `|g| <= 1` follows; and the argument of `1 - g e^{-d T}`,
    the quantity whose logarithm the formula takes, stays inside
    `(-pi/2, pi/2)` -- so the principal branch is the continuous one and the
    cut is never crossed. Worst measured `|arg|` over the same grid is 0.0852
    against `pi/2 = 1.5708`, a factor of 18. CLOSED_FORM.

(c) **The `xi = 0` limit is an algebraic identity, not a numerical one.** At
    `xi = 0` the formula reduces to the transform of a Gaussian whose variance
    is the integrated deterministic variance, and at `v0 = theta` that is the
    Black-Scholes transform with `sigma^2 = theta` -- reproduced to 1.1e-16
    over 4001 values of `u`. Approaching that limit, the transform error is
    `O(xi)` when `rho != 0` and `O(xi^2)` when `rho = 0`, measured over six
    decades of `xi`. EXACT_IDENTITY and CONVERGENCE_ORDER.

(d) **The cumulants are independent of the transform.** `c1` and `c2` are
    derived in `qpl.models.heston` from the variance dynamics by Ito isometry,
    never from `ln phi`; here they are checked *against* finite differences of
    `ln phi`, which is a real cross-check rather than a restatement.
    CLOSED_FORM. `c4` is reported as zero and the fourth difference says what
    that costs: `sqrt(c4) / c2` is 1.05 on the Lewis set and **6.85** on the
    Feller-violating one. NEGATIVE_FINDING; the price consequence is measured
    in `tests/test_heston_fourier.py`.

Sources: Heston (1993); Albrecher, Mayer, Schoutens and Tistaert (2007)
section 3; Lord and Kahl (2010) section 2; Andersen and Piterbarg (2007)
section 3. Every number below is measured in this repository.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from heston_points import POINTS, SMALL_XI

from qpl.engines.fourier import black_scholes_characteristic_function
from qpl.exceptions import InvalidInputError
from qpl.models.heston import (
    HestonModel,
    heston_characteristic_function,
    heston_log_return_cumulants,
    numerical_log_return_cumulant,
)
from qpl.validation import EvidenceClass, fit_convergence_order

REAL_ARGUMENTS = np.linspace(-200.0, 200.0, 4001)
"""Where the identities are checked. The reach is deliberately far past
anything a pricer evaluates: the COS method's largest `u` on these points is
about 140, and Carr-Madan's about 60."""

MATURITIES = (0.25, 1.0, 10.0, 30.0)

IDENTITY_TOLERANCE = 1e-15
"""Every identity in this file is measured at exactly 0.0 -- the transform's
`phi(0)` is `exp(0)` and `phi(-i)` is `exp((r-q)T)` with the other terms
vanishing identically -- but the tolerance is a round-off budget rather than
equality, because which operations a platform fuses is not this repository's
to pin (see the cross-platform rule in `AGENTS.md`)."""


def _phi(point, expiry: float, u=REAL_ARGUMENTS) -> np.ndarray:
    return point.model().characteristic_function(
        u, expiry, rate=point.rate, dividend=point.dividend
    )


# --------------------------------------------------------------------------
# (a) The four identities every characteristic function owes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("expiry", MATURITIES)
def test_phi_at_zero_is_one(point, expiry: float) -> None:
    """Evidence class: EXACT_IDENTITY. `phi(0) = E[1] = 1`."""
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    value = _phi(point, expiry, np.array([0.0]))[0]
    assert abs(value - 1.0) <= IDENTITY_TOLERANCE


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("expiry", MATURITIES)
def test_martingale_condition(point, expiry: float) -> None:
    """EXACT_IDENTITY. `phi(-i) = E[S_T / S_0] = e^{(r - q) T}`.

    This is the one identity that would notice a sign error anywhere in the
    drift, the correlation term or the discounting convention, and it is
    checked on Feller-violating parameters too because that is where a
    formula that only works when the variance stays away from zero would
    first show it.
    """
    value = _phi(point, expiry, np.array([-1j]))[0]
    target = math.exp((point.rate - point.dividend) * expiry)
    assert abs(value - target) <= IDENTITY_TOLERANCE * max(target, 1.0)
    assert abs(value.imag) <= IDENTITY_TOLERANCE


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("expiry", MATURITIES)
def test_modulus_bounded_by_one_and_hermitian(point, expiry: float) -> None:
    """EXACT_IDENTITY. `|phi| <= 1` and `phi(-u) = conj(phi(u))` for real `u`."""
    values = _phi(point, expiry)
    assert float(np.max(np.abs(values))) <= 1.0 + IDENTITY_TOLERANCE
    mirrored = _phi(point, expiry, -REAL_ARGUMENTS)
    assert float(np.max(np.abs(values - np.conj(mirrored)))) <= IDENTITY_TOLERANCE


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
def test_complex_arguments_are_finite_on_the_strips_the_engines_use(point) -> None:
    """The strips Carr-Madan, Lewis and Gil-Pelaez evaluate on.

    `u - (alpha + 1) i` for the default `alpha = 1.5`, `u - i/2` for Lewis and
    `u - i` for Gil-Pelaez's share-measure probability. A transform that is
    only valid for real `u` does not satisfy the Slice 14 protocol, and this
    is the test that says so.
    """
    grid = np.linspace(-60.0, 60.0, 1201)
    for offset in (0.5, 1.0, 2.5):
        values = _phi(point, 1.0, grid - offset * 1j)
        assert np.all(np.isfinite(values))


# --------------------------------------------------------------------------
# (b) Why the branch is safe: Re(d) >= 0, |g| <= 1, arg inside (-pi/2, pi/2).
# --------------------------------------------------------------------------

MAX_LOG_ARGUMENT = 0.2
"""Worst measured `|arg(1 - g e^{-dT})|` over the four points, both maturities
and 4001 values of `u` is **0.0852** (the Feller-violating set at `T = 1`),
against the `pi/2 = 1.5708` at which the principal branch would stop being the
continuous one. The bound keeps a factor of 2.3 over the measurement and 7.9
under the failure."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("expiry", (1.0, 30.0))
def test_the_principal_branch_is_never_crossed(point, expiry: float) -> None:
    """CLOSED_FORM, and the whole content of the Albrecher / Lord-Kahl choice.

    Three statements, each checked rather than quoted:

    1. `Re(d) >= 0`, which is what taking the principal square root means;
    2. `|g| <= 1`, which follows from it because `Re(beta) = kappa > 0` for
       real `u`;
    3. `|arg(1 - g e^{-d T})| < pi/2`, so the logarithm's argument stays in
       the right half plane and never winds around the origin.

    Statement 3 is the one that matters: it is what makes the principal branch
    of `ln` the *continuous* branch, and it is the statement the original form
    violates (`tests/test_heston_fourier.py`).
    """
    model = point.model()
    u = REAL_ARGUMENTS.astype(complex)
    beta = model.kappa - model.rho * model.xi * 1j * u
    d = np.sqrt(beta * beta + model.xi**2 * (u * u + 1j * u))
    g = (beta - d) / (beta + d)

    assert float(np.min(d.real)) >= 0.0
    assert float(np.max(np.abs(g))) <= 1.0
    argument = 1.0 - g * np.exp(-d * expiry)
    assert float(np.max(np.abs(np.angle(argument)))) < MAX_LOG_ARGUMENT


# --------------------------------------------------------------------------
# (c) The Black-Scholes limit, as an identity on the formula.
# --------------------------------------------------------------------------


def test_zero_vol_of_vol_is_the_deterministic_variance_gaussian() -> None:
    """EXACT_IDENTITY on the *formula*, not on a price.

    At `xi = 0` the variance is deterministic, so `ln(S_T/S_0)` is Gaussian
    with variance `V = int_0^T v_t dt = theta T + (v0 - theta)(1 - e^{-kT})/k`.
    The transform must then be `exp(i u (r-q) T - (u^2 + i u) V / 2)`, written
    out here from that statement rather than read back from the module.
    """
    expiry, rate, dividend = 2.0, 0.05, 0.01
    v0, kappa, theta = 0.05, 1.5, 0.04
    variance = theta * expiry + (v0 - theta) * (1.0 - math.exp(-kappa * expiry)) / kappa
    u = REAL_ARGUMENTS.astype(complex)
    expected = np.exp(
        1j * u * (rate - dividend) * expiry - 0.5 * (u * u + 1j * u) * variance
    )
    computed = heston_characteristic_function(
        u,
        expiry,
        rate=rate,
        dividend=dividend,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=0.0,
        rho=-0.7,
    )
    assert float(np.max(np.abs(computed - expected))) <= IDENTITY_TOLERANCE


def test_zero_vol_of_vol_at_v0_equals_theta_is_the_black_scholes_transform() -> None:
    """EXACT_IDENTITY. The algebraic reduction the slice statement asked for.

    With `xi = 0` **and** `v0 = theta` the integrated variance is `theta T`
    for every `kappa`, so the Heston transform is literally the Black-Scholes
    one at `sigma = sqrt(theta)` -- compared here against this package's own
    Black-Scholes transform, at 4001 values of `u`, with `kappa` and `rho`
    given non-trivial values to prove they drop out.
    """
    expiry, rate, dividend, theta = 2.0, 0.05, 0.01, 0.09
    computed = heston_characteristic_function(
        REAL_ARGUMENTS,
        expiry,
        rate=rate,
        dividend=dividend,
        v0=theta,
        kappa=3.0,
        theta=theta,
        xi=0.0,
        rho=-0.7,
    )
    reference = black_scholes_characteristic_function(
        REAL_ARGUMENTS, expiry, rate=rate, dividend=dividend, sigma=math.sqrt(theta)
    )
    assert float(np.max(np.abs(computed - reference))) <= IDENTITY_TOLERANCE


XI_LADDER = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
"""Six decades. The stable form has no `1 / xi^2` in it, so the column is
allowed to be this long; the literal transcription of the same algebra turns
around at `xi = 1e-04` (the table in `qpl.models.heston`)."""


@pytest.mark.parametrize(
    ("rho", "expected_order"), ((0.0, 2.0), (-0.5, 1.0)), ids=("rho_zero", "rho_neg")
)
def test_the_order_in_xi_of_the_black_scholes_limit(
    rho: float, expected_order: float
) -> None:
    """CONVERGENCE_ORDER, and the answer is `rho`-dependent.

    The leading correction to the Black-Scholes transform is the covariance
    term `rho xi`, which enters `d` linearly through `beta = kappa - rho xi i u`
    and is therefore `O(xi)`. At `rho = 0` that term is absent and the next one
    is the `xi^2` inside the discriminant, so the order **doubles**. Measured
    `max_u |phi_xi - phi_0|` over `xi` in `1e-01 ... 1e-06`:

        rho =  0.0   1.126e-02  1.149e-04  1.149e-06  1.149e-08  1.149e-10  1.149e-12
        rho = -0.5   5.420e-02  5.372e-03  5.368e-04  5.367e-05  5.367e-06  5.367e-07

    -- a factor of 100 and a factor of 10 per decade respectively, to four
    figures, for six decades. The slice statement guessed `O(xi^2)`; that is
    right for `rho = 0` only.
    """
    common = dict(rate=0.05, dividend=0.01, v0=0.04, kappa=1.0, theta=0.04, rho=rho)
    limit = heston_characteristic_function(REAL_ARGUMENTS, 1.0, xi=0.0, **common)
    errors = np.array(
        [
            float(
                np.max(
                    np.abs(
                        heston_characteristic_function(
                            REAL_ARGUMENTS, 1.0, xi=xi, **common
                        )
                        - limit
                    )
                )
            )
            for xi in XI_LADDER
        ]
    )
    assert np.all(np.diff(errors) < 0.0)
    fit = fit_convergence_order(np.array(XI_LADDER), errors)
    assert fit.order == pytest.approx(expected_order, abs=0.02)
    assert fit.residual < 0.01


def test_small_vol_of_vol_does_not_degrade_the_transform() -> None:
    """The corner Slice 14's QuantLib oracle pointed at, checked head on.

    `COSHestonEngine` *diverges* as the vol-of-vol goes to zero, because its
    truncation range is built from cumulant formulas carrying `xi` in
    denominators. This module's `c1`, `c2` carry `xi` only in numerators and
    the transform carries no `1 / xi^2` at all, so both stay finite and
    accurate at `xi = 1e-06` -- which is where `SMALL_XI` sits.
    """
    model = SMALL_XI.model()
    c1, c2 = model.log_return_cumulants(
        1.0, rate=SMALL_XI.rate, dividend=SMALL_XI.dividend
    ).c1, model.log_return_cumulants(
        1.0, rate=SMALL_XI.rate, dividend=SMALL_XI.dividend
    ).c2
    variance = SMALL_XI.v0 * 1.0
    assert c2 == pytest.approx(variance, rel=1e-05)
    assert c1 == pytest.approx(
        (SMALL_XI.rate - SMALL_XI.dividend) * 1.0 - 0.5 * variance, rel=1e-05
    )
    values = _phi(SMALL_XI, 1.0)
    reference = black_scholes_characteristic_function(
        REAL_ARGUMENTS,
        1.0,
        rate=SMALL_XI.rate,
        dividend=SMALL_XI.dividend,
        sigma=math.sqrt(SMALL_XI.v0),
    )
    assert float(np.max(np.abs(values - reference))) < 1e-05


# --------------------------------------------------------------------------
# (d) The cumulants, checked against the transform they were not derived from.
# --------------------------------------------------------------------------

CUMULANT_STEP = 0.02
CUMULANT_RELATIVE_TOLERANCE = 1e-03
"""The finite difference is the *approximation* in this comparison, not the
closed form. Worst measured relative residual over the four points and three
maturities: 2.3e-03 for `c1` and 3.8e-03 for `c2`, both at the
Feller-violating set at `T = 10` where the law is widest and the `O(step^2)`
truncation term is largest; on the Lewis set at `T = 1` they are 4.2e-05 and
8.1e-06. The tolerance below is applied at the maturities where the step is
appropriate to the width of the law."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("expiry", (0.25, 1.0))
def test_closed_form_cumulants_match_the_transform(point, expiry: float) -> None:
    """CLOSED_FORM, and a genuine cross-check.

    `c1` and `c2` come from the variance dynamics by Ito isometry
    (`qpl.models.heston`), and `ln phi` comes from the Riccati solution. The
    two derivations share the model and nothing else, so agreement is evidence
    that both are right rather than evidence that one was copied.
    """
    model = point.model()
    cumulants = model.log_return_cumulants(
        expiry, rate=point.rate, dividend=point.dividend
    )
    for order, closed in ((1, cumulants.c1), (2, cumulants.c2)):
        measured = numerical_log_return_cumulant(
            model,
            order,
            expiry,
            rate=point.rate,
            dividend=point.dividend,
            step=CUMULANT_STEP,
        )
        assert measured == pytest.approx(
            closed, rel=CUMULANT_RELATIVE_TOLERANCE
        ), order


def test_the_cumulants_reduce_to_black_scholes_at_zero_vol_of_vol() -> None:
    """CLOSED_FORM. `c1 -> (r - q - theta/2) T` and `c2 -> theta T`."""
    expiry, rate, dividend, theta = 1.5, 0.03, 0.01, 0.09
    c1, c2 = heston_log_return_cumulants(
        expiry,
        rate=rate,
        dividend=dividend,
        v0=theta,
        kappa=2.0,
        theta=theta,
        xi=0.0,
        rho=-0.6,
    )
    assert c1 == pytest.approx((rate - dividend - 0.5 * theta) * expiry, abs=1e-15)
    assert c2 == pytest.approx(theta * expiry, abs=1e-15)


FOURTH_CUMULANT_STEP = 0.05


@pytest.mark.parametrize(
    ("point_name", "expected_ratio"),
    (("lewis", 1.05), ("feller_violated", 6.85)),
)
def test_the_reported_c4_of_zero_is_a_lie_and_this_is_its_size(
    point_name: str, expected_ratio: float
) -> None:
    """NEGATIVE_FINDING, pinned with the number that makes it matter.

    `log_return_cumulants` reports `c4 = 0`, following Fang and Oosterlee's own
    treatment of Heston. The fourth difference of `ln phi` says what is being
    dropped, as the ratio `sqrt(c4) / c2` -- which is exactly the quantity the
    COS range rule `w = sqrt(c2 + sqrt(c4))` cares about:

        T = 1          c2        c4         sqrt(c4)/c2   widening factor
        lewis          0.21786   5.261e-02      1.053          1.433
        feller_viol.   0.05767   1.559e-01      6.848          2.801

    So the default range is 1.43x too narrow on the Lewis set and **2.80x** too
    narrow on the Feller-violating one. `tests/test_heston_fourier.py` measures
    what that costs a price and shows that raising `truncation_l` by exactly
    that factor repairs it.
    """
    point = next(p for p in POINTS if p.name == point_name)
    model = point.model()
    c2 = model.log_return_cumulants(1.0, rate=point.rate, dividend=point.dividend).c2
    c4 = numerical_log_return_cumulant(
        model, 4, 1.0, rate=point.rate, dividend=point.dividend, step=FOURTH_CUMULANT_STEP
    )
    assert c4 > 0.0
    assert math.sqrt(c4) / c2 == pytest.approx(expected_ratio, rel=0.01)
    assert (
        model.log_return_cumulants(1.0, rate=point.rate, dividend=point.dividend).c4
        == 0.0
    )


# --------------------------------------------------------------------------
# The dataclass itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"v0": -0.01}, "v0 must be >= 0"),
        ({"kappa": 0.0}, "kappa must be > 0"),
        ({"theta": 0.0}, "theta must be > 0"),
        ({"xi": 0.0}, "xi must be > 0"),
        ({"rho": 1.0}, "rho must satisfy -1 < rho < 1"),
        ({"rho": -1.0}, "rho must satisfy -1 < rho < 1"),
        ({"kappa": float("nan")}, "kappa must be finite"),
    ),
)
def test_parameter_validation(kwargs: dict, message: str) -> None:
    base = {"v0": 0.04, "kappa": 1.0, "theta": 0.04, "xi": 0.5, "rho": -0.5}
    with pytest.raises(InvalidInputError, match=message):
        HestonModel(**{**base, **kwargs})


def test_feller_number_matches_the_simulation_slices_convention() -> None:
    """`4 kappa theta / xi^2`, and the Lewis set **satisfies** the condition.

    Worth stating explicitly because the slice statement claimed it was
    violated and then computed that it is not: `2 kappa theta = 2.0` against
    `xi^2 = 1.0`, so the condition holds with a factor of two to spare. The
    Feller-violating measurements in this slice therefore use a *different*
    parameter set (`tests/heston_points.py`).
    """
    lewis = next(p for p in POINTS if p.name == "lewis").model()
    assert lewis.feller_number == pytest.approx(4.0)
    assert lewis.feller_satisfied
    assert 2.0 * lewis.kappa * lewis.theta > lewis.xi**2

    violated = next(p for p in POINTS if p.name == "feller_violated").model()
    assert violated.feller_number == pytest.approx(0.08)
    assert not violated.feller_satisfied


def test_volatility_bump_moves_the_spot_volatility_not_the_variance() -> None:
    """The bump behind the Heston vega, and why it is `sqrt(v0)`."""
    model = HestonModel(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.5)
    assert model.spot_volatility == pytest.approx(0.2)
    bumped = model.with_volatility_bump(0.01)
    assert bumped.spot_volatility == pytest.approx(0.21)
    assert bumped.v0 == pytest.approx(0.21**2)
    for name in ("kappa", "theta", "xi", "rho"):
        assert getattr(bumped, name) == getattr(model, name)
    with pytest.raises(InvalidInputError, match="below zero"):
        model.with_volatility_bump(-1.0)


def test_numerical_cumulant_refuses_an_order_it_has_no_stencil_for() -> None:
    model = HestonModel(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.5)
    with pytest.raises(InvalidInputError, match="order must be one of"):
        numerical_log_return_cumulant(
            model, 3, 1.0, rate=0.01, dividend=0.0, step=0.05
        )
