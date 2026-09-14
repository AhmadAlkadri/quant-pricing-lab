"""Heston Monte Carlo through the registry: wiring, bias, and what did not hold.

Four families here.

1. **The Black-Scholes route is untouched.** Slice 16 adds engines; it must not
   move a single existing Monte Carlo number. Four pinned `(value, stderr)`
   pairs, compared with `math.isclose(rel_tol=1e-12)` so the claim is
   "bit-for-bit" without pinning a machine-precision literal by equality.

2. **The registry wiring.** Four instrument types resolve to the Heston Monte
   Carlo engines; `n_steps = 1` is refused with the message that says what
   `n_steps` now means; the Greeks that are refused are refused with a reason
   rather than with "unsupported combination".

3. **The bias study (c) and (e).** Measured weak error against the transform
   price, on the reference set and on the Feller-violating one, for QE and for
   full-truncation Euler. Path counts are modest and the estimator is the
   conditional one; the full ladder with fitted orders is
   `examples/heston_mc_qe.py` and `docs/notes/heston_monte_carlo_qe.md`.

4. **Two things the slice statement got wrong**, both pinned as measurements:
   - it expected QE's bias to be "far smaller" than Euler's at coarse steps on
     the reference set. Measured ratio at `dt = 1/4`: **2.6**, not orders of
     magnitude. The dramatic separation is on the *Feller-violating* set, where
     it is **160**.
   - it expected the `rho = 0` price to agree with the transform "within noise
     at coarse dt". It does not: the `rho = 0` discretisation bias at
     `dt = 1/4` is **-0.244**, larger in absolute value than the `rho = -0.5`
     one (-0.102). Correlation is not what the coarse-step bias is about, and
     the sharp correlation check lives in `tests/test_mc_heston.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from heston_points import FELLER_VIOLATED, LEWIS

from qpl.engines.fourier.pricers import FourierConfig
from qpl.engines.mc.heston import EULER_FULL_TRUNCATION, QE
from qpl.engines.mc.pricers import MCConfig
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import (
    AsianOption,
    BarrierOption,
    DigitalOption,
    EuropeanOption,
    uniform_fixing_times,
    uniform_monitoring_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.models.heston import HestonModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass

# --------------------------------------------------------------------------
# (1) The Black-Scholes Monte Carlo route is unchanged.
# --------------------------------------------------------------------------

_BS_MODEL = BlackScholesModel(sigma=0.2)
_BS_MARKET = Market(
    spot=100.0,
    rate_curve=FlatRateCurve(0.05),
    dividend_curve=FlatDividendCurve(0.01),
)

# Recorded from the engines *before* this slice touched anything in
# `qpl.engines.mc`. The only edit there was splitting `bridge_survival` into a
# thin wrapper over `bridge_survival_from_step_variance`, which passes exactly
# the expression it used to compute inline -- so the barrier row below is the
# one that would catch a change of arithmetic, not just a change of behaviour.
BLACK_SCHOLES_PINS = (
    ("european", 9.87581435958162, 0.1010585432987553),
    ("european_antithetic_control", 9.840199631254132, 0.01955015979139077),
    ("digital", 0.5148053645397864, 0.0033517508970401927),
    ("asian_control", 5.832855791702162, 0.001609822756307165),
    ("barrier_brownian_bridge", 8.002338626161404, 0.09760156732788065),
)


def _black_scholes_case(name: str):
    """`(instrument, cfg)` for one pinned Black-Scholes Monte Carlo row."""
    if name == "european":
        return EuropeanOption(kind="call", strike=100.0, expiry=1.0), MCConfig(
            n_paths=20_000, n_steps=1, seed=123
        )
    if name == "european_antithetic_control":
        return EuropeanOption(kind="call", strike=100.0, expiry=1.0), MCConfig(
            n_paths=20_000,
            n_steps=1,
            seed=123,
            variance_reduction=("antithetic", "control_variate"),
        )
    if name == "digital":
        return DigitalOption(
            kind="call", strike=100.0, expiry=1.0, cash=1.0
        ), MCConfig(n_paths=20_000, n_steps=1, seed=123)
    if name == "asian_control":
        return AsianOption(
            kind="call",
            strike=100.0,
            expiry=1.0,
            fixing_times=uniform_fixing_times(1.0, 12),
            averaging="arithmetic",
        ), MCConfig(
            n_paths=20_000, n_steps=1, seed=123, variance_reduction="control_variate"
        )
    return BarrierOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        barrier=90.0,
        barrier_type="down-and-out",
        monitoring=uniform_monitoring_times(1.0, 25),
    ), MCConfig(
        n_paths=20_000, n_steps=1, seed=123, barrier_correction="brownian_bridge"
    )


@pytest.mark.parametrize(
    ("name", "value", "stderr"), BLACK_SCHOLES_PINS, ids=[p[0] for p in BLACK_SCHOLES_PINS]
)
def test_the_black_scholes_mc_route_is_unchanged(
    name: str, value: float, stderr: float
) -> None:
    """Evidence class: EXACT_IDENTITY (a regression pin on this repository).

    Adding a model must not move an existing number. `MCConfig` grew two
    fields, `qpl.pricing` grew four registrations and `qpl.engines.mc.barrier`
    grew one function; none of them is on the Black-Scholes path, and this is
    what says so.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    instrument, cfg = _black_scholes_case(name)
    result = price(instrument, _BS_MODEL, _BS_MARKET, method="mc", cfg=cfg)
    assert math.isclose(result.value, value, rel_tol=1e-12)
    assert math.isclose(result.stderr, stderr, rel_tol=1e-12)


def test_the_heston_fields_do_not_reach_the_black_scholes_engines() -> None:
    """Evidence class: EXACT_IDENTITY.

    `heston_scheme` and `heston_conditional` are read by the Heston engines and
    by nothing else, so setting them to anything must leave a Black-Scholes
    price bit-for-bit alone.
    """
    instrument, cfg = _black_scholes_case("european")
    baseline = price(instrument, _BS_MODEL, _BS_MARKET, method="mc", cfg=cfg)
    noisy = price(
        instrument,
        _BS_MODEL,
        _BS_MARKET,
        method="mc",
        cfg=MCConfig(
            n_paths=20_000,
            n_steps=1,
            seed=123,
            heston_scheme=EULER_FULL_TRUNCATION,
            heston_conditional=True,
        ),
    )
    assert math.isclose(noisy.value, baseline.value, rel_tol=1e-12)
    assert math.isclose(noisy.stderr, baseline.stderr, rel_tol=1e-12)


# --------------------------------------------------------------------------
# (2) Registry wiring and the input contract.
# --------------------------------------------------------------------------

_MODEL = LEWIS.model()
_MARKET = LEWIS.market()
_CALL = EuropeanOption(kind="call", strike=100.0, expiry=1.0)


def _cfg(**changes) -> MCConfig:
    base = {
        "n_paths": 20_000,
        "n_steps": 16,
        "seed": 20240913,
        "variance_reduction": "antithetic",
        "heston_conditional": True,
    }
    base.update(changes)
    return MCConfig(**base)


def test_every_instrument_type_resolves_to_a_heston_mc_engine() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the registry)."""
    cases = (
        _CALL,
        DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0),
        AsianOption(
            kind="call",
            strike=100.0,
            expiry=1.0,
            fixing_times=uniform_fixing_times(1.0, 6),
            averaging="arithmetic",
        ),
        BarrierOption(
            kind="call",
            strike=100.0,
            expiry=1.0,
            barrier=80.0,
            barrier_type="down-and-out",
            monitoring=uniform_monitoring_times(1.0, 6),
        ),
    )
    for instrument in cases:
        conditional = isinstance(instrument, EuropeanOption | DigitalOption)
        result = price(
            instrument,
            _MODEL,
            _MARKET,
            method="mc",
            cfg=_cfg(n_paths=4_000, n_steps=8, heston_conditional=conditional),
        )
        assert result.meta["model"] == "Heston"
        assert result.meta["heston_scheme"] == QE
        assert result.value > 0.0
        assert result.stderr is not None and result.stderr > 0.0


def test_n_steps_one_is_refused_and_the_message_says_what_n_steps_means() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract).

    `MCConfig.n_steps` defaults to 1 because that is terminal sampling under
    Black-Scholes. Under Heston it is the discretisation, and none of the three
    schemes is exact in one step, so the default is refused rather than
    silently returning a one-step answer.
    """
    with pytest.raises(InvalidInputError, match="TIME DISCRETISATION"):
        price(_CALL, _MODEL, _MARKET, method="mc", cfg=MCConfig())


def test_stratified_sampling_is_refused_under_heston() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    with pytest.raises(NotSupportedError, match="2 \\* n_steps normals"):
        price(
            _CALL,
            _MODEL,
            _MARKET,
            method="mc",
            cfg=_cfg(variance_reduction="stratified", n_strata=16),
        )


def test_antithetic_is_refused_with_the_exact_variance_scheme() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    with pytest.raises(NotSupportedError, match="noncentral chi-square"):
        price(
            _CALL,
            _MODEL,
            _MARKET,
            method="mc",
            cfg=_cfg(
                variance_reduction="antithetic",
                heston_scheme="exact_variance_euler_log_spot",
            ),
        )


def test_the_likelihood_ratio_greek_is_refused_with_its_reason() -> None:
    """Evidence class: NEGATIVE_FINDING (a documented gap, not a bug).

    The slice statement left this open ("refused or supported with a stated
    reason"). It is refused, and the reason is that the *model's* transition
    density is the thing Broadie & Kaya can only reach by inverting the
    characteristic function of the integrated variance -- the cost QE exists to
    avoid. The QE step's conditional law does have a score, but it is the score
    of the discretisation, which is not what "unbiased likelihood ratio" claims.
    """
    with pytest.raises(NotSupportedError, match="Broadie & Kaya"):
        greeks(
            _CALL,
            _MODEL,
            _MARKET,
            method="mc",
            cfg=_cfg(greeks_estimator="likelihood_ratio"),
        )


@pytest.mark.parametrize(
    ("instrument", "match"),
    (
        (
            DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0),
            "zero almost everywhere",
        ),
        (
            AsianOption(
                kind="call",
                strike=100.0,
                expiry=1.0,
                fixing_times=uniform_fixing_times(1.0, 6),
                averaging="arithmetic",
            ),
            "path-dependent",
        ),
        (
            BarrierOption(
                kind="call",
                strike=100.0,
                expiry=1.0,
                barrier=80.0,
                barrier_type="down-and-out",
                monitoring=uniform_monitoring_times(1.0, 6),
            ),
            "path-dependent",
        ),
    ),
    ids=["digital", "asian", "barrier"],
)
def test_the_other_greeks_are_refused_with_a_reason(instrument, match) -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract).

    Registered-and-raising: an unregistered key would say "Unsupported
    instrument/model/market combination", which is false when the price engine
    beside it prices exactly that combination.
    """
    with pytest.raises(NotSupportedError, match=match):
        greeks(instrument, _MODEL, _MARKET, method="mc", cfg=_cfg())


def test_bgk_is_refused_under_heston_because_it_names_one_sigma() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    option = BarrierOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        barrier=80.0,
        barrier_type="down-and-out",
        monitoring=uniform_monitoring_times(1.0, 6),
    )
    with pytest.raises(NotSupportedError, match="single constant sigma"):
        price(
            option,
            _MODEL,
            _MARKET,
            method="mc",
            cfg=_cfg(barrier_correction="bgk", heston_conditional=False),
        )


# --------------------------------------------------------------------------
# (c) The bias study on the reference set.
# --------------------------------------------------------------------------

_BIAS_PATHS = 100_000
_BIAS_SEED = 20240913

_TRANSFORM = FourierConfig(method="lewis")
"""The reference. Slice 15's oracle conclusion: Lewis and Gil-Pelaez are the
only two transform methods with no parameter to get wrong, worst residual
2.4e-10 and 1.2e-11 against QuantLib's `AnalyticHestonEngine` over 24 cells
spanning both Feller regimes. COS needs `L = 28` on the Feller-violating set
and its *call* has no usable setting there at `T = 10`, so it is not used as a
reference anywhere in this file -- though at `T = 1` it agrees with Lewis to
1.2e-10 at `L = 28, N = 4096`, which is checked below."""


def _reference(model: HestonModel, market: Market, option: EuropeanOption) -> float:
    return price(option, model, market, method="fourier", cfg=_TRANSFORM).value


def _mc_bias(
    model: HestonModel,
    market: Market,
    option: EuropeanOption,
    *,
    scheme: str,
    n_steps: int,
    n_paths: int = _BIAS_PATHS,
) -> tuple[float, float]:
    """`(bias, stderr)` of the conditional antithetic estimator, no control.

    Deliberately **no** control variate: its mean is the model's forward, not
    the scheme's, so it would remove the part of the bias collinear with the
    scheme's own martingale defect. Measured shift on full-truncation Euler at
    `dt = 1/4`: -5.2e-03. That is a legitimate bias reduction and an
    illegitimate bias measurement.
    """
    result = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=_BIAS_SEED,
            variance_reduction="antithetic",
            heston_conditional=True,
            heston_scheme=scheme,
        ),
    )
    return result.value - _reference(model, market, option), float(result.stderr)


def test_the_cos_reference_agrees_with_lewis_on_the_feller_violating_set() -> None:
    """Evidence class: INDEPENDENT_ENGINE (two payoff transforms, one model).

    Slice 15 derived `HESTON_TRUNCATION_L_FELLER_VIOLATED = 28` from
    `sqrt(c4)/c2` and measured it. At `T = 1` it works: the COS call at
    `L = 28, N = 4096` agrees with a Lewis integral of the same transform to
    1.2e-10, which is what licenses using Lewis as *the* reference below
    without re-arguing it.
    """
    model = FELLER_VIOLATED.model()
    market = FELLER_VIOLATED.market()
    lewis = _reference(model, market, _CALL)
    cos = price(
        _CALL,
        model,
        market,
        method="fourier",
        cfg=FourierConfig(method="cos", truncation_l=28.0, n_terms=4096),
    ).value
    assert abs(cos - lewis) < 1e-08


@pytest.mark.parametrize("scheme", (QE, EULER_FULL_TRUNCATION))
def test_the_weak_bias_falls_with_the_step_size(scheme: str) -> None:
    """Evidence class: CONVERGENCE_ORDER, read qualitatively.

    The full ladder with fitted orders and constants is in
    `examples/heston_mc_qe.py`; measured there at 1,000,000 antithetic
    conditional paths on the reference set:

        dt      QE bias      Euler bias
        1/4     -1.017e-01   +2.601e-01
        1/8     -2.557e-02   +3.693e-02
        1/16    -8.40e-03    -5.9e-03
        1/32    -4.9e-03     -7.9e-03
        1/64    +4.2e-03     +5e-04

    with a standard error of about 3.2e-03 (QE) and 3.3e-03 to 5.8e-03
    (Euler), so only the two coarsest levels clear the noise for either scheme
    and a five-point fitted order is reported next to the count of resolved
    levels rather than on its own. This test asserts the part that is
    unambiguous: the coarse-step bias is many standard errors from zero and the
    next level is at least twice smaller.
    """
    coarse, coarse_stderr = _mc_bias(_MODEL, _MARKET, _CALL, scheme=scheme, n_steps=4)
    finer, _ = _mc_bias(_MODEL, _MARKET, _CALL, scheme=scheme, n_steps=8)
    assert abs(coarse) > 5.0 * coarse_stderr
    assert abs(finer) < 0.5 * abs(coarse)

    fine, fine_stderr = _mc_bias(_MODEL, _MARKET, _CALL, scheme=scheme, n_steps=32)
    assert abs(fine) < 6.0 * fine_stderr


def test_qe_beats_euler_at_dt_a_quarter_but_only_by_a_factor_of_two_and_a_half() -> None:
    """Evidence class: NEGATIVE_FINDING against the slice statement.

    The slice expected QE's bias to be "far smaller" than Euler's at coarse
    steps. On the reference set (Feller **satisfied**, number 4.0) the measured
    ratio at `dt = 1/4` is **2.6** -- better, and not by an order of magnitude.
    The dramatic separation is a Feller-regime effect, not a scheme effect in
    general, and it is measured in the next test.

    Note also what is *not* being compared: QE carries Andersen's martingale
    correction and full-truncation Euler carries no correction at all, because
    its drift is not built from a conditional moment generating function. Each
    scheme is measured as it is actually used.
    """
    qe_bias, _ = _mc_bias(_MODEL, _MARKET, _CALL, scheme=QE, n_steps=4)
    euler_bias, _ = _mc_bias(
        _MODEL, _MARKET, _CALL, scheme=EULER_FULL_TRUNCATION, n_steps=4
    )
    ratio = abs(euler_bias) / abs(qe_bias)
    assert 1.5 < ratio < 5.0
    # And the signs differ, which is the other half of the finding: the two
    # schemes do not err in the same direction, so a bias measured on one says
    # nothing about the other even qualitatively.
    assert qe_bias < 0.0 < euler_bias


# --------------------------------------------------------------------------
# (e) The Feller-violating set.
# --------------------------------------------------------------------------

_FELLER_MODEL = FELLER_VIOLATED.model()
_FELLER_MARKET = FELLER_VIOLATED.market()


def test_the_feller_violating_set_separates_the_two_schemes_by_two_orders() -> None:
    """Evidence class: NEGATIVE_FINDING for Euler, STATISTICAL for QE.

    `v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1, rho = -0.9`; Feller number
    **0.08**, so the origin is attainable and the variance spends most of its
    time near it. Measured at 500,000 conditional paths against the Lewis
    reference 3.591838:

        dt      QE bias      Euler bias   Euler negative-variance fraction
        1/4     -2.14e-02    +3.418       0.539
        1/8     -7.8e-03     +2.137       0.586
        1/16    -7.7e-03     +1.150       0.611
        1/32    -2.3e-03     +0.538       0.611

    Euler's error at `dt = 1/4` is **95% of the price**. That is the
    Lord-Koekkoek-van Dijk ordering at its most extreme: full truncation is the
    least-biased *Euler* fix and is still unusable here, because more than half
    of its variance draws are negative and the truncation that repairs them
    injects variance the model does not have.
    """
    qe_bias, qe_stderr = _mc_bias(
        _FELLER_MODEL, _FELLER_MARKET, _CALL, scheme=QE, n_steps=4
    )
    euler_bias, _ = _mc_bias(
        _FELLER_MODEL, _FELLER_MARKET, _CALL, scheme=EULER_FULL_TRUNCATION, n_steps=4
    )
    assert abs(qe_bias) < 0.06
    assert euler_bias > 3.0
    assert abs(euler_bias) / abs(qe_bias) > 50.0


def test_the_engine_reports_eulers_negative_variance_fraction() -> None:
    """Evidence class: NEGATIVE_FINDING, surfaced in the result's metadata.

    A caller who chooses `euler_full_truncation` on a Feller-violating set gets
    the frequency in `meta` rather than having to know to look for it.
    """
    result = price(
        _CALL,
        _FELLER_MODEL,
        _FELLER_MARKET,
        method="mc",
        cfg=MCConfig(
            n_paths=20_000,
            n_steps=8,
            seed=_BIAS_SEED,
            variance_reduction="antithetic",
            heston_conditional=True,
            heston_scheme=EULER_FULL_TRUNCATION,
        ),
    )
    assert 0.4 < float(result.meta["negative_variance_fraction"]) < 0.8
    qe_result = price(
        _CALL,
        _FELLER_MODEL,
        _FELLER_MARKET,
        method="mc",
        cfg=MCConfig(
            n_paths=20_000,
            n_steps=8,
            seed=_BIAS_SEED,
            variance_reduction="antithetic",
            heston_conditional=True,
        ),
    )
    assert "negative_variance_fraction" not in qe_result.meta


# --------------------------------------------------------------------------
# (f) rho = 0, which is where the slice statement was wrong.
# --------------------------------------------------------------------------


def test_the_uncorrelated_price_does_not_agree_within_noise_at_coarse_dt() -> None:
    """Evidence class: NEGATIVE_FINDING against the slice statement.

    Measured at 500,000 conditional antithetic paths with `rho = 0` and
    everything else at the reference set: bias **-0.2439** at `dt = 1/4`
    (stderr 1.4e-03), -0.0652 at 1/8 and -0.0154 at 1/16. The `rho = 0` bias is
    *larger* than the `rho = -0.5` bias (-0.1017), so "the price agrees at
    `rho = 0`" is not a test of the correlation wiring -- it is a test of the
    time discretisation with the correlation switched off, and at `rho = 0` the
    log-spot step reduces to the trapezoidal rule for the integrated variance,
    whose error at `kappa dt = 1` is exactly this size.

    The correlation claim is carried instead by
    `tests/test_mc_heston.py::test_the_increment_correlation_recovers_rho`,
    which measures `corr(dX, dv)` directly against `rho` with a sampling
    standard error of 2e-05.
    """
    model = HestonModel(v0=0.04, kappa=4.0, theta=0.25, xi=1.0, rho=0.0)
    coarse, coarse_stderr = _mc_bias(model, _MARKET, _CALL, scheme=QE, n_steps=4)
    assert coarse < 0.0
    assert abs(coarse) > 20.0 * coarse_stderr
    assert coarse == pytest.approx(-0.244, abs=0.03)

    fine, fine_stderr = _mc_bias(model, _MARKET, _CALL, scheme=QE, n_steps=32)
    assert abs(fine) < 8.0 * fine_stderr


# --------------------------------------------------------------------------
# Variance reduction: what composes and what does not.
# --------------------------------------------------------------------------


def _stderr(**changes) -> float:
    result = price(_CALL, _MODEL, _MARKET, method="mc", cfg=_cfg(**changes))
    return float(result.stderr)


def test_conditioning_is_worth_an_order_of_magnitude_in_variance() -> None:
    """Evidence class: STATISTICAL.

    Same estimator mean, measured factor **54x** in variance for QE on the
    reference set at 200k antithetic paths (19x to 48x for full-truncation
    Euler, rising with the step count). The two estimates must also agree,
    which is checked within the *plain* estimator's standard error since that
    is the only tolerance available.
    """
    plain = price(
        _CALL, _MODEL, _MARKET, method="mc", cfg=_cfg(heston_conditional=False)
    )
    conditional = price(
        _CALL, _MODEL, _MARKET, method="mc", cfg=_cfg(heston_conditional=True)
    )
    assert abs(conditional.value - plain.value) < 4.0 * float(plain.stderr)
    factor = (float(plain.stderr) / float(conditional.stderr)) ** 2
    assert factor > 20.0


def test_the_terminal_spot_control_and_conditioning_are_substitutes() -> None:
    """Evidence class: NEGATIVE_FINDING, and a genuinely useful one.

    The discounted terminal spot is a strong control for the **plain**
    estimator (measured correlation 0.79 to 0.89, predicted factor 2.7 to 5.0)
    and a nearly worthless one for the **conditional** estimator (0.34 to 0.49,
    factor 1.1 to 1.3). The reason is structural rather than numerical: what
    the control was correlated with is the spot diffusion, and conditioning has
    already integrated that out exactly. Stacking them buys almost nothing, and
    a reader who assumes variance reductions multiply would over-promise by an
    order of magnitude.
    """
    plain = price(
        _CALL,
        _MODEL,
        _MARKET,
        method="mc",
        cfg=_cfg(
            heston_conditional=False,
            variance_reduction=("antithetic", "control_variate"),
        ),
    )
    conditional = price(
        _CALL,
        _MODEL,
        _MARKET,
        method="mc",
        cfg=_cfg(
            heston_conditional=True,
            variance_reduction=("antithetic", "control_variate"),
        ),
    )
    assert float(plain.meta["control_correlation"]) > 0.7
    assert float(conditional.meta["control_correlation"]) < 0.6
    assert "control_variate_caveat" in conditional.meta


def test_antithetic_reduces_the_conditional_estimators_variance() -> None:
    """Evidence class: STATISTICAL.

    Reflecting both drivers is worth a measured **7.7x** in variance on the
    conditional European estimator at equal path count -- large because the
    conditional payoff is a smooth, nearly monotone function of the variance
    path, which is exactly the case antithetic sampling is built for.
    """
    plain = _stderr(variance_reduction="none")
    antithetic = _stderr(variance_reduction="antithetic")
    assert (plain / antithetic) ** 2 > 2.0


# --------------------------------------------------------------------------
# Determinism and metadata.
# --------------------------------------------------------------------------


def test_the_heston_engines_are_deterministic_at_a_fixed_seed() -> None:
    """Evidence class: EXACT_IDENTITY (determinism)."""
    first = price(_CALL, _MODEL, _MARKET, method="mc", cfg=_cfg())
    second = price(_CALL, _MODEL, _MARKET, method="mc", cfg=_cfg())
    assert math.isclose(first.value, second.value, rel_tol=1e-12)
    assert math.isclose(first.stderr, second.stderr, rel_tol=1e-12)


def test_the_result_metadata_names_the_scheme_and_the_feller_regime() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the metadata)."""
    result = price(_CALL, _MODEL, _MARKET, method="mc", cfg=_cfg())
    assert result.meta["heston_scheme"] == QE
    assert result.meta["martingale_correction"] is True
    assert result.meta["feller_satisfied"] is True
    assert result.meta["feller_number"] == pytest.approx(4.0)
    assert result.meta["n_normal_draws"] == 2 * (20_000 // 2) * 16

    violating = price(
        _CALL, _FELLER_MODEL, _FELLER_MARKET, method="mc", cfg=_cfg(n_paths=2_000)
    )
    assert violating.meta["feller_satisfied"] is False
    assert violating.meta["feller_number"] == pytest.approx(0.08)


def test_the_european_greeks_agree_with_the_transform_greeks() -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    A common-random-numbers bump on a simulated Heston price against closed
    forms of the cosine expansion. Measured at 20,000 antithetic paths and
    `n_steps = 16`: delta 0.5997 against 0.5995, gamma 9.288e-03 against
    9.288e-03, vega 4.079 against 4.072, theta -9.409 against -9.433, rho
    43.90 against 43.88. Vega is `dV/d sqrt(v0)` on both sides, which is the
    thing that has to be arranged for the comparison to mean anything.
    """
    mc = greeks(_CALL, _MODEL, _MARKET, method="mc", cfg=_cfg())
    transform = greeks(
        _CALL, _MODEL, _MARKET, method="fourier", cfg=FourierConfig(method="cos")
    )
    assert mc.meta["vega_convention"].startswith("d/d sqrt(v0)")
    assert mc.delta == pytest.approx(transform.delta, abs=0.01)
    assert mc.gamma == pytest.approx(transform.gamma, abs=5e-04)
    assert mc.vega == pytest.approx(transform.vega, abs=0.1)
    assert mc.theta == pytest.approx(transform.theta, abs=0.3)
    assert mc.rho == pytest.approx(transform.rho, abs=0.5)


def test_the_pathwise_delta_is_available_and_the_rest_falls_back_to_a_bump() -> None:
    """Evidence class: STATISTICAL.

    `S_T` is homogeneous of degree one in `S_0` under Heston -- the variance
    dynamics never see the spot -- so `dS_T/dS_0 = S_T/S_0` exactly and the
    pathwise delta of a call costs one multiplication. The other four Greeks
    still come from the bump and `meta["estimator"]` says so per Greek, the
    same mixed-result convention the Asian engine uses.
    """
    result = greeks(
        _CALL,
        _MODEL,
        _MARKET,
        method="mc",
        cfg=_cfg(greeks_estimator="pathwise", heston_conditional=False),
    )
    assert result.meta["estimator"]["delta"] == "pathwise"
    assert result.meta["estimator"]["gamma"] == "bump"
    transform = greeks(
        _CALL, _MODEL, _MARKET, method="fourier", cfg=FourierConfig(method="cos")
    )
    stderr = float(result.meta["stderr"]["delta"])
    assert abs(result.delta - transform.delta) < 4.0 * stderr + 0.01


def test_the_digital_price_agrees_with_the_transform() -> None:
    """Evidence class: STATISTICAL.

    The digital is the payoff conditioning helps most: a Bernoulli variable
    becomes a smooth `Phi(d2)`. Compared against Gil-Pelaez, which prices the
    terminal law directly (the Lewis transform is a call transform and has no
    cash-or-nothing payoff, which is a Slice 14 fact this test inherits).
    """
    option = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0)
    mc = price(option, _MODEL, _MARKET, method="mc", cfg=_cfg(n_steps=32))
    reference = price(
        option,
        _MODEL,
        _MARKET,
        method="fourier",
        cfg=FourierConfig(method="gil_pelaez"),
    ).value
    assert abs(mc.value - reference) < 5.0 * float(mc.stderr) + 0.005


def test_the_asian_reports_that_the_geometric_control_does_not_exist() -> None:
    """Evidence class: NEGATIVE_FINDING, surfaced in the metadata.

    Kemna-Vorst needs the geometric average to be lognormal, which it is under
    Black-Scholes (Slice 8, measured correlation 0.9996 and a factor of 1277)
    and is not under Heston -- it is lognormal only conditional on the variance
    path, so there is no closed form to supply the control's exact mean. The
    fallback is the discounted terminal spot, and the result says which one is
    in play and why the good one is not.
    """
    option = AsianOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        fixing_times=uniform_fixing_times(1.0, 12),
        averaging="arithmetic",
    )
    result = price(
        option,
        _MODEL,
        _MARKET,
        method="mc",
        cfg=_cfg(
            n_steps=24,
            heston_conditional=False,
            variance_reduction=("antithetic", "control_variate"),
        ),
    )
    assert result.meta["control_variate"] == "discounted_terminal_spot"
    assert "lognormal only" in result.meta["geometric_control_unavailable"]
    assert 0.0 < float(result.meta["control_correlation"]) < 1.0


def test_the_conditional_estimator_is_refused_for_path_payoffs() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract)."""
    asian = AsianOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        fixing_times=uniform_fixing_times(1.0, 6),
        averaging="arithmetic",
    )
    with pytest.raises(NotSupportedError, match="TERMINAL payoff"):
        price(asian, _MODEL, _MARKET, method="mc", cfg=_cfg(heston_conditional=True))


def test_the_contract_dates_land_exactly_on_the_simulation_grid() -> None:
    """Evidence class: EXACT_IDENTITY.

    `heston_time_grid` unions the uniform grid with the contract's own dates,
    so a fixing schedule that does not divide the step count is still read
    where the contract says it is -- and the grid is then strictly finer than
    `n_steps` asks, which the metadata reports.
    """
    from qpl.engines.mc.heston_pricers import heston_time_grid

    option = AsianOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        fixing_times=uniform_fixing_times(1.0, 7),
        averaging="arithmetic",
    )
    grid = heston_time_grid(1.0, 8, dates=option.fixing_times)
    for date in option.fixing_times:
        assert np.min(np.abs(grid - date)) < 1e-14
    assert grid.size - 1 > 8
    result = price(
        option,
        _MODEL,
        _MARKET,
        method="mc",
        cfg=_cfg(n_paths=2_000, n_steps=8, heston_conditional=False),
    )
    assert result.meta["n_grid"] > result.meta["n_steps"]
