"""`qpl.cases.heston_calibration` checked against the calibrator.

Forty-nine rows in six families. The heavy studies (the single-maturity start
sweep, the noise sweep, the objective sweep and the start grid) are each run
**once** in a module-scoped fixture and read by every row that quotes them, so
the file costs one pass over each experiment rather than one per row.

What is deliberately *not* here: a second copy of the assertions in
`tests/test_heston_calibration.py`. That file measures the behaviour and
derives its own bounds; this one checks that the published rows agree with the
behaviour and that the rows themselves are well-formed -- unique ids, an
evidence class on every claim, a citation on every expectation, a positive
tolerance wherever a tolerance means anything, and disjointness from the other
seven id spaces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.calibration import (
    HESTON_PARAMETERS,
    calibrate_heston,
    heston_quote_values,
    parameter_covariance,
    residual_jacobian,
    vega_weights,
)
from qpl.cases import (
    ALL_AMERICAN_CASES,
    ALL_ASIAN_CASES,
    ALL_BARRIER_CASES,
    ALL_CASES,
    ALL_DIGITAL_CASES,
    ALL_HESTON_CALIBRATION_CASES,
    ALL_HESTON_CASES,
    ALL_SDE_CASES,
    HESTON_CALIBRATION_CONDITION_CASES,
    HESTON_CALIBRATION_GLOBAL_COUNT,
    HESTON_CALIBRATION_GLOBAL_TOLERANCE,
    HESTON_CALIBRATION_INITIALISATION_CASES,
    HESTON_CALIBRATION_MARKET,
    HESTON_CALIBRATION_MULTISTART_COUNT,
    HESTON_CALIBRATION_NOISE_CASES,
    HESTON_CALIBRATION_NOISE_SEEDS,
    HESTON_CALIBRATION_OBJECTIVE_CASES,
    HESTON_CALIBRATION_OBJECTIVE_SEEDS,
    HESTON_CALIBRATION_RECOVERY_CASES,
    HESTON_CALIBRATION_REFERENCE,
    HESTON_CALIBRATION_SAFETY_FACTOR,
    HESTON_CALIBRATION_SINGLE_MATURITY_CASES,
    HESTON_CALIBRATION_SINGLE_MATURITY_STARTS,
    HESTON_CALIBRATION_START_GRID,
    noisy_quotes,
)
from qpl.cases.heston_calibration import _INITIALISATION_SPEC, _SINGLE_MATURITY_SPEC
from qpl.engines.analytic.black_scholes import implied_volatility
from qpl.instruments.options import EuropeanOption
from qpl.models.black_scholes import bs_price
from qpl.models.heston import HestonModel
from qpl.validation import BenchmarkRow, EvidenceClass

MARKET = HESTON_CALIBRATION_MARKET
TRUE = np.array(HESTON_CALIBRATION_REFERENCE)


def _ids(cases):
    return [case.row.id for case in cases]


# --------------------------------------------------------------------------
# Metadata contract.
# --------------------------------------------------------------------------


def test_every_row_is_well_formed() -> None:
    """Ids unique, evidence class present, citation present, tolerance sane."""
    seen = set()
    for case in ALL_HESTON_CALIBRATION_CASES:
        row = case.row
        assert isinstance(row, BenchmarkRow)
        assert row.id not in seen, row.id
        seen.add(row.id)
        assert row.id.startswith("heston_cal_")
        assert row.description
        assert isinstance(row.evidence, EvidenceClass)
        assert row.source
        assert row.notes
        assert math.isfinite(row.expected)
        assert row.tolerance >= 0.0
        if row.tolerance == 0.0:
            # A zero tolerance is only defensible for an integer-valued claim.
            assert row.expected == float(int(row.expected)), row.id
        assert case.parameter in (None, *HESTON_PARAMETERS)


def test_the_family_lists_partition_the_whole() -> None:
    families = (
        HESTON_CALIBRATION_RECOVERY_CASES,
        HESTON_CALIBRATION_CONDITION_CASES,
        HESTON_CALIBRATION_SINGLE_MATURITY_CASES,
        HESTON_CALIBRATION_NOISE_CASES,
        HESTON_CALIBRATION_OBJECTIVE_CASES,
        HESTON_CALIBRATION_INITIALISATION_CASES,
    )
    assert sum(len(f) for f in families) == len(ALL_HESTON_CALIBRATION_CASES)
    assert _ids(ALL_HESTON_CALIBRATION_CASES) == [
        case.row.id for family in families for case in family
    ]


def test_the_eighth_id_space_is_disjoint_from_the_other_seven() -> None:
    """Eight id spaces now, asserted pairwise disjoint.

    A duplicate id across spaces makes two different claims share one pytest
    parameter id, which is how a failing row silently becomes someone else's
    failing row.
    """
    spaces = {
        "european": {c.row.id for c in ALL_CASES},
        "american": {c.row.id for c in ALL_AMERICAN_CASES},
        "digital": {c.row.id for c in ALL_DIGITAL_CASES},
        "asian": {c.row.id for c in ALL_ASIAN_CASES},
        "sde": {c.row.id for c in ALL_SDE_CASES},
        "barrier": {c.row.id for c in ALL_BARRIER_CASES},
        "heston": {c.row.id for c in ALL_HESTON_CASES},
        "heston_calibration": {c.row.id for c in ALL_HESTON_CALIBRATION_CASES},
    }
    names = sorted(spaces)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert spaces[left].isdisjoint(spaces[right]), (left, right)


def test_no_recovery_row_claims_to_be_evidence_of_identifiability() -> None:
    """The one thing this case family must never be read as saying.

    Every clean-recovery row is `CLOSED_FORM` (a statement about the solver
    reaching the minimum of a noise-free problem) and none is `STATISTICAL`.
    The statistical claims live in the noise family, and the identifiability
    claims live in the conditioning and single-maturity families, which are
    `CLOSED_FORM` and `NEGATIVE_FINDING`.
    """
    for case in HESTON_CALIBRATION_RECOVERY_CASES:
        assert case.row.evidence is EvidenceClass.CLOSED_FORM
        assert "NOT evidence of identifiability" in case.row.notes
    assert any(
        case.row.evidence is EvidenceClass.NEGATIVE_FINDING
        for case in HESTON_CALIBRATION_SINGLE_MATURITY_CASES
    )
    assert all(
        case.row.evidence is EvidenceClass.STATISTICAL
        for case in HESTON_CALIBRATION_NOISE_CASES
    )


# --------------------------------------------------------------------------
# (1) Clean recovery.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recovery_fits():
    """One calibration per distinct recovery spec, keyed by `id(spec)`."""
    fits = {}
    for case in HESTON_CALIBRATION_RECOVERY_CASES:
        key = (case.spec.parameters, case.spec.objective)
        if key not in fits:
            fits[key] = calibrate_heston(
                case.spec.quotes(),
                MARKET,
                initial=case.spec.start,
                objective=case.spec.objective,
            )
    return fits


@pytest.mark.parametrize(
    "case", HESTON_CALIBRATION_RECOVERY_CASES, ids=_ids(HESTON_CALIBRATION_RECOVERY_CASES)
)
def test_clean_recovery_rows(case, recovery_fits) -> None:
    """Evidence class: CLOSED_FORM, tolerance from the fit's own Jacobian."""
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    fit = recovery_fits[(case.spec.parameters, case.spec.objective)]
    assert fit.success
    index = HESTON_PARAMETERS.index(case.parameter)
    error = abs(fit.parameters[index] - case.spec.parameters[index])
    budget = (
        HESTON_CALIBRATION_SAFETY_FACTOR
        * float(np.linalg.norm(fit.residuals))
        / float(fit.singular_values[-1])
    )
    assert error == pytest.approx(case.row.expected, abs=budget)
    assert budget < 1e-03


def test_the_feller_violating_fit_is_flagged_and_not_forbidden(recovery_fits) -> None:
    fit = recovery_fits[((0.04, 0.5, 0.04, 1.0, -0.9), "implied_vol")]
    assert not fit.feller_satisfied
    assert fit.feller_number == pytest.approx(0.08, abs=1e-03)
    assert fit.model is not None


# --------------------------------------------------------------------------
# (2) Conditioning.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    HESTON_CALIBRATION_CONDITION_CASES,
    ids=_ids(HESTON_CALIBRATION_CONDITION_CASES),
)
def test_condition_number_rows(case) -> None:
    """Evidence class: CLOSED_FORM -- linear algebra at the true parameters."""
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    residuals, jacobian = residual_jacobian(
        case.spec.quotes(),
        MARKET,
        case.spec.parameters,
        objective=case.spec.objective,
    )
    _, condition, _, _ = parameter_covariance(jacobian, residuals)
    assert condition == pytest.approx(case.row.expected, abs=case.row.tolerance)


def test_the_published_condition_numbers_carry_the_ordering_finding() -> None:
    """The implied-volatility column falls; the price column does not.

    Asserted on the published `expected` values themselves, so that the rows
    and the finding cannot drift apart: a future edit that "fixed" the price
    column to be monotone would fail here as well as in the behaviour test.
    """
    by_key = {
        (case.spec.objective, len(case.spec.maturities)): case.row.expected
        for case in HESTON_CALIBRATION_CONDITION_CASES
    }
    implied = [by_key[("implied_vol", n)] for n in (1, 3, 6)]
    priced = [by_key[("price", n)] for n in (1, 3, 6)]
    assert implied[0] > implied[1] > implied[2]
    assert priced[0] > priced[1]
    assert priced[2] > priced[1]


# --------------------------------------------------------------------------
# (3) The single-maturity degeneracy.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_maturity_fits():
    quotes = _SINGLE_MATURITY_SPEC.quotes()
    return [
        calibrate_heston(
            quotes,
            MARKET,
            initial=start,
            objective=_SINGLE_MATURITY_SPEC.objective,
        )
        for start in HESTON_CALIBRATION_SINGLE_MATURITY_STARTS
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    HESTON_CALIBRATION_SINGLE_MATURITY_CASES,
    ids=_ids(HESTON_CALIBRATION_SINGLE_MATURITY_CASES),
)
def test_single_maturity_rows(case, single_maturity_fits) -> None:
    """NEGATIVE_FINDING for three of the four rows; CLOSED_FORM for `rho`."""
    fits = single_maturity_fits
    if case.row.id.endswith("smile_is_recovered"):
        measured = max(fit.rmse for fit in fits)
    elif case.parameter in ("kappa", "xi"):
        values = np.array([fit.parameter(case.parameter) for fit in fits])
        measured = float(values.max() / values.min())
    else:
        index = HESTON_PARAMETERS.index(case.parameter)
        measured = float(
            np.max(
                np.abs(
                    np.array([fit.parameter(case.parameter) for fit in fits])
                    - TRUE[index]
                )
            )
        )
    assert measured == pytest.approx(case.row.expected, abs=case.row.tolerance)


@pytest.mark.slow
def test_every_single_maturity_fit_is_at_the_same_objective(
    single_maturity_fits,
) -> None:
    """The degeneracy in one line: six answers, one objective value.

    Measured spread 1.0e-12 to 2.1e-11 -- one decimal order, all of it at the
    numerical floor -- while `kappa` spans a factor of 2.5. No single result
    among the six carries any sign of this.
    """
    objectives = np.array([fit.objective for fit in single_maturity_fits])
    assert objectives.max() < 1e-10
    assert objectives.max() / objectives.min() < 1e02


# --------------------------------------------------------------------------
# (4) Noise and the Jacobian error bars.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def noise_study():
    """`{noise_bp: (empirical rms, mean predicted stderr)}` over 20 draws."""
    out = {}
    for case in HESTON_CALIBRATION_NOISE_CASES:
        noise_bp = case.spec.noise_bp
        if noise_bp in out:
            continue
        errors = []
        predicted = []
        for seed in range(HESTON_CALIBRATION_NOISE_SEEDS):
            quotes = noisy_quotes(
                HestonModel(*case.spec.parameters),
                case.spec.maturities,
                case.spec.strikes,
                noise_bp=noise_bp,
                seed=1000 + seed,
            )
            fit = calibrate_heston(
                quotes,
                MARKET,
                initial=case.spec.start,
                objective=case.spec.objective,
            )
            errors.append(np.array(fit.parameters) - np.array(case.spec.parameters))
            predicted.append(fit.standard_errors)
        out[noise_bp] = (
            np.sqrt((np.array(errors) ** 2).mean(axis=0)),
            np.array(predicted).mean(axis=0),
        )
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "case", HESTON_CALIBRATION_NOISE_CASES, ids=_ids(HESTON_CALIBRATION_NOISE_CASES)
)
def test_standard_error_rows(case, noise_study) -> None:
    """Evidence class: STATISTICAL -- the honest identifiability statement."""
    assert case.row.evidence is EvidenceClass.STATISTICAL
    empirical, predicted = noise_study[case.spec.noise_bp]
    index = HESTON_PARAMETERS.index(case.parameter)
    ratio = float(empirical[index] / predicted[index])
    assert ratio == pytest.approx(case.row.expected, abs=case.row.tolerance)


@pytest.mark.slow
def test_the_predicted_error_bars_are_consistently_on_the_small_side(
    noise_study,
) -> None:
    """A directional claim, not just a band.

    Gauss-Newton drops the second-order term of the Hessian, so its covariance
    understates the spread of a nonlinear fit. Measured: the empirical spread
    exceeds the predicted standard error for four of the five parameters at
    both noise levels, and `theta` -- the parameter the term structure pins
    best -- is the one that sits at 1.00.
    """
    for noise_bp in noise_study:
        empirical, predicted = noise_study[noise_bp]
        ratios = empirical / predicted
        assert int(np.sum(ratios > 1.0)) >= 4
        assert float(np.max(ratios)) < 1.4


# --------------------------------------------------------------------------
# (5) The objective choice.
# --------------------------------------------------------------------------


def _price_quotes(volatility_quotes):
    from qpl.calibration import OptionQuote

    return [
        OptionQuote(
            strike=q.strike,
            expiry=q.expiry,
            kind="call",
            value=float(
                bs_price(
                    S=MARKET.spot,
                    K=q.strike,
                    T=q.expiry,
                    r=MARKET.rate(q.expiry),
                    sigma=q.value,
                    q=MARKET.dividend_yield(q.expiry),
                    kind="call",
                )
            ),
            value_type="price",
        )
        for q in volatility_quotes
    ]


def _rmse_pair(parameters, volatility_quotes) -> tuple[float, float]:
    model = HestonModel(*parameters)
    prices = heston_quote_values(model, MARKET, volatility_quotes)
    vol_sq = 0.0
    price_sq = 0.0
    for quote, price in zip(volatility_quotes, prices, strict=True):
        option = EuropeanOption(kind="call", strike=quote.strike, expiry=quote.expiry)
        vol_sq += (implied_volatility(float(price), option, MARKET) - quote.value) ** 2
        target = float(
            bs_price(
                S=MARKET.spot,
                K=quote.strike,
                T=quote.expiry,
                r=MARKET.rate(quote.expiry),
                sigma=quote.value,
                q=MARKET.dividend_yield(quote.expiry),
                kind="call",
            )
        )
        price_sq += (float(price) - target) ** 2
    n = len(volatility_quotes)
    return math.sqrt(vol_sq / n), math.sqrt(price_sq / n)


@pytest.fixture(scope="module")
def objective_study():
    """`{name: (iv rmse, price rmse, mean |parameter error|)}` over 10 draws."""
    spec = HESTON_CALIBRATION_OBJECTIVE_CASES[0].spec
    names = ("price", "vega_price", "implied_vol")
    rmse = {name: np.zeros(2) for name in names}
    errors = {name: np.zeros(5) for name in names}
    for seed in range(HESTON_CALIBRATION_OBJECTIVE_SEEDS):
        volatility_quotes = noisy_quotes(
            HestonModel(*spec.parameters),
            spec.maturities,
            spec.strikes,
            noise_bp=spec.noise_bp,
            seed=3000 + seed,
        )
        price_quotes = _price_quotes(volatility_quotes)
        fits = {
            "price": calibrate_heston(
                price_quotes, MARKET, initial=spec.start, objective="price"
            ),
            "vega_price": calibrate_heston(
                price_quotes,
                MARKET,
                initial=spec.start,
                objective="price",
                weights=vega_weights(price_quotes, MARKET),
            ),
            "implied_vol": calibrate_heston(
                volatility_quotes, MARKET, initial=spec.start, objective="implied_vol"
            ),
        }
        for name, fit in fits.items():
            rmse[name] += np.array(_rmse_pair(fit.parameters, volatility_quotes))
            errors[name] += np.abs(np.array(fit.parameters) - TRUE)
    return {
        name: (
            rmse[name][0] / HESTON_CALIBRATION_OBJECTIVE_SEEDS,
            rmse[name][1] / HESTON_CALIBRATION_OBJECTIVE_SEEDS,
            errors[name] / HESTON_CALIBRATION_OBJECTIVE_SEEDS,
        )
        for name in names
    }


@pytest.mark.parametrize(
    "case",
    [c for c in HESTON_CALIBRATION_OBJECTIVE_CASES if "_rmse" in c.row.id],
    ids=[c.row.id for c in HESTON_CALIBRATION_OBJECTIVE_CASES if "_rmse" in c.row.id],
)
def test_objective_rmse_rows(case, objective_study) -> None:
    """Evidence class: STATISTICAL."""
    assert case.row.evidence is EvidenceClass.STATISTICAL
    stem = case.row.id.removeprefix("heston_cal_objective_").removesuffix("_rmse")
    if stem.endswith("_implied_vol"):
        name, metric = stem.removesuffix("_implied_vol"), 0
    else:
        name, metric = stem.removesuffix("_price"), 1
    measured = objective_study[name][metric]
    assert measured == pytest.approx(case.row.expected, abs=case.row.tolerance)


def test_the_objective_does_not_separate_the_parameters(objective_study) -> None:
    """NEGATIVE_FINDING, and the row that carries it."""
    case = next(
        c
        for c in HESTON_CALIBRATION_OBJECTIVE_CASES
        if c.row.id.endswith("does_not_separate_the_parameters")
    )
    assert case.row.evidence is EvidenceClass.NEGATIVE_FINDING
    ratios = objective_study["price"][2] / objective_study["implied_vol"][2]
    for ratio in ratios:
        assert ratio == pytest.approx(case.row.expected, abs=case.row.tolerance)
    # Straddling 1 is the claim: neither objective is uniformly better.
    assert float(np.max(ratios)) > 1.0
    assert float(np.min(ratios)) < 1.0
    # ... while vega weighting and implied volatility give the same answer.
    vega = objective_study["vega_price"][2] / objective_study["implied_vol"][2]
    assert np.all(np.abs(vega - 1.0) < 0.02)


# --------------------------------------------------------------------------
# (6) Initialisation.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def start_grid_fits():
    quotes = _INITIALISATION_SPEC.quotes()
    return quotes, [
        calibrate_heston(
            quotes, MARKET, initial=start, objective=_INITIALISATION_SPEC.objective
        )
        for start in HESTON_CALIBRATION_START_GRID
    ]


@pytest.mark.slow
def test_the_global_basin_count_row(start_grid_fits) -> None:
    """NEGATIVE_FINDING: 11 of 14, pinned with a zero tolerance."""
    case = HESTON_CALIBRATION_INITIALISATION_CASES[0]
    assert case.row.evidence is EvidenceClass.NEGATIVE_FINDING
    _, fits = start_grid_fits
    best = min(fit.objective for fit in fits)
    reached = sum(
        1
        for fit in fits
        if fit.objective <= best * (1.0 + HESTON_CALIBRATION_GLOBAL_TOLERANCE)
    )
    assert float(reached) == case.row.expected
    assert reached == HESTON_CALIBRATION_GLOBAL_COUNT


@pytest.mark.slow
def test_the_multistart_row(start_grid_fits) -> None:
    quotes, fits = start_grid_fits
    case = HESTON_CALIBRATION_INITIALISATION_CASES[1]
    best = min(fit.objective for fit in fits)
    dead = (0.50, 0.1, 0.80, 3.0, -0.99)
    multi = calibrate_heston(
        quotes,
        MARKET,
        initial=dead,
        objective=_INITIALISATION_SPEC.objective,
        n_starts=HESTON_CALIBRATION_MULTISTART_COUNT,
    )
    gap = abs(multi.objective - best) / best
    assert gap == pytest.approx(case.row.expected, abs=case.row.tolerance)
    assert len(multi.starts) == HESTON_CALIBRATION_MULTISTART_COUNT
