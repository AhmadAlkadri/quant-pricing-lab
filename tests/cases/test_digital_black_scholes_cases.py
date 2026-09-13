"""Evaluate the cash-or-nothing digital benchmark rows.

Every assertion reads its expected value, tolerance, evidence class and
citation from `qpl.cases.digital_black_scholes`; nothing numeric is hardcoded
here except the per-engine budgets in the cross-engine test, and each of those
is a named constant from the cases layer with its derivation in the constant's
docstring.

The detailed studies live in `tests/test_digital_analytic.py`,
`tests/test_digital_tree_convergence.py`, `tests/test_digital_pde.py` and
`tests/test_digital_mc.py`. This file exists so that the cases layer carries
the claims as *data* and so that the four engines are checked against one
another in one place.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.cases import (
    ALL_AMERICAN_CASES,
    ALL_CASES,
    ALL_DIGITAL_CASES,
    DIGITAL_CROSS_ENGINE_CASES,
    DIGITAL_GREEKS_MC_PATHS,
    DIGITAL_GREEKS_MC_STDERR_MULTIPLE,
    DIGITAL_IDENTITY_CASES,
    DIGITAL_KNOWN_VALUE_CASES,
    DIGITAL_MC_GREEKS_CASES,
    DIGITAL_MC_PATHS,
    DIGITAL_MC_SEED,
    DIGITAL_MC_STDERR_MULTIPLE,
    DIGITAL_ODD_LEVELS,
    DIGITAL_PDE_LEVELS,
    DIGITAL_PDE_N,
    DIGITAL_PDE_ORDER_CASES,
    DIGITAL_PDE_PAYOFF_PROJECTION,
    DIGITAL_PDE_STRIKE_ALIGNMENT,
    DIGITAL_PDE_TIME_STEPPING,
    DIGITAL_PDE_TOLERANCE,
    DIGITAL_STRIKE_BUMP,
    DIGITAL_STRIKE_DERIVATIVE_CASES,
    DIGITAL_TREE_LR_N_STEPS,
    DIGITAL_TREE_LR_TOLERANCE,
    DIGITAL_TREE_ORDER_CASES,
    DigitalBSCase,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.exceptions import NotSupportedError
from qpl.models.black_scholes import bs_price
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order


def _ids(cases: tuple[DigitalBSCase, ...]) -> list[str]:
    return [case.row.id for case in cases]


def _analytic(spec) -> float:
    return price(spec.option(), spec.model(), spec.market()).value


def _pde_cfg(n: int) -> PDEConfig:
    """The one remedied grid every PDE row uses."""
    return PDEConfig(
        n_s=n,
        n_t=n,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=DIGITAL_PDE_STRIKE_ALIGNMENT,  # type: ignore[arg-type]
        time_stepping=DIGITAL_PDE_TIME_STEPPING,  # type: ignore[arg-type]
        payoff_projection=DIGITAL_PDE_PAYOFF_PROJECTION,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Identities and closed forms.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", DIGITAL_IDENTITY_CASES, ids=_ids(DIGITAL_IDENTITY_CASES))
def test_static_replication_rows(case: DigitalBSCase) -> None:
    """A digital call plus a digital put is a zero-coupon bond."""
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY

    spec = case.spec
    flipped = spec.flipped()
    residual = _analytic(spec) + _analytic(flipped) - spec.riskless()

    assert abs(residual - case.row.expected) <= case.row.tolerance, case.row.source
    # Not vacuous: both legs carry real value at these points.
    assert 0.0 < _analytic(spec) < spec.riskless()


@pytest.mark.parametrize(
    "case", DIGITAL_STRIKE_DERIVATIVE_CASES, ids=_ids(DIGITAL_STRIKE_DERIVATIVE_CASES)
)
def test_strike_derivative_rows(case: DigitalBSCase) -> None:
    """The digital is minus the strike-derivative of the vanilla."""
    assert case.row.evidence is EvidenceClass.CLOSED_FORM

    spec = case.spec
    h = DIGITAL_STRIKE_BUMP
    kwargs = {
        "S": spec.spot,
        "T": spec.expiry,
        "r": spec.rate,
        "sigma": spec.sigma,
        "q": spec.dividend,
        "kind": spec.kind,
    }
    slope = (
        float(bs_price(K=spec.strike + h, **kwargs))
        - float(bs_price(K=spec.strike - h, **kwargs))
    ) / (2.0 * h)
    expected = spec.cash * (-slope if spec.kind == "call" else slope)

    residual = _analytic(spec) - expected
    assert abs(residual - case.row.expected) <= case.row.tolerance, case.row.source


@pytest.mark.parametrize(
    "case", DIGITAL_KNOWN_VALUE_CASES, ids=_ids(DIGITAL_KNOWN_VALUE_CASES)
)
def test_known_value_rows(case: DigitalBSCase) -> None:
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    assert "not a published table" in case.row.source
    assert _analytic(case.spec) == pytest.approx(
        case.row.expected, abs=case.row.tolerance
    )


# --------------------------------------------------------------------------
# Measured rates.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", DIGITAL_TREE_ORDER_CASES, ids=_ids(DIGITAL_TREE_ORDER_CASES))
def test_tree_order_rows(case: DigitalBSCase) -> None:
    """Evaluate the tree convergence rows.

    Two shapes share this test. The Leisen-Reimer rows are CONVERGENCE_ORDER
    and are fitted at the six standard odd levels directly. The CRR row is a
    NEGATIVE_FINDING about a *block-RMS* rate, because the per-`n` error is a
    sawtooth and a fit through six single points measures nothing (it returns
    0.3569 with residual 0.5064); the block-RMS windows are the honest
    statement, and the per-`n` non-power-law is asserted alongside it.
    """
    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()
    exact = _analytic(spec)

    def tree(n: int, scheme: str) -> float:
        return price(
            option,
            model,
            market,
            method="tree",
            cfg=TreeConfig(n_steps=n, scheme=scheme),  # type: ignore[arg-type]
        ).value

    if case.row.evidence is EvidenceClass.CONVERGENCE_ORDER:
        errors = [tree(n, "leisen-reimer") - exact for n in DIGITAL_ODD_LEVELS]
        fit = fit_convergence_order(
            [1.0 / n for n in DIGITAL_ODD_LEVELS], [abs(e) for e in errors]
        )
        assert abs(fit.order - case.row.expected) <= case.row.tolerance, case.row.source
        assert fit.residual < 0.05, fit.residual
        assert all(e > 0.0 for e in errors) or all(e < 0.0 for e in errors), errors
        assert all(abs(a) > abs(b) for a, b in pairwise(errors)), errors
        return

    assert case.row.evidence is EvidenceClass.NEGATIVE_FINDING
    windows = ((25, 50), (51, 100), (101, 200), (201, 400), (401, 801))
    odd = [n for n in range(25, 802, 2)]
    signed = np.array([tree(n, "crr") - exact for n in odd])

    rms, centres = [], []
    for low, high in windows:
        chosen = [i for i, n in enumerate(odd) if low <= n <= high]
        rms.append(float(np.sqrt(np.mean(signed[chosen] ** 2))))
        centres.append(float(np.exp(np.mean(np.log([odd[i] for i in chosen])))))

    fit = fit_convergence_order([1.0 / c for c in centres], rms)
    assert abs(fit.order - case.row.expected) <= case.row.tolerance, case.row.source
    # The sawtooth the row's notes describe.
    assert int(np.sum(np.diff(np.sign(signed)) != 0)) >= 5, signed
    assert np.max(np.abs(signed)) > 1_000.0 * np.min(np.abs(signed))


@pytest.mark.parametrize("case", DIGITAL_PDE_ORDER_CASES, ids=_ids(DIGITAL_PDE_ORDER_CASES))
def test_pde_order_rows(case: DigitalBSCase) -> None:
    """Evaluate the PDE pathology and remedy rows."""
    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()

    remedied = case.row.evidence is EvidenceClass.CONVERGENCE_ORDER
    quantity = "delta" if case.row.id.endswith("delta_order_two") else "price"
    exact = (
        getattr(greeks(option, model, market), quantity)
        if quantity != "price"
        else _analytic(spec)
    )

    def value(n: int) -> float:
        cfg = (
            _pde_cfg(n)
            if remedied
            else PDEConfig(
                n_s=n,
                n_t=n,
                theta=0.5,
                s_max_multiplier=4.0,
                strike_alignment="none",
                time_stepping="theta",
                payoff_projection="none",
            )
        )
        if quantity == "price":
            return price(option, model, market, method="pde", cfg=cfg).value
        return getattr(greeks(option, model, market, method="pde", cfg=cfg), quantity)

    errors = [value(n) - exact for n in DIGITAL_PDE_LEVELS]
    fit = fit_convergence_order(
        [1.0 / n for n in DIGITAL_PDE_LEVELS], [abs(e) for e in errors]
    )
    assert abs(fit.order - case.row.expected) <= case.row.tolerance, case.row.source
    assert fit.residual < 0.05, fit.residual

    if remedied:
        assert all(abs(a) > abs(b) for a, b in pairwise(errors)), errors
    else:
        # The pathology is about the price being wrong, not only slow: 0.88%
        # of the value at the finest grid in this sequence.
        assert abs(errors[-1]) > 0.008 * abs(exact), errors


# --------------------------------------------------------------------------
# Four engines, one number.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", DIGITAL_CROSS_ENGINE_CASES, ids=_ids(DIGITAL_CROSS_ENGINE_CASES)
)
def test_cross_engine_agreement(case: DigitalBSCase) -> None:
    """Four independent routes to the same digital price.

    Evidence class: INDEPENDENT_ENGINE for the tree and PDE legs (different
    numerical methods reaching the same value), STATISTICAL for the Monte
    Carlo leg (the tolerance is a multiple of its own reported standard error,
    not an accuracy claim).

    Per-engine tolerances, every one derived from a measurement:

    - analytic: 1e-15. It *is* the reference, so this only pins that the two
      call sites agree.
    - Leisen-Reimer tree at `n_steps = 2001`: `DIGITAL_TREE_LR_TOLERANCE`
      (2e-08). Measured errors -7.152e-10, +1.938e-09, -5.284e-09.
    - PDE at `n_s = n_t = 800`, midpoint-aligned, Rannacher:
      `DIGITAL_PDE_TOLERANCE` (8e-05). Measured errors +2.229e-07, -1.152e-05,
      +2.331e-05.
    - Monte Carlo at 200_000 paths, seed 123: 4 standard errors. Measured |z|
      of 1.218, 1.644 and 0.495.

    The gap between the two *deterministic* numerical engines is three decimal
    orders of magnitude, and that is the finding rather than an embarrassment:
    both are order 2, but the lattice's error constant is tiny here because the
    digital price is exactly the binomial tail the Leisen-Reimer construction
    is built to match, while the grid still has to represent a step function.
    """
    assert case.row.evidence is EvidenceClass.INDEPENDENT_ENGINE

    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()
    exact = price(option, model, market).value

    tree = price(
        option,
        model,
        market,
        method="tree",
        cfg=TreeConfig(n_steps=DIGITAL_TREE_LR_N_STEPS, scheme="leisen-reimer"),
    ).value
    assert tree == pytest.approx(exact, abs=DIGITAL_TREE_LR_TOLERANCE)

    pde = price(option, model, market, method="pde", cfg=_pde_cfg(DIGITAL_PDE_N)).value
    assert pde == pytest.approx(exact, abs=DIGITAL_PDE_TOLERANCE)

    mc = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(n_paths=DIGITAL_MC_PATHS, n_steps=1, seed=DIGITAL_MC_SEED),
    )
    assert mc.stderr is not None
    assert abs(mc.value - exact) <= DIGITAL_MC_STDERR_MULTIPLE * mc.stderr

    # Every pairwise gap between the deterministic legs is inside the row's
    # own tolerance, which is what makes this a cross-engine claim rather than
    # three separate comparisons against the closed form.
    for a, b in ((tree, pde), (tree, exact), (pde, exact)):
        assert abs(a - b) <= case.row.tolerance, (case.row.id, a, b)

    # And the order-2 lattice is three decimal orders ahead of the order-2 grid.
    assert abs(tree - exact) * 100.0 < abs(pde - exact) + 1e-12


def test_monte_carlo_greeks_are_by_likelihood_ratio_and_pathwise_is_refused() -> None:
    """The fourth engine differentiates now, by exactly one estimator.

    Evidence class: NEGATIVE_FINDING on the pathwise half, STATISTICAL on the
    likelihood-ratio half. Slice 6 recorded that Monte Carlo digital Greeks
    were refused and named the Phase 3 replacement; Slice 10 delivers it, and
    what survives as a refusal is narrower and sharper: the **pathwise**
    estimator, whose almost-everywhere payoff derivative is identically zero,
    so it would return `0.0` rather than fail. The likelihood-ratio estimator
    covers the closed form at every cross-engine point (measured over 40 seeds
    at the reference point in `tests/test_digital_mc.py`), and the bump is
    available and measured to be the wrong tool there.
    """
    for case in DIGITAL_CROSS_ENGINE_CASES:
        spec = case.spec
        triple = (spec.option(), spec.model(), spec.market())
        with pytest.raises(NotSupportedError, match="biased to exactly 0.0"):
            greeks(
                *triple,
                method="mc",
                cfg=MCConfig(
                    n_paths=1_000, seed=DIGITAL_MC_SEED, greeks_estimator="pathwise"
                ),
            )
        exact = greeks(*triple, method="analytic")
        result = greeks(
            *triple,
            method="mc",
            cfg=MCConfig(
                n_paths=DIGITAL_GREEKS_MC_PATHS,
                seed=DIGITAL_MC_SEED,
                greeks_estimator="likelihood_ratio",
            ),
        )
        assert result.meta is not None
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            stderr = result.meta["stderr"][name]
            z = abs(getattr(result, name) - getattr(exact, name)) / stderr
            assert z <= DIGITAL_GREEKS_MC_STDERR_MULTIPLE, (case.row.id, name, z)


@pytest.mark.parametrize("case", DIGITAL_MC_GREEKS_CASES, ids=lambda c: c.row.id)
def test_digital_monte_carlo_likelihood_ratio_greek_rows(case: DigitalBSCase) -> None:
    """Fifteen cells of the estimator Slice 6 named and could not ship.

    Evidence class: STATISTICAL. `row.tolerance` is a z-score against the
    estimator's own reported standard error, the same shape the vanilla rows
    use -- which is the point of stating them the same way: the likelihood
    ratio meets a 4-sigma budget on a jump exactly as the pathwise estimator
    does on a smooth payoff, while the bump meets it on one and misses it by
    two orders of magnitude on the other.
    """
    assert case.row.evidence is EvidenceClass.STATISTICAL
    greek = case.row.id.split("_")[4]

    spec = case.spec
    triple = (spec.option(), spec.model(), spec.market())
    analytic = greeks(*triple, method="analytic")
    result = greeks(
        *triple,
        method="mc",
        cfg=MCConfig(
            n_paths=DIGITAL_GREEKS_MC_PATHS,
            n_steps=1,
            seed=DIGITAL_MC_SEED,
            greeks_estimator="likelihood_ratio",
        ),
    )
    assert result.meta is not None
    stderr = result.meta["stderr"][greek]
    assert stderr > 0.0, (case.row.id, stderr)
    z = (getattr(result, greek) - getattr(analytic, greek)) / stderr
    assert abs(z) <= case.row.tolerance, (case.row.id, z, case.row.notes)
    assert getattr(result, greek) != getattr(analytic, greek)


# --------------------------------------------------------------------------
# Hygiene.
# --------------------------------------------------------------------------


def test_every_digital_row_states_its_evidence_and_source() -> None:
    seen: set[str] = set()
    for case in ALL_DIGITAL_CASES:
        row = case.row
        assert row.id not in seen, f"duplicate case id {row.id}"
        seen.add(row.id)
        assert isinstance(row.evidence, EvidenceClass)
        assert row.description.strip()
        assert row.source.strip()
        assert row.tolerance > 0.0
        assert case.specs


def test_the_three_case_id_spaces_are_disjoint() -> None:
    """`ALL_CASES` stays European-only, as `ALL_AMERICAN_CASES` stays American.

    Parametrised test ids are built from these, so a collision would silently
    merge two different claims in a failure report.
    """
    european = {case.row.id for case in ALL_CASES}
    american = {case.row.id for case in ALL_AMERICAN_CASES}
    digital = {case.row.id for case in ALL_DIGITAL_CASES}

    assert european.isdisjoint(american)
    assert european.isdisjoint(digital)
    assert american.isdisjoint(digital)


def test_specs_round_trip_through_their_domain_objects() -> None:
    spec = DIGITAL_CROSS_ENGINE_CASES[0].spec
    option = spec.option()
    assert option.strike == spec.strike
    assert option.cash == spec.cash
    assert spec.flipped().flipped() == spec
    assert spec.riskless() == pytest.approx(
        spec.cash * math.exp(-spec.rate * spec.expiry), rel=0.0, abs=0.0
    )
