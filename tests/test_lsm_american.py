"""Least-squares Monte Carlo against the published table and the other engines.

Four claims, four kinds of evidence:

(b) PUBLISHED_BENCHMARK -- Longstaff & Schwartz (2001) Table 1 row 1 reports a
    least-squares Monte Carlo value of 4.472 with a standard error of 0.010 at
    their own settings (100 000 paths as 50 000 antithetic pairs, 50 exercise
    dates, a constant plus the first three weighted Laguerre functions, policy
    and valuation on the same paths). This file runs exactly that and checks
    the estimate is within three of its own standard errors of the published
    number, and that the standard error is of the published size. The value is
    a **Bermudan** value; the companion out-of-sample run is checked against
    the 50-date lattice Bermudan instead.

(c) CONVERGENCE_ORDER -- the gap between the Bermudan value and the
    continuously-exercisable one, measured on the lattice at 10, 50 and 250
    exercise dates, is first order in `1/m`. The LSM estimator is then checked
    against the lattice Bermudan at each of those frequencies, and the date at
    which its own noise swallows the gap is recorded rather than asserted
    away.

(d) INDEPENDENT_ENGINE -- the ATM American put from three discretisations:
    the Leisen-Reimer lattice, the PSOR grid, and a path sample. The budget is
    derived term by term, and one of its three terms is a contradiction of
    what this slice expected; see
    `test_atm_put_agrees_with_the_lattice_and_the_grid`.

(g) STATISTICAL -- the exercise boundary implied by the fitted policy against
    the boundaries the lattice and the grid report.

Every LSM number here was measured in this repository. The only quoted figures
are Longstaff & Schwartz's 4.472 and its standard error, used as a fixture with
the citation in `qpl.cases.american_black_scholes`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.cases import (
    AMERICAN_BRACKETED_LIMIT,
    AMERICAN_LSM_ATM_BERMUDAN_GAP,
    AMERICAN_LSM_ATM_DATES,
    AMERICAN_LSM_ATM_FINITE_SAMPLE_BIAS,
    AMERICAN_LSM_ATM_PATHS,
    AMERICAN_LSM_ATM_SEED,
    AMERICAN_LSM_BOUNDARY_TIMES,
    AMERICAN_LSM_CASES,
    AMERICAN_REFERENCE_SPEC,
    LS2001_BRACKETED_LIMIT,
    LS2001_LSM_PATHS,
    LS2001_LSM_SEED,
    LS2001_PUBLISHED_LSM_STDERR,
    LS2001_ROW1,
    LSM_BASIS,
    LSM_DEGREE,
    LSM_EXERCISE_FREQUENCIES,
    LSM_LATTICE_N_STEPS,
    LSM_STDERR_MULTIPLE,
    AmericanBSSpec,
    bermudan_value_on_lattice,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order

# --------------------------------------------------------------------------
# Shared settings. Every number comes from `qpl.cases`; nothing numeric is
# chosen in this file except the seeds of the two studies that need several.
# --------------------------------------------------------------------------

LATTICE_N_STEPS = LSM_LATTICE_N_STEPS
EXERCISE_FREQUENCIES = LSM_EXERCISE_FREQUENCIES
LS_PAPER_DATES = 50
"""Longstaff & Schwartz's exercise frequency for a one-year option; the same
`LS2001_BERMUDAN_EXERCISES_PER_YEAR` the Slice 2 rows use."""


def _row(case_id: str):
    return next(c.row for c in AMERICAN_LSM_CASES if c.row.id == case_id)


def _lsm(spec: AmericanBSSpec, **kwargs) -> object:
    cfg = MCConfig(n_steps=1, **kwargs)
    return price(spec.option(), spec.model(), spec.market(), method="mc", cfg=cfg)


FREQUENCY_STUDY_PATHS = 50_000
FREQUENCY_STUDY_SEED = 4


@pytest.fixture(scope="module")
def frequency_study() -> dict[int, object]:
    """One LSM run per exercise frequency, shared by the two tests that read it.

    Module-scoped because the two claims below -- "the estimator tracks the
    lattice Bermudan at each frequency" and "its own noise hides the gap at the
    finest one" -- are two readings of the *same* three runs, and running them
    twice would double the slowest part of this file for no extra evidence.
    """
    return {
        m: _lsm(
            LS2001_ROW1,
            n_paths=FREQUENCY_STUDY_PATHS,
            seed=FREQUENCY_STUDY_SEED,
            variance_reduction="antithetic",
            exercise_dates=m,
            lsm_degree=3,
        )
        for m in EXERCISE_FREQUENCIES
    }


# --------------------------------------------------------------------------
# (b) The published row.
# --------------------------------------------------------------------------


def test_longstaff_schwartz_row_one_in_sample() -> None:
    """Evidence class: PUBLISHED_BENCHMARK.

    Their settings, reproduced: `S = 36, K = 40, T = 1, r = 6%, q = 0,
    sigma = 20%`, 100 000 paths drawn as 50 000 antithetic pairs, 50 exercise
    dates, a constant plus the first three weighted Laguerre functions, and the
    policy fitted and applied on the same paths -- which is the estimator their
    table reports and which is biased **high** (Glasserman 2003, section 8.6).

    Measured at seed 20260913: **4.467550** with a standard error of
    **0.006032**, against their 4.472. The gap is 4.45e-03, which is **0.74**
    of this run's own standard error and 0.44 of the 0.010 printed in the
    table. Runtime 0.33 s.

    Two things are asserted and one is deliberately not. Asserted: the estimate
    is within three standard errors of 4.472, and the standard error is itself
    in `[0.004, 0.010]` -- the upper end is what the table prints, the lower is
    a guard against a standard error computed over 100 000 supposedly
    independent paths instead of 50 000 antithetic pairs, which would report
    about 0.0043 for a sample whose true precision is worse than that. Not
    asserted: that 4.4676 *equals* 4.472. It does not, it should not, and a run
    that reproduced a published Monte Carlo number to four figures at a
    different seed would be evidence of a copied answer rather than of a
    working estimator.

    What the number is: a **Bermudan** value on 50 dates, not an American one.
    The continuously-exercisable value at this specification is 4.4867
    (`LS2001_BRACKETED_LIMIT`), and Slice 2 pinned that distinction as a
    negative finding.
    """
    row = _row("ls2001_row1_lsm_reproduces_the_published_simulation_value")
    assert row.evidence is EvidenceClass.PUBLISHED_BENCHMARK
    assert "Longstaff" in row.source

    result = _lsm(
        LS2001_ROW1,
        n_paths=LS2001_LSM_PATHS,
        seed=LS2001_LSM_SEED,
        variance_reduction="antithetic",
        exercise_dates=LS_PAPER_DATES,
        lsm_basis=LSM_BASIS,
        lsm_degree=LSM_DEGREE,
        lsm_in_sample=True,
    )

    assert result.meta is not None
    assert result.meta["lsm_in_sample"] is True
    assert result.meta["n_basis_functions"] == LSM_DEGREE + 1
    assert result.meta["n_estimator_units"] == LS2001_LSM_PATHS // 2

    # Two readings of the same claim: the row's fixed budget (three times the
    # standard error this engine was measured to report) and the budget this
    # run's own standard error implies. Both, so that a change which quietly
    # inflated the reported standard error could not widen the test with it.
    assert abs(result.value - row.expected) <= row.tolerance, (result.value, row.notes)
    assert abs(result.value - row.expected) < LSM_STDERR_MULTIPLE * result.stderr
    assert 0.004 < result.stderr <= LS2001_PUBLISHED_LSM_STDERR, result.stderr


def test_longstaff_schwartz_row_one_out_of_sample_matches_the_lattice_bermudan() -> None:
    """Evidence class: STATISTICAL, against an in-repo lattice reference.

    The same settings with the policy fitted on one path set and applied to an
    independent one -- Glasserman's (2003, section 8.7) low-biased estimator.
    Its reference is not the published number but the 50-date Bermudan value on
    the CRR lattice, 4.477922 at `n = 5000`, because that is the thing this
    estimator is estimating.

    Measured at seed 20260913: **4.472996** with a standard error of
    **0.006057**; the gap to the lattice is -4.93e-03, or **0.81** standard
    errors. Over ten seeds at 100 000 paths the mean out-of-sample estimate is
    4.47634 with a standard error of the mean of 0.00143, so the residual low
    bias at this path count is about -1.5e-03 -- a tenth of a percent, and
    about one standard error of a single run.
    """
    row = _row("ls2001_row1_lsm_out_of_sample_matches_the_50_date_bermudan")
    assert row.evidence is EvidenceClass.STATISTICAL

    lattice = bermudan_value_on_lattice(
        LS2001_ROW1, n_steps=LATTICE_N_STEPS, n_exercise=LS_PAPER_DATES
    )
    result = _lsm(
        LS2001_ROW1,
        n_paths=LS2001_LSM_PATHS,
        seed=LS2001_LSM_SEED,
        variance_reduction="antithetic",
        exercise_dates=LS_PAPER_DATES,
        lsm_degree=LSM_DEGREE,
    )

    assert result.meta is not None
    assert result.meta["lsm_in_sample"] is False
    gap = result.value - lattice
    assert abs(gap - row.expected) <= row.tolerance, (gap, row.notes)
    assert abs(gap) < LSM_STDERR_MULTIPLE * result.stderr, (gap, result.stderr)
    # And it is below the continuously-exercisable value by about the Bermudan
    # gap, which is the point of (c) below.
    assert result.value < LS2001_BRACKETED_LIMIT


# --------------------------------------------------------------------------
# (c) The Bermudan gap and how it shrinks with the exercise frequency.
# --------------------------------------------------------------------------


def test_the_bermudan_gap_is_first_order_in_one_over_the_exercise_count() -> None:
    """Evidence class: CONVERGENCE_ORDER, measured on the lattice.

    This is a property of the *instrument*, not of the simulation, so it is
    measured where there is no noise: on the CRR lattice at `n = 5000`, with
    the exercise restriction applied at 10, 50 and 250 dates and the reference
    the continuously-exercisable limit `LS2001_BRACKETED_LIMIT = 4.4866721476`.

    Measured gaps: **4.4029e-02**, **8.750e-03**, **1.672e-03** at `m = 10,
    50, 250`. Successive ratios over a fivefold refinement are **5.03** and
    **5.23**, and the fitted order against `1/m` is **1.0176** with a log-space
    RMS residual of **0.0080**. First order, and the `m = 50` gap is the
    8.8e-03 the Slice 2 notes pinned.

    The band is 0.9 to 1.1. It admits the measured 1.0176 and excludes both
    order 1/2 (which is what a *barrier*'s discrete-monitoring bias would give,
    and is the reason to state the order rather than assume it) and order 2.
    """
    row = _row("lsm_bermudan_gap_is_first_order_in_the_exercise_count")
    gaps = [
        LS2001_BRACKETED_LIMIT
        - bermudan_value_on_lattice(LS2001_ROW1, n_steps=LATTICE_N_STEPS, n_exercise=m)
        for m in EXERCISE_FREQUENCIES
    ]
    assert all(gap > 0.0 for gap in gaps), gaps
    assert gaps[0] > gaps[1] > gaps[2]

    fit = fit_convergence_order([1.0 / m for m in EXERCISE_FREQUENCIES], gaps)
    assert abs(fit.order - row.expected) <= row.tolerance, (fit.order, row.notes)
    assert fit.residual < 0.05, fit.residual
    assert gaps[1] == pytest.approx(8.8e-3, abs=2e-4), gaps[1]


@pytest.mark.parametrize("n_exercise", EXERCISE_FREQUENCIES)
def test_lsm_tracks_the_lattice_bermudan_at_each_exercise_frequency(
    n_exercise: int, frequency_study: dict[int, object]
) -> None:
    """Evidence class: STATISTICAL.

    At each frequency the LSM estimate must sit within three of its own
    standard errors of the lattice Bermudan at the *same* frequency. That is
    the check that the simulation is solving the Bermudan problem it claims to
    -- comparing it against the continuous American value instead would confuse
    a Monte Carlo error with an exercise-frequency error.

    Measured at 50 000 antithetic paths, ten seeds, degree-3 Laguerre
    (mean over seeds, then the signed bias against the lattice):

        m = 10    4.44103    lattice 4.442643    bias -1.5e-03
        m = 50    4.47527    lattice 4.477922    bias -2.6e-03
        m = 250   4.48554    lattice 4.485000    bias +6.2e-04

    all inside one standard error of a single run (about 8.6e-03). The
    single-seed assertion below is therefore comfortable; the mean-over-seeds
    numbers are what say the estimator has no *systematic* problem at this
    point.
    """
    lattice = bermudan_value_on_lattice(
        LS2001_ROW1, n_steps=LATTICE_N_STEPS, n_exercise=n_exercise
    )
    result = frequency_study[n_exercise]
    assert abs(result.value - lattice) < 3.0 * result.stderr, (
        n_exercise, result.value, lattice, result.stderr,
    )


def test_the_lsm_estimator_cannot_resolve_the_bermudan_gap_at_250_dates(
    frequency_study: dict[int, object],
) -> None:
    """Evidence class: NEGATIVE_FINDING -- a measured noise floor.

    The slice asked for the Bermudan gap to be measured *through the LSM
    estimator* at 10, 50 and 250 dates and for an order in `1/m` to be fitted
    if a clean one exists. It does not, and the reason is arithmetic rather
    than numerical. At 50 000 antithetic paths a single run of this option
    reports a standard error of about **8.5e-03**, while the true gaps are

        m = 10    4.403e-02     5.2 standard errors -- resolvable
        m = 50    8.751e-03     1.0 standard errors -- marginal
        m = 250   1.672e-03     0.2 standard errors -- invisible

    so only the coarsest frequency is measurable at a cost this suite can
    carry. Fitting an order through three points of which two are noise would
    be fitting the seed. Resolving the `m = 250` gap to a third of its own size
    would need roughly five million paths.

    The order is therefore fitted on the lattice, where it is clean (1.0176,
    residual 0.0080), and the simulation is checked against the lattice
    frequency by frequency. What is asserted here is the honest residue: the
    `m = 10` gap comes out of the simulation at more than three standard
    errors, and the `m = 250` gap comes out statistically indistinguishable
    from zero even though the instrument being priced is definitely not the
    continuously-exercisable one.

    Measured at seed 4: `m = 10` gap +5.50e-02 (+6.39 sigma), `m = 50`
    +1.72e-02 (+2.03 sigma), `m = 250` **-1.38e-02** (-1.61 sigma) -- the last
    is on the wrong side of the true value, which is exactly what a quantity
    buried in noise looks like.
    """
    results = frequency_study
    gaps = {m: LS2001_BRACKETED_LIMIT - r.value for m, r in results.items()}
    true_gaps = {
        m: LS2001_BRACKETED_LIMIT
        - bermudan_value_on_lattice(LS2001_ROW1, n_steps=LATTICE_N_STEPS, n_exercise=m)
        for m in EXERCISE_FREQUENCIES
    }

    # The deterministic half of the claim: the true gap at 250 dates is well
    # below one standard error, and the one at 10 dates is well above three.
    assert true_gaps[250] < 0.5 * results[250].stderr, true_gaps[250]
    assert true_gaps[10] > 3.0 * results[10].stderr, true_gaps[10]

    # The measured half: resolvable at 10 dates, not at 250.
    assert gaps[10] > 3.0 * results[10].stderr, gaps[10]
    assert abs(gaps[250]) < 3.0 * results[250].stderr, gaps[250]


# --------------------------------------------------------------------------
# (d) Three discretisations of the ATM American put.
# --------------------------------------------------------------------------

def test_atm_put_agrees_with_the_lattice_and_the_grid() -> None:
    """Evidence class: INDEPENDENT_ENGINE, third discretisation: simulation.

    `S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`, American put, from three
    engines that share nothing but the model:

        Leisen-Reimer lattice, n = 8001      6.09033758
        PSOR grid, n_s = n_t = 800           6.08995244
        LSM, 250 dates, 100 000 paths        6.05330  (stderr 1.31e-02)

    **The budget, term by term.** The slice statement expected the tolerance to
    be "its stderr plus the Bermudan gap". Measured, that is not enough, and
    the missing term is the larger of the three:

        Bermudan gap at m = 250                      2.46e-03
        finite-sample low bias at 100 000 paths      1.89e-02
        three standard errors of one run             3.93e-02
        ------------------------------------------------------
        budget                                       6.07e-02

    The middle term is the out-of-sample estimator's residual low bias at a
    finite path count: the policy is fitted on 100 000 paths, so it is a noisy
    policy, and a noisy policy is a suboptimal policy. Measured over ten seeds
    it is 1.89e-02 at this point and decays like `N^{-1/2}` -- 3.29e-02 at
    20 000 paths, 2.30e-02 at 50 000, 1.89e-02 at 100 000. It is **not** a
    property of this implementation: QuantLib's `MCAmericanEngine` at
    comparable settings lands at 6.0782 +- 1.29e-02 against the same lattice
    Bermudan of 6.0880, a bias of -9.8e-03 with a standard error that covers
    ours (`tests/oracle/test_lsm_vs_quantlib.py`).

    It is also strongly point-dependent, which is the part worth remembering:
    at the Longstaff-Schwartz point the same measurement gives -1.5e-03, an
    order of magnitude smaller at the same path count and exercise frequency.
    An LSM tolerance is not transferable between specifications.

    The budget of 6.07e-02 is 1.0% of the value. That is a weak statement next
    to the 1.5e-03 the tree-and-grid row of `qpl.cases` gets, and saying so is
    the point: a third discretisation that agrees to 1% is still a third
    discretisation, and the reason it is only 1% is written above rather than
    hidden in a round number.
    """
    row = _row("atm_american_put_lsm_agrees_with_the_lattice_and_the_grid")
    assert row.evidence is EvidenceClass.INDEPENDENT_ENGINE

    spec = AMERICAN_REFERENCE_SPEC
    tree = price(
        spec.option(), spec.model(), spec.market(), method="tree",
        cfg=TreeConfig(n_steps=8001, scheme="leisen-reimer"),
    ).value
    pde = price(
        spec.option(), spec.model(), spec.market(), method="pde",
        cfg=PDEConfig(
            n_s=800, n_t=800, strike_alignment="midpoint", time_stepping="rannacher"
        ),
    ).value
    lsm = _lsm(
        spec,
        n_paths=AMERICAN_LSM_ATM_PATHS,
        seed=AMERICAN_LSM_ATM_SEED,
        variance_reduction="antithetic",
        exercise_dates=AMERICAN_LSM_ATM_DATES,
        lsm_degree=LSM_DEGREE,
    )

    bermudan = bermudan_value_on_lattice(
        spec, n_steps=LATTICE_N_STEPS, n_exercise=AMERICAN_LSM_ATM_DATES
    )
    bermudan_gap = AMERICAN_BRACKETED_LIMIT - bermudan
    assert bermudan_gap == pytest.approx(AMERICAN_LSM_ATM_BERMUDAN_GAP, abs=2e-4)

    worst = max(abs(lsm.value - tree), abs(lsm.value - pde))
    assert abs(worst - row.expected) <= row.tolerance, (worst, row.notes)
    # The same budget rebuilt from this run's own standard error, so an
    # inflated stderr cannot widen the test.
    budget = (
        bermudan_gap
        + AMERICAN_LSM_ATM_FINITE_SAMPLE_BIAS
        + LSM_STDERR_MULTIPLE * lsm.stderr
    )
    assert worst < budget, (worst, budget)
    # The two deterministic engines agree far better than either agrees with
    # the simulation, which is what makes the wide budget a statement about
    # Monte Carlo and not about them.
    assert abs(tree - pde) < 1e-3


# --------------------------------------------------------------------------
# (g) The exercise boundary implied by the regression.
# --------------------------------------------------------------------------

def test_exercise_boundary_tracks_the_lattice_and_the_grid() -> None:
    """Evidence class: STATISTICAL. Noisy by construction; reported as such.

    The LSM boundary at a date is the largest sampled spot at which exercising
    beat the fitted continuation (for a put; the smallest, for a call). It is
    not the same object as the lattice's or the grid's boundary and cannot be
    compared to them tightly, for three reasons that are each worth a line:

    1. It is an **upper order statistic** of a sample, so it is biased upward
       by roughly the sample's spacing near the boundary, and the bias shrinks
       only as the sample thickens there.
    2. It is a **Bermudan** boundary. At an exercise date the holder commits
       for the whole interval to the next one, so exercise is worthwhile a
       little higher than under continuous exercise; that effect is `O(dt)`.
    3. The fitted continuation carries the regression's approximation error,
       and near the boundary intrinsic and continuation differ by very little
       -- which is exactly where a small error moves the crossing a long way.

    Measured at the Longstaff-Schwartz point, 100 000 antithetic paths, 50
    dates, degree-3 Laguerre, over seeds 1, 2, 3, against the PSOR boundary at
    `n_s = n_t = 800` (the lattice's own boundary sits 0.10 to 0.28 below the
    grid's, which sets the scale of what "agreement" can mean here):

        t      PDE      tree     LSM (seeds 1/2/3)       worst gap to PDE
        0.10   33.079   32.973   33.704 33.676 33.726    +0.65
        0.25   33.438   33.200   33.740 33.517 33.741    +0.30
        0.50   33.978   33.782   34.285 34.468 34.349    +0.49
        0.75   35.056   34.841   35.198 35.192 35.215    +0.16
        0.90   36.315   36.034   35.951 36.086 36.030    -0.36

    Worst deviation 0.65 on a boundary of 33 to 36, i.e. **2.0%**. The sign is
    informative rather than random: the LSM boundary sits *above* the true one
    early in the option's life, which says the fitted policy exercises too
    eagerly there, which is the same suboptimality the low bias in the price
    measures. The tolerance is 1.2, 1.8 times the worst measured deviation and
    about 3% of the strike; a tighter budget would be asserting that a sample's
    extreme order statistic is a boundary estimator, which it is not.
    """
    row = _row("lsm_exercise_boundary_tracks_the_grid_and_the_lattice")
    spec = LS2001_ROW1
    pde = price(
        spec.option(), spec.model(), spec.market(), method="pde",
        cfg=PDEConfig(
            n_s=800, n_t=800, strike_alignment="midpoint", time_stepping="rannacher"
        ),
    )
    tree = price(
        spec.option(), spec.model(), spec.market(), method="tree",
        cfg=TreeConfig(n_steps=2001, scheme="leisen-reimer"),
    )
    lsm = _lsm(
        spec, n_paths=LS2001_LSM_PATHS, seed=1, variance_reduction="antithetic",
        exercise_dates=LS_PAPER_DATES, lsm_degree=LSM_DEGREE,
    )

    assert pde.meta is not None and tree.meta is not None and lsm.meta is not None
    pde_boundary = np.asarray(pde.meta["exercise_boundary"], dtype=float)
    pde_times = np.asarray(pde.meta["exercise_boundary_times"], dtype=float)
    tree_boundary = np.asarray(tree.meta["exercise_boundary"], dtype=float)
    tree_times = np.linspace(0.0, spec.expiry, tree_boundary.size)
    lsm_boundary = np.asarray(lsm.meta["exercise_boundary"], dtype=float)
    lsm_times = np.asarray(lsm.meta["exercise_boundary_times"], dtype=float)

    for t in AMERICAN_LSM_BOUNDARY_TIMES:
        i = int(np.argmin(np.abs(pde_times - t)))
        j = int(np.argmin(np.abs(tree_times - t)))
        k = int(np.argmin(np.abs(lsm_times - t)))
        assert math.isfinite(lsm_boundary[k]), t
        for reference in (pde_boundary[i], tree_boundary[j]):
            deviation = abs(lsm_boundary[k] - reference)
            assert abs(deviation - row.expected) <= row.tolerance, (
                t, lsm_boundary[k], reference, row.notes,
            )

    # The boundary rises toward the strike as expiry approaches: a put is
    # exercised at higher spots when there is less time left. Checked on the
    # whole LSM column, not only the sampled dates, as a shape statement that
    # survives the noise.
    finite = lsm_boundary[np.isfinite(lsm_boundary)]
    assert finite[0] < finite[-1]
    assert finite[-1] <= spec.strike
