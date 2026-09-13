"""What the basis and the sampling scheme cost: two biases, measured.

Slice item (f), and the measured half of the `lsm_in_sample` contract. Two
effects live here and they point in opposite directions:

**The in-sample high bias** (Glasserman 2003, section 8.6). Fitting the
exercise policy and valuing it on the same paths lets each path's exercise
decision be informed by that path's own realised future, through the fitted
coefficients. The result is biased **up**, the bias grows with the number of
basis functions, and it decays as the sample grows.

**The out-of-sample low bias** (Glasserman section 8.7). A policy fitted on an
independent sample is a genuine, suboptimal stopping rule, so valuing it gives
a lower bound in expectation. Its size is driven by how badly the fitted policy
misses the true exercise boundary, which has two causes -- too few basis
functions to describe the continuation value, and too few paths to fit the ones
there are. The second is why *more* basis functions can make this bias worse.

How the measurement is built, and why it is not just "run the engine twice"
-------------------------------------------------------------------------
The naive comparison -- run `lsm_in_sample=True`, run `lsm_in_sample=False`,
subtract -- measures the bias plus the full sampling noise of two independent
valuations. Over 20 seeds at 50 000 paths that reads `-0.0022 +- 0.0028`: the
wrong sign and not significant, on an effect that is really `+0.0021`.

What is measured here instead: for each seed, simulate two independent path
sets `A` and `B`, fit a policy on each, and value **both policies on `A`**. The
in-sample estimator is `A`'s policy on `A`; the out-of-sample estimator is
`B`'s policy on `A`, which has the same law as the engine's out-of-sample
output. Their difference shares the valuation sample, so the valuation noise
cancels and what is left is the bias. The same seeds also drive every
`(basis, degree)` configuration, so the comparisons between configurations are
paired too.

This is why the file reaches into `qpl.engines.mc.american.lsm_rollback` and
`simulate_exercise_grid` rather than calling `price`: the engine has no way to
report "this policy, valued on that sample", and it should not -- that is a
measurement, not a price.

Point: the Longstaff-Schwartz row 1 specification at its own 50 exercise
dates. Reference for the out-of-sample bias: the 50-date Bermudan value on the
CRR lattice, `qpl.cases.bermudan_value_on_lattice`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.cases import (
    AMERICAN_LSM_CASES,
    LS2001_BERMUDAN_EXERCISES_PER_YEAR,
    LS2001_ROW1,
    LSM_LATTICE_N_STEPS,
    bermudan_value_on_lattice,
)
from qpl.engines.mc.american import lsm_rollback, simulate_exercise_grid
from qpl.engines.mc.greeks import reduce_to_units
from qpl.validation import EvidenceClass

IN_SAMPLE_BIAS_ROW = next(
    case.row
    for case in AMERICAN_LSM_CASES
    if case.row.id == "lsm_in_sample_estimator_is_biased_high"
)
"""The row this file evaluates: its `tolerance` is the number of standard
errors every configuration's in-sample gap must exceed."""

EXERCISE_DATES = LS2001_BERMUDAN_EXERCISES_PER_YEAR
SMALL_PATHS = 2_000
LARGE_PATHS = 8_000
N_SEEDS = 24
SEED_BASE = 3000
METHODS = ("antithetic",)

BASES = ("laguerre", "polynomial")
DEGREES = (2, 3, 5)
CONFIGS = tuple((basis, degree) for basis in BASES for degree in DEGREES)

LATTICE_N_STEPS = LSM_LATTICE_N_STEPS


def _mean_and_stderr(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _paired_study(n_paths: int, configs: tuple[tuple[str, int], ...]) -> dict:
    """Per configuration: out-of-sample values and the paired in-sample gaps.

    One pair of path sets per seed, shared by every configuration. See the
    module docstring for why the in-sample gap is `A|A - B|A` rather than
    `A|A - B|B`.
    """
    spec = LS2001_ROW1
    times = np.linspace(0.0, spec.expiry, EXERCISE_DATES + 1)[1:]
    discounts = np.exp(-spec.rate * times)
    out: dict = {config: {"out": [], "gap": []} for config in configs}

    for offset in range(N_SEEDS):
        streams = [
            np.random.default_rng(child)
            for child in np.random.SeedSequence(SEED_BASE + offset).spawn(2)
        ]
        samples = [
            simulate_exercise_grid(
                s0=spec.spot, mu=spec.rate - spec.dividend, sigma=spec.sigma,
                times=times, n_paths=n_paths, seed=stream, methods=METHODS,
            )[0]
            for stream in streams
        ]
        a_paths, b_paths = samples
        for basis, degree in configs:
            kwargs = dict(
                times=times, discounts=discounts, kind=spec.kind, strike=spec.strike,
                basis=basis, degree=degree,
            )
            policy_a = lsm_rollback(a_paths, **kwargs)
            policy_b = lsm_rollback(b_paths, **kwargs)
            b_on_a = lsm_rollback(a_paths, coefficients=policy_b.coefficients, **kwargs)

            def mean_of(cashflows: np.ndarray) -> float:
                return float(np.mean(reduce_to_units(cashflows, methods=METHODS)))

            in_sample = mean_of(policy_a.cashflows)
            out_of_sample = mean_of(b_on_a.cashflows)
            out[(basis, degree)]["out"].append(out_of_sample)
            out[(basis, degree)]["gap"].append(in_sample - out_of_sample)

    return {
        config: {key: np.asarray(values) for key, values in record.items()}
        for config, record in out.items()
    }


@pytest.fixture(scope="module")
def small_sample_study() -> dict:
    """All six `(basis, degree)` configurations at 2 000 paths. About 2.5 s."""
    return _paired_study(SMALL_PATHS, CONFIGS)


@pytest.fixture(scope="module")
def large_sample_study() -> dict:
    """Laguerre degree 3 only, at 8 000 paths, for the `1/N` decay. About 1.5 s."""
    return _paired_study(LARGE_PATHS, (("laguerre", 3),))


@pytest.fixture(scope="module")
def lattice_bermudan() -> float:
    return bermudan_value_on_lattice(
        LS2001_ROW1, n_steps=LATTICE_N_STEPS, n_exercise=EXERCISE_DATES
    )


# --------------------------------------------------------------------------
# The in-sample high bias.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS, ids=[f"{b}_d{d}" for b, d in CONFIGS])
def test_the_in_sample_estimator_is_biased_high(
    config: tuple[str, int], small_sample_study: dict
) -> None:
    """Evidence class: STATISTICAL. Sign, not size.

    Measured at 2 000 paths over 24 paired seeds (`in - out`, and how many of
    its own standard errors that is):

        laguerre   d=2   +0.03794 +- 0.00663    5.7
        laguerre   d=3   +0.04585 +- 0.00614    7.5
        laguerre   d=5   +0.07413 +- 0.00729   10.2
        polynomial d=2   +0.04396 +- 0.00612    7.2
        polynomial d=3   +0.05219 +- 0.00538    9.7
        polynomial d=5   +0.07436 +- 0.00831    8.9

    Every one positive at better than five standard errors, which is the
    direction Glasserman section 8.6 predicts and the reason
    `MCConfig.lsm_in_sample` defaults to `False`.
    """
    assert IN_SAMPLE_BIAS_ROW.evidence is EvidenceClass.STATISTICAL
    mean, stderr = _mean_and_stderr(small_sample_study[config]["gap"])
    z_score = mean / stderr
    assert z_score > IN_SAMPLE_BIAS_ROW.expected + IN_SAMPLE_BIAS_ROW.tolerance, (
        config, mean, stderr, IN_SAMPLE_BIAS_ROW.notes,
    )


def test_the_in_sample_bias_decays_like_one_over_the_path_count(
    small_sample_study: dict, large_sample_study: dict
) -> None:
    """Evidence class: CONVERGENCE_ORDER, in the sample size.

    The in-sample bias is an overfitting bias: with `k` coefficients fitted on
    `N` in-the-money paths it is the usual `O(k/N)`, not `O(N^{-1/2})`, so it
    disappears much faster than the standard error does. That is what makes the
    in-sample estimator usable at all at a hundred thousand paths, and useless
    at two thousand.

    Measured on laguerre degree 3 over the same 24 seeds: **+0.04585** at
    2 000 paths and **+0.01114 +- 0.00212** at 8 000. A fourfold increase in
    paths divides the bias by **4.11**, an order of **1.020** in `1/N`.

    The band is 0.7 to 1.3 on the fitted order. It admits the measured 1.020,
    excludes the `1/2` that the *standard error* obeys -- which is the point:
    the two shrink at different rates, so at large `N` the bias is invisible
    inside the noise and at small `N` it dominates it. At 2 000 paths the bias
    is 0.046 against a standard error of about 0.043 for a single run; at
    100 000 it is 0.0021 against 0.006.
    """
    small, _ = _mean_and_stderr(small_sample_study[("laguerre", 3)]["gap"])
    large, large_stderr = _mean_and_stderr(large_sample_study[("laguerre", 3)]["gap"])

    assert large > 3.0 * large_stderr, (large, large_stderr)
    order = math.log(small / large) / math.log(LARGE_PATHS / SMALL_PATHS)
    assert 0.7 <= order <= 1.3, order


def test_the_in_sample_bias_grows_with_the_number_of_basis_functions(
    small_sample_study: dict,
) -> None:
    """Evidence class: STATISTICAL -- the `k` in `O(k/N)`, measured.

    Going from four coefficients (degree 3) to six (degree 5) raises the
    in-sample bias by, paired seed by seed,

        laguerre     +0.02827 +- 0.00680   (4.2 standard errors)
        polynomial   +0.02218 +- 0.00679   (3.3)

    The degree-2-to-degree-3 step is **not** resolved at this seed count
    (+0.00791 +- 0.00776 for laguerre, 1.0 standard errors) and is deliberately
    not asserted: three coefficients to four is a smaller step than four to
    six, and a basis with only three functions has a second, larger problem
    that the next test measures.
    """
    for basis in BASES:
        contrast = (
            small_sample_study[(basis, 5)]["gap"] - small_sample_study[(basis, 3)]["gap"]
        )
        mean, stderr = _mean_and_stderr(contrast)
        assert mean > 2.0 * stderr, (basis, mean, stderr)


# --------------------------------------------------------------------------
# The out-of-sample low bias: the basis/degree table.
# --------------------------------------------------------------------------


def test_the_basis_degree_table_against_the_lattice_bermudan(
    small_sample_study: dict, lattice_bermudan: float
) -> None:
    """Evidence class: STATISTICAL. The table, and that every entry is a loss.

    Out-of-sample bias against the 50-date lattice Bermudan (4.477922), at
    2 000 paths over 24 seeds:

        laguerre   d=2   -0.02630 +- 0.00905
        laguerre   d=3   -0.01946 +- 0.00814
        laguerre   d=5   -0.03416 +- 0.00791
        polynomial d=2   -0.02716 +- 0.00866
        polynomial d=3   -0.02447 +- 0.00751
        polynomial d=5   -0.03370 +- 0.00832

    Every entry is negative, which is the direction a suboptimal stopping rule
    has to produce. Nothing tighter is asserted per entry, because at 2 000
    paths a single entry is a two-to-three standard error statement; the
    *contrasts* in the next two tests are where the seed pairing buys
    resolution.
    """
    for config in CONFIGS:
        mean, stderr = _mean_and_stderr(small_sample_study[config]["out"])
        assert mean - lattice_bermudan < 0.0, (config, mean, lattice_bermudan)
        # And not absurdly so: a loss of more than 2% of the value would mean
        # the policy had stopped resembling an exercise rule.
        assert lattice_bermudan - mean < 0.02 * lattice_bermudan, (config, mean)


def test_degree_five_overfits_at_two_thousand_paths(small_sample_study: dict) -> None:
    """Evidence class: STATISTICAL -- the other half of Glasserman section 8.6.

    More basis functions buy a better description of the continuation value and
    cost a noisier fit. At 2 000 paths the second effect wins: valued out of
    sample, degree 3 beats degree 5 by, paired seed by seed,

        laguerre     +0.01470 +- 0.00544   (2.7 standard errors)
        polynomial   +0.00923 +- 0.00457   (2.0)

    so the richer basis produces a **worse** policy, not a better one. The
    direction reverses with enough paths: at 20 000 the same contrast for
    laguerre is `-0.00016 +- 0.00111` -- the penalty has gone and nothing has
    replaced it, because four functions already describe this continuation
    value well enough.

    Read together with the previous test, this is the whole tuning problem in
    two numbers: at 2 000 paths degree 5 raises the in-sample bias by +0.028
    *and* lowers the out-of-sample value by 0.015. There is no setting of
    `lsm_in_sample` that makes a basis too rich for the sample a good idea.
    """
    for basis in BASES:
        contrast = (
            small_sample_study[(basis, 3)]["out"] - small_sample_study[(basis, 5)]["out"]
        )
        mean, stderr = _mean_and_stderr(contrast)
        assert mean > 0.0, (basis, mean, stderr)
        assert mean > 1.5 * stderr, (basis, mean, stderr)


def test_the_basis_family_does_not_matter_but_the_count_does(
    small_sample_study: dict,
) -> None:
    """Evidence class: NEGATIVE_FINDING.

    Longstaff & Schwartz use weighted Laguerre functions and say in section 2
    that other families give similar results. Measured here, paired seed by
    seed at 2 000 paths, laguerre minus polynomial out-of-sample:

        d = 2    +0.00086 +- 0.00299   (0.3 standard errors)
        d = 3    +0.00500 +- 0.00328   (1.5)
        d = 5    -0.00046 +- 0.00212   (-0.2)

    Not one of the three is resolved, while the degree-3-to-degree-5 contrast
    *within* a family is (2.0 to 2.7 standard errors). The count of functions
    is what the estimator responds to; which spanning set they come from is
    not measurable here. Pinned as a negative finding so that `lsm_basis` is
    not mistaken for a tuning knob with a known best setting -- on this
    problem, at these degrees, it is not one.

    The caveat that keeps this honest: both families are applied to the
    **moneyness**, and the Laguerre one carries `exp(-x/2)`. The claim is about
    two well-conditioned spanning sets of the same size, not about the raw
    weighting, whose absence is a rank-one design
    (`tests/test_mc_american_lsm.py`).
    """
    for degree in DEGREES:
        contrast = (
            small_sample_study[("laguerre", degree)]["out"]
            - small_sample_study[("polynomial", degree)]["out"]
        )
        mean, stderr = _mean_and_stderr(contrast)
        assert abs(mean) < 3.0 * stderr, (degree, mean, stderr)
