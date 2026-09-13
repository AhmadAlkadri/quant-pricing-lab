"""American (Bermudan) options by least-squares Monte Carlo.

Reference: Longstaff, F.A. and Schwartz, E.S. (2001), "Valuing American
options by simulation: a simple least-squares approach", *Review of Financial
Studies* 14(1), 113-147, sections 1 (the algorithm, stated on their two-date
example) and 2 (the numerical results and the basis they use); Glasserman
(2003), *Monte Carlo Methods in Financial Engineering*, section 8.6
(regression-based methods and the in-sample bias of a fitted continuation
value) and section 8.7 (high- and low-biased estimators). Convergence of the
method in the number of paths and the number of basis functions:
Clement, E., Lamberton, D. and Protter, P. (2002), "An analysis of a least
squares regression method for American option pricing", *Finance and
Stochastics* 6, 449-471. Everything below is re-derived; no prose, code or
table is reproduced from any of them.

The problem
-----------
A Bermudan option with exercise dates `0 < t_1 < ... < t_m = T` is a
finite-horizon optimal stopping problem. Writing `g` for the intrinsic value
and `D(a, b)` for the discount factor from `b` back to `a`, the value function
satisfies the Bellman recursion

    V(t_m, S) = g(S),
    V(t_j, S) = max( g(S),  C(t_j, S) ),
    C(t_j, S) = E[ D(t_j, t_{j+1}) V(t_{j+1}, S_{t_{j+1}}) | S_{t_j} = S ].

That is the same recursion `qpl.engines.tree.american` solves on a lattice and
`qpl.engines.pde.american` solves as a linear complementarity problem. What is
different here is that the state is a *sample*, not a grid: there is no set of
nodes at `t_{j+1}` to average over, because each simulated path visits exactly
one point at each date.

Why the generic backward-induction helper is not reused
-------------------------------------------------------
`qpl.engines.dp.backward_induction_optimal_stopping` takes the exercise values
level by level and a `continuation_operator(j, values_at_j_plus_1)` returning
one continuation value per node at level `j`, with level `j` required to have
exactly `j + 1` entries. That signature is the *recombining lattice*: the
number of states grows by one per level and the operator is a local average
over two children. Neither holds here. A path sample has `n_paths` states at
every date, not `j + 1`, and the continuation operator is not a function of the
values at `t_{j+1}` alone -- it is a regression of those values on the
**current** spots, so the operator needs `S_{t_j}` as well, which that
signature has no way to pass. Fitting LSM into it would mean widening the
contract to "any number of states, plus a second state array", at which point
the helper is a `for` loop with extra validation. The recursion above is five
lines; it is written out.

The estimator
-------------
Following Longstaff & Schwartz section 1, the continuation value is
approximated by projecting the **realised** discounted cashflow from
continuing onto a finite set of basis functions of the current spot, using
ordinary least squares over the paths that are **in the money** at that date:

1. Start with the terminal cashflow `g(S_T)` on every path, dated `t_m`.
2. At each earlier date `t_j`, take the paths with `g(S_{t_j}) > 0` and
   regress their cashflow, discounted back to `t_j`, on `phi(S_{t_j})`.
3. Exercise on a path when `g(S_{t_j})` exceeds the **fitted** continuation
   value there; on those paths the cashflow is replaced by `g(S_{t_j})` and
   redated to `t_j`.
4. The price is the sample mean of the discounted **realised** cashflows.

Three details in that list carry the method, and each is a place an
implementation can be quietly wrong:

*The in-the-money filter.* Out of the money the intrinsic value is zero, so
the exercise decision is settled without any regression; including those paths
spends the basis functions on describing a region where the answer is known
and makes the fit over the region that matters worse. Longstaff & Schwartz
make this restriction explicitly (section 1).

*The regression is used for the decision only.* The reported value is the mean
of realised cashflows, **not** the mean of the fitted continuation values at
`t_1`. The fitted value is a projection of the cashflow onto a small function
space and its sample mean absorbs the regression's own fitting error; the
realised cashflow under the resulting policy is the value of an actual
(suboptimal) stopping rule, which is what makes it interpretable.

*Which sample the regression is fitted on.* If the policy is estimated and
applied on the same paths, the exercise decision at each date is made with
knowledge of that path's own realised future -- through the fitted
coefficients -- and the estimator is **biased high** (Glasserman section 8.6).
Fitting on one set of paths and valuing on an independent set removes that: the
policy is then fixed before the valuation paths are seen, so the valuation
average is the value of a genuine, suboptimal, stopping rule and is **biased
low** (Glasserman section 8.7). `MCConfig.lsm_in_sample` selects, and defaults
to `False` -- the low-biased estimator, which is the honest one to report
alone. The high-biased one is what Longstaff & Schwartz's own table reports
and is what `lsm_in_sample=True` reproduces. Both biases are `O(1)` in the
basis, not `O(1/N)`: more paths shrink the noise, not the approximation.

The basis and its scaling
-------------------------
Both bases are built on the **moneyness** `x = S / K`, never on the raw spot.
Two reasons, and the first is fatal without it: the weighted Laguerre
functions carry a factor `exp(-x/2)`, which at a spot of 100 is `exp(-50)` and
underflows any distinction between the basis functions. The second is
conditioning -- a polynomial in raw spot has columns of order `1, 10^2, 10^4,
10^6`, and the Gram matrix of that design is numerically singular long before
the basis is.

    "polynomial", degree d:  1, x, x^2, ..., x^d           (d + 1 columns)
    "laguerre",   degree d:  1, w L_0(x), ..., w L_{d-1}(x), w = exp(-x/2)

so both have `d + 1` columns at degree `d` and are comparable at equal degree.
The Laguerre case at `d = 3` is exactly the basis Longstaff & Schwartz report
in section 2: a constant plus the first three weighted Laguerre functions.

`numpy.linalg.lstsq` is used rather than solving the normal equations: it is a
least-squares solve of the design matrix itself, so the conditioning that
matters is `cond(A)` and not its square. The condition number is reported
anyway -- per date, summarised in `meta` -- because a degree-5 polynomial on a
sample that has collapsed into a narrow range of moneyness is exactly the
situation where the fit is meaningless and the price still looks plausible.

Antithetic sampling
-------------------
`variance_reduction="antithetic"` composes with LSM through the Slice 7
estimator layer, and it composes because the pairing is preserved *through* the
exercise decision rather than around it. One regression is fitted across the
whole sample (both a path and its reflection contribute to it), each path makes
its own exercise decision, and only then are the realised discounted cashflows
averaged within each `(Z, -Z)` pair to form the estimator unit. The pair
average is i.i.d. across pairs, so
`qpl.engines.mc.variance_reduction.estimate_from_sample` produces the same
`ddof=1`-over-pairs standard error it does for a European. What would *not* be
legitimate is reducing to pairs before the decision, or fitting two separate
regressions on the two halves.

`control_variate` and `stratified` are refused; see `_VR_REFUSAL`.

Greeks
------
Refused. Differentiating an LSM price means differentiating through an
exercise policy that is itself a step function of the fitted coefficients,
which are themselves functions of the sample. The pathwise derivative of the
stopped payoff exists (the exercise boundary is hit with probability zero for
a fixed policy), but the fitted policy moves with the bump, and a
finite-difference bump that changes the policy is estimating something that is
not the derivative of the price. That is its own topic and its own slice.

Measured tables, the bias directions, and the comparison against the lattice
and the PSOR grid: `docs/notes/lsm_american_monte_carlo.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import AmericanOption
from ...instruments.payoffs import call_payoff, put_payoff
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .greeks import reduce_to_units
from .pricers import MCConfig
from .processes import gbm_paths_from_normals
from .variance_reduction import (
    ANTITHETIC,
    CONTROL_VARIATE,
    STRATIFIED,
    TerminalSample,
    estimate_from_sample,
    normalise_variance_reduction,
    validate_sampler,
)

__all__ = [
    "LSM_BASES",
    "LSMFit",
    "basis_matrix",
    "greeks_american",
    "lsm_rollback",
    "price_american",
    "simulate_exercise_grid",
]

LAGUERRE = "laguerre"
POLYNOMIAL = "polynomial"
LSM_BASES = (LAGUERRE, POLYNOMIAL)
"""Basis families `MCConfig.lsm_basis` selects between."""

_VR_REFUSAL = (
    "variance_reduction '{name}' is not available for an American option "
    "priced by least-squares Monte Carlo. 'stratified' stratifies the single "
    "normal that drives a terminal price, and a Bermudan payoff is driven by "
    "one normal per exercise date with no single scalar to partition (the "
    "same refusal Slice 7 and the Asian engine already make). "
    "'control_variate' would need a control whose mean is known *under the "
    "stopping rule the regression produced*: the discounted spot is a "
    "martingale only up to a fixed date, and stopping it at a data-dependent "
    "time destroys the known mean that makes it a control. The European twin "
    "of the same option is the usable control there, and it is a later slice. "
    "'antithetic' does compose and is supported."
)

_GREEKS_REFUSAL = (
    "Greeks are not available for an American option priced by least-squares "
    "Monte Carlo. The price depends on the spot both through the payoff and "
    "through the fitted exercise policy, which is a step function of "
    "regression coefficients estimated from the sample; a bump moves the "
    "policy, so a difference quotient of two LSM prices is not an estimate of "
    "dV/dS, and the pathwise derivative of the stopped payoff is only valid "
    "for a policy held fixed. Use method='tree' or method='pde', which have "
    "American Greeks, or price the LSM value and differentiate a policy you "
    "hold fixed yourself. Pathwise/likelihood-ratio Greeks through an "
    "exercise policy are a later slice."
)


def _laguerre_columns(x: np.ndarray, degree: int) -> list[np.ndarray]:
    """`exp(-x/2) L_k(x)` for `k = 0 ... degree - 1`, by the standard recurrence.

    `L_0 = 1`, `L_1 = 1 - x`, and
    `(k + 1) L_{k+1} = (2k + 1 - x) L_k - k L_{k-1}`.
    The recurrence is used rather than expanded coefficients because the
    expanded form of `L_5` already alternates in sign with coefficients of
    order `1/120` and loses digits on a sample whose moneyness spans a decade.
    """
    weight = np.exp(-0.5 * x)
    previous = np.ones_like(x)
    columns = [weight * previous]
    if degree == 1:
        return columns
    current = 1.0 - x
    columns.append(weight * current)
    for k in range(1, degree - 1):
        previous, current = current, ((2 * k + 1 - x) * current - k * previous) / (k + 1)
        columns.append(weight * current)
    return columns


def basis_matrix(spots: np.ndarray, *, strike: float, basis: str, degree: int) -> np.ndarray:
    """Design matrix `(n, degree + 1)` of basis functions of the moneyness.

    The first column is the constant, so a degree-`d` fit has `d + 1` free
    coefficients in either family and the two are comparable at equal degree.
    Scaling by the strike is not cosmetic; see the module docstring.
    """
    x = np.asarray(spots, dtype=float) / strike
    columns = [np.ones_like(x)]
    if basis == POLYNOMIAL:
        power = np.ones_like(x)
        for _ in range(degree):
            power = power * x
            columns.append(power)
    else:
        columns.extend(_laguerre_columns(x, degree))
    return np.column_stack(columns)


def _design_condition(design: np.ndarray) -> float:
    """`cond(A)` from the Gram matrix, which is `(k, k)` rather than `(n, k)`.

    `cond(A) = sqrt(cond(A^T A))` for a full-rank `A`, and the Gram matrix costs
    `n k^2` flops against the `n k^2` an SVD of `A` would cost plus a copy of
    the whole design; at `n = 100_000` and fifty dates that copy is the
    expensive part. The number is reported, not acted on, so losing the last
    digits of a condition number that is being read as an order of magnitude is
    the right trade. Returns `inf` for a numerically singular design.
    """
    gram = design.T @ design
    try:
        return float(math.sqrt(np.linalg.cond(gram)))
    except np.linalg.LinAlgError:  # pragma: no cover - cond raises only on non-finite input
        return float("inf")


@dataclass(frozen=True)
class LSMFit:
    """What one backward induction over a path sample produced.

    Parameters
    ----------
    cashflows
        `(n_paths,)` realised cashflows, **discounted to time 0**.
    stop_index
        `(n_paths,)` index into the exercise grid at which each path stopped.
    coefficients
        Regression coefficients per exercise date, indexed by the grid index of
        that date. Dates at which no path was in the money are absent.
    condition_numbers
        `cond(A)` of the fitted design at each date that was fitted.
    boundary
        Estimated exercise boundary per exercise date, `nan` where the sample
        gives none.
    early_exercise_fraction
        Share of paths that stopped strictly before the last date.
    """

    cashflows: np.ndarray
    stop_index: np.ndarray
    coefficients: dict[int, np.ndarray]
    condition_numbers: dict[int, float]
    boundary: np.ndarray
    early_exercise_fraction: float


def _intrinsic(spots: np.ndarray, *, kind: str, strike: float) -> np.ndarray:
    values = call_payoff(spots, strike) if kind == "call" else put_payoff(spots, strike)
    return np.asarray(values, dtype=float)


def _boundary_from_sample(
    spots: np.ndarray, exercised: np.ndarray, *, kind: str
) -> float:
    """The exercise boundary implied by one date's decisions, or `nan`.

    For a put the exercise region is `S <= B(t)`, so the boundary is the
    **largest** sampled spot at which exercising was chosen; for a call it is
    `S >= B(t)` and therefore the smallest. Same convention as
    `qpl.engines.tree.american._boundary_node`, which is what makes the two
    columns comparable.

    The slice brief asked for "the smallest spot at which intrinsic exceeds the
    fitted continuation" for both kinds. That is right for a call and wrong for
    a put -- for a put it returns the smallest spot in the sample, which is a
    statement about the sample's lower tail and not about the boundary. The
    kind-dependent rule is what is implemented.
    """
    where = np.flatnonzero(exercised)
    if where.size == 0:
        return math.nan
    chosen = spots[where]
    return float(chosen.max() if kind == "put" else chosen.min())


def lsm_rollback(
    paths: np.ndarray,
    *,
    times: np.ndarray,
    discounts: np.ndarray,
    kind: str,
    strike: float,
    basis: str,
    degree: int,
    coefficients: dict[int, np.ndarray] | None = None,
) -> LSMFit:
    """Backward induction over a path sample; fit a policy, or apply a given one.

    Parameters
    ----------
    paths
        `(n_paths, m)` spot values at the exercise dates, without the `t = 0`
        column.
    times, discounts
        The `m` exercise dates and `df_r` at each of them.
    kind, strike
        The contract.
    basis, degree
        The regression basis; see :func:`basis_matrix`.
    coefficients
        When given, the policy is **applied** rather than fitted: no regression
        is run and the continuation value at date `j` is `phi(S) @ coefficients[j]`.
        This is the out-of-sample (low-biased) valuation pass of Glasserman
        section 8.7. Dates absent from the mapping are treated as dates at
        which continuing is always chosen, which is what a training pass with
        no in-the-money paths concluded.

    Returns
    -------
    LSMFit
        Cashflows discounted to time 0 plus the policy and its diagnostics.
    """
    n_paths, n_dates = paths.shape
    fitting = coefficients is None
    fitted: dict[int, np.ndarray] = {} if fitting else dict(coefficients or {})
    conditions: dict[int, float] = {}
    boundary = np.full(n_dates, math.nan)

    terminal = _intrinsic(paths[:, -1], kind=kind, strike=strike)
    # Everything is carried discounted to time 0, so redating a cashflow is one
    # multiplication and no cashflow date has to be tracked alongside it.
    cashflows = discounts[-1] * terminal
    stop_index = np.full(n_paths, n_dates - 1, dtype=int)
    boundary[-1] = _boundary_from_sample(paths[:, -1], terminal > 0.0, kind=kind)

    for j in range(n_dates - 2, -1, -1):
        spots = paths[:, j]
        intrinsic = _intrinsic(spots, kind=kind, strike=strike)
        itm = intrinsic > 0.0
        if not np.any(itm):
            continue

        design = basis_matrix(spots[itm], strike=strike, basis=basis, degree=degree)
        if fitting:
            # The regressand is the realised cashflow discounted back to t_j:
            # `cashflows` is already at time 0, so one division restores it.
            target = cashflows[itm] / discounts[j]
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
            fitted[j] = beta
            conditions[j] = _design_condition(design)
        else:
            beta = fitted.get(j)
            if beta is None:
                continue
            conditions[j] = _design_condition(design)

        continuation = design @ beta
        exercise = np.zeros(n_paths, dtype=bool)
        exercise[itm] = intrinsic[itm] > continuation
        cashflows = np.where(exercise, discounts[j] * intrinsic, cashflows)
        stop_index = np.where(exercise, j, stop_index)
        boundary[j] = _boundary_from_sample(spots, exercise, kind=kind)

    return LSMFit(
        cashflows=cashflows,
        stop_index=stop_index,
        coefficients=fitted,
        condition_numbers=conditions,
        boundary=boundary,
        early_exercise_fraction=float(np.mean(stop_index < n_dates - 1)),
    )


def simulate_exercise_grid(
    *,
    s0: float,
    mu: float,
    sigma: float,
    times: np.ndarray,
    n_paths: int,
    seed: int | np.random.Generator,
    methods: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    """`(paths, n_normal_draws)` at the exercise dates, under the sampler.

    Exact lognormal stepping between the dates
    (`qpl.engines.mc.processes.gbm_paths_from_normals`), so the only error in
    an LSM price at a *given* exercise grid is statistical plus the
    regression's approximation error: there is no time-discretisation bias to
    separate from either. The antithetic branch reproduces the Asian engine's
    `(Z, -Z)` block layout, so the first `n_paths // 2` rows are the base draws
    and the second half their reflections -- which is the layout
    `qpl.engines.mc.greeks.reduce_to_units` expects when it forms pair averages.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_dates = times.size
    if ANTITHETIC in methods:
        n_pairs = n_paths // 2
        base = rng.normal(size=(n_pairs, n_dates))
        z = np.concatenate([base, -base], axis=0)
        n_draws = n_pairs * n_dates
    else:
        z = rng.normal(size=(n_paths, n_dates))
        n_draws = n_paths * n_dates
    return gbm_paths_from_normals(z, s0=s0, mu=mu, sigma=sigma, times=times), n_draws


def _validated_lsm_config(cfg: MCConfig) -> tuple[int, str, int, bool]:
    """`(exercise_dates, basis, degree, in_sample)` after validation."""
    if not isinstance(cfg.exercise_dates, int) or isinstance(cfg.exercise_dates, bool):
        raise InvalidInputError("exercise_dates must be an int")
    if cfg.exercise_dates < 1:
        raise InvalidInputError(
            "exercise_dates must be >= 1: a Bermudan needs at least the "
            "terminal exercise date, and exercise_dates=1 is the European"
        )
    if not isinstance(cfg.lsm_basis, str) or cfg.lsm_basis not in LSM_BASES:
        raise InvalidInputError(
            f"unknown lsm_basis {cfg.lsm_basis!r}; expected one of {LSM_BASES}"
        )
    if not isinstance(cfg.lsm_degree, int) or isinstance(cfg.lsm_degree, bool):
        raise InvalidInputError("lsm_degree must be an int")
    if cfg.lsm_degree < 1:
        raise InvalidInputError("lsm_degree must be >= 1")
    if not isinstance(cfg.lsm_in_sample, bool):
        raise InvalidInputError("lsm_in_sample must be a bool")
    return cfg.exercise_dates, cfg.lsm_basis, cfg.lsm_degree, cfg.lsm_in_sample


def _degenerate_value(option: AmericanOption, market: Market) -> float:
    """The value when there is nothing to simulate: `T = 0` or `sigma = 0`.

    At `T = 0` the option is its intrinsic value. At `sigma = 0` the spot
    follows the deterministic forward, so the optimal stopping problem is a
    maximum over the exercise dates of the discounted intrinsic value along it
    -- the same statement `qpl.engines.tree.american._degenerate_value` makes,
    restricted here to the Bermudan grid rather than to a continuum.
    """
    t = option.expiry
    if t == 0.0:
        return float(_intrinsic(np.array([market.spot]), kind=option.kind,
                                strike=option.strike)[0])
    return 0.0


def price_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price an American option by least-squares Monte Carlo on a Bermudan grid.

    What is priced is a **Bermudan** option with `cfg.exercise_dates` equally
    spaced dates `t_i = i T / m`, `i = 1 ... m`, exercise at `t = 0` excluded.
    That is deliberately not "an American option computed by simulation": the
    value it converges to as `n_paths -> inf` with a rich enough basis is the
    Bermudan value, which sits **below** the continuously-exercisable one by a
    gap that shrinks as `m` grows (measured in
    `docs/notes/lsm_american_monte_carlo.md`). The instrument type is
    `AmericanOption` because that is the registry key early exercise lives
    under; `meta["exercise_style"]` says what was actually computed, and the
    reported value should not be compared with a tree or PSOR American value
    without accounting for that gap.

    Parameters
    ----------
    option, model, market
        The contract, the Black-Scholes volatility, and the spot plus curves.
    cfg
        Monte Carlo settings. Read here: `n_paths`, `seed`,
        `variance_reduction` (`"none"` or `"antithetic"`), `exercise_dates`,
        `lsm_basis`, `lsm_degree`, `lsm_in_sample`. `n_steps` must be its
        default `1`: the simulation grid **is** the exercise grid, and exact
        lognormal stepping means subdividing between exercise dates changes
        nothing but the cost.

    Returns
    -------
    PriceResult
        Value, its standard error, and `meta` carrying `lsm_in_sample`,
        `lsm_basis`, `lsm_degree`, `exercise_dates`, `early_exercise_fraction`,
        `regression_condition_max` / `_median`, `exercise_boundary` and
        `exercise_boundary_times`.

    Raises
    ------
    InvalidInputError
        `n_paths < 2`, `n_steps != 1`, an odd `n_paths` under antithetic, or an
        unusable `exercise_dates` / `lsm_basis` / `lsm_degree` /
        `lsm_in_sample`.
    NotSupportedError
        `variance_reduction` including `"control_variate"` or `"stratified"`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps != 1:
        raise InvalidInputError(
            "n_steps must be 1 for an American option priced by LSM: the "
            "simulation grid is the exercise grid (cfg.exercise_dates), and "
            "exact lognormal stepping means subdividing between exercise "
            "dates changes nothing but the cost"
        )
    n_dates, basis, degree, in_sample = _validated_lsm_config(cfg)
    methods = normalise_variance_reduction(cfg.variance_reduction)
    for name in (CONTROL_VARIATE, STRATIFIED):
        if name in methods:
            raise NotSupportedError(_VR_REFUSAL.format(name=name))
    validate_sampler(methods=methods, n_paths=cfg.n_paths, n_steps=1, n_strata=cfg.n_strata)

    t = option.expiry
    sigma = model.sigma
    s0 = market.spot
    r = market.rate(t) if t > 0.0 else 0.0
    q = market.dividend_yield(t) if t > 0.0 else 0.0

    meta: dict[str, Any] = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "american",
        "estimator": "longstaff_schwartz",
        "exercise_style": f"bermudan({n_dates} dates)",
        "exercise_dates": n_dates,
        "lsm_basis": basis,
        "lsm_degree": degree,
        "lsm_in_sample": in_sample,
        "n_basis_functions": degree + 1,
        "n_paths": cfg.n_paths,
        "n_steps": n_dates,
        "seed": cfg.seed,
        "variance_reduction": methods if methods else "none",
    }

    if t == 0.0 or sigma == 0.0:
        times = np.linspace(0.0, t, n_dates + 1)[1:] if t > 0.0 else np.array([0.0])
        if t == 0.0:
            value = _degenerate_value(option, market)
        else:
            forwards = s0 * np.exp((r - q) * times)
            intrinsic = _intrinsic(forwards, kind=option.kind, strike=option.strike)
            value = float(np.max([market.df_r(u) * v for u, v in zip(times, intrinsic)]))
        meta["degenerate"] = "T=0" if t == 0.0 else "sigma=0"
        meta["early_exercise_fraction"] = 0.0
        return PriceResult(value=value, stderr=0.0, meta=meta)

    times = np.linspace(0.0, t, n_dates + 1)[1:]
    discounts = np.array([market.df_r(float(u)) for u in times])
    mu = r - q

    # Two independent streams from one seed. `SeedSequence.spawn` is used rather
    # than `seed` and `seed + 1` because neighbouring seeds are not a documented
    # guarantee of independent streams, and the whole point of the out-of-sample
    # estimator is that the valuation paths are independent of the ones the
    # policy was fitted on.
    train_seed, value_seed = (
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(2)
    )

    train_paths, n_draws = simulate_exercise_grid(
        s0=s0, mu=mu, sigma=sigma, times=times, n_paths=cfg.n_paths,
        seed=train_seed, methods=methods,
    )
    fit = lsm_rollback(
        train_paths, times=times, discounts=discounts, kind=option.kind,
        strike=option.strike, basis=basis, degree=degree,
    )

    if in_sample:
        valued = fit
    else:
        value_paths, value_draws = simulate_exercise_grid(
            s0=s0, mu=mu, sigma=sigma, times=times, n_paths=cfg.n_paths,
            seed=value_seed, methods=methods,
        )
        n_draws += value_draws
        valued = lsm_rollback(
            value_paths, times=times, discounts=discounts, kind=option.kind,
            strike=option.strike, basis=basis, degree=degree,
            coefficients=fit.coefficients,
        )

    units = reduce_to_units(valued.cashflows, methods=methods)
    sample = TerminalSample(
        y=units,
        x=np.zeros_like(units),
        x_mean=0.0,
        stratum=None,
        n_strata=0,
        n_normal_draws=n_draws,
        n_paths=cfg.n_paths,
        control_name="unused",
    )
    estimate = estimate_from_sample(sample, methods=methods)
    meta.update(estimate.meta)
    meta["variance_reduction"] = methods if methods else "none"

    conditions = np.array(sorted(fit.condition_numbers.values())) if fit.condition_numbers \
        else np.array([math.nan])
    meta.update(
        {
            "n_normal_draws": n_draws,
            "early_exercise_fraction": valued.early_exercise_fraction,
            "n_regressions": len(fit.condition_numbers),
            "regression_condition_max": float(np.max(conditions)),
            "regression_condition_median": float(np.median(conditions)),
            "exercise_boundary": valued.boundary,
            "exercise_boundary_times": times,
            "intrinsic_at_inception": float(
                _intrinsic(np.array([s0]), kind=option.kind, strike=option.strike)[0]
            ),
        }
    )
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def greeks_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises `NotSupportedError`; see :data:`_GREEKS_REFUSAL`.

    Registered rather than omitted so that the caller is told *why* early
    exercise by simulation has no Greeks here, instead of being told that the
    instrument/model/market combination is unsupported -- which would be false,
    since the price engine right above prices it. Same pattern as the Slice 6
    Monte Carlo digital Greeks.
    """
    raise NotSupportedError(_GREEKS_REFUSAL)
