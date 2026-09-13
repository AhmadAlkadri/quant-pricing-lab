"""Scalar Ito SDE simulation: Euler-Maruyama, Milstein, and exact transitions.

The object of study is the one-dimensional Ito SDE

    dX_t = a(X_t, t) dt + b(X_t, t) dW_t,      X_0 = x0,

discretised on a grid ``0 = t_0 < t_1 < ... < t_m = T`` with steps
``h_i = t_{i+1} - t_i`` and Brownian increments ``dW_i = sqrt(h_i) Z_i``,
``Z_i`` independent standard normals.

Schemes
-------
Euler-Maruyama (Glasserman 2003, section 6.1; Kloeden & Platen 1992,
chapter 9):

    X_{i+1} = X_i + a(X_i, t_i) h_i + b(X_i, t_i) dW_i.

Milstein (Glasserman 2003, section 6.2; Kloeden & Platen 1992, chapter 10)
adds the next term of the stochastic Taylor expansion, which comes from
applying Ito's lemma to ``b`` inside the diffusion integral:

    X_{i+1} = X_i + a h_i + b dW_i + (1/2) b (db/dx) (dW_i^2 - h_i),

with ``a``, ``b`` and ``db/dx`` all evaluated at ``(X_i, t_i)``.  The extra
term has mean zero, which is exactly why it buys a *strong* order and not a
*weak* one: it corrects each path, not the law's low moments.  Orders (for
Lipschitz, sufficiently smooth coefficients): Euler is strong 1/2 and weak 1;
Milstein is strong 1 and weak 1.  Those are the claims `tests/
test_sde_convergence.py` measures rather than assumes.

Why the increments can be supplied
----------------------------------
`simulate` accepts an explicit ``normals`` block instead of a seed.  That is
not a convenience: the strong error ``E|X_T^h - X_T|`` is only defined once
the approximation and the reference are driven by the *same* Brownian path,
so a coarse grid must use increments obtained by **summing** the fine ones
(Higham 2001, SIAM Review 43(3), section 5).  `coarsen_normals` does that
summation, in units of standard normals rather than of increments, so the
result is again a standard normal block.  Without this, differencing two
independently seeded runs measures ``sqrt(2) * std(X_T)`` at every step size
and fits an order of zero.

State constraints
-----------------
`SDEModel.state_floor` records a lower bound the *true* process respects and
the *scheme* need not.  ``truncation='full'`` is Andersen's (2008, Journal of
Computational Finance 11(3), section 3) full truncation: evaluate the drift
and the diffusion at ``max(X_i, floor)`` while leaving the state itself
unfloored.  It keeps the scheme **defined** (no square root of a negative
number); it does not keep it **positive**, and `tests/test_mc_sde.py` pins
that distinction with a measured frequency.

Models supplied here
--------------------
- `gbm_sde(mu, sigma)`: ``a = mu x``, ``b = sigma x``, exact lognormal
  transition.  The reference case, because everything about it is known in
  closed form.
- `cir_sde(kappa, theta, xi)`: ``a = kappa (theta - v)``, ``b = xi sqrt(v)``
  (Cox, Ingersoll & Ross 1985, Econometrica 53(2), equation 17), floor 0, and
  the exact noncentral chi-square transition of Broadie & Kaya (2006,
  Operations Research 54(2), section 2.1); see `cir_transition_parameters`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

from ...exceptions import InvalidInputError

__all__ = [
    "SCHEMES",
    "TRUNCATIONS",
    "SDEModel",
    "SDEPaths",
    "cir_expected_excess",
    "cir_moments",
    "cir_sde",
    "cir_transition_parameters",
    "coarsen_normals",
    "gbm_sde",
    "simulate",
    "uniform_time_grid",
]

SCHEMES: tuple[str, ...] = ("euler", "milstein", "exact")
"""Schemes `simulate` accepts.  ``'exact'`` requires ``model.exact_step``."""

TRUNCATIONS: tuple[str, ...] = ("none", "full")
"""``'none'`` evaluates the coefficients at the state as it is; ``'full'``
evaluates them at ``max(state, model.state_floor)`` (Andersen 2008)."""

Coefficient = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class SDEModel:
    """A scalar Ito SDE, as the pieces a discretisation scheme actually reads.

    Parameters
    ----------
    name
        Short identifier, carried into `SDEPaths.meta` so a table of results
        says which process produced it.
    drift, diffusion
        ``a(x, t)`` and ``b(x, t)``, vectorised over ``x``.
    diffusion_derivative
        ``db/dx (x, t)``, required by the Milstein scheme unless
        `milstein_coefficient` is supplied or ``db_dx=`` is passed to
        `simulate`.
    milstein_coefficient
        ``(1/2) b (db/dx) (x, t)``, supplied directly for processes where the
        *product* is finite but the *factors* are not.  Complexity receipt:
        this field exists for exactly one observed bug.  For CIR,
        ``b = xi sqrt(v)`` and ``db/dx = xi / (2 sqrt(v))``, so the product is
        the constant ``xi^2 / 4`` everywhere but the generic
        ``0.5 * b * db_dx`` evaluates ``0 * inf = nan`` at ``v = 0`` -- which
        full truncation makes a *common* state, not a measure-zero one.
        Without the field, Milstein on CIR silently returns NaN paths.  Cost:
        one optional field and a three-branch resolution order in `simulate`.
    exact_step
        ``exact_step(x_prev, dt, rng=None, normals=None) -> x_next``, the exact
        transition law of the process where one is known.  It takes *either* a
        generator *or* a block of standard normals, because the GBM transition
        is a function of the Brownian increment (so it can be coupled to the
        discretised runs) while the CIR transition is not (it is a noncentral
        chi-square draw, and no deterministic function of one normal per step
        produces it).
    state_floor
        A bound the true process respects, used by ``truncation='full'``.
        ``None`` means the process is unconstrained and ``'full'`` is refused.
    params
        The parameters the model was built from, for reporting.
    """

    name: str
    drift: Coefficient
    diffusion: Coefficient
    diffusion_derivative: Coefficient | None = None
    milstein_coefficient: Coefficient | None = None
    exact_step: Callable[..., np.ndarray] | None = None
    state_floor: float | None = None
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SDEPaths:
    """Simulated paths plus the normals that produced them.

    ``values`` is ``(n_paths, len(times))`` with column 0 equal to ``x0``;
    ``normals`` is ``(n_paths, len(times) - 1)``.  The normals are always
    carried, not optionally returned: every coupled experiment in this
    repository needs them, they are the same size as ``values`` (so the
    memory cost is a factor of two, not an order), and an optional return
    that half the callers forget to ask for is how a coupled study quietly
    becomes an uncoupled one.

    For ``scheme='exact'`` on a model whose transition is not a function of a
    normal increment (CIR), ``normals`` is an empty ``(n_paths, 0)`` array:
    there were none.
    """

    times: np.ndarray
    values: np.ndarray
    normals: np.ndarray
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def terminal(self) -> np.ndarray:
        """Terminal state of every path, ``values[:, -1]``."""
        return self.values[:, -1]


def uniform_time_grid(t: float, n_steps: int) -> np.ndarray:
    """``n_steps`` equal steps from 0 to ``t``, as ``t * arange(n+1) / n``.

    Written this way rather than as `numpy.linspace` so that a coarse grid is
    a bit-for-bit sub-grid of any finer grid whose step count is a multiple of
    this one: ``t * (i*r) / (n*r)`` and ``t * i / n`` agree when ``i*r`` and
    ``n*r`` are exact, which they are for the power-of-two refinement ladders
    used in the convergence studies.  `linspace` computes ``start + i*step``
    with a separately rounded ``step`` and does not guarantee that.
    """
    if not math.isfinite(t) or t <= 0.0:
        raise InvalidInputError("t must be finite and > 0")
    if n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    return t * np.arange(n_steps + 1, dtype=float) / float(n_steps)


def coarsen_normals(normals: np.ndarray, n_coarse_steps: int) -> np.ndarray:
    """Standard normals for a coarse grid, built from the fine ones.

    Higham's (2001) coupling: the coarse Brownian increment over a block of
    ``r`` fine steps is the **sum** of the fine increments in that block.  In
    units of standard normals, with a uniform grid of fine step ``h`` and
    coarse step ``r h``,

        dW_coarse = sum_{j<r} sqrt(h) Z_j = sqrt(r h) * (sum_j Z_j / sqrt(r)),

    so the coarse normal is the block sum divided by ``sqrt(r)`` -- again a
    standard normal, and driven by the same Brownian path.  This function is
    the reason the strong-error studies measure an order at all.

    Parameters
    ----------
    normals
        ``(n_paths, n_fine_steps)`` standard normals.
    n_coarse_steps
        Must divide ``n_fine_steps``.
    """
    arr = np.asarray(normals, dtype=float)
    if arr.ndim != 2:
        raise InvalidInputError("normals must be a 2D (n_paths, n_steps) array")
    n_fine = arr.shape[1]
    if n_coarse_steps < 1:
        raise InvalidInputError("n_coarse_steps must be >= 1")
    if n_fine % n_coarse_steps != 0:
        raise InvalidInputError("n_coarse_steps must divide the number of fine steps")
    ratio = n_fine // n_coarse_steps
    blocks = arr.reshape(arr.shape[0], n_coarse_steps, ratio)
    return blocks.sum(axis=2) / math.sqrt(ratio)


def _clean_grid(t_grid: np.ndarray | list[float]) -> np.ndarray:
    grid = np.asarray(t_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise InvalidInputError("t_grid must be a 1D grid with at least two times")
    if not np.all(np.isfinite(grid)):
        raise InvalidInputError("t_grid must be finite")
    if grid[0] != 0.0:
        raise InvalidInputError("t_grid must start at 0")
    if np.any(np.diff(grid) <= 0.0):
        raise InvalidInputError("t_grid must be strictly increasing")
    return grid


def _resolve_normals(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
    normals: np.ndarray | None,
    n_paths: int,
    n_steps: int,
) -> tuple[np.ndarray, np.random.Generator | None]:
    given = sum(x is not None for x in (seed, rng, normals))
    if given != 1:
        raise InvalidInputError("pass exactly one of seed, rng or normals")
    if normals is not None:
        arr = np.asarray(normals, dtype=float)
        if arr.ndim != 2:
            raise InvalidInputError("normals must be a 2D (n_paths, n_steps) array")
        if arr.shape != (n_paths, n_steps):
            raise InvalidInputError(
                f"normals must have shape ({n_paths}, {n_steps}), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise InvalidInputError("normals must be finite")
        return arr, None
    generator = rng if rng is not None else np.random.default_rng(seed)
    return generator.normal(size=(n_paths, n_steps)), generator


def simulate(
    model: SDEModel,
    x0: float,
    t_grid: np.ndarray | list[float],
    n_paths: int,
    *,
    scheme: Literal["euler", "milstein", "exact"] = "euler",
    truncation: Literal["none", "full"] = "none",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    normals: np.ndarray | None = None,
    db_dx: Coefficient | None = None,
) -> SDEPaths:
    """Simulate a scalar Ito SDE on ``t_grid``.

    Parameters
    ----------
    model
        The process.  See `gbm_sde` and `cir_sde`.
    x0
        Initial state, finite.  Must be at or above ``model.state_floor`` when
        the model has one.
    t_grid
        Strictly increasing times starting at ``0.0``; see `uniform_time_grid`.
    n_paths
        Number of paths, ``>= 1``.
    scheme
        ``'euler'``, ``'milstein'`` or ``'exact'`` (see `SCHEMES`).
    truncation
        ``'none'`` or ``'full'`` (see `TRUNCATIONS`).  ``'full'`` requires a
        ``state_floor`` and is ignored by ``scheme='exact'``, whose transition
        law respects the constraint by construction -- passing it there is an
        error rather than a no-op, because it would otherwise read as if the
        exact sampler needed fixing.
    seed, rng, normals
        Exactly one.  ``normals`` is ``(n_paths, n_steps)`` standard normals,
        which is how a coarse run is coupled to a fine one; see
        `coarsen_normals`.
    db_dx
        Overrides the model's diffusion derivative for ``scheme='milstein'``.

    Returns
    -------
    SDEPaths
    """
    if scheme not in SCHEMES:
        raise InvalidInputError(f"scheme must be one of {SCHEMES}")
    if truncation not in TRUNCATIONS:
        raise InvalidInputError(f"truncation must be one of {TRUNCATIONS}")
    if not math.isfinite(x0):
        raise InvalidInputError("x0 must be finite")
    if n_paths < 1:
        raise InvalidInputError("n_paths must be >= 1")
    if model.state_floor is not None and x0 < model.state_floor:
        raise InvalidInputError(f"x0 must be >= the model state floor {model.state_floor}")
    if truncation == "full":
        if model.state_floor is None:
            raise InvalidInputError("truncation='full' needs a model with a state_floor")
        if scheme == "exact":
            raise InvalidInputError("truncation is meaningless for scheme='exact'")

    grid = _clean_grid(t_grid)
    n_steps = grid.size - 1
    dt = np.diff(grid)

    if scheme == "exact":
        return _simulate_exact(
            model, x0, grid, dt, n_paths, seed=seed, rng=rng, normals=normals
        )

    correction = _milstein_correction(model, db_dx) if scheme == "milstein" else None

    z, _ = _resolve_normals(
        seed=seed, rng=rng, normals=normals, n_paths=n_paths, n_steps=n_steps
    )
    values = np.empty((n_paths, n_steps + 1), dtype=float)
    values[:, 0] = float(x0)

    floor = model.state_floor if truncation == "full" else None
    for i in range(n_steps):
        h = float(dt[i])
        t_i = float(grid[i])
        x = values[:, i]
        x_eval = x if floor is None else np.maximum(x, floor)
        dw = math.sqrt(h) * z[:, i]
        with np.errstate(invalid="ignore"):
            step = model.drift(x_eval, t_i) * h + model.diffusion(x_eval, t_i) * dw
            if correction is not None:
                step = step + correction(x_eval, t_i) * (dw * dw - h)
        values[:, i + 1] = x + step

    meta = {
        "model": model.name,
        "scheme": scheme,
        "truncation": truncation,
        "n_steps": n_steps,
        "n_paths": n_paths,
        **dict(model.params),
    }
    return SDEPaths(times=grid, values=values, normals=z, meta=meta)


def _milstein_correction(model: SDEModel, db_dx: Coefficient | None) -> Coefficient:
    """Resolve ``(1/2) b db/dx``: explicit override, model product, model factor."""
    if db_dx is not None:
        diffusion = model.diffusion

        def from_override(x: np.ndarray, t: float) -> np.ndarray:
            return 0.5 * diffusion(x, t) * db_dx(x, t)

        return from_override
    if model.milstein_coefficient is not None:
        return model.milstein_coefficient
    if model.diffusion_derivative is not None:
        diffusion = model.diffusion
        derivative = model.diffusion_derivative

        def from_factors(x: np.ndarray, t: float) -> np.ndarray:
            return 0.5 * diffusion(x, t) * derivative(x, t)

        return from_factors
    raise InvalidInputError(
        "scheme='milstein' needs the diffusion derivative: pass db_dx= or use a "
        "model that supplies diffusion_derivative or milstein_coefficient"
    )


def _simulate_exact(
    model: SDEModel,
    x0: float,
    grid: np.ndarray,
    dt: np.ndarray,
    n_paths: int,
    *,
    seed: int | None,
    rng: np.random.Generator | None,
    normals: np.ndarray | None,
) -> SDEPaths:
    if model.exact_step is None:
        raise InvalidInputError(f"model {model.name!r} has no exact transition sampler")
    n_steps = grid.size - 1

    if normals is None and seed is None and rng is None:
        raise InvalidInputError("pass exactly one of seed, rng or normals")
    if normals is not None and (seed is not None or rng is not None):
        raise InvalidInputError("pass exactly one of seed, rng or normals")

    values = np.empty((n_paths, n_steps + 1), dtype=float)
    values[:, 0] = float(x0)

    if normals is not None:
        z, _ = _resolve_normals(
            seed=None, rng=None, normals=normals, n_paths=n_paths, n_steps=n_steps
        )
        for i in range(n_steps):
            values[:, i + 1] = model.exact_step(values[:, i], float(dt[i]), normals=z[:, i])
    else:
        generator = rng if rng is not None else np.random.default_rng(seed)
        z = np.empty((n_paths, 0), dtype=float)
        for i in range(n_steps):
            values[:, i + 1] = model.exact_step(values[:, i], float(dt[i]), rng=generator)

    meta = {
        "model": model.name,
        "scheme": "exact",
        "truncation": "none",
        "n_steps": n_steps,
        "n_paths": n_paths,
        **dict(model.params),
    }
    return SDEPaths(times=grid, values=values, normals=z, meta=meta)


# --------------------------------------------------------------------------
# Geometric Brownian motion
# --------------------------------------------------------------------------


def gbm_sde(mu: float, sigma: float) -> SDEModel:
    """``dS = mu S dt + sigma S dW``, with its exact lognormal transition.

    The reference process for a discretisation study: ``a = mu x`` and
    ``b = sigma x`` are Lipschitz on the relevant domain, ``db/dx = sigma`` is
    constant, and the transition

        S_{t+h} = S_t exp((mu - sigma^2/2) h + sigma sqrt(h) Z)

    is exact, so the strong and weak errors of a scheme can be measured
    against the truth rather than against a finer run.

    The exact step is written as a function of the normal ``Z`` precisely so
    that it can be driven by the same increments as the Euler and Milstein
    runs; that coupling is what makes ``E|X^h_T - X_T|`` a meaningful number.
    """
    if not math.isfinite(mu):
        raise InvalidInputError("mu must be finite")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise InvalidInputError("sigma must be finite and >= 0")

    def drift(x: np.ndarray, t: float) -> np.ndarray:
        return mu * x

    def diffusion(x: np.ndarray, t: float) -> np.ndarray:
        return sigma * x

    def diffusion_derivative(x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), sigma)

    def exact_step(
        x_prev: np.ndarray,
        dt: float,
        *,
        rng: np.random.Generator | None = None,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        if (normals is None) == (rng is None):
            raise InvalidInputError("exact_step needs exactly one of rng or normals")
        x = np.asarray(x_prev, dtype=float)
        z = rng.normal(size=x.shape) if normals is None else np.asarray(normals, dtype=float)
        return x * np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)

    return SDEModel(
        name="gbm",
        drift=drift,
        diffusion=diffusion,
        diffusion_derivative=diffusion_derivative,
        exact_step=exact_step,
        state_floor=0.0,
        params={"mu": mu, "sigma": sigma},
    )


# --------------------------------------------------------------------------
# Cox-Ingersoll-Ross
# --------------------------------------------------------------------------


def cir_transition_parameters(
    v0: np.ndarray | float,
    dt: float,
    *,
    kappa: float,
    theta: float,
    xi: float,
) -> tuple[float, np.ndarray, float]:
    """``(df, nc, scale)`` of the exact CIR transition, as a scaled ncx2.

    For ``dv = kappa (theta - v) dt + xi sqrt(v) dW`` the transition density is
    known in closed form (Cox, Ingersoll & Ross 1985, Econometrica 53(2),
    equation 18): ``v_{t+h}`` given ``v_t`` is a **scaled noncentral
    chi-square**,

        v_{t+h} = c * X,      X ~ chi'^2(d, lambda),
        c      = xi^2 (1 - e^{-kappa h}) / (4 kappa),
        d      = 4 kappa theta / xi^2,
        lambda = v_t e^{-kappa h} / c.

    Derivation check, not quotation: ``E[chi'^2(d, lambda)] = d + lambda`` and
    ``Var = 2 (d + 2 lambda)``, so

        E[v_{t+h}]   = c d + v_t e^{-kappa h}
                     = theta (1 - e^{-kappa h}) + v_t e^{-kappa h},
        Var[v_{t+h}] = 2 c^2 d + 4 c v_t e^{-kappa h}
                     = theta xi^2 (1 - e^{-kappa h})^2 / (2 kappa)
                       + v_t (xi^2/kappa) (e^{-kappa h} - e^{-2 kappa h}),

    which are the CIR conditional moments.  `cir_moments` returns exactly
    these, and `tests/test_mc_sde.py` checks the sampler against them rather
    than against the formula it was built from.

    The degrees of freedom ``d`` is the Feller number: ``2 kappa theta >= xi^2``
    (the Feller condition, so that 0 is unattainable) is exactly ``d >= 2``.
    Sampling: Broadie & Kaya (2006), Operations Research 54(2), section 2.1;
    Glasserman (2003), section 3.4.  `scipy.stats.ncx2` requires ``d > 0``,
    which holds whenever ``kappa``, ``theta`` and ``xi`` are positive, so the
    sampler is exact on **both** sides of the Feller condition.
    """
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise InvalidInputError("kappa must be finite and > 0")
    if theta <= 0.0 or not math.isfinite(theta):
        raise InvalidInputError("theta must be finite and > 0")
    if xi <= 0.0 or not math.isfinite(xi):
        raise InvalidInputError("xi must be finite and > 0")
    if not math.isfinite(dt) or dt <= 0.0:
        raise InvalidInputError("dt must be finite and > 0")

    v = np.asarray(v0, dtype=float)
    if np.any(v < 0.0):
        raise InvalidInputError("v0 must be >= 0")

    decay = math.exp(-kappa * dt)
    scale = xi * xi * (1.0 - decay) / (4.0 * kappa)
    df = 4.0 * kappa * theta / (xi * xi)
    nc = v * decay / scale
    return df, nc, scale


def cir_moments(
    v0: float,
    t: float,
    *,
    kappa: float,
    theta: float,
    xi: float,
) -> tuple[float, float]:
    """Closed-form conditional mean and variance of ``v_t`` given ``v_0``.

    Derived in `cir_transition_parameters`; restated here because a test that
    checks the sampler against the moments should not have to re-derive them
    inline.
    """
    df, nc, scale = cir_transition_parameters(v0, t, kappa=kappa, theta=theta, xi=xi)
    mean = float(scale * (df + nc))
    variance = float(scale * scale * 2.0 * (df + 2.0 * nc))
    return mean, variance


def cir_expected_excess(
    v0: float,
    t: float,
    *,
    kappa: float,
    theta: float,
    xi: float,
    strike: float,
) -> float:
    """``E[(v_t - strike)^+]`` under the exact CIR transition law.

    Computed by quadrature on the survival function, using
    ``E[(X - a)^+] = int_a^inf P(X > x) dx`` (the layer-cake identity for a
    non-negative random variable), with ``P(v_t > x)`` the scaled noncentral
    chi-square tail of `cir_transition_parameters`.

    This is a *deterministic* reference for the weak-error study.  The slice
    that motivated it said "versus the exact sampler"; sampling the exact law
    would put a Monte Carlo error bar on the reference as well as on the
    scheme, and the two would have to be disentangled.  Integrating the same
    law instead removes the reference's noise entirely, which is strictly
    better evidence for the same claim.
    """
    if strike < 0.0 or not math.isfinite(strike):
        raise InvalidInputError("strike must be finite and >= 0")
    df, nc, scale = cir_transition_parameters(v0, t, kappa=kappa, theta=theta, xi=xi)
    nc_value = float(np.asarray(nc).reshape(()))

    def tail(x: float) -> float:
        return float(ncx2.sf(x / scale, df, nc_value))

    mean = float(scale * (df + nc_value))
    variance = float(scale * scale * 2.0 * (df + 2.0 * nc_value))
    upper = strike + mean + 20.0 * math.sqrt(variance)
    value, _ = quad(tail, strike, upper, limit=200)
    return float(value)


def cir_sde(kappa: float, theta: float, xi: float) -> SDEModel:
    """``dv = kappa (theta - v) dt + xi sqrt(v) dW`` (Cox-Ingersoll-Ross 1985).

    The square-root diffusion is the standard counterexample to "Euler is
    fine": it is not Lipschitz at ``v = 0`` and it is not defined for
    ``v < 0``, so the Euler scheme can step to a state at which it cannot be
    evaluated again.  ``truncation='full'`` (Andersen 2008) evaluates both
    coefficients at ``max(v, 0)``, which keeps the recursion defined; it does
    **not** make the scheme positive.

    `milstein_coefficient` is supplied because the Milstein correction
    ``(1/2) b db/dx`` is the constant ``xi^2/4`` while its two factors are
    ``0`` and ``inf`` at ``v = 0``; see `SDEModel`.
    """
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise InvalidInputError("kappa must be finite and > 0")
    if theta <= 0.0 or not math.isfinite(theta):
        raise InvalidInputError("theta must be finite and > 0")
    if xi <= 0.0 or not math.isfinite(xi):
        raise InvalidInputError("xi must be finite and > 0")

    def drift(x: np.ndarray, t: float) -> np.ndarray:
        return kappa * (theta - x)

    def diffusion(x: np.ndarray, t: float) -> np.ndarray:
        return xi * np.sqrt(x)

    def diffusion_derivative(x: np.ndarray, t: float) -> np.ndarray:
        return xi / (2.0 * np.sqrt(x))

    def milstein_coefficient(x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), 0.25 * xi * xi)

    def exact_step(
        x_prev: np.ndarray,
        dt: float,
        *,
        rng: np.random.Generator | None = None,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        if normals is not None:
            raise InvalidInputError(
                "the CIR transition is a noncentral chi-square draw, not a "
                "function of one normal per step; pass rng= instead"
            )
        if rng is None:
            raise InvalidInputError("exact_step needs rng=")
        df, nc, scale = cir_transition_parameters(
            x_prev, dt, kappa=kappa, theta=theta, xi=xi
        )
        return np.asarray(
            ncx2.rvs(df, nc, scale=scale, size=nc.shape, random_state=rng), dtype=float
        )

    return SDEModel(
        name="cir",
        drift=drift,
        diffusion=diffusion,
        diffusion_derivative=diffusion_derivative,
        milstein_coefficient=milstein_coefficient,
        exact_step=exact_step,
        state_floor=0.0,
        params={"kappa": kappa, "theta": theta, "xi": xi},
    )
