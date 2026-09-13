"""Benchmark cases for SDE discretisation: strong/weak orders and the CIR boundary.

A sixth id space, and the second keyed by a *method* rather than an instrument
(`mc_variance_reduction` was the first). A row here says "the Milstein scheme
converges strongly at order 1 on this process, measured this way, at this
budget" -- a claim about a discretisation, which no instrument-keyed module
has a place for.

What a row asserts
------------------
For an order row, `expected` is a convergence order and `tolerance` is the band
around it. Where the theory gives an order, `expected` is the theoretical value
and `source` cites it; where this repository measured something the theory does
not predict at this budget (the CIR truncation rows), `expected` is the
measurement and `source` says so. A measured `expected` pins the measurement;
it does not explain it.

For a `NEGATIVE_FINDING` row, `expected` is a measured frequency at fixed
settings, and the row exists so that the frequency cannot silently move.

Bands are not round numbers picked for comfort. Each one is stated in its
`notes` as a multiple of the two things that actually move a fitted order here:
the seed-to-seed spread of the fit, and the distance between the fit and the
theoretical value caused by starting the ladder at a finite step size. Where
those two disagree about which dominates, the notes say which.

What is deliberately **not** a row
----------------------------------
The CIR exact sampler's mean and variance. Their expected values are computed
from `(kappa, theta, xi, v0, t)` by `qpl.engines.mc.sde.cir_moments` at test
time, so a row would carry a copy of a formula rather than a claim, and the two
copies could drift apart. The test checks the sampler against the formula and
the formula against the CIR conditional moments written out independently.

Sources. Orders: Kloeden & Platen (1992), *Numerical Solution of Stochastic
Differential Equations*, chapters 9-10; Glasserman (2003), *Monte Carlo Methods
in Financial Engineering*, sections 6.1-6.2. Measurement design: Higham (2001),
SIAM Review 43(3), section 5. CIR process: Cox, Ingersoll & Ross (1985),
Econometrica 53(2). Exact CIR transition sampling: Broadie & Kaya (2006),
Operations Research 54(2); Glasserman section 3.4. Full truncation: Andersen
(2008), Journal of Computational Finance 11(3), section 3. Every number below
was produced in this repository; none is quoted from those sources.
Derivation and full tables: `docs/notes/sde_discretization.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_SDE_CASES",
    "CIR_FELLER_OK",
    "CIR_FELLER_VIOLATED",
    "CIR_LEVELS",
    "CIR_NEGATIVITY_PATHS",
    "CIR_NEGATIVITY_SEED",
    "CIR_NEGATIVITY_STEPS",
    "CIR_PATHS",
    "CIR_SEED",
    "GBM_LEVELS",
    "GBM_PATHS",
    "GBM_SEED",
    "GBM_SMALL_DRIFT_MU",
    "GBM_STUDY_SPEC",
    "SDE_CIR_CALL_CASES",
    "SDE_CIR_MEAN_CASES",
    "SDE_NEGATIVITY_CASES",
    "SDE_PRICE_BIAS_CASES",
    "SDE_STRONG_CASES",
    "SDE_WEAK_CALL_CASES",
    "SDE_WEAK_IDENTITY_CASES",
    "CIRSpec",
    "GBMSpec",
    "SDECase",
]


@dataclass(frozen=True)
class GBMSpec:
    """``dS = mu S dt + sigma S dW`` plus the strike the weak study reads it at.

    Read as a market it is ``r = mu``, ``q = 0``: the price row below is the
    discounted call payoff on exactly this process.
    """

    s0: float
    mu: float
    sigma: float
    expiry: float
    strike: float


@dataclass(frozen=True)
class CIRSpec:
    """``dv = kappa (theta - v) dt + xi sqrt(v) dW`` with its study strike."""

    v0: float
    kappa: float
    theta: float
    xi: float
    expiry: float
    strike: float

    @property
    def feller_number(self) -> float:
        """``4 kappa theta / xi^2``: the ncx2 degrees of freedom.

        The Feller condition ``2 kappa theta >= xi^2`` is exactly ``>= 2``
        here, which is why the sampler's parameters and the boundary behaviour
        are the same statement read two ways.
        """
        return 4.0 * self.kappa * self.theta / (self.xi * self.xi)

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_number >= 2.0

    @property
    def params(self) -> dict[str, float]:
        """Keyword arguments for `qpl.engines.mc.sde.cir_sde` and friends."""
        return {"kappa": self.kappa, "theta": self.theta, "xi": self.xi}


GBM_STUDY_SPEC = GBMSpec(s0=100.0, mu=0.2, sigma=0.2, expiry=1.0, strike=100.0)
"""The GBM point every order below is measured at.

``mu = 0.2`` is load-bearing, not decorative. The coupled weak-error estimator's
noise floor is the scheme's own **strong** error, so its signal-to-noise ratio
is proportional to ``C_weak / c_strong``, and to leading order ``C_weak`` is set
by the drift (``S0 e^{mu T} mu^2 T / 2``) while ``c_strong`` is set by the
diffusion (``0.8 sigma^2 S0 e^{(mu + sigma^2/2) T} sqrt(T/2)``). At
``mu = 0.05`` with the same ``sigma`` the Euler weak error is *below* the noise
floor at 100 000 paths and no order can be fitted -- pinned in
`tests/test_sde_convergence.py`. A study that reported an order there would be
reporting noise."""

GBM_LEVELS: tuple[int, ...] = (8, 16, 32, 64, 128)
"""Step counts. Powers of two so every coarse grid is a sub-grid of the finest
one and `coarsen_normals` can sum whole blocks of fine increments."""

GBM_PATHS = 100_000
GBM_SEED = 20250913

GBM_SMALL_DRIFT_MU = 0.05
"""The drift at which the same experiment stops working; see `GBM_STUDY_SPEC`."""

CIR_FELLER_OK = CIRSpec(
    v0=0.04, kappa=2.0, theta=0.04, xi=0.2, expiry=1.0, strike=0.05
)
"""Feller number 8.0. ``v0 = theta`` on purpose: the exact ``E[v_T]`` is then
``theta`` for every ``T``, and the untruncated Euler recursion has the same
fixed point, so any measured bias in the mean is the truncation's alone."""

CIR_FELLER_VIOLATED = CIRSpec(
    v0=0.04, kappa=0.5, theta=0.04, xi=1.0, expiry=1.0, strike=0.05
)
"""Feller number 0.08. Zero is attainable and the scheme meets the boundary on
most paths."""

CIR_LEVELS: tuple[int, ...] = (4, 8, 16, 32)
"""Coarser and shorter than the GBM ladder: the CIR weak error is measured
against a *known mean* rather than a coupled reference (the exact transition is
a noncentral chi-square draw, not a function of the Brownian increment), so the
noise floor is the full ``std(v_T)/sqrt(N)`` and does not shrink with ``h``.
Four levels at 400 000 paths keeps every level above five standard errors where
an order is claimed."""

CIR_PATHS = 400_000
CIR_SEED = 4242

CIR_NEGATIVITY_STEPS = 50
CIR_NEGATIVITY_PATHS = 50_000
CIR_NEGATIVITY_SEED = 11
"""Settings for the negativity rows. Fixed rather than refined: the frequency
of negative states is a property of a scheme *at a step size*, not a limit."""


@dataclass(frozen=True)
class SDECase:
    """One discretisation claim, with the experiment that produces it.

    Parameters
    ----------
    row
        The claim.
    study
        ``'gbm'`` (coupled refinement on the GBM spec), ``'gbm_price'`` (the
        same samples read as a discounted call price), or ``'cir'``.
    scheme, truncation
        As passed to `qpl.engines.mc.sde.simulate`.
    payoff
        What the error is measured in: ``'path'`` (strong error), ``'identity'``,
        ``'call'``, ``'mean'``, ``'price'``, ``'negative_fraction'`` or
        ``'nan_fraction'``.
    regime
        ``'feller_ok'`` / ``'feller_violated'`` for CIR rows, ``''`` otherwise.
    """

    row: BenchmarkRow
    study: str
    scheme: str
    truncation: str
    payoff: str
    regime: str = ""

    @property
    def cir_spec(self) -> CIRSpec:
        if self.regime == "feller_ok":
            return CIR_FELLER_OK
        if self.regime == "feller_violated":
            return CIR_FELLER_VIOLATED
        raise ValueError(f"{self.row.id} is not a CIR case")


_THEORY_SOURCE = (
    "Kloeden & Platen (1992), 'Numerical Solution of Stochastic Differential "
    "Equations', chapters 9-10 (strong and weak order definitions and the "
    "orders of the Euler-Maruyama and Milstein schemes); Glasserman (2003), "
    "'Monte Carlo Methods in Financial Engineering', sections 6.1-6.2. The "
    "measurement design (coarse increments summed from the fine ones) is "
    "Higham (2001), SIAM Review 43(3), section 5. Derived and re-measured "
    "in-repo; no table or code is reproduced from any of these."
)

_MEASURED_SOURCE = (
    "derived in-repo: tests/test_sde_convergence.py. Full truncation is "
    "Andersen (2008), Journal of Computational Finance 11(3), section 3, but "
    "no order is claimed there for a Feller-violating parameter set, so this "
    "row pins the measurement rather than a theory."
)

_NEGATIVE_SOURCE = (
    "derived in-repo: tests/test_mc_sde.py, at CIR_NEGATIVITY_* settings. The "
    "failure mode is the one Andersen (2008) section 3 fixes; the frequencies "
    "are this repository's measurements."
)


# (id, study, scheme, truncation, payoff, regime, expected, tolerance,
#  evidence, source, description, notes)
_ORDER_ROWS: tuple[tuple, ...] = (
    (
        "sde_gbm_euler_strong",
        "gbm",
        "euler",
        "none",
        "path",
        "",
        0.5,
        0.05,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Euler-Maruyama strong order on GBM, E|X^h_T - X_T| on a shared path",
        "Measured 0.5154, log-space residual 0.0062, constant 2.9624; levels "
        "317 to 389 standard errors from zero. The band is set by the gap to "
        "the theoretical 1/2 (0.015, a pre-asymptotic effect of starting at "
        "h = 1/8), not by noise: the seed-to-seed spread over {20250913, 7, "
        "99} is 0.0017.",
    ),
    (
        "sde_gbm_milstein_strong",
        "gbm",
        "milstein",
        "none",
        "path",
        "",
        1.0,
        0.05,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Milstein strong order on GBM, same coupled increments",
        "Measured 0.9932, residual 0.0025, constant 4.2643. The constant is "
        "LARGER than Euler's 2.9624: at h = 1/8 the two schemes are 1.90x "
        "apart and the whole gain is the exponent, reaching 7.12x by h = 1/128.",
    ),
    (
        "sde_gbm_euler_weak_identity",
        "gbm",
        "euler",
        "none",
        "identity",
        "",
        1.0,
        0.08,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Euler weak order on GBM for f(x) = x",
        "Measured 1.0041, residual 0.0087, constant 2.4142 against the exact "
        "leading constant S0 e^{mu T} mu^2 T / 2 = 2.4428. E[X^h_T] is exactly "
        "S0 (1 + mu h)^n, so this error is deterministic and the Monte Carlo "
        "only confirms it.",
    ),
    (
        "sde_gbm_milstein_weak_identity",
        "gbm",
        "milstein",
        "none",
        "identity",
        "",
        1.0,
        0.03,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Milstein weak order on GBM for f(x) = x",
        "Measured 0.9944, residual 0.0017, constant 2.3916 -- the SAME error "
        "as Euler, because the Milstein correction (1/2) b b' (dW^2 - h) has "
        "conditional mean zero and both recursions give E[X_{i+1}] = "
        "(1 + mu h) E[X_i]. This row is why a strong order is not a weak one.",
    ),
    (
        "sde_gbm_euler_weak_call",
        "gbm",
        "euler",
        "none",
        "call",
        "",
        1.0,
        0.08,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Euler weak order on GBM for f(x) = (x - K)^+",
        "Measured 1.0088, residual 0.0109, constant 2.5642, error negative at "
        "every level. Signal-to-noise falls across the ladder (|z| 73, 55, 40, "
        "29, 20) because the coupled estimator's noise floor is Euler's own "
        "strong error; the band is three times the 0.03 seed-to-seed spread "
        "that causes.",
    ),
    (
        "sde_gbm_milstein_weak_call",
        "gbm",
        "milstein",
        "none",
        "call",
        "",
        1.0,
        0.03,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Milstein weak order on GBM for f(x) = (x - K)^+",
        "Measured 0.9946, residual 0.0018, constant 3.1096. |z| is flat at 207 "
        "to 211 across the ladder because p_weak = p_strong = 1 here, which is "
        "what lets the band be 0.03 rather than Euler's 0.08.",
    ),
    (
        "sde_gbm_euler_price_bias",
        "gbm_price",
        "euler",
        "none",
        "price",
        "",
        1.0,
        0.08,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Order of the Euler discretisation bias in a European call price",
        "Measured 1.0088 with constant 2.0994 = e^{-rT} * 2.5642. Bias -0.2560 "
        "at n = 8 (1.30% of the 19.6299 analytic price) to -0.0155 at n = 128, "
        "always negative. Measurable only because the coupled exact payoff is "
        "used as a control variate with its known mean: 3.5e-03 -> 7.8e-04 "
        "standard error against the plain estimator's flat 5.7e-02.",
    ),
    (
        "sde_cir_full_truncation_mean_feller_violated",
        "cir",
        "euler",
        "full",
        "mean",
        "feller_violated",
        0.68,
        0.15,
        EvidenceClass.CONVERGENCE_ORDER,
        _MEASURED_SOURCE,
        "Weak order of full-truncation Euler in E[v_T], Feller condition violated",
        "Measured 0.6834 (residual 0.1261; 0.6906 and 0.7037 at two other "
        "seeds) -- DEGRADED below first order. The bias is negative at every "
        "level and reaches 28% of the true mean at n = 4. The Feller-satisfied "
        "counterpart is deliberately absent: there the bias is under the noise "
        "floor from h = 1/8 on and no order exists to pin.",
    ),
    (
        "sde_cir_full_truncation_call_feller_ok",
        "cir",
        "euler",
        "full",
        "call",
        "feller_ok",
        1.0,
        0.15,
        EvidenceClass.CONVERGENCE_ORDER,
        _THEORY_SOURCE,
        "Weak order of full-truncation Euler in E[(v_T - K)^+], Feller satisfied",
        "Measured 0.9729, residual 0.0705, constant 3.8638e-03. The band is "
        "wide because the fit moves 0.9729 / 1.0404 / 1.0850 over seeds "
        "{4242, 77, 5150}: this budget cannot distinguish 0.97 from 1.09, and "
        "saying so is the honest form of the claim.",
    ),
    (
        "sde_cir_full_truncation_call_feller_violated",
        "cir",
        "euler",
        "full",
        "call",
        "feller_violated",
        0.94,
        0.06,
        EvidenceClass.CONVERGENCE_ORDER,
        _MEASURED_SOURCE,
        "Weak order of full-truncation Euler in E[(v_T - K)^+], Feller violated",
        "Measured 0.9389, residual 0.0606, constant 0.1329 (0.9472 and 0.9407 "
        "at two other seeds, |z| 19 to 118). The narrow band is a real claim "
        "that the order is BELOW one. The damage of violating Feller shows up "
        "in the constant -- 34x the satisfied regime's 3.8638e-03 -- far more "
        "than in the exponent.",
    ),
)

_NEGATIVITY_ROWS: tuple[tuple, ...] = (
    (
        "sde_cir_negative_fraction_feller_ok",
        "negative_fraction",
        "feller_ok",
        2.6e-04,
        1.0e-04,
        "Fraction of Euler paths reaching a negative variance, Feller satisfied",
        "Rare but not absent: 13 paths in 50 000 at 50 steps. 'The Feller "
        "condition holds so the scheme stays positive' is false -- the "
        "condition is about the process, not the discretisation.",
    ),
    (
        "sde_cir_negative_fraction_feller_violated",
        "negative_fraction",
        "feller_violated",
        0.9117,
        0.01,
        "Fraction of Euler paths reaching a negative variance, Feller violated",
        "Identical under truncation='none' and truncation='full', and it must "
        "be: the two schemes are pathwise identical until the first negative "
        "state, and after that plain Euler has no next state at all. Full "
        "truncation keeps the recursion DEFINED, not positive.",
    ),
    (
        "sde_cir_nan_fraction_feller_ok",
        "nan_fraction",
        "feller_ok",
        2.6e-04,
        1.0e-04,
        "Fraction of plain-Euler CIR paths that become NaN, Feller satisfied",
        "Equal to the negative fraction to the resolution of this sample.",
    ),
    (
        "sde_cir_nan_fraction_feller_violated",
        "nan_fraction",
        "feller_violated",
        0.9097,
        0.01,
        "Fraction of plain-Euler CIR paths that become NaN, Feller violated",
        "Slightly BELOW the negative fraction (0.9117) for the only reason it "
        "can be: a path whose first negative state is the terminal one is "
        "never evaluated again. Under full truncation this is exactly 0.",
    ),
)


def _order_case(row: tuple) -> SDECase:
    (
        row_id,
        study,
        scheme,
        truncation,
        payoff,
        regime,
        expected,
        tolerance,
        evidence,
        source,
        description,
        notes,
    ) = row
    return SDECase(
        row=BenchmarkRow(
            id=row_id,
            description=description,
            expected=expected,
            tolerance=tolerance,
            evidence=evidence,
            source=source,
            notes=notes,
        ),
        study=study,
        scheme=scheme,
        truncation=truncation,
        payoff=payoff,
        regime=regime,
    )


_ORDER_CASES: tuple[SDECase, ...] = tuple(_order_case(r) for r in _ORDER_ROWS)

SDE_NEGATIVITY_CASES: tuple[SDECase, ...] = tuple(
    SDECase(
        row=BenchmarkRow(
            id=row_id,
            description=description,
            expected=expected,
            tolerance=tolerance,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source=_NEGATIVE_SOURCE,
            notes=notes,
        ),
        study="cir",
        scheme="euler",
        truncation="none",
        payoff=payoff,
        regime=regime,
    )
    for row_id, payoff, regime, expected, tolerance, description, notes in _NEGATIVITY_ROWS
)


def _select(study: str, payoff: str) -> tuple[SDECase, ...]:
    return tuple(c for c in _ORDER_CASES if c.study == study and c.payoff == payoff)


SDE_STRONG_CASES = _select("gbm", "path")
SDE_WEAK_IDENTITY_CASES = _select("gbm", "identity")
SDE_WEAK_CALL_CASES = _select("gbm", "call")
SDE_PRICE_BIAS_CASES = _select("gbm_price", "price")
SDE_CIR_MEAN_CASES = _select("cir", "mean")
SDE_CIR_CALL_CASES = _select("cir", "call")

ALL_SDE_CASES: tuple[SDECase, ...] = _ORDER_CASES + SDE_NEGATIVITY_CASES
"""Fourteen rows: ten measured convergence orders and four negative findings."""
