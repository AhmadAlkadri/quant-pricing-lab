import os
import re
import subprocess
import sys

import pytest

# (script name, command-line arguments, output keys that must appear).
#
# A script may appear more than once, once per curated invocation: the same
# example run two ways is two smoke cases, not one. The pytest id is built from
# the script name plus the arguments so the two are distinguishable in a
# failure report.
_SMOKE_CASES = [
    ("bs_analytic_greeks.py", [], ["example=bs_analytic_greeks", "price=", "delta=", "gamma="]),
    (
        "mc_pricing_and_stderr.py",
        [],
        ["example=mc_pricing_and_stderr", "analytic_price=", "mc_price=", "mc_stderr=", "z_score="],
    ),
    (
        "pde_theta_scheme.py",
        [],
        ["example=pde_theta_scheme", "analytic_price=", "theta=0.0", "theta=0.5", "theta=1.0"],
    ),
    (
        "tree_convergence.py",
        [],
        [
            "example=tree_convergence",
            "black_scholes=",
            "odd_order=",
            "even_order=",
            "odd_sign=above",
            "even_sign=below",
            "richardson_order=",
        ],
    ),
    (
        # The Leisen-Reimer leg of the same example. `sign=below` and the two
        # ratio keys are the curated part: a run that had lost the scheme would
        # still print an order, but not an order-2 sequence sitting three
        # decimal orders of magnitude inside CRR's.
        "tree_convergence.py",
        ["--scheme", "leisen-reimer"],
        [
            "example=tree_convergence",
            "scheme=leisen-reimer",
            "black_scholes=",
            "order=1.9",
            "sign=below",
            "error_ratio_vs_crr_at_n101=",
            "error_ratio_vs_crr_at_n801=",
            "richardson_order=",
        ],
    ),
    (
        # The smooth case: both time steppings land on order two in all three
        # grid Greeks, which is the baseline the start-up case departs from.
        "pde_greeks_demo.py",
        [],
        [
            "example=pde_greeks_demo",
            "case=smooth",
            "smooth_order theta_gamma=+2.",
            "smooth_order rannacher_gamma=+2.",
            "smooth_order rannacher_delta=+2.",
        ],
    ),
    (
        # The start-up case. `theta_gamma=-1.` is the curated part: a run that
        # had lost the pathology would still print an order, but not a NEGATIVE
        # one, and `rannacher_gamma=+1.8` pins the repair on the same grids.
        "pde_greeks_demo.py",
        ["--case", "startup"],
        [
            "example=pde_greeks_demo",
            "case=startup",
            "startup_order theta_gamma=-1.",
            "startup_order theta_theta=-1.",
            "startup_order rannacher_gamma=+1.8",
            "startup_order rannacher_delta=+2.",
        ],
    ),
    (
        # Three engines on the same American put. The curated keys are the
        # ones a run that had quietly lost a leg would not print:
        # `psor_mean_sweeps` and `lcp_max_complementarity` exist only on the
        # finite-difference leg, `engine_gap=1.3` pins the two deterministic
        # engines' agreement to one figure, and `pde_boundary_first_time=0.0000`
        # next to a non-zero `tree_boundary_first_time` is the grid-versus-
        # lattice difference the example exists to show.
        #
        # Slice 11 adds the simulation leg. `bermudan_gap=1.185e-02` and
        # `lsm_finite_sample_bias=-1.279e-02` are the two corrections that turn
        # "6.066 against 6.090" from a failure into a measurement, and
        # `lsm_z_vs_bermudan=-0.94` is the number that says the simulation
        # agrees with what it is actually estimating. A run that had lost the
        # out-of-sample split would print an in-sample value equal to the
        # out-of-sample one and a z-score with the wrong sign.
        "american_put_cross_method.py",
        [],
        [
            "example=american_put_cross_method",
            "pde_american=6.0888",
            "tree_american=6.0902",
            "engine_gap=1.3",
            "early_exercise_premium=",
            "psor_mean_sweeps=",
            "psor_converged=True",
            "lcp_max_complementarity=",
            "lcp_min_constraint_slack=0.000e+00",
            "lsm_out_of_sample=6.0657",
            "lsm_in_sample=6.0848",
            "lsm_lattice_bermudan=6.0785",
            "bermudan_gap=1.185e-02",
            "lsm_finite_sample_bias=-1.279e-02",
            "lsm_z_vs_bermudan=-0.94",
            "lsm_regressions=49",
            "boundary t=",
            "pde_boundary_first_time=0.0000",
        ],
    ),
    (
        # One digital, four engines. The curated keys are the ones a run that
        # had lost the point of the example would not print: `tree_order
        # leisen-reimer=+1.99` next to `pde_order plain_cn_unaligned=+1.0` is
        # the whole slice in two lines -- the order-2 lattice on a jump and the
        # order-1 grid -- and `sawtooth_sign_changes=2` pins that CRR's error
        # off the money changes sign inside a single window of consecutive odd
        # n. The `mc_greeks` lines pin Slice 10's estimators: the pathwise one
        # is refused because its payoff derivative is identically zero, and the
        # bumped theta's z-score is three orders of magnitude larger than the
        # likelihood-ratio one on the same seed and the same paths.
        "digital_option_cross_method.py",
        [],
        [
            "example=digital_option_cross_method",
            "analytic=0.5323248155",
            "replication_residual=+0.000e+00",
            "tree_order crr=+1.0",
            "tree_order leisen-reimer=+1.99",
            "tree_error_ratio_crr_over_lr_at_n801=4546.4",
            "sawtooth_sign_changes=2",
            "pde_order plain_cn_unaligned=+1.0",
            "pde_order cn_unaligned_projected=+2.0",
            "pde_order rannacher_aligned=+1.9",
            "mc_stderr_order=+0.4999",
            "mc_greeks pathwise=refused_zero_derivative",
            "mc_greeks likelihood_ratio delta estimate=",
            "mc_greeks bump theta estimate=",
            "z=+531.",
        ],
    ),
    (
        # The finite-difference table on its own, which is where the
        # discontinuity pathology and its two repairs sit side by side.
        "digital_option_cross_method.py",
        ["--case", "pde"],
        [
            "case=pde",
            "pde plain_cn_unaligned n=800 price=0.5276356328",
            "pde rannacher_aligned n=800 price=0.5323250384",
            "pde_order plain_cn_unaligned=+1.0",
            "pde_order rannacher_unaligned_projected=+2.0",
        ],
    ),
    (
        # Six estimators on one price at a fixed cost. The curated keys are the
        # ones a run that had lost the point would not print: `paths= 40960
        # normals= 20480` is the equal-cost bookkeeping (antithetic buys two
        # paths per normal, so the comparison is at equal normals), and
        # `predicted_factor antithetic=4.0` next to the measured `seed_factor`
        # column is theory and measurement in the same table.
        "mc_variance_reduction.py",
        [],
        [
            "example=mc_variance_reduction",
            "case=vanilla",
            "analytic=10.4505835722",
            "pilot_rho_antithetic=-0.50",
            "predicted_factor antithetic=4.0",
            "predicted_factor control_variate=6.7",
            "estimator=none",
            "paths= 40960 normals= 20480",
            "estimator=stratified+control",
            "seed_factor=",
        ],
    ),
    (
        # The digital leg, which exists for the stratum scan:
        # `strata_gain_monotone_in_K=False` with `K= 16` beating `K= 20` by
        # 2.9x is the finding -- for a discontinuous payoff the gain is set by
        # where the jump falls inside its stratum, so more strata can mean less
        # gain, and the printed `predicted` column says so before the measured
        # one does.
        "mc_variance_reduction.py",
        ["--case", "digital"],
        [
            "case=digital",
            "analytic=0.5323248155",
            "pilot_rho_antithetic=-0.78",
            "jump_at_u=0.440382",
            "strata_scan K= 16 f=0.0461",
            "strata_scan K= 20 f=0.8076",
            "strata_gain_monotone_in_K=False",
            "strata_gain_K16_over_K20=2.9",
        ],
    ),
    (
        # The Kemna-Vorst control variate on an arithmetic Asian, at two fixing
        # counts and equal normal draws. The curated keys are the ones a run
        # that had lost the point would not print: `pilot_rho=0.999608` next to
        # `predicted_factor=   1276.7` is the whole slice in one line (the
        # control is worth three orders of magnitude because rho is 0.9996,
        # against the 7.6 the terminal spot buys on a vanilla in Slice 7), and
        # `tw_minus_reference=+0.017777` pins the approximation's bias with its
        # sign -- a number that must not quietly become zero.
        "asian_option_control_variate.py",
        [],
        [
            "example=asian_option_control_variate",
            "case=control_variate",
            "fixings= 10",
            "fixings= 52",
            "geometric_closed_form=6.0191160793",
            "geometric_closed_form=5.6374316204",
            "pilot_rho=0.999608",
            "predicted_factor=   1276.7",
            "estimator=none",
            "estimator=antithetic+control",
            "paths=  4160 normals= 41600",
            "paths=   800 normals= 41600",
            "turnbull_wakeman=6.2523156852",
            "tw_minus_reference=+0.017777",
            "arithmetic_minus_geometric=+0.215",
        ],
    ),
    (
        # The fixing-refinement leg, which exists for one number:
        # `fixing_order=+1.0002`. A right-endpoint fixing convention approaches
        # the continuous-averaging limit at order ONE, and
        # `errors_halve_at_each_level=True` is the assumption-free form of that.
        "asian_option_control_variate.py",
        ["--case", "fixings"],
        [
            "case=fixings",
            "continuous_limit=5.5468186338",
            "fixings=   20 geometric=5.7826160816",
            "fixings= 2560 geometric=5.5486582749",
            "fixing_order=+1.0002",
            "errors_halve_at_each_level=True",
        ],
    ),
    (
        # Euler against Milstein on one GBM, at five step counts, with the
        # exact sampler on the SAME Brownian path as both. The curated keys
        # are the ones a run that had lost the coupling or the point would not
        # print: `strong_order euler=+0.51` next to `milstein=+0.99` is the
        # slice in two lines, and `weak_call_z_decay euler=0.276` against
        # `milstein=0.981` is the finding -- the coupled weak estimator's
        # noise floor is the scheme's own STRONG error, so Euler's
        # signal-to-noise falls by 3.6x down the same ladder on which
        # Milstein's is flat.
        "sde_convergence.py",
        [],
        [
            "example=sde_convergence",
            "case=gbm",
            "analytic_call=19.6298871",
            "strong_order euler=+0.51",
            "strong_order milstein=+0.99",
            "weak_identity_order euler=+0.99",
            "weak_identity_order milstein=+0.99",
            "weak_call_order euler=+1.00",
            "weak_call_order milstein=+0.99",
            "weak_call_z_decay euler=0.27",
            "weak_call_z_decay milstein=0.98",
            "weak_identity_closed_form_constant=2.4428",
            "price_bias_order=+1.00",
            "bias_pct=-1.29",
        ],
    ),
    (
        # The square-root diffusion, where Euler meets a boundary it cannot
        # cross. `plain_negative=0.911` equal to `full_negative=0.911` with
        # `full_nan=0.00000` is the whole CIR finding: full truncation keeps
        # the recursion defined and changes the negative frequency not at all.
        "sde_convergence.py",
        ["--case", "cir"],
        [
            "case=cir",
            "feller_satisfied=True",
            "feller_satisfied=False",
            "negativity feller_ok feller_number=8.00 plain_negative=0.00026",
            "negativity feller_violated feller_number=0.08 plain_negative=0.911",
            "full_negative=0.911",
            "plain_nan=0.909",
            "full_nan=0.00000",
            "cir_order feller_ok call=+1.0",
            "cir_order feller_violated call=+0.96",
            "cir_order feller_violated mean=+0.64",
        ],
    ),
    (
        # Slice 10's three Greek estimators on the same two payoffs. The
        # curated keys are the ones that would survive a run which had lost
        # the point: `pathwise_lr_mixed` next to `call pathwise gamma` pins
        # that a pathwise result does NOT carry a pathwise gamma; the digital
        # `refused=payoff_derivative_is_zero_ae` line pins the one estimator
        # that is refused and why; and the two spread ratios are the slice's
        # headline read in both directions -- the likelihood ratio costs 5.03
        # in variance on a smooth delta and buys 983.07 on a discontinuous one.
        "mc_greeks_estimators.py",
        [],
        [
            "example=mc_greeks_estimators",
            "source=pathwise_lr_mixed",
            "digital pathwise refused=payoff_derivative_is_zero_ae",
            "call_spread delta likelihood_ratio sd=9.4740e-03 variance_ratio=5.03",
            "call_spread gamma bump sd=7.3400e-03 variance_ratio=805.52",
            "digital_spread delta bump sd=6.3260e-03 variance_ratio=983.07",
        ],
    ),
    (
        # The bump-size study, where the trade-off has an interior optimum.
        # `bump_h_optimal=3` is the curated part: it is 300x the package
        # default, and a run that had lost the variance branch would report
        # the smallest h instead.
        "mc_greeks_estimators.py",
        ["--case", "h"],
        [
            "case=h",
            "bump_h_sd_order=-0.4263",
            "bump_h_bias_order=+1.9979",
            "bump_h_optimal=3",
        ],
    ),
    (
        # One barrier, two discretisations, one half-order. The curated keys
        # are the ones a run that had lost the point would not print:
        # `plain_bias_order=+0.51` next to a plain bias of +1.17 at m = 25 is
        # the monitoring bias and its order in one line; `single_observation
        # bridge=` with `plain=7.849428` is the slice's most surprising result
        # -- at ONE observation date the Brownian-bridge estimator already
        # prices the continuous barrier while the plain estimator returns the
        # vanilla; and `bgk_bias=+0.070633` against `plain_bias=+1.171070` at
        # the same m is the continuity correction's size.
        "barrier_option_monitoring_bias.py",
        [],
        [
            "example=barrier_option_monitoring_bias",
            "case=monitoring",
            "analytic_continuous=4.5125986078",
            "bgk_beta=0.5825971579",
            "monitoring m=  25 plain=5.6836",
            "plain_bias=+1.1710",
            "bgk_bias=+0.0706",
            "bgk_closed_form=5.619546",
            "effective_barrier=96.977095",
            "monitoring m= 400",
            "plain_bias_order=+0.51",
            "single_observation bridge=4.5198",
            "plain=7.849428",
        ],
    ),
    (
        # The lattice leg. `sawtooth_order crr=+0.45` next to
        # `sawtooth_order leisen-reimer=+0.45` is the whole negative finding --
        # half an order for both schemes -- and `lr_over_crr=0.97` pins that
        # Leisen-Reimer buys 2% of the amplitude where it buys two decimal
        # orders on a vanilla. `err_times_n=` on the Boyle-Lau rows is the
        # bounded-but-erratic constant.
        "barrier_option_monitoring_bias.py",
        ["--case", "sawtooth"],
        [
            "case=sawtooth",
            "window n0=  100 period=  70",
            "crr_amplitude=0.93510",
            "lr_amplitude=0.91326",
            "lr_over_crr=0.97",
            "sawtooth_order crr=+0.45",
            "sawtooth_order leisen-reimer=+0.45",
            "boyle_lau k=  7 n=   582 err=-6.38024e-05",
            "err_times_n=",
            "misalignment=",
            "nearby_unaligned_err=+3.19468e-01",
        ],
    ),
    (
        # The barrier on a grid. The curated keys are the ones a run that had
        # lost the point would not print: `grid_order uniform_off_node=+0.70`
        # next to `+2.06` and `+1.99` is the whole slice in three lines -- the
        # same contract, the same scheme, and a full order of difference that
        # comes from nothing but where the nodes are -- and
        # `effective_barrier=93.87` at `n=100` on the off-node row is the
        # mechanism, a barrier displaced by more than a whole spacing.
        "barrier_pde_grid.py",
        [],
        [
            "example=barrier_pde_grid",
            "case=grid",
            "analytic_continuous=4.5125986078",
            "grid_order uniform_on_node=+2.06",
            "grid_order uniform_off_node=+0.70",
            "grid_order sinh_on_node=+1.99",
            "effective_barrier=93.87",
            "off_node_over_on_node=182.1",
            "uniform_over_sinh=18.0 13.0",
        ],
    ),
    (
        # The concentration scan, which exists for one contrast:
        # `barrier_gain= 22.1` against `vanilla_gain=  0.1` in the same row.
        # A mesh that concentrates at the barrier is not free accuracy, and the
        # example says so in the same table rather than in a footnote.
        "barrier_pde_grid.py",
        ["--case", "mesh"],
        [
            "case=mesh",
            "barrier_gain= 22.1",
            "vanilla_gain=  0.1",
            "default_concentration=0.05",
            "mesh_reading=",
        ],
    ),
    (
        # Discrete monitoring. `discrete_gap_order=+0.44` next to
        # `bgk_gap_order=+0.43` is the finding that the deficit from one half
        # belongs to the continuity correction and not to the grid;
        # `gap_over_bgk=0.99` is the correction predicting this engine's own
        # gap; and `relative=` is in-out parity on the discrete system (round-off digits vary by platform).
        "barrier_pde_grid.py",
        ["--case", "discrete"],
        [
            "case=discrete",
            "discrete m= 160",
            "projections=160",
            "gap_over_bgk=0.99",
            "discrete_gap_order=+0.44",
            "bgk_gap_order=+0.43",
            "parity n=  100",
            "relative=",  # round-off residual; its digits are platform-dependent
        ],
    ),
    (
        "american_put_binomial_dp.py",
        [],
        [
            "example=american_put_binomial_dp",
            "american_put=",
            "european_put=",
            "early_exercise_premium=",
            "early_exercise_nodes=",
            "boundary_first_time=",
            "boundary t=",
        ],
    ),
    ("quadrature_demo.py", [], ["example=quadrature_demo", "exact=", "trap=", "simpson=", "gauss="]),
    (
        "laplace_inversion_demo.py",
        [],
        ["example=laplace_inversion_demo", "max_abs_error=", "mean_abs_error="],
    ),
    (
        "copula_gaussian_demo.py",
        [],
        ["example=copula_gaussian_demo", "tau_emp=", "spearman_emp=", "coexceedance_corr=", "coexceedance_ind="],
    ),
    (
        # Slice 14: the four transform methods side by side. The curated keys
        # are the ones a run that had lost the point would not print.
        # `abs_error=6.4e-04` next to three methods at the floating-point floor
        # is the whole FFT finding in one line, and it is deliberately the only
        # error digit pinned here -- the other three are round-off and their
        # leading digit is not a cross-platform constant. `fft_on_grid` and
        # `fft_off_grid` on one line carry the `--case fft` headline into this
        # invocation, so that study needs no curated run of its own.
        "fourier_methods_bs.py",
        [],
        [
            "example=fourier_methods_bs",
            "case=methods",
            "analytic=9.8262977827",
            "method=cos",
            "method=carr_madan_fft",
            "abs_error=6.4e-04",
            "method=lewis",
            "method=gil_pelaez",
            "carr_madan_fft_minus_quadrature=+6.447e-04",
            "fft_on_grid=2.1e-07 fft_off_grid=6.4e-04",
        ],
    ),
    (
        # The COS leg. `cos_decay_ratio=1.1` is the curated part: the measured
        # decay of log10(error) in N^2 against the derived -pi^2/(8 L^2 ln 10),
        # so a run that had lost the cumulant-based range would still print an
        # error column but not a ratio near one. `cos_range_L4_error_flat_in_N`
        # pins that a too-narrow range is an error no term count removes.
        "fourier_methods_bs.py",
        ["--case", "cos"],
        [
            "case=cos",
            "cos_error N=  16 err=5.2e-02",
            "cos_error N=  32 err=1.3e-06",
            "cos_decay_ratio=1.1",
            "cos_range L=  4.0 err=6.2e-04",
            "cos_range L=  6.0 err=2.0e-08",
            "cos_range_L4_error_flat_in_N=0.0e+00",
        ],
    ),
    (
        # The damping sweep. The curated cells are the two *failures*, which
        # are both outside the range the slice statement predicted would fail:
        # `a=0.25 1e-04` at the small end and `alpha=40.00 err=3.6e-03` at the
        # large one, with `S/K=1.667` printed next to the row that fails first.
        "fourier_methods_bs.py",
        ["--case", "alpha"],
        [
            "case=alpha",
            "S/K=1.667",
            "a=0.25 1e-04",
            "alpha= 0.10 err=4.7e-01",
            "alpha= 1.50 err=",  # machine-precision error; digits are platform-dependent
            "alpha=40.00 err=",  # cancellation-dominated; digits are platform-dependent
            "cancellation_bound=",
        ],
    ),
    (
        # Slice 15: Heston. The curated keys are the ones a run that had lost
        # the point would not print. `call=  16.070155 put=  17.055271` is the
        # published reference row reproduced by the COS engine; `skew_signs`
        # printing `---` and then `+++` when rho flips is the whole smile
        # claim in two lines; and `parity=` is deliberately keyless on its
        # digits -- the residual is a truncation number, not round-off, and
        # the study that pins its size lives in `--case cos`.
        "heston_smile.py",
        [],
        [
            "example=heston_smile",
            "case=smile",
            "feller_number=4.00 feller_satisfied=True log_multiplier=2.00",
            "call=  16.070155 put=  17.055271",
            "implied_vol=0.4244",
            "skew=-0.0942",
            "skew_signs=--- skew_flattens=True",
            "rho=+0.5 skew_signs=+++",
            "parity=",  # truncation residual; its digits are studied in --case cos
        ],
    ),
    (
        # The range study. `widen_L_by=2.80 suggested_L= 28.0` is the derived
        # repair for `c4 = 0`, and the flat `9.1e-04` column next to it is
        # what identifies a COS error that ignores the term count as a RANGE
        # error. `cos_call_window_empty=True` is the slice's hardest finding.
        "heston_smile.py",
        ["--case", "cos"],
        [
            "case=cos",
            "widen_L_by=1.43 suggested_L= 14.3",
            "widen_L_by=2.80 suggested_L= 28.0",
            "feller_violated  L= 10.0 N=  512:9.1e-04",
            "N= 4096:1.2e-10",
            "L=  8.0 call_err=7.2e+00 put_err=2.7e-02",
            "cos_call_window_empty=True cos_put_then_parity=recommended",
        ],
    ),
    (
        # The little Heston trap. `rel=8.6e-01` at T = 2 on a model that
        # differs from the reference one in `theta` alone, next to
        # `trap_invisible_on_reference_set=True`, is the finding: the branch
        # jump multiplies the transform by exp(-4 pi i kappa theta / xi^2),
        # which is 1 when 2 kappa theta / xi^2 is an integer -- and it is, on
        # the reference set, exactly.
        "heston_smile.py",
        ["--case", "trap"],
        [
            "case=trap",
            "model=trap 2*kappa*theta/xi^2=2.50",
            "T=  2.0 stable=  26.180405",
            "rel=8.6e-01",
            "first_break_u=19.4150",
            "u_times_T= 5.100",
            "trap_invisible_on_reference_set=True reason=integer_log_multiplier",
        ],
    ),
    (
        # Carr-Madan against the moment-explosion bound.
        # `alpha= 10.60 explosion_time=   1.0233` marked ok next to
        # `alpha= 10.65 explosion_time=   1.0103` marked BROKEN is the bound
        # and the measurement in two lines -- and it is the constraint Slice
        # 14 predicted and could not produce under Black-Scholes.
        "heston_smile.py",
        ["--case", "alpha"],
        [
            "case=alpha",
            "critical_moment=11.6905 alpha_max=10.6905",
            "alpha= 10.60 explosion_time=   1.0233",
            "alpha= 10.65 explosion_time=   1.0103",
            "BROKEN",
            "alpha_failure_is_moment_explosion=True",
        ],
    ),
    (
        # Slice 16: Heston by simulation. The curated keys are the ones a run
        # that had lost the point would not print. `resolved=1` next to a `z`
        # column is the honest half of the bias table -- only the coarse levels
        # clear their own noise at the default path count, and the example says
        # so per row rather than fitting a slope through five points of which
        # three are noise. `qe_bias_far_smaller_than_euler_on_reference_set=
        # False` is the slice statement's expectation being contradicted in one
        # line, and `vr_control_and_conditioning_are_substitutes=True` is the
        # variance-reduction finding: the control variate is worth 0.80 in
        # correlation on the plain estimator and 0.49 on the conditional one,
        # because conditioning already integrated out what it was correlated
        # with. Digits are kept to truncation level throughout.
        "heston_mc_qe.py",
        [],
        [
            "example=heston_mc_qe",
            "case=bias",
            "feller_number=4.00 feller_satisfied=True",
            "estimator=antithetic+conditional",
            "reference_transform_price=16.07015",
            "resolved=1",
            "bias_ratio_1_4_over_1_8=",
            "reference_euler_over_qe_at_dt_quarter=",
            "qe_bias_far_smaller_than_euler_on_reference_set=False",
            "uncorrected=+1.0",
            "martingale_correction_removes_the_defect=True",
            "vr_factor conditional=",
            "vr_control_rho plain=0.7",
            "vr_control_and_conditioning_are_substitutes=True",
        ],
    ),
    (
        # The Feller-violating leg, which is where the two schemes actually
        # separate. `feller_violated_euler_relative_error_at_dt_quarter=0.9`
        # is the headline -- full-truncation Euler is 95% wrong on a price of
        # 3.59 -- and `resolved=5 order=+0.9` next to it is the one fully
        # resolved convergence ladder in this example, because an error that
        # large clears its noise at every level. `qe_negative=0.00000` beside
        # `euler_negative=0.54` is the mechanism in one line.
        "heston_mc_qe.py",
        ["--case", "feller"],
        [
            "case=feller",
            "feller_number=0.08 feller_satisfied=False",
            "reference_used=lewis",
            "feller_violated_euler_over_qe_at_dt_quarter=",
            "feller_violated_euler_relative_error_at_dt_quarter=0.9",
            "resolved=5 order=+0.9",
            "qe_negative=0.00000 euler_negative=0.54",
            "qe_variance_non_negative_by_construction=True",
        ],
    ),
    (
        # The Chapter 6 rules on a pricing integral. `simpson_identity` printing
        # two equal columns is the mechanism, and
        # `simpson_over_trapezoid_at_n128=7.7e+03` is the consequence: the
        # nominally order-4 rule is four decimal orders worse than the
        # nominally order-2 one at the same evaluation count.
        "fourier_methods_bs.py",
        ["--case", "quadrature"],
        [
            "case=quadrature",
            "rule=trapezoid",
            "n=128 2e-07",
            "rule=simpson",
            "n=256 6e-08",
            "rule=gauss_legendre",
            "simpson_identity n= 256 simpson_err=-6.117e-08",
            "minus_third_of_trapezoid_at_half=-6.117e-08",
            "simpson_over_trapezoid_at_n128=7.7e+03",
        ],
    ),
]

_SMOKE_IDS = [
    name if not args else f"{name}{''.join(' ' + a for a in args)}"
    for name, args, _ in _SMOKE_CASES
]


def _run(script_name: str, script_args: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(repo_root, "examples", script_name)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    return subprocess.run(
        [sys.executable, script_path, *script_args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("script_name", "script_args", "required_keys"), _SMOKE_CASES, ids=_SMOKE_IDS
)
def test_public_examples_smoke_and_determinism(
    script_name: str, script_args: list[str], required_keys: list[str]
) -> None:
    """Curated keys and byte-identical repetition, from the **same two runs**.

    This used to be two parametrised tests, one asserting the keys on a single
    run and one comparing two more runs to each other: three subprocess
    invocations of every example, of which the first was thrown away after one
    `in` check. The determinism claim does not need its own pair -- a second
    run compared against the keyed one says exactly as much -- so the third
    invocation was pure cost. Merging them cut the example harness from 72
    invocations to 48 and this file's runtime from 42.2 s to 28.4 s (measured,
    same machine, `pytest -q -p no:randomly tests/test_examples_smoke.py`),
    about 9% of the whole suite. Nothing an example prints changed.

    Both runs are still checked for a zero exit status, so a script that fails
    only on a second invocation (a stray cache write, say) is still caught.
    """
    first = _run(script_name, script_args)
    assert first.returncode == 0, first.stderr

    stdout = first.stdout
    for key in required_keys:
        assert key in stdout

    if "mc_stderr=" in stdout:
        match = re.search(r"mc_stderr=([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)", stdout)
        assert match is not None
        stderr_val = float(match.group(1))
        assert stderr_val >= 0.0

    second = _run(script_name, script_args)
    assert second.returncode == 0, second.stderr
    assert second.stdout == stdout
