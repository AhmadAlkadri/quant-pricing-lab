# Strike alignment in the finite-difference grid

## Why a kink on a node hurts

The terminal condition for a vanilla option is continuous but not smooth: at
`S = K` the first derivative jumps and the second derivative is a point mass.
A central finite-difference stencil is second-order accurate only where the
Taylor expansion that justifies it is valid, so near the strike the local
truncation error is far larger than the formal `O(ds^2)`. Diffusion spreads
that local damage over the whole solution, and what survives is a degraded
*error constant*: the scheme still converges, but from a worse starting point,
and the constant depends on where the kink sits relative to the nodes.

The worst placement is the kink exactly on a node. That node then carries the
full one-sided discrepancy between the payoff and any smooth interpolant of
it, and both stencils that touch it inherit the error. If the kink instead
falls midway between two nodes, the two neighbouring payoff values are
symmetric about it and the leading payoff-induced term cancels between them.

This is the cheapest of the standard non-smooth-payoff remedies. See Pooley,
Forsyth and Vetzal (2003), *Convergence remedies for non-smooth payoffs in
option pricing*, Journal of Computational Finance 6(4), and the
grid-construction discussion in Tavella and Randall (2000), *Pricing Financial
Instruments: The Finite Difference Method*.

## What `strike_alignment="midpoint"` does

From the nominal spacing `ds0 = s_max / n_s`, the strike sits `K / ds0`
spacings from the origin. Take the nearest half-integer position,
`j + 1/2` with `j = max(round(K / ds0 - 1/2), 0)`, and solve `K = (j + 1/2) ds`
for the spacing: `ds = K / (j + 1/2)`, `s_max = ds * n_s`. The node count
`n_s` is preserved and `s_max` absorbs the adjustment, moving by a factor of
at most `0.5 / (j + 1/2)`. Default is `"none"`, which leaves existing output
bit-identical.

## Measured effect

Case: `S = K = 100`, `r = 5%`, `q = 0`, `sigma = 20%`, `T = 1`, European call.
Crank-Nicolson (`theta = 0.5`), `n_s = n_t = n`, `s_max_multiplier = 4.0`.
Error is the absolute difference from the closed form, `10.450583572185565`.

| n | unaligned error | aligned error | aligned `s_max` |
|---:|---:|---:|---:|
| 50 | 7.608e-03 | 7.608e-03 | 400.000 |
| 100 | 3.979e-02 | 2.034e-03 | 408.163 |
| 200 | 9.892e-03 | 4.820e-04 | 396.040 |
| 400 | 2.470e-03 | 1.219e-04 | 398.010 |
| 800 | 6.172e-04 | 3.065e-05 | 399.002 |

Least-squares slope of `log(error)` on `log(1/n)`:

| grid | fitted order | log-space RMS residual |
|---|---:|---:|
| unaligned | 1.126 | 0.860 |
| aligned | 1.997 | 0.022 |

The unaligned sequence is not merely noisy, it is non-monotone: refining from
`n = 50` to `n = 100` makes the error five times *worse*. At `n = 50` the
spacing is 8 and `K = 100` already falls halfway between nodes 12 and 13; at
`n = 100` the spacing is 4 and `K` lands exactly on node 25. The residual of
0.86 is the fit reporting that a single power law does not describe that data.
The aligned sequence recovers the formal second order of the scheme, and the
`n = 50` row is identical between the two columns because that grid was
already aligned by accident.

Time order, isolated: `theta = 1.0` (fully implicit), `n_s = 800` aligned and
held fixed, refining only `n_t`.

| n_t | error |
|---:|---:|
| 25 | 4.190e-02 |
| 50 | 2.099e-02 |
| 100 | 1.052e-02 |
| 200 | 5.276e-03 |

Fitted order 0.997, residual 3e-04. The spatial error on that grid is a floor
the refinement cannot cross; it was measured at 1.95e-04 by driving `n_t` to
6400 on the same grid, which is a factor of 27 below the smallest error in the
fit, so the slope is the temporal order and not the floor.

## What this does not fix

Crank-Nicolson damps high-frequency modes only marginally, so a non-smooth
terminal condition still produces oscillations in the Greeks near the strike
even on an aligned grid. The prices above are fine; Delta and especially Gamma
are not the same question.

Slice 4 implemented the first remedy, `PDEConfig(time_stepping="rannacher")`,
and measured what alignment does and does not buy: on `T = 0.05`,
`n_s = 80 n_t`, alignment improves the plain-Crank-Nicolson gamma error by a
factor of about nine but leaves its fitted order *negative* (-0.915 aligned
against -1.043 unaligned) - refinement still makes it worse. Rannacher restores
order two on both grids (1.888 aligned, 1.933 unaligned). Alignment fixes where
the kink sits; it does nothing about what the scheme does to it. Full tables:
`docs/notes/pde_greeks_and_rannacher.md`.

The second remedy, a non-uniform grid concentrated at the strike, is still not
implemented; it remains a Phase 2 item.

## Reproducing

`tests/test_pde_ch4.py` pins all three findings:
`test_pde_cn_unaligned_strike_pathology_negative_finding`,
`test_pde_cn_aligned_strike_order_two`, and
`test_pde_implicit_euler_first_order_in_time`. The fitting is
`qpl.validation.fit_convergence_order`.
