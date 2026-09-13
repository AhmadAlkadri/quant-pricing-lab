"""Four Fourier pricing methods on one Black-Scholes call, and what each costs.

The point of running transform methods against a model with a closed form is
that the error against that closed form is the *method's own* error with
nothing else mixed in. Five cases:

    --case methods     all four methods at the package defaults, side by side
    --case cos         COS error against the term count and the range L
    --case alpha       the Carr-Madan damping parameter, at three moneyness
    --case fft         the FFT grid: on a node, between two nodes, and eta
    --case quadrature  the Chapter 6 rules driving the Carr-Madan integral

Everything printed is deterministic: no sampling anywhere in this file.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from qpl.engines.fourier import FourierConfig, carr_madan_fft, characteristic_function_model
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
DIVIDEND = 0.01
SIGMA = 0.20

MONEYNESS_STRIKES = (60.0, 100.0, 160.0)
"""Deep in the money, at the money, out of the money -- the axis the Carr-Madan
damping study turns out to depend on."""


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _option(strike: float = STRIKE) -> EuropeanOption:
    return EuropeanOption(kind="call", strike=strike, expiry=EXPIRY)


def _analytic(strike: float = STRIKE) -> float:
    return price(_option(strike), BlackScholesModel(sigma=SIGMA), _market()).value


def _fourier(strike: float = STRIKE, **kwargs) -> float:
    return price(
        _option(strike),
        BlackScholesModel(sigma=SIGMA),
        _market(),
        method="fourier",
        cfg=FourierConfig(**kwargs),
    ).value


def _header(case: str) -> None:
    print("example=fourier_methods_bs")
    print(f"case={case}")
    print(
        f"spec S={SPOT:g} K={STRIKE:g} T={EXPIRY:g} r={RATE:g} q={DIVIDEND:g} "
        f"sigma={SIGMA:g}"
    )
    print(f"analytic={_analytic():.10f}")


def case_methods() -> None:
    _header("methods")
    exact = _analytic()
    rows = (
        ("cos", "N=256,L=10", {"method": "cos"}),
        (
            "carr_madan_fft",
            "N=4096,eta=0.25",
            {"method": "carr_madan", "carr_madan_transform": "fft"},
        ),
        (
            "carr_madan_quad",
            "trapezoid,n=512",
            {"method": "carr_madan", "carr_madan_transform": "quadrature"},
        ),
        ("lewis", "quad,tol=1e-10", {"method": "lewis"}),
        ("gil_pelaez", "quad,tol=1e-10", {"method": "gil_pelaez"}),
    )
    for name, setting, kwargs in rows:
        value = _fourier(**kwargs)
        print(
            f"method={name:16s} setting={setting:16s} price={value:.10f} "
            f"abs_error={abs(value - exact):.1e}"
        )
    # The one line that says which of the five is not like the others: the FFT
    # variant's number is an interpolation between two log-strike nodes, and the
    # gap is the interpolation, not the integration.
    fft = _fourier(method="carr_madan", carr_madan_transform="fft")
    quad = _fourier(method="carr_madan", carr_madan_transform="quadrature")
    print(f"carr_madan_fft_minus_quadrature={fft - quad:+.3e}")
    on_grid, off_grid = _fft_on_and_off_grid(0.25)
    print(f"fft_on_grid={on_grid:.1e} fft_off_grid={off_grid:.1e}")


COS_TERMS = (8, 12, 16, 20, 24, 28, 32, 48, 64)
COS_RANGES = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 20.0)


def case_cos() -> None:
    _header("cos")
    exact = _analytic()
    for terms in COS_TERMS:
        err = abs(_fourier(method="cos", n_terms=terms) - exact)
        print(f"cos_error N={terms:4d} err={err:.1e}")
    # The decay is Gaussian in N, not merely exponential: the k-th cosine
    # coefficient of a Gaussian density on a range of width 2 L sigma sqrt(T)
    # is damped by exp(-k^2 pi^2 / (8 L^2)), which has no sigma and no T in it.
    fit_terms = [float(n) for n in COS_TERMS if 1e-9 < abs(_fourier(method="cos", n_terms=n) - exact) < 1.0]
    logs = [math.log10(abs(_fourier(method="cos", n_terms=int(n)) - exact)) for n in fit_terms]
    slope, _ = np.polyfit(np.array(fit_terms) ** 2, np.array(logs), 1)
    predicted = -(math.pi**2) / (8.0 * 10.0**2) / math.log(10.0)
    print(f"cos_decay_per_N2_measured={slope:+.6f} predicted={predicted:+.6f}")
    print(f"cos_decay_ratio={slope / predicted:.3f}")
    for level in COS_RANGES:
        err = abs(_fourier(method="cos", n_terms=256, truncation_l=level) - exact)
        print(f"cos_range L={level:5.1f} err={err:.1e}")
    # Too narrow a range is an error the term count cannot remove.
    at_l4 = [_fourier(method="cos", n_terms=n, truncation_l=4.0) for n in (32, 256, 1024)]
    print(f"cos_range_L4_error_flat_in_N={max(at_l4) - min(at_l4):.1e}")


ALPHAS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0, 20.0, 30.0, 40.0)


def case_alpha() -> None:
    _header("alpha")
    for strike in MONEYNESS_STRIKES:
        exact = _analytic(strike)
        cells = []
        for alpha in ALPHAS:
            value = _fourier(
                strike,
                method="carr_madan",
                carr_madan_transform="quadrature",
                alpha=alpha,
            )
            cells.append(f"a={alpha:<5g}{abs(value - exact):.0e}")
        print(f"alpha_sweep K={strike:6.1f} S/K={SPOT / strike:.3f} " + " ".join(cells))
    exact = _analytic()
    for alpha in (0.1, 1.5, 40.0):
        value = _fourier(
            method="carr_madan", carr_madan_transform="quadrature", alpha=alpha
        )
        print(f"alpha={alpha:5.2f} err={abs(value - exact):.1e}")
    # The large-alpha bound: the price is a tiny prefactor times an enormous
    # integral, and the relative round-off of that product is what fails.
    for alpha in (20.0, 40.0):
        bound = (
            (SPOT / STRIKE) ** alpha
            * math.exp(0.5 * alpha**2 * SIGMA**2 * EXPIRY)
            * SPOT
            / exact
            * float(np.finfo(float).eps)
        )
        print(f"alpha={alpha:5.2f} cancellation_bound={bound:.1e}")


ETAS = (0.05, 0.10, 0.25, 0.50)


def _fft_on_and_off_grid(eta: float) -> tuple[float, float]:
    """|error| at a strike that IS a grid node, and at the one asked for.

    Same FFT output read two ways; the gap is the linear interpolation between
    two log-strike nodes and nothing else.
    """
    cf = characteristic_function_model(BlackScholesModel(sigma=SIGMA))
    grid = carr_madan_fft(
        cf,
        s0=SPOT,
        expiry=EXPIRY,
        rate=RATE,
        dividend=DIVIDEND,
        alpha=1.5,
        n_grid=4096,
        eta=eta,
    )
    node = int(np.argmin(np.abs(grid.log_strikes - math.log(STRIKE))))
    on_grid = abs(grid.values[node] - _analytic(math.exp(grid.log_strikes[node])))
    off_grid = abs(
        _fourier(method="carr_madan", carr_madan_transform="fft", eta=eta) - _analytic()
    )
    return on_grid, off_grid


def case_fft() -> None:
    _header("fft")
    cf = characteristic_function_model(BlackScholesModel(sigma=SIGMA))
    for eta in ETAS:
        spacing = 2.0 * math.pi / (4096 * eta)
        on_grid, off_grid = _fft_on_and_off_grid(eta)
        print(
            f"fft eta={eta:5.2f} lambda={spacing:.6f} on_grid={on_grid:.1e} "
            f"off_grid={off_grid:.1e}"
        )
    for eta in (0.25, 0.50):
        cells = []
        for weights in ("simpson", "trapezoid"):
            grid = carr_madan_fft(
                cf,
                s0=SPOT,
                expiry=EXPIRY,
                rate=RATE,
                dividend=DIVIDEND,
                alpha=1.5,
                n_grid=4096,
                eta=eta,
                weights=weights,
            )
            node = int(np.argmin(np.abs(grid.log_strikes - math.log(STRIKE))))
            cells.append(abs(grid.values[node] - _analytic(math.exp(grid.log_strikes[node]))))
        print(
            f"fft_weights eta={eta:5.2f} simpson={cells[0]:.1e} "
            f"trapezoid={cells[1]:.1e} ratio={cells[0] / cells[1]:.1e}"
        )


RULE_NODES = (8, 16, 32, 64, 128, 256)


def case_quadrature() -> None:
    _header("quadrature")
    exact = _analytic()
    for rule in ("trapezoid", "simpson", "gauss_legendre"):
        cells = []
        for nodes in RULE_NODES:
            value = _fourier(
                method="carr_madan",
                carr_madan_transform="quadrature",
                quadrature=rule,
                n_quad=nodes,
            )
            cells.append(f"n={nodes:<4d}{abs(value - exact):.0e}")
        print(f"rule={rule:15s} " + " ".join(cells))
    # Composite Simpson at n intervals IS (4 T_n - T_{n/2}) / 3. When the
    # trapezoid rule is already spectrally accurate, that combination throws
    # away the accurate term and keeps a third of the inaccurate one.
    for nodes in (128, 256):
        simpson = _fourier(
            method="carr_madan",
            carr_madan_transform="quadrature",
            quadrature="simpson",
            n_quad=nodes,
        )
        coarse = _fourier(
            method="carr_madan",
            carr_madan_transform="quadrature",
            quadrature="trapezoid",
            n_quad=nodes // 2,
        )
        print(
            f"simpson_identity n={nodes:4d} simpson_err={simpson - exact:+.3e} "
            f"minus_third_of_trapezoid_at_half={-(coarse - exact) / 3.0:+.3e}"
        )
    # The headline is taken at n = 128, where both rules are still well above
    # the floating-point floor, so the ratio is a property of the rules and not
    # of the last few bits of a converged answer.
    fine = abs(
        _fourier(
            method="carr_madan",
            carr_madan_transform="quadrature",
            quadrature="trapezoid",
            n_quad=128,
        )
        - exact
    )
    simpson = abs(
        _fourier(
            method="carr_madan",
            carr_madan_transform="quadrature",
            quadrature="simpson",
            n_quad=128,
        )
        - exact
    )
    print(f"simpson_over_trapezoid_at_n128={simpson / fine:.1e}")


CASES = {
    "methods": case_methods,
    "cos": case_cos,
    "alpha": case_alpha,
    "fft": case_fft,
    "quadrature": case_quadrature,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="methods")
    CASES[parser.parse_args().case]()


if __name__ == "__main__":
    main()
