from __future__ import annotations

from qpl.engines.pde.pricers import PDEConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price


def main() -> None:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.20)
    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.01),
    )

    analytic = price(option, model, market, method="analytic").value

    print("example=pde_theta_scheme")
    print(f"analytic_price={analytic:.6f}")

    for theta in (0.0, 0.5, 1.0):
        cfg = PDEConfig(n_s=50, n_t=50, theta=theta, s_max_multiplier=4.0)
        pde = price(option, model, market, method="pde", cfg=cfg).value
        abs_error = abs(pde - analytic)
        print(f"theta={theta:.1f} pde_price={pde:.6f} abs_error={abs_error:.6f}")


if __name__ == "__main__":
    main()
