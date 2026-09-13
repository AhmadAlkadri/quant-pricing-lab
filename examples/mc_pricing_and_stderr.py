from __future__ import annotations

from qpl.engines.mc.pricers import MCConfig
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

    cfg = MCConfig(n_paths=20_000, n_steps=40, seed=123)

    analytic = price(option, model, market, method="analytic")
    mc = price(option, model, market, method="mc", cfg=cfg)

    if mc.stderr is None:
        raise RuntimeError("Expected Monte Carlo stderr")

    z_score = (mc.value - analytic.value) / mc.stderr if mc.stderr > 0.0 else 0.0

    print("example=mc_pricing_and_stderr")
    print(f"analytic_price={analytic.value:.6f}")
    print(f"mc_price={mc.value:.6f}")
    print(f"mc_stderr={mc.stderr:.6f}")
    print(f"z_score={z_score:.4f}")


if __name__ == "__main__":
    main()
