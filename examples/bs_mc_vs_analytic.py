from qpl.engines.mc.pricers import MCConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price


def main() -> None:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.01),
    )

    mc_cfg = MCConfig(n_paths=100_000, seed=7)
    analytic = price(option, model, market, method="analytic")
    mc = price(option, model, market, method="mc", cfg=mc_cfg)

    if mc.stderr is None:
        raise RuntimeError("MC stderr is None; expected a Monte Carlo result.")
    if mc.stderr == 0.0:
        raise RuntimeError("MC stderr is zero; z-score is undefined.")

    z_score = (mc.value - analytic.value) / mc.stderr

    print(f"analytic={analytic.value:.6f}")
    print(f"mc={mc.value:.6f} stderr={mc.stderr:.6f}")
    print(f"z={z_score:.3f}")

    bumps = {"spot": 1e-2, "sigma": 1e-4, "r": 1e-5}
    g_an = greeks(option, model, market, method="analytic")
    g_mc = greeks(option, model, market, method="mc", cfg=mc_cfg, bumps=bumps)
    print(
        "greeks_analytic="
        f"delta:{g_an.delta:.6f} gamma:{g_an.gamma:.6f} vega:{g_an.vega:.6f} rho:{g_an.rho:.6f}"
    )
    print(
        "greeks_mc="
        f"delta:{g_mc.delta:.6f} gamma:{g_mc.gamma:.6f} vega:{g_mc.vega:.6f} rho:{g_mc.rho:.6f}"
    )


if __name__ == "__main__":
    main()
