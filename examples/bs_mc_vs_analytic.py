from qpl.engines.mc.pricers import MCConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price


def main() -> None:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.01),
    )

    analytic = price(option, model, market, method="analytic")
    mc = price(option, model, market, method="mc", cfg=MCConfig(n_paths=100_000, seed=7))

    if mc.stderr is None:
        raise RuntimeError("MC stderr is None; expected a Monte Carlo result.")
    if mc.stderr == 0.0:
        raise RuntimeError("MC stderr is zero; z-score is undefined.")

    z_score = (mc.value - analytic.value) / mc.stderr

    print(f"analytic={analytic.value:.6f}")
    print(f"mc={mc.value:.6f} stderr={mc.stderr:.6f}")
    print(f"z={z_score:.3f}")


if __name__ == "__main__":
    main()
