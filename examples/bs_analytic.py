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
        dividend_curve=FlatDividendCurve(0.0),
    )

    px = price(option, model, market, method="analytic")
    gr = greeks(option, model, market, method="analytic")

    print(f"price={px.value:.6f}")
    print(
        f"delta={gr.delta:.6f} gamma={gr.gamma:.6f} vega={gr.vega:.6f} "
        f"theta={gr.theta:.6f} rho={gr.rho:.6f}"
    )


if __name__ == "__main__":
    main()
