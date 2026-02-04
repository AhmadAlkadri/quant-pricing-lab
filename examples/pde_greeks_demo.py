from qpl.engines.pde.pricers import PDEConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price


def main():
    # 1. Market / Model Setup
    spot = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.20

    market = Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )
    model = BlackScholesModel(sigma=sigma)

    # 2. Instrument
    call = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    put = EuropeanOption(kind="put", strike=100.0, expiry=1.0)

    # 3. PDE Config
    pde_cfg = PDEConfig(n_s=300, n_t=400, theta=0.5, s_max_multiplier=4.0)

    # 4. Compare Analytic vs PDE Greeks
    print(f"Pricing Call Option (K=100, T=1.0) with Spot={spot}, r={r}, q={q}, sigma={sigma}")
    print("-" * 80)
    print(f"{'Metric':<10} | {'Analytic':<15} | {'PDE':<15} | {'Diff':<15}")
    print("-" * 80)

    for opt_name, opt in [("Call", call), ("Put", put)]:
        # Analytic
        res_a_price = price(opt, model, market, method="analytic")
        res_a_greeks = greeks(opt, model, market, method="analytic")

        # PDE
        res_p_price = price(opt, model, market, method="pde", cfg=pde_cfg)
        res_p_greeks = greeks(opt, model, market, method="pde", cfg=pde_cfg)

        # Print Price
        print(f"{opt_name} Price | {res_a_price.value:15.6f} | {res_p_price.value:15.6f} | {res_p_price.value - res_a_price.value:15.6e}")
        
        # Print Delta
        print(f"{opt_name} Delta | {res_a_greeks.delta:15.6f} | {res_p_greeks.delta:15.6f} | {res_p_greeks.delta - res_a_greeks.delta:15.6e}")
        
        # Print Gamma
        print(f"{opt_name} Gamma | {res_a_greeks.gamma:15.6f} | {res_p_greeks.gamma:15.6f} | {res_p_greeks.gamma - res_a_greeks.gamma:15.6e}")
        print("-" * 80)

if __name__ == "__main__":
    main()
