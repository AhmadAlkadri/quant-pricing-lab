from qpl.engines.analytic.black_scholes import bs_price, implied_volatility
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel

def main():
    print("--- Implied Volatility Solver Demo ---")
    
    # 1. Setup
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.01
    sigma_true = 0.25
    
    market = Market(
        spot=S,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )
    option = EuropeanOption(kind="call", strike=K, expiry=T)
    
    # 2. Compute "Market" Price
    print(f"Inputs:\n Spot={S}, Strike={K}, T={T}\n r={r}, q={q}")
    print(f"True Sigma: {sigma_true}")
    
    price = float(bs_price(
        S=S, K=K, T=T, r=r, q=q, sigma=sigma_true, kind="call"
    ))
    print(f"Computed Option Price: {price:.6f}")
    
    # 3. Solve for Implied Vol
    print("\nSolving for Implied Volatility...")
    try:
        iv = implied_volatility(price, option, market)
        print(f"Implied Vol: {iv:.6f}")
        print(f"Error:       {iv - sigma_true:.2e}")
        
    except Exception as e:
        print(f"Solver failed: {e}")
        
    # 4. Demonstrate Sensitivity
    print("\nSensitivities:")
    for bump in [-1.0, 1.0]:
        perturbed_price = price + bump
        iv_p = implied_volatility(perturbed_price, option, market)
        print(f" Price {perturbed_price:.2f} -> IV {iv_p:.4f}")

if __name__ == "__main__":
    main()
