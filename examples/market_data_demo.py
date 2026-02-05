"""
Demo of Market Data Retrieval and Caching (Slice 3a).
"""
import shutil
import time
import pandas as pd
from qpl.market.data import get_prices

def main():
    print("--- Market Data Demo ---")
    
    # Using a fake cache dir for demo purposes to show behavior clearly
    # In real usage, default is '.market_cache'
    CACHE_DIR = ".demo_cache"
    
    # Clean up previous run
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    
    ticker = "SPY"
    start = "2023-01-01"
    end = "2023-01-10"
    
    print(f"\n1. Fetching {ticker} (Network call expected)...")
    start_t = time.time()
    try:
        df1 = get_prices(ticker, start, end, cache_dir=CACHE_DIR)
        print(f"   Got {len(df1)} rows in {time.time() - start_t:.2f}s")
        print(f"   Head:\n{df1.head(2)}")
    except Exception as e:
        print(f"   Error fetching data: {e}. (This could be network or yfinance issue)")
        return

    print(f"\n2. Fetching {ticker} again (Cache hit expected)...")
    start_t = time.time()
    df2 = get_prices(ticker, start, end, cache_dir=CACHE_DIR)
    print(f"   Got {len(df2)} rows in {time.time() - start_t:.2f}s")
    
    # Verify equality
    if df1.equals(df2):
        print("\nSUCCESS: Dataframes match.")
    else:
        print("\nWARNING: Dataframes do not match!")
        
    print("\nDemo complete. Cleaning up...")
    shutil.rmtree(CACHE_DIR)

if __name__ == "__main__":
    main()
