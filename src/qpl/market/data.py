"""
Market data retrieval with local caching.

This module is optional-dependency gated: fetching and caching prices needs
pandas, pyarrow, and yfinance (the ``data`` extra). Those packages are
imported lazily, inside functions, so that ``import qpl`` and
``import qpl.market.data`` both succeed with only the core dependencies
(numpy/scipy/matplotlib) installed. A clear ``NotSupportedError`` is raised
only when a data-dependent function is actually called without the extra
installed.
"""

import hashlib
import os

from qpl.exceptions import NotSupportedError

# Common interval alias mapping to standardized filename part
INTERVAL_ALIASES = {
    "1d": "1d",
    "daily": "1d"
}

_INSTALL_HINT = 'pip install "qpl[data]"'


def _require_data_deps():
    """Lazily import pandas and yfinance, raising a clear error if absent."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise NotSupportedError(
            "qpl.market.data requires the optional 'data' extra (pandas). "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc

    try:
        import yfinance as yf
    except ImportError as exc:
        raise NotSupportedError(
            "qpl.market.data requires the optional 'data' extra (yfinance). "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc

    return pd, yf


def get_prices(
    ticker: str,
    start: str,
    end: str,
    *,
    source: str = "yahoo",
    interval: str = "1d",
    cache_dir: str = ".market_cache"
):
    """
    Fetch historical prices for a ticker, using local disk cache if available.

    Requires the optional ``data`` extra (``pip install "qpl[data]"``).

    Parameters
    ----------
    ticker : str
        The ticker symbol (e.g. "SPY", "^GSPC").
    start : str
        Start date string (YYYY-MM-DD).
    end : str
        End date string (YYYY-MM-DD).
    source : str, default "yahoo"
        The data source. Currently only "yahoo" is supported.
    interval : str, default "1d"
        Data interval (e.g. "1d").
    cache_dir : str, default ".market_cache"
        Directory to store cached data files.

    Returns
    -------
    pandas.DataFrame
        DataFrame with DatetimeIndex and at least a 'Close' column.

    Raises
    ------
    ValueError
        If inputs are invalid or source is unsupported.
    IOError
        If data cannot be fetched and is not in cache.
    NotSupportedError
        If the optional 'data' extra (pandas/yfinance/pyarrow) is not installed.
    """
    if source != "yahoo":
        raise ValueError(f"Unsupported data source: {source}")

    pd, yf = _require_data_deps()

    interval = INTERVAL_ALIASES.get(interval, interval)

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Create deterministic cache filename
    # We hash the inputs to handle special characters in tickers/dates safely
    key_str = f"{source}_{ticker}_{start}_{end}_{interval}"
    key_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{ticker}_{key_hash}.parquet")

    # 1. Try to load from cache
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)

            # Restore frequency if possible (parquet does not persist it)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq:
                    df.index.freq = inferred_freq

            print(f"[MarketData] Loaded {ticker} from cache: {cache_path}")
            return df
        except Exception as e:  # noqa: BLE001 -- any cache-read failure (corrupt/partial
            # parquet file, schema drift) should fall through to a clean refetch below.
            print(f"[MarketData] Cache load failed, refetching. Error: {e}")
            # If cache is corrupted, proceed to fetch

    # 2. Fetch from network
    print(f"[MarketData] Fetching {ticker} from {source}...")
    try:
        # yfinance download
        # auto_adjust=True gives adjusted close as 'Close'
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise OSError(f"No data found for {ticker} from {source}")

        # Ensure we have a Close column
        if "Close" not in df.columns:
            # Fallback if auto_adjust failed to behave as expected
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            elif "Close" not in df.columns:
                 raise OSError(f"Data for {ticker} missing 'Close' column")

        # 3. Save to cache
        # Use parquet for efficiency and type preservation
        df.to_parquet(cache_path)
        print(f"[MarketData] Saved {ticker} to cache: {cache_path}")

        return df

    except Exception as e:  # noqa: BLE001 -- deliberately translates any fetch failure
        # (network, yfinance, parquet write) into the single documented OSError below.
        raise OSError(f"Failed to fetch data for {ticker}: {e}")
