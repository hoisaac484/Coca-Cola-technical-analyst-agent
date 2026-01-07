# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd


# -----------------------------
# Fetch OHLCV
# -----------------------------
def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV for a ticker using yfinance.

    Returns a DataFrame indexed by date with columns:
      open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from e

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}.")

    # yfinance sometimes returns a MultiIndex for columns (e.g., with multiple tickers).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]

    # Standardize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Got: {list(df.columns)}")

    df = df[needed].copy()

    # Ensure clean datetime index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df.index = df.index.tz_localize(None).normalize()

    return df


# -----------------------------
# Validate OHLCV
# -----------------------------
@dataclass
class ValidationReport:
    ok: bool
    issues: List[str]
    fixes: List[str]
    n_rows: int
    start: Optional[str]
    end: Optional[str]


def validate_ohlcv(df: pd.DataFrame, *, min_rows: int = 252 * 8) -> Tuple[pd.DataFrame, Dict]:
    """
    Basic sanity checks + light cleaning for OHLCV.

    - Enforces columns: open/high/low/close/volume
    - Drops bad rows (NaNs in OHLC, non-positive prices, inconsistent high/low)
    - Removes duplicate dates
    - Ensures a monotonic DatetimeIndex
    """
    if df is None or df.empty:
        raise ValueError("validate_ohlcv() received an empty dataframe.")

    issues: List[str] = []
    fixes: List[str] = []

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Index -> datetime, drop invalid timestamps
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("Index was not DatetimeIndex; converting.")
        df.index = pd.to_datetime(df.index, errors="coerce")
        fixes.append("Converted index to DatetimeIndex.")

    bad_idx = int(df.index.isna().sum())
    if bad_idx:
        issues.append(f"Found {bad_idx} rows with invalid dates; dropped.")
        df = df[~df.index.isna()]
        fixes.append("Dropped invalid date rows.")

    # Sort + de-duplicate
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        fixes.append("Sorted index ascending.")

    dup = int(df.index.duplicated().sum())
    if dup:
        issues.append(f"Found {dup} duplicate dates; kept first occurrence.")
        df = df[~df.index.duplicated(keep="first")]
        fixes.append("Dropped duplicate rows.")

    # Convert to numeric (coerce bad values to NaN)
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in OHLC
    ohlc = ["open", "high", "low", "close"]
    nan_ohlc = int(df[ohlc].isna().any(axis=1).sum())
    if nan_ohlc:
        issues.append(f"Found {nan_ohlc} rows with NaNs in OHLC; dropped.")
        df = df.dropna(subset=ohlc)
        fixes.append("Dropped NaN OHLC rows.")

    # Volume: fill NaNs with 0 (common for holidays / bad vendor rows)
    nan_vol = int(df["volume"].isna().sum())
    if nan_vol:
        issues.append(f"Found {nan_vol} rows with NaN volume; filled with 0.")
        df["volume"] = df["volume"].fillna(0.0)
        fixes.append("Filled NaN volume with 0.")

    # Drop non-positive prices
    nonpos = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    nonpos_n = int(nonpos.sum())
    if nonpos_n:
        issues.append(f"Found {nonpos_n} rows with non-positive prices; dropped.")
        df = df.loc[~nonpos].copy()
        fixes.append("Dropped non-positive price rows.")

    # Basic OHLC consistency:
    # - high should be >= open/close/low
    # - low should be <= open/close/high
    bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    bad = bad_high | bad_low
    bad_n = int(bad.sum())
    if bad_n:
        issues.append(f"Found {bad_n} rows with inconsistent OHLC; dropped.")
        df = df.loc[~bad].copy()
        fixes.append("Dropped inconsistent OHLC rows.")

    # Normalize index (keeps things consistent for rolling windows)
    df.index = df.index.tz_localize(None).normalize()

    if len(df) < min_rows:
        raise ValueError(
            f"Not enough usable rows after cleaning: {len(df)} < {min_rows}. "
            "Use a longer date range or a different data source."
        )

    report = ValidationReport(
        ok=(len(issues) == 0),
        issues=issues,
        fixes=fixes,
        n_rows=int(len(df)),
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
    )

    return df, report.__dict__
