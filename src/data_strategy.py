# data_strategy.py
from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf


def download_price_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError(f"No data returned from yfinance for {ticker}.")
    return raw


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure single-level OHLCV columns from yfinance output (handles MultiIndex)."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance often returns (field, ticker)
        tickers = list(dict.fromkeys([c[1] for c in df.columns]))
        t0 = tickers[0]
        df = df.xs(t0, axis=1, level=1, drop_level=True)

    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Got: {list(df.columns)}")

    df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    return df


# -----------------------------
# Indicators
# -----------------------------
def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return rma(tr, length)


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = rma(up, length) / rma(down, length)
    return 100 - (100 / (1 + rs))


def smma(series: pd.Series, length: int) -> pd.Series:
    # Wilder-style smoothing ~= RMA
    return rma(series, length)


def cmo(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(length).sum()
    down = (-delta).clip(lower=0).rolling(length).sum()
    return 100 * (up - down) / (up + down)


def supertrend(df: pd.DataFrame, period: int = 1, multiplier: float = 0.1) -> pd.DataFrame:
    """
    Minimal Supertrend:
      - direction: 1 bullish, -1 bearish
      - buy_signal: bearish -> bullish flip
    """
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    _atr = atr(df, period)
    hl2 = (h + l) / 2.0
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(df)):
        if pd.isna(_atr.iat[i]):
            continue

        if (upperband.iat[i] < final_upper.iat[i - 1]) or (c.iat[i - 1] > final_upper.iat[i - 1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if (lowerband.iat[i] > final_lower.iat[i - 1]) or (c.iat[i - 1] < final_lower.iat[i - 1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

    st = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")

    direction.iat[0] = 1
    st.iat[0] = final_lower.iat[0]

    for i in range(1, len(df)):
        prev_dir = int(direction.iat[i - 1])
        prev_st = float(st.iat[i - 1])

        if prev_dir == 1:
            if c.iat[i] < final_lower.iat[i]:
                direction.iat[i] = -1
                st.iat[i] = float(final_upper.iat[i])
            else:
                direction.iat[i] = 1
                st.iat[i] = float(max(final_lower.iat[i], prev_st))
        else:
            if c.iat[i] > final_upper.iat[i]:
                direction.iat[i] = 1
                st.iat[i] = float(final_lower.iat[i])
            else:
                direction.iat[i] = -1
                st.iat[i] = float(min(final_upper.iat[i], prev_st))

    buy_signal = (direction.shift(1) == -1) & (direction == 1)
    return pd.DataFrame({"st": st, "direction": direction, "buy_signal": buy_signal})


# -----------------------------
# Strategy features
# -----------------------------
def build_features(
    df_in: pd.DataFrame,
    *,
    atr_len: int = 100,
    dist_threshold_pct: float = 0.5,
    candle_height_limit_pct: float = 2.0,
) -> pd.DataFrame:
    """
    Adds all columns used by the backtest + trade note prompt:
      buy_signal, rsi50, ma200, dist_pct, dist_ok,
      cmo15, cmo_smma4, cmo_ok,
      candle_ht_pct, candle_ok,
      atr, entry_cond
    """
    df = normalize_ohlc(df_in)

    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
    hlcc4 = (df["High"] + df["Low"] + df["Close"] + df["Close"]) / 4.0

    st = supertrend(df, period=1, multiplier=0.1)
    df["buy_signal"] = st["buy_signal"]

    df["rsi50"] = rsi(hlc3, 50)

    df["ma200"] = smma(df["Close"], 200)
    df["dist_pct"] = (df["Close"] / df["ma200"] - 1.0) * 100.0
    df["dist_ok"] = df["dist_pct"].abs() > float(dist_threshold_pct)

    df["cmo15"] = cmo(hlcc4, 15)
    df["cmo_smma4"] = smma(df["cmo15"], 4)
    df["cmo_ok"] = df["cmo15"] > df["cmo_smma4"]

    df["candle_ht_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
    df["candle_ok"] = df["candle_ht_pct"] <= float(candle_height_limit_pct)

    df["atr"] = atr(df, int(atr_len))

    df["entry_cond"] = (
        df["buy_signal"].fillna(False)
        & (df["rsi50"] < 50)
        & df["dist_ok"].fillna(False)
        & df["cmo_ok"].fillna(False)
        & df["candle_ok"].fillna(False)
    )

    return df
