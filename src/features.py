# src/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add technical features to a validated OHLCV dataframe.

    Expected input columns:
      open, high, low, close, volume

    Adds (baseline):
      rsi14
      bb_mid, bb_upper, bb_lower, bb_width
      sma50, sma200
      atr14
      adx14
      ret_1d, vol_20d

    Adds (for Strategy D - volatility breakout):
      donchian_high_20

    Notes:
    - We don't drop rows here; the backtest can drop/ignore early NaNs.
    - All calculations use rolling windows (no lookahead).
    """
    df = df.copy()

    # --- config with safe defaults ---
    rsi_period = int(cfg.get("rsi_period", 14))
    bb_window = int(cfg.get("bb_window", 20))
    bb_nstd = float(cfg.get("bb_nstd", 2.0))
    sma_fast = int(cfg.get("sma_fast", 50))
    sma_slow = int(cfg.get("sma_slow", 200))
    atr_period = int(cfg.get("atr_period", 14))
    adx_period = int(cfg.get("adx_period", 14))
    donchian_window = int(cfg.get("donchian_window", 20))
    vol_window = int(cfg.get("vol_window", 20))

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- returns / volatility ---
    df["ret_1d"] = close.pct_change()
    df["vol_20d"] = df["ret_1d"].rolling(vol_window, min_periods=vol_window).std()

    # --- moving averages ---
    df["sma50"] = close.rolling(sma_fast, min_periods=sma_fast).mean()
    df["sma200"] = close.rolling(sma_slow, min_periods=sma_slow).mean()

    # --- bollinger bands ---
    bb_mid = close.rolling(bb_window, min_periods=bb_window).mean()
    bb_std = close.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + bb_nstd * bb_std
    df["bb_lower"] = bb_mid - bb_nstd * bb_std

    # width is useful for "compression" in Strategy D
    # (avoid division by zero; bb_mid is price so shouldn't be 0 in normal markets)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # --- RSI (Wilder's smoothing) ---
    df["rsi14"] = _rsi_wilder(close, rsi_period)

    # --- ATR (Wilder) ---
    df["atr14"] = _atr_wilder(high, low, close, atr_period)

    # --- ADX (Wilder) ---
    df["adx14"] = _adx_wilder(high, low, close, adx_period)

    # --- Donchian high (for breakout trigger) ---
    df["donchian_high_20"] = close.rolling(donchian_window, min_periods=donchian_window).max()

    return df


# -----------------------------
# Helpers (kept simple + readable)
# -----------------------------
def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """
    RSI using Wilder's smoothing (EMA-like).

    Returns values in [0, 100].
    """
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # If avg_loss is 0, rs becomes inf and rsi becomes 100 — that’s fine.
    return rsi


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Average True Range (ATR) using Wilder smoothing.
    """
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def _adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    ADX (Average Directional Index) using Wilder's method.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing for TR and DM
    tr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx
