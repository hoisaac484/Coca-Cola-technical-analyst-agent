# src/strategies.py
from __future__ import annotations

from typing import Dict, Callable, List
import numpy as np
import pandas as pd


# Keeps regime stable when ADX sits in the "gray zone".
# This is intentionally simple and deterministic for a single-run backtest.
_LAST_REGIME = "UNCLEAR"


# -----------------------------
# Regime snapshot (long-only) with hysteresis
# -----------------------------
def regime_snapshot(df_feat: pd.DataFrame, cfg: dict) -> dict:
    """
    Classify the current market regime using the latest available data.

    Hysteresis logic (Option A):
      - If ADX <= adx_range_enter: regime becomes RANGE
      - If ADX >= adx_trend_enter AND SMA50 > SMA200: regime becomes UPTREND
      - If adx_range_enter < ADX < adx_trend_enter: keep the previous regime

    Defaults:
      adx_range_enter = 18
      adx_trend_enter = 28

    Returns:
      {
        "regime": "RANGE" | "UPTREND" | "UNCLEAR",
        "flags": [...],
        "evidence": {...}
      }
    """
    global _LAST_REGIME

    if df_feat is None or df_feat.empty:
        _LAST_REGIME = "UNCLEAR"
        return {"regime": "UNCLEAR", "flags": ["NO_DATA"], "evidence": {}}

    last = df_feat.iloc[-1]

    adx = float(last.get("adx14", np.nan))
    sma50 = float(last.get("sma50", np.nan))
    sma200 = float(last.get("sma200", np.nan))
    vol = float(last.get("vol_20d", np.nan))

    flags: List[str] = []

    need = ["adx14", "sma50", "sma200"]
    if any(pd.isna(last.get(c, np.nan)) for c in need):
        flags.append("INSUFFICIENT_FEATURES")

    # If features missing, be conservative and reset hysteresis state.
    if "INSUFFICIENT_FEATURES" in flags:
        _LAST_REGIME = "UNCLEAR"
        return {
            "regime": "UNCLEAR",
            "flags": flags,
            "evidence": {"adx14": adx, "sma50": sma50, "sma200": sma200, "vol_20d": vol},
        }

    adx_range_enter = float(cfg.get("adx_range_enter", 18))
    adx_trend_enter = float(cfg.get("adx_trend_enter", 28))

    # Decide regime with hysteresis
    if adx <= adx_range_enter:
        regime = "RANGE"
    elif adx >= adx_trend_enter and sma50 > sma200:
        regime = "UPTREND"
    else:
        # Gray zone: keep last regime to reduce regime flicker
        regime = _LAST_REGIME

        # If last regime was UPTREND but the MA condition no longer holds,
        # fall back to UNCLEAR (prevents "stuck in uptrend" when trend breaks).
        if regime == "UPTREND" and not (sma50 > sma200):
            regime = "UNCLEAR"

    _LAST_REGIME = regime

    return {
        "regime": regime,
        "flags": flags,
        "evidence": {
            "adx14": adx,
            "sma50": sma50,
            "sma200": sma200,
            "sma50_gt_sma200": bool(sma50 > sma200),
            "vol_20d": vol,
            "adx_range_enter": adx_range_enter,
            "adx_trend_enter": adx_trend_enter,
            "prev_regime_used": True,
        },
    }


# -----------------------------
# Strategy selection (sticky)
# -----------------------------
def propose_strategy(regime_info: dict, cfg: dict) -> str:
    """
    Mapping:
      RANGE   -> mean_reversion
      UPTREND -> trend_follow
      UNCLEAR -> cash
    """
    regime = regime_info.get("regime", "UNCLEAR")
    flags = set(regime_info.get("flags", []))

    if "INSUFFICIENT_FEATURES" in flags:
        return "cash"

    if regime == "RANGE":
        return "mean_reversion"

    if regime == "UPTREND":
        return "trend_follow"

    return "cash"


def is_strategy_valid(active_strategy: str, df_feat: pd.DataFrame, cfg: dict) -> bool:
    """
    Sticky strategy validity check.
    """
    reg = regime_snapshot(df_feat, cfg)
    regime = reg["regime"]
    flags = set(reg["flags"])
    last = df_feat.iloc[-1]

    adx = float(last.get("adx14", np.nan))
    sma50 = float(last.get("sma50", np.nan))
    sma200 = float(last.get("sma200", np.nan))

    # You can keep your original invalidation threshold (25) for MR.
    # It is independent from the hysteresis entry thresholds (18/28).
    adx_trend_min = float(cfg.get("adx_trend_min", 25))

    if "INSUFFICIENT_FEATURES" in flags:
        return active_strategy == "cash"

    if active_strategy == "mean_reversion":
        if not np.isnan(adx) and adx >= adx_trend_min:
            return False
        return True

    if active_strategy == "trend_follow":
        if np.isnan(sma50) or np.isnan(sma200):
            return False
        return sma50 > sma200

    if active_strategy == "cash":
        return regime == "UNCLEAR"

    return False


# -----------------------------
# Human-readable rules
# -----------------------------
def explain_rules(strategy_name: str, cfg: dict) -> str:
    if strategy_name == "mean_reversion":
        rsi_entry = float(cfg.get("rsi_entry", 35))
        mr_atr_k = float(cfg.get("mr_atr_stop_k", 2.0))
        mr_max_hold = int(cfg.get("mr_max_hold", 10))
        return (
            "Mean Reversion (Long-only)\n"
            f"- Enter LONG when Close < Lower Bollinger Band AND RSI(14) < {rsi_entry:.0f}.\n"
            "- Exit when price reverts to the Bollinger mid-band, OR\n"
            f"  when price falls below Entry − {mr_atr_k:.1f}×ATR(14), OR\n"
            f"  after {mr_max_hold} trading days."
        )

    if strategy_name == "trend_follow":
        return (
            "Trend Follow (Long-only)\n"
            "- Hold LONG when SMA(50) > SMA(200).\n"
            "- Exit to CASH when SMA(50) ≤ SMA(200)."
        )

    return "Cash\n- Stay in CASH (no position)."


# -----------------------------
# Signal generators (0/1)
# -----------------------------
def signal_cash(df_feat: pd.DataFrame, cfg: dict) -> pd.Series:
    return pd.Series(0, index=df_feat.index, dtype=int)


def signal_trend_follow(df_feat: pd.DataFrame, cfg: dict) -> pd.Series:
    cond = df_feat["sma50"] > df_feat["sma200"]
    sig = cond.astype(int)
    sig = sig.where(~sig.isna(), 0).astype(int)
    return sig


def signal_mean_reversion(df_feat: pd.DataFrame, cfg: dict) -> pd.Series:
    rsi_entry = float(cfg.get("rsi_entry", 35))
    mr_atr_stop_k = float(cfg.get("mr_atr_stop_k", 2.0))
    mr_max_hold = int(cfg.get("mr_max_hold", 10))

    sig = pd.Series(0, index=df_feat.index, dtype=int)

    in_pos = False
    entry_price = np.nan
    entry_i = -1

    for i in range(len(df_feat)):
        row = df_feat.iloc[i]

        close = row.get("close", np.nan)
        rsi = row.get("rsi14", np.nan)
        bb_lower = row.get("bb_lower", np.nan)
        bb_mid = row.get("bb_mid", np.nan)
        atr = row.get("atr14", np.nan)

        if any(pd.isna(x) for x in [close, rsi, bb_lower, bb_mid, atr]):
            sig.iloc[i] = 1 if in_pos else 0
            continue

        if not in_pos:
            if close < bb_lower and rsi < rsi_entry:
                in_pos = True
                entry_price = float(close)
                entry_i = i
                sig.iloc[i] = 1
            else:
                sig.iloc[i] = 0
        else:
            held = i - entry_i
            stop_price = entry_price - mr_atr_stop_k * atr

            if close >= bb_mid or close <= stop_price or held >= mr_max_hold:
                in_pos = False
                entry_price = np.nan
                entry_i = -1
                sig.iloc[i] = 0
            else:
                sig.iloc[i] = 1

    return sig


# -----------------------------
# Strategy registry
# -----------------------------
STRATEGY_MAP: Dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {
    "mean_reversion": signal_mean_reversion,
    "trend_follow": signal_trend_follow,
    "cash": signal_cash,
}
