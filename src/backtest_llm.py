# src/backtest_llm.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from src.strategies import (
    regime_snapshot,
    is_strategy_valid,
    STRATEGY_MAP,
    propose_strategy,
)
from src.backtest import extract_trades, compute_metrics


# -----------------------------
# Simple JSONL cache
# -----------------------------
@dataclass
class DecisionCache:
    path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._map: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    k = obj.get("key")
                    v = obj.get("value")
                    if isinstance(k, str) and isinstance(v, dict):
                        self._map[k] = v
                except Exception:
                    continue

    def get(self, key: str) -> Optional[dict]:
        return self._map.get(key)

    def set(self, key: str, value: dict) -> None:
        self._map[key] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _stable_key(payload: dict) -> str:
    # Stable hash of request payload => reproducible cache key
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------
# LLM-driven backtest (Mode B)
# -----------------------------
def run_backtest_llm(
    df_feat: pd.DataFrame,
    cfg: dict,
    selector,  # your LLMStrategySelector instance
    *,
    cache_path: str = ".cache/llm_decisions.jsonl",
    max_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Mode B: LLM decides KEEP/SWITCH daily throughout history.

    Convention A:
      - At day t close: LLM decides strategy, strategy produces signal[t]
      - Position for day t+1 becomes signal[t]
      - Return at t+1 uses pos[t]
    """
    if df_feat is None or df_feat.empty:
        raise ValueError("run_backtest_llm() received empty df_feat")

    if "ret_1d" not in df_feat.columns:
        raise ValueError("df_feat must include 'ret_1d'")

    # Keep rows where returns exist
    df = df_feat.loc[~df_feat["ret_1d"].isna()].copy()
    if len(df) < 50:
        raise ValueError("Not enough rows after dropping NaN returns.")

    if max_days is not None:
        df = df.iloc[:max_days].copy()

    # Reset hysteresis regime memory if your strategies.py supports it
    try:
        from src.strategies import reset_regime_state  # optional helper you may have
        reset_regime_state()
    except Exception:
        # fallback: calling snapshot on empty should reset to UNCLEAR in your hysteresis implementation
        regime_snapshot(pd.DataFrame(), cfg)

    cost_rate = float(cfg.get("cost_bps", 5.0)) / 10000.0
    ticker = str(cfg.get("ticker", "TICKER"))

    cache = DecisionCache(cache_path)

    active_strategy = "cash"
    pos = 0
    equity = 1.0

    logs: List[dict] = []

    # Loop to len(df)-1 because we need t+1 return
    for i in range(len(df) - 1):
        today = df.index[i]
        tomorrow = df.index[i + 1]

        hist = df.iloc[: i + 1]

        # 1) Deterministic snapshot and constraints
        reg = regime_snapshot(hist, cfg)
        forced_switch = not is_strategy_valid(active_strategy, hist, cfg)

        # 2) Build LLM request payload (used for cache key)
        payload = {
            "ticker": ticker,
            "as_of_date": str(today.date()),
            "sticky_strategy_current": active_strategy,
            "regime_snapshot": reg,
            "allowed_strategies": list(STRATEGY_MAP.keys()),
            "forced_switch": forced_switch,
            # include key thresholds to avoid cache mixing across config changes
            "cfg": {
                "adx_range_enter": cfg.get("adx_range_enter"),
                "adx_trend_enter": cfg.get("adx_trend_enter"),
                "adx_trend_min": cfg.get("adx_trend_min"),
                "rsi_entry": cfg.get("rsi_entry"),
                "mr_atr_stop_k": cfg.get("mr_atr_stop_k"),
                "mr_max_hold": cfg.get("mr_max_hold"),
            },
        }
        key = _stable_key(payload)

        # 3) Cache lookup or call LLM
        cached = cache.get(key)
        if cached is None:
            llm_dec = selector.decide(
                ticker=ticker,
                date=str(today.date()),
                active_strategy=active_strategy,
                regime_info=reg,
                cfg=cfg,
                forced_switch=forced_switch,
            )
            # store as plain dict for portability
            cached = {
                "action": llm_dec.action,
                "chosen_strategy": llm_dec.chosen_strategy,
                "confidence": llm_dec.confidence,
                "reason": llm_dec.reason,
                "evidence_bullets": getattr(llm_dec, "evidence_bullets", []),
            }
            cache.set(key, cached)

        action = str(cached.get("action", "KEEP")).upper()
        chosen = str(cached.get("chosen_strategy", active_strategy))
        confidence = float(cached.get("confidence", 0.0))
        reason = str(cached.get("reason", ""))

        # 4) Apply guardrails and sticky policy
        allowed = set(STRATEGY_MAP.keys())
        if chosen not in allowed:
            chosen = "cash"
            action = "SWITCH"
            confidence = 0.0
            reason = "Invalid strategy from cached/LLM output; fallback to cash."

        if forced_switch:
            action = "SWITCH"

        if action == "SWITCH":
            active_strategy = chosen  # switch to LLM-chosen strategy
        else:
            # KEEP -> do nothing
            pass

        # Safety fallback if something odd happens
        if active_strategy not in allowed:
            active_strategy = propose_strategy(reg, cfg)

        # 5) Compute today's signal from the ACTIVE strategy, then apply to tomorrow (Convention A)
        sig_today = int(STRATEGY_MAP[active_strategy](hist, cfg).iloc[-1])

        turnover = abs(sig_today - pos)
        cost = turnover * cost_rate

        asset_ret = float(df.iloc[i + 1]["ret_1d"])
        strat_ret = pos * asset_ret - cost
        equity *= (1.0 + strat_ret)

        logs.append(
            {
                "date": tomorrow,
                "regime": reg.get("regime", "UNCLEAR"),
                "active_strategy": active_strategy,
                "pos": pos,
                "turnover": turnover,
                "cost": cost,
                "asset_ret": asset_ret,
                "strat_ret": strat_ret,
                "equity": equity,
                # optional audit fields (useful)
                "llm_action": action,
                "llm_chosen_strategy": chosen,
                "llm_confidence": confidence,
                "llm_reason": reason[:200],
            }
        )

        # Position updates for next day
        pos = sig_today

    equity_df = pd.DataFrame(logs).set_index("date")
    trades_df = extract_trades(df_feat, equity_df)
    metrics = compute_metrics(equity_df, trades_df)

    return {"equity": equity_df, "trades": trades_df, "metrics": metrics}
