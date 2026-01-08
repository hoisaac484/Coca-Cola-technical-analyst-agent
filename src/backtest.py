# src/backtest.py
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

from src.strategies import (
    regime_snapshot,
    propose_strategy,
    is_strategy_valid,
    STRATEGY_MAP,
)


def run_backtest(df_feat: pd.DataFrame, cfg: dict) -> Dict:
    """
    Convention A:
    - Decision made at close of day t
    - Position takes effect on day t+1
    - Return on day t+1 uses pos[t]
    """

    if "ret_1d" not in df_feat.columns:
        raise ValueError("df_feat must include 'ret_1d'")

    df = df_feat.copy()
    df = df.loc[~df["ret_1d"].isna()].copy()

    cost_rate = float(cfg.get("cost_bps", 5.0)) / 10000.0

    active_strategy = "cash"
    pos = 0
    equity = 1.0

    logs = []

    for i in range(len(df) - 1):
        today = df.index[i]
        tomorrow = df.index[i + 1]

        hist = df.iloc[: i + 1]

        # --- regime & strategy decision at today's close ---
        reg = regime_snapshot(hist, cfg)
        regime = reg["regime"]

        if not is_strategy_valid(active_strategy, hist, cfg):
            active_strategy = propose_strategy(reg, cfg)

        sig_fn = STRATEGY_MAP[active_strategy]
        sig_today = int(sig_fn(hist, cfg).iloc[-1])

        # --- position change happens for tomorrow ---
        turnover = abs(sig_today - pos)
        cost = turnover * cost_rate

        asset_ret = float(df.iloc[i + 1]["ret_1d"])
        strat_ret = pos * asset_ret - cost

        equity *= (1.0 + strat_ret)

        logs.append(
            {
                "date": tomorrow,
                "active_strategy": active_strategy,
                "regime": regime,
                "pos": pos,
                "turnover": turnover,
                "cost": cost,
                "asset_ret": asset_ret,
                "strat_ret": strat_ret,
                "equity": equity,
            }
        )

        pos = sig_today

    equity_df = pd.DataFrame(logs).set_index("date")

    trades_df = extract_trades(df, equity_df)
    metrics = compute_metrics(equity_df, trades_df)

    return {
        "equity": equity_df,
        "trades": trades_df,
        "metrics": metrics,
    }


def extract_trades(df_feat: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    pos = equity_df["pos"].astype(int)
    close = df_feat["close"].reindex(equity_df.index)

    trades = []
    in_trade = False

    for i in range(len(pos)):
        if not in_trade and pos.iloc[i] == 1:
            in_trade = True
            entry_i = i
            entry_date = pos.index[i]
            entry_price = close.iloc[i]

        elif in_trade and pos.iloc[i] == 0:
            exit_date = pos.index[i]
            exit_price = close.iloc[i]
            ret = exit_price / entry_price - 1.0

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": ret,
                    "bars_held": i - entry_i,
                }
            )
            in_trade = False

    if in_trade:
        exit_date = pos.index[-1]
        exit_price = close.iloc[-1]
        ret = exit_price / entry_price - 1.0
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": ret,
                "bars_held": len(pos) - entry_i - 1,
            }
        )

    return pd.DataFrame(trades)


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    eq = equity_df["equity"]
    r = equity_df["strat_ret"]

    years = len(eq) / 252.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    sharpe = (
        r.mean() / r.std(ddof=0) * np.sqrt(252)
        if r.std(ddof=0) > 0
        else np.nan
    )

    dd = eq / eq.cummax() - 1
    max_dd = dd.min()

    return {
        "final_equity": float(eq.iloc[-1]),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "num_trades": int(len(trades_df)),
        "exposure": float(equity_df["pos"].mean()),
        "strategy_switches": int(
            equity_df["active_strategy"]
            .ne(equity_df["active_strategy"].shift(1))
            .sum()
            - 1
        ),
    }
