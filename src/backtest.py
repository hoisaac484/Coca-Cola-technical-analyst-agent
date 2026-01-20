# backtest.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_strategy import build_features, normalize_ohlc


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    shares: float
    sl: float
    tp: float
    risk_amount: float  # equity * risk_pct at entry (for R-multiple)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    r_mult: Optional[float] = None
    outcome: Optional[str] = None  # TP / SL / EOD


MaxOpenTradesType = Union[int, bool, None]


def compute_drawdown(curve: pd.Series) -> pd.Series:
    peak = curve.cummax()
    return curve / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = compute_drawdown(equity)
    return float(dd.min()) if len(dd) else float("nan")


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float((r.mean() / r.std(ddof=1)) * math.sqrt(periods_per_year))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def run_backtest_virtual_capital(
    df_in: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
    risk_pct: float = 0.01,
    max_open_trades: MaxOpenTradesType = 1,  # int cap, or False/None for unlimited
    allow_fractional_shares: bool = True,
    atr_len: int = 100,
    sl_atr_mult: float = 4.0,
    rr: float = 1.4,
    # IMPORTANT: commission_rate is % of notional per side.
    # If you want 0.001% per trade side, that is 0.00001.
    commission_rate: float = 0.00001,
    slippage_bps: float = 0.0,
    dist_threshold_pct: float = 0.5,
    candle_height_limit_pct: float = 2.0,
) -> Dict[str, Any]:
    # Build all indicators + entry_cond
    df = build_features(
        df_in,
        atr_len=atr_len,
        dist_threshold_pct=dist_threshold_pct,
        candle_height_limit_pct=candle_height_limit_pct,
    )

    def apply_slippage(px: float, side: str) -> float:
        adj = px * (slippage_bps / 10000.0)
        return px + adj if side == "buy" else px - adj

    UNLIMITED = max_open_trades in (False, None)
    max_open_trades_int = int(max_open_trades) if not UNLIMITED else None
    if (not UNLIMITED) and max_open_trades_int is not None and max_open_trades_int < 1:
        raise ValueError("max_open_trades must be >= 1, or set to False/None for unlimited.")

    equity = float(initial_equity)
    trades: List[Trade] = []
    open_trades: List[Trade] = []

    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    gross_leverage_curve: List[Tuple[pd.Timestamp, float]] = []
    open_trades_curve: List[Tuple[pd.Timestamp, int]] = []

    idx = df.index.to_list()

    for i in range(len(df) - 1):  # entries at next bar open
        t = idx[i]
        row = df.iloc[i]

        bar_low = float(row["Low"])
        bar_high = float(row["High"])

        # 1) Exit logic (SL/TP in current bar)
        closed: List[Trade] = []
        for tr in open_trades:
            hit_sl = bar_low <= tr.sl
            hit_tp = bar_high >= tr.tp
            if not (hit_sl or hit_tp):
                continue

            # Conservative if both hit: SL first
            if hit_sl and hit_tp:
                exit_px = tr.sl
                outcome = "SL"
            elif hit_sl:
                exit_px = tr.sl
                outcome = "SL"
            else:
                exit_px = tr.tp
                outcome = "TP"

            exit_px = apply_slippage(float(exit_px), "sell")
            exit_notional = exit_px * tr.shares
            exit_fee = exit_notional * commission_rate
            pnl = (exit_px - tr.entry_price) * tr.shares - exit_fee

            equity += float(pnl)

            tr.exit_time = t
            tr.exit_price = float(exit_px)
            tr.pnl = float(pnl)
            tr.outcome = outcome
            tr.r_mult = (float(pnl) / tr.risk_amount) if tr.risk_amount > 0 else float("nan")
            closed.append(tr)

        if closed:
            open_trades = [tr for tr in open_trades if tr not in closed]
            trades.extend(closed)

        # 2) Entry logic (next open) with cap/unlimited
        slots_ok = UNLIMITED or (len(open_trades) < int(max_open_trades_int))  # type: ignore[arg-type]
        can_enter = bool(row["entry_cond"]) and (not pd.isna(row["atr"])) and (equity > 0) and slots_ok

        if can_enter:
            next_open = float(df.iloc[i + 1]["Open"])
            entry_px = apply_slippage(next_open, "buy")
            a = float(row["atr"])

            sl = entry_px - sl_atr_mult * a
            tp = entry_px + (sl_atr_mult * rr) * a

            risk_per_share = entry_px - sl
            if risk_per_share > 0:
                risk_amount = equity * risk_pct
                shares = risk_amount / risk_per_share

                if not allow_fractional_shares:
                    shares = math.floor(shares)

                if shares > 0:
                    entry_notional = entry_px * shares
                    entry_fee = entry_notional * commission_rate
                    equity -= float(entry_fee)

                    open_trades.append(
                        Trade(
                            entry_time=idx[i + 1],
                            entry_price=float(entry_px),
                            shares=float(shares),
                            sl=float(sl),
                            tp=float(tp),
                            risk_amount=float(risk_amount),
                        )
                    )

        # 3) Mark-to-market + leverage + open-trades tracking
        close_px = float(row["Close"])

        unreal = 0.0
        notional = 0.0
        for tr in open_trades:
            unreal += (close_px - tr.entry_price) * tr.shares
            notional += abs(close_px * tr.shares)

        mtm_equity = float(equity + unreal)
        equity_curve.append((t, mtm_equity))

        gl = float(notional / mtm_equity) if mtm_equity > 0 else float("nan")
        gross_leverage_curve.append((t, gl))

        open_trades_curve.append((t, int(len(open_trades))))

    # 4) Close remaining at last close
    last_t = df.index[-1]
    last_close = apply_slippage(float(df.iloc[-1]["Close"]), "sell")
    for tr in open_trades:
        exit_notional = last_close * tr.shares
        exit_fee = exit_notional * commission_rate
        pnl = (last_close - tr.entry_price) * tr.shares - exit_fee

        equity += float(pnl)

        tr.exit_time = last_t
        tr.exit_price = float(last_close)
        tr.pnl = float(pnl)
        tr.outcome = "EOD"
        tr.r_mult = (float(pnl) / tr.risk_amount) if tr.risk_amount > 0 else float("nan")
        trades.append(tr)

    equity_series = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity")
    gross_lev_series = pd.Series([v for _, v in gross_leverage_curve], index=[d for d, _ in gross_leverage_curve],
                                 name="gross_leverage")
    open_trades_series = pd.Series([v for _, v in open_trades_curve], index=[d for d, _ in open_trades_curve],
                                   name="open_trades")

    rets = equity_series.pct_change()

    wins = [tr for tr in trades if tr.pnl is not None and tr.pnl > 0]
    losses = [tr for tr in trades if tr.pnl is not None and tr.pnl < 0]
    gross_profit = float(sum(tr.pnl for tr in wins)) if wins else 0.0
    gross_loss = float(-sum(tr.pnl for tr in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    r_list = [tr.r_mult for tr in trades if tr.r_mult is not None and math.isfinite(tr.r_mult)]
    avg_r = float(np.mean(r_list)) if r_list else float("nan")

    observed_max_open_trades = int(open_trades_series.max()) if len(open_trades_series) else 0
    observed_max_leverage = float(np.nanmax(gross_lev_series.values)) if len(gross_lev_series) else float("nan")

    max_open_trades_repr: Union[int, str] = "unlimited" if UNLIMITED else int(max_open_trades_int)  # type: ignore[arg-type]

    metrics = {
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "bars": int(len(df)),
        "initial_equity": float(initial_equity),
        "final_equity": float(equity_series.iloc[-1]) if len(equity_series) else float("nan"),
        "num_trades": int(len(trades)),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "profit_factor": float(profit_factor) if math.isfinite(profit_factor) else float("inf"),
        "avg_R": avg_r,
        "CAGR": cagr(equity_series),
        "Sharpe": sharpe_ratio(rets, 252),
        "MaxDrawdown": max_drawdown(equity_series) if len(equity_series) else float("nan"),
        "risk_pct": float(risk_pct),
        "max_open_trades": max_open_trades_repr,
        "fractional_shares": bool(allow_fractional_shares),
        "observed_max_open_trades": observed_max_open_trades,
        "observed_max_leverage": observed_max_leverage,
        "atr_len": int(atr_len),
        "sl_atr_mult": float(sl_atr_mult),
        "rr": float(rr),
        "slippage_bps": float(slippage_bps),
        "dist_threshold_pct": float(dist_threshold_pct),
        "candle_height_limit_pct": float(candle_height_limit_pct),
        "commission_rate": float(commission_rate),
    }

    return {
        "metrics": metrics,
        "equity": equity_series,
        "gross_leverage": gross_lev_series,
        "open_trades": open_trades_series,
        "trades": trades,
        "df": df,
    }