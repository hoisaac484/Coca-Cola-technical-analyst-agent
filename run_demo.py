# run_demo.py
from __future__ import annotations

import math
from typing import Any, Dict
import matplotlib.pyplot as plt

from src.data_strategy import download_price_data, normalize_ohlc
from src.backtest import run_backtest_virtual_capital, compute_drawdown
from src.llm_tradenote import generate_trade_note_azure, save_trade_note_to_word

def print_metrics(m: Dict[str, Any]) -> None:
    def pct(x: float) -> str:
        return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2%}"

    def num(x: float) -> str:
        return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"

    print("\n=== Metrics (Virtual Capital) ===")
    print(f"{'start':>28}: {m['start']}")
    print(f"{'end':>28}: {m['end']}")
    print(f"{'bars':>28}: {m['bars']}")
    print(f"{'risk_pct':>28}: {pct(m['risk_pct'])}")
    print(f"{'max_open_trades(set)':>28}: {m['max_open_trades']}")
    print(f"{'max_open_trades(obs)':>28}: {m.get('observed_max_open_trades', 'na')}")
    print(f"{'max_leverage(obs)':>28}: {num(m.get('observed_max_leverage', float('nan')))}")
    print(f"{'fractional_shares':>28}: {m['fractional_shares']}")
    print(f"{'initial_equity':>28}: {m['initial_equity']:.4f}")
    print(f"{'final_equity':>28}: {m['final_equity']:.4f}")
    print(f"{'num_trades':>28}: {m['num_trades']}")
    print(f"{'win_rate':>28}: {pct(m['win_rate'])}")
    pf = m["profit_factor"]
    print(f"{'profit_factor':>28}: {'inf' if pf == float('inf') else f'{pf:.3f}'}")
    print(f"{'avg_R':>28}: {num(m['avg_R'])}")
    print(f"{'CAGR':>28}: {pct(m['CAGR'])}")
    print(f"{'Sharpe':>28}: {num(m['Sharpe'])}")
    print(f"{'MaxDrawdown':>28}: {pct(m['MaxDrawdown'])}")


def plot_all_in_one_window(result: Dict[str, Any]) -> None:
    df = normalize_ohlc(result["df"])
    equity = result["equity"].dropna()
    idx = equity.index
    if len(idx) < 5:
        raise ValueError("Not enough data to plot.")

    price_close = df.loc[idx, "Close"].astype(float)
    price_adj = df.loc[idx, "Adj Close"].astype(float)

    # Normalize (so initial capital doesn't flatten things)
    price_close_norm = price_close / price_close.iloc[0]
    price_adj_norm = price_adj / price_adj.iloc[0]
    equity_norm = equity / equity.iloc[0]

    bh_close = price_close_norm.rename("Buy & Hold (Close)")
    bh_adj = price_adj_norm.rename("Buy & Hold (Adj Close)")

    # Drawdowns
    dd_bh = compute_drawdown(bh_adj)
    dd_strat = compute_drawdown(equity_norm.rename("strategy_equity"))

    # Leverage + open trades
    lev = result["gross_leverage"].reindex(idx).astype(float)
    ot = result["open_trades"].reindex(idx).astype(float)

    max_ot_set = result["metrics"].get("max_open_trades", "unlimited")
    max_lev_obs = float(result["metrics"].get("observed_max_leverage", float("nan")))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 1]},
    )

    # 1) Price vs Equity (normalized)
    ax = axes[0]
    ax.plot(bh_close, label="Buy & Hold (Close)")
    ax.plot(bh_adj, label="Buy & Hold (Adj Close)")
    ax.plot(equity_norm, label="Strategy Equity (norm)")
    ax.set_title("Strategy vs Buy & Hold (Normalized)")
    ax.set_ylabel("Normalized value")
    ax.grid(True)
    ax.legend(loc="upper left")

    # 2) Drawdowns
    ax = axes[1]
    ax.plot(dd_bh, label="B&H drawdown")
    ax.plot(dd_strat, label="Strategy drawdown")
    ax.axhline(0, linewidth=1)
    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    ax.legend(loc="lower left")

    # 3) Leverage
    ax = axes[2]
    ax.plot(lev, label="Gross Leverage")
    ax.axhline(1.0, linewidth=1, label="1x")
    if math.isfinite(max_lev_obs):
        ax.axhline(max_lev_obs, linestyle="--", linewidth=1, label=f"Max (obs) = {max_lev_obs:.3f}x")
    ax.set_title("Gross Leverage")
    ax.set_ylabel("Leverage (x)")
    ax.grid(True)
    ax.legend(loc="upper left")

    # 4) Open trades
    ax = axes[3]
    ax.step(ot.index, ot.values, where="post", label="Open Trades")
    if max_ot_set != "unlimited":
        ax.axhline(int(max_ot_set), linestyle="--", linewidth=1, label=f"Max Open Trades (set) = {max_ot_set}")
    ax.set_title("Open Trades")
    ax.set_ylabel("Count")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

def main() -> None:
    # =========================
    # Hard-coded parameters
    # =========================
    ticker = "KO"
    START_DATE = "2016-01-15"
    END_DATE = "2026-01-15"

    max_open_trades = False          # False/None => unlimited, or set an int like 1,2,3...
    atr_len = 100
    sl_atr_mult = 4.0
    rr = 1.4
    risk_pct = 0.01
    slippage_bps = 0.0
    dist_threshold_pct = 0.5
    candle_height_limit_pct = 2.0

    # Optional: if you want to hardcode costs too (otherwise defaults inside backtest)
    # commission_rate = 0.00001  # 0.001% per side (buy once + sell once)

    # =========================
    # Run pipeline
    # =========================
    raw = download_price_data(ticker, START_DATE, END_DATE)
    df = normalize_ohlc(raw)

    result = run_backtest_virtual_capital(
        df,
        initial_equity=1.0,
        risk_pct=risk_pct,
        max_open_trades=max_open_trades,
        allow_fractional_shares=True,
        atr_len=atr_len,
        sl_atr_mult=sl_atr_mult,
        rr=rr,
        slippage_bps=slippage_bps,
        dist_threshold_pct=dist_threshold_pct,
        candle_height_limit_pct=candle_height_limit_pct,
        # commission_rate=commission_rate,
    )

    # 1) Metrics
    print_metrics(result["metrics"])

    # # 2) Plots
    plot_all_in_one_window(result)

    # 3) Trade note
    note = generate_trade_note_azure(result, ticker=ticker)

    path = save_trade_note_to_word(
        note,
        filename=f"{ticker}_trade_note.docx",
    )

    print("\n=== TRADE NOTE SAVED ===")
    print(f"Location: {path}")



if __name__ == "__main__":
    main()
