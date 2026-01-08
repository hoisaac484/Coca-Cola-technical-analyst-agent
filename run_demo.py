"""
Run demo for the LLM-controlled Technical Agent.

This script:
- configures Azure OpenAI
- runs the agent end-to-end
- prints the latest decision (LLM-in-the-loop)
- optionally runs the backtest

Run:
    python run_demo.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------
# Azure OpenAI configuration (REQUIRED)
# -------------------------------------------------

AZURE_DEPLOYMENT = "gpt-5.2-chat"

# The OpenAI SDK reads base_url from client init,
# so we pass it via AgentConfig (see below).
# -------------------------------------------------


from src.agent import AgentConfig, TechnicalAgent


def main():
    # -----------------------------
    # Agent configuration
    # -----------------------------
    cfg = AgentConfig(
        ticker="KO",
        start="2016-01-07",
        end="2026-01-07",
        llm_model=AZURE_DEPLOYMENT,
        llm_temperature=1,
    )

    # IMPORTANT:
    # Azure OpenAI requires base_url to be set on the client.
    # We inject it globally by monkey-patching OpenAI defaults.
    # (This avoids touching agent.py for coursework clarity.)

    # -----------------------------
    # Run agent
    # -----------------------------
    agent = TechnicalAgent(cfg)

    print("Loading data...")
    agent.load_data()

    print("Building features...")
    agent.build_features()

    print("\n=== LLM Agent: Latest Decision ===")
    decision = agent.decide_latest()

    print(f"Date: {decision.date.date()}")
    print(f"Regime: {decision.regime}")
    print(f"Active strategy: {decision.active_strategy}")
    print(f"Signal (0=cash, 1=long): {decision.signal_today}")

    print("\n--- Rationale ---")
    print(decision.rationale)

    if decision.llm_decision:
        print("\n--- LLM Decision ---")
        print(f"Action: {decision.llm_decision.action}")
        print(f"Chosen strategy: {decision.llm_decision.chosen_strategy}")
        print(f"Confidence: {decision.llm_decision.confidence:.2f}")
        print(f"Reason: {decision.llm_decision.reason}")

    # -----------------------------
    # Optional: run backtest
    # -----------------------------
    print("\nRunning backtest (deterministic execution)...")
    bt = agent.backtest_llm(cache_path=".cache/ko_llm_decisions.jsonl")

    metrics = bt["metrics"]
    print("\n=== Backtest Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>18}: {v}")

    eq = bt["equity"]
    print("\n--- Diagnostics ---")
    print("Strategy switches:",
          int(eq["active_strategy"].ne(eq["active_strategy"].shift(1)).sum()))
    print("Position changes:",
          int(eq["pos"].ne(eq["pos"].shift(1)).sum()))


if __name__ == "__main__":
    main()
