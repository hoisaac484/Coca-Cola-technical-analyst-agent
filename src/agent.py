# src/agent.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal, List

import pandas as pd

from openai import OpenAI  # pip install openai  :contentReference[oaicite:1]{index=1}

from src.data import get_ohlcv, validate_ohlcv
from src.features import compute_indicators
from src.backtest import run_backtest
from src.strategies import (
    regime_snapshot,
    is_strategy_valid,
    propose_strategy,
    STRATEGY_MAP,
    explain_rules,
)
from src.backtest_llm import run_backtest_llm


AllowedStrategy = Literal["mean_reversion", "trend_follow", "cash"]


# -----------------------------
# Config + state
# -----------------------------
@dataclass
class AgentConfig:
    ticker: str = "KO"
    start: str = "2015-01-01"
    end: str = "2026-01-07"

    # Backtest
    cost_bps: float = 5.0

    # Regime hysteresis (Option A)
    adx_range_enter: float = 18.0
    adx_trend_enter: float = 28.0
    adx_trend_min: float = 25.0  # MR invalidation threshold

    # Mean reversion
    rsi_entry: float = 35.0
    mr_atr_stop_k: float = 2.0
    mr_max_hold: int = 10

    # Indicator windows
    rsi_period: int = 14
    bb_window: int = 20
    bb_nstd: float = 2.0
    sma_fast: int = 50
    sma_slow: int = 200
    atr_period: int = 14
    adx_period: int = 14
    vol_window: int = 20

    # LLM
    llm_model: str = "gpt-5.1-mini"  # configurable; pick what you have access to
    llm_temperature: float = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "start": self.start,
            "end": self.end,
            "cost_bps": self.cost_bps,
            "adx_range_enter": self.adx_range_enter,
            "adx_trend_enter": self.adx_trend_enter,
            "adx_trend_min": self.adx_trend_min,
            "rsi_entry": self.rsi_entry,
            "mr_atr_stop_k": self.mr_atr_stop_k,
            "mr_max_hold": self.mr_max_hold,
            "rsi_period": self.rsi_period,
            "bb_window": self.bb_window,
            "bb_nstd": self.bb_nstd,
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "atr_period": self.atr_period,
            "adx_period": self.adx_period,
            "vol_window": self.vol_window,
        }


@dataclass
class AgentState:
    active_strategy: AllowedStrategy = "cash"
    last_decision_date: Optional[pd.Timestamp] = None


@dataclass
class LLMDecision:
    action: Literal["KEEP", "SWITCH"]
    chosen_strategy: AllowedStrategy
    confidence: float  # 0..1
    reason: str
    # Optional: a human-readable list of evidence claims (kept short)
    evidence_bullets: List[str] = field(default_factory=list)


@dataclass
class DailyDecision:
    date: pd.Timestamp
    regime: str
    active_strategy: AllowedStrategy
    signal_today: int
    rationale: str
    llm_decision: Optional[LLMDecision] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# LLM Strategy Selector (Agent)
# -----------------------------
# inside src/agent.py (replace the existing LLMStrategySelector)

class LLMStrategySelector:
    """
    Azure OpenAI compatible:
    - uses chat.completions.create()
    - enforces JSON via prompt + parsing + validation
    """

    def __init__(self, *, base_url: str, api_key: str, deployment_name: str, temperature: float = 1):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.deployment_name = deployment_name
        self.temperature = temperature

    def decide(
        self,
        *,
        ticker: str,
        date: str,
        active_strategy: str,
        regime_info: dict,
        cfg: dict,
        forced_switch: bool,
    ):
        allowed = list(STRATEGY_MAP.keys())

        rules_text = "\n\n".join([explain_rules(s, cfg) for s in allowed])

        system = (
            "You are an investment strategy controller agent.\n"
            "You must output VALID JSON only (no markdown, no extra text).\n"
            "You decide whether to KEEP the current sticky strategy or SWITCH.\n"
            "You must obey constraints. If forced_switch is true, action MUST be SWITCH.\n"
        )

        user_payload = {
            "ticker": ticker,
            "as_of_date": date,
            "sticky_strategy_current": active_strategy,
            "regime_snapshot": regime_info,
            "allowed_strategies": allowed,
            "forced_switch": forced_switch,
            "strategy_rules_reference": rules_text,
        }

        # Strict output contract (we'll validate)
        json_contract = {
            "action": "KEEP or SWITCH",
            "chosen_strategy": f"one of {allowed}",
            "confidence": "number 0..1",
            "reason": "short string grounded in regime evidence",
            "evidence_bullets": "optional list of up to 5 short strings",
        }

        prompt = (
            "Return JSON ONLY.\n"
            "Required keys: action, chosen_strategy, confidence, reason.\n"
            "Optional key: evidence_bullets.\n"
            f"Output contract: {json.dumps(json_contract)}\n\n"
            f"INPUT:\n{json.dumps(user_payload)}\n"
        )

        resp = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        raw = _extract_text_from_completion(resp).strip()
        data = _safe_json_loads(raw)

        # Validate + enforce constraints
        action = str(data.get("action", "")).upper()
        chosen = str(data.get("chosen_strategy", "")).strip()
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", "")).strip()
        bullets = data.get("evidence_bullets", []) or []

        if action not in {"KEEP", "SWITCH"}:
            action = "SWITCH" if forced_switch else "KEEP"

        if forced_switch:
            action = "SWITCH"

        if chosen not in allowed:
            # safe fallback
            chosen = "cash"
            action = "SWITCH"
            confidence = 0.0
            reason = "Invalid strategy from LLM; fallback to cash."

        confidence = min(max(confidence, 0.0), 1.0)

        return LLMDecision(
            action=action,
            chosen_strategy=chosen,  # type: ignore[arg-type]
            confidence=confidence,
            reason=reason if reason else "No reason provided.",
            evidence_bullets=[str(b) for b in bullets][:5] if isinstance(bullets, list) else [],
        )


def _safe_json_loads(text: str) -> dict:
    """
    Attempts to parse JSON even if the model wraps it in extra text.
    We try:
      1) direct json.loads
      2) extract first {...} block
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to extract the first JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    return {}

def _extract_text_from_completion(resp) -> str:
    # Case 1: already a string
    if isinstance(resp, str):
        return resp

    # Case 2: dict-like
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""

    # Case 3: object with .choices
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        pass

    # Fallback: try string conversion
    try:
        return str(resp)
    except Exception:
        return ""


# -----------------------------
# Technical Agent (Orchestrator)
# -----------------------------
class TechnicalAgent:
    """
    Project-level agent that orchestrates:
    - data ingestion
    - feature engineering
    - regime snapshot
    - (LLM) sticky strategy control
    - signal generation
    - backtesting
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.state = AgentState()
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_feat: Optional[pd.DataFrame] = None

        # in TechnicalAgent.__init__(...)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable for Azure OpenAI.")

        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("Missing OPENAI_BASE_URL environment variable (Azure endpoint).")

        self.selector = LLMStrategySelector(
            base_url=base_url,
            api_key=api_key,
            deployment_name=self.cfg.llm_model,     # store deployment name in cfg.llm_model
            temperature=self.cfg.llm_temperature,
        )


    # ---- pipeline ----
    def load_data(self) -> pd.DataFrame:
        c = self.cfg.to_dict()
        df = get_ohlcv(c["ticker"], c["start"], c["end"])
        df, _ = validate_ohlcv(df)
        self.df_raw = df
        return df

    def build_features(self) -> pd.DataFrame:
        if self.df_raw is None:
            raise ValueError("No raw data. Call load_data() first.")
        self.df_feat = compute_indicators(self.df_raw, self.cfg.to_dict())
        return self.df_feat

    # ---- decision at latest close ----
    def decide_latest(self) -> DailyDecision:
        if self.df_feat is None or self.df_feat.empty:
            raise ValueError("No features. Call build_features() first.")

        cfg = self.cfg.to_dict()
        df = self.df_feat.dropna(subset=["ret_1d"]).copy()
        if df.empty:
            raise ValueError("Not enough rows after dropping NaN returns.")

        asof_date = df.index[-1]
        regime_info = regime_snapshot(df, cfg)
        regime = regime_info["regime"]

        active: AllowedStrategy = self.state.active_strategy

        # Hard-rule: if current strategy is invalid, we must switch.
        forced_switch = not is_strategy_valid(active, df, cfg)

        # LLM agent decides KEEP/SWITCH (within constraints).
        llm_dec = self.selector.decide(
            ticker=self.cfg.ticker,
            date=str(asof_date.date()),
            active_strategy=active,
            regime_info=regime_info,
            cfg=cfg,
            forced_switch=forced_switch,
        )

        # Apply sticky policy:
        # - If forced switch: take LLM chosen strategy (still constrained).
        # - If not forced:
        #     - KEEP -> stay
        #     - SWITCH -> switch
        if forced_switch:
            active = llm_dec.chosen_strategy
        else:
            if llm_dec.action == "SWITCH":
                active = llm_dec.chosen_strategy

        # Additional safety net: if model chooses something inconsistent, fall back to rule selector
        # (rare, but good for robustness).
        if active not in STRATEGY_MAP:
            active = propose_strategy(regime_info, cfg)  # deterministic fallback

        # Signal for today (as-of latest close): 0/1
        sig_today = int(STRATEGY_MAP[active](df, cfg).iloc[-1])

        self.state.active_strategy = active
        self.state.last_decision_date = asof_date

        rationale = self._build_rationale(
            active_strategy=active,
            regime=regime,
            regime_info=regime_info,
            llm_decision=llm_dec,
        )

        return DailyDecision(
            date=asof_date,
            regime=regime,
            active_strategy=active,
            signal_today=sig_today,
            rationale=rationale,
            llm_decision=llm_dec,
            evidence=regime_info.get("evidence", {}),
        )

    def _build_rationale(
        self,
        *,
        active_strategy: AllowedStrategy,
        regime: str,
        regime_info: dict,
        llm_decision: LLMDecision,
    ) -> str:
        flags = regime_info.get("flags", [])
        lines = []
        lines.append(f"Regime: {regime}")
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
        lines.append(f"Active strategy (sticky): {active_strategy}")
        lines.append("")
        lines.append("LLM controller decision:")
        lines.append(f"- action: {llm_decision.action}")
        lines.append(f"- chosen_strategy: {llm_decision.chosen_strategy}")
        lines.append(f"- confidence: {llm_decision.confidence:.2f}")
        lines.append(f"- reason: {llm_decision.reason}")
        if llm_decision.evidence_bullets:
            lines.append("- evidence:")
            for b in llm_decision.evidence_bullets:
                lines.append(f"  - {b}")
        lines.append("")
        lines.append("Strategy rules:")
        lines.append(explain_rules(active_strategy, self.cfg.to_dict()))
        return "\n".join(lines)

    # ---- backtest ----
    def backtest(self) -> Dict[str, Any]:
        if self.df_feat is None:
            raise ValueError("No features. Call build_features() first.")
        return run_backtest(self.df_feat, self.cfg.to_dict())

    def backtest_llm(self, *, cache_path: str = ".cache/llm_decisions.jsonl") -> Dict[str, Any]:
        if self.df_feat is None:
            raise ValueError("No features. Call build_features() first.")
        return run_backtest_llm(
            self.df_feat,
            self.cfg.to_dict(),
            self.selector,
            cache_path=cache_path,
        )


