"""
agent/agent.py – F1 prediction agent.

In "stub" mode (no OPENAI_API_KEY), the agent routes keyword-matched queries
to the right tool and returns structured responses.

In "llm" mode (OPENAI_API_KEY set and configs.agent.use_llm = true), the
agent uses OpenAI function-calling to let the LLM decide which tool to call.

TODO: Extend with RAG over race reports and driver history for richer answers.
"""
from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

from agent.tools import (
    explain_prediction,
    generate_preview,
    health_check,
    predict_race,
    retrain_pipeline,
)
from src.utils import configure_logging, load_config

configure_logging()

# ── Stub agent (no LLM required) ─────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {
    "predict_race": predict_race,
    "explain_prediction": explain_prediction,
    "generate_preview": generate_preview,
    "retrain_pipeline": retrain_pipeline,
    "health_check": health_check,
}


def run_stub_agent(query: str, context: dict[str, Any] | None = None) -> str:
    """
    Route a natural-language query to the appropriate tool using keyword
    matching and return a formatted string response.

    *context* may contain race-specific info (circuit_id, season, etc.).
    """
    q = query.lower()
    ctx = context or {}

    if any(w in q for w in ["health", "status", "alive", "running"]):
        result = health_check()
        return f"API status: {json.dumps(result, indent=2)}"

    if any(w in q for w in ["retrain", "train", "update model", "re-train"]):
        result = retrain_pipeline()
        return f"Retrain result: {result['status']}\n{result.get('output', '')}"

    if any(w in q for w in ["explain", "why", "feature", "reason"]):
        if not ctx.get("driver_id"):
            return (
                "Please provide driver_id, constructor_id, grid, "
                "qualifying_position, circuit_id, season, and round to explain a prediction."
            )
        result = explain_prediction(
            driver_id=ctx["driver_id"],
            constructor_id=ctx.get("constructor_id", "unknown"),
            grid=ctx.get("grid", 10),
            qualifying_position=ctx.get("qualifying_position", 10),
            circuit_id=ctx.get("circuit_id", "unknown"),
            season=ctx.get("season", 2025),
            round_num=ctx.get("round", 1),
        )
        top_features = list(result["feature_importances"].items())[:3]
        summary = ", ".join(f"{k}={v}" for k, v in top_features)
        return (
            f"Predicted position for {ctx['driver_id']}: "
            f"{result['predicted_position']}.\n"
            f"Top driving factors: {summary}."
        )

    if any(w in q for w in ["preview", "preview race", "race preview", "storyline"]):
        if not ctx.get("circuit_id") or not ctx.get("drivers"):
            return (
                "Please provide circuit_id, season, round, and a list of drivers "
                "to generate a race preview."
            )
        result = generate_preview(
            circuit_id=ctx["circuit_id"],
            season=ctx.get("season", 2025),
            round_num=ctx.get("round", 1),
            drivers=ctx["drivers"],
        )
        return result.get("preview_text", json.dumps(result, indent=2))

    if any(w in q for w in ["predict", "race", "winner", "podium", "top 10", "top10"]):
        if not ctx.get("circuit_id") or not ctx.get("drivers"):
            return (
                "Please provide circuit_id, season, round, and a list of drivers "
                "to get race predictions."
            )
        result = predict_race(
            circuit_id=ctx["circuit_id"],
            season=ctx.get("season", 2025),
            round_num=ctx.get("round", 1),
            drivers=ctx["drivers"],
        )
        top3 = result["predictions"][:3]
        podium_str = " | ".join(
            f"P{r['predicted_position']} {r['driver_id']}" for r in top3
        )
        return f"Predicted podium: {podium_str}"

    return (
        "I can help with: race predictions, prediction explanations, race previews, "
        "retraining the model, or checking system health. "
        "Please refine your query or provide more context."
    )


# ── LLM agent ────────────────────────────────────────────────────────────────

def run_llm_agent(query: str, context: dict[str, Any] | None = None) -> str:
    """
    Use OpenAI function-calling to route the query to the right tool.
    Falls back to stub agent if the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; falling back to stub agent.")
        return run_stub_agent(query, context)

    try:
        import openai  # noqa: PLC0415

        client = openai.OpenAI(api_key=api_key)
        cfg = load_config()
        model = cfg["agent"].get("openai_model", "gpt-4o-mini")

        system_prompt = (
            "You are an F1 race prediction assistant. "
            "Use the provided tools to answer user questions about race predictions, "
            "driver performance, and system status. "
            "Be concise and data-driven."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        if context:
            messages.insert(
                1,
                {"role": "system", "content": f"Context: {json.dumps(context)}"},
            )

        # TODO: Add OpenAI function definitions for full agentic loop.
        # For now, use a single completion with context.
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    except Exception as exc:
        logger.error(f"LLM agent error: {exc}. Falling back to stub agent.")
        return run_stub_agent(query, context)


# ── Unified entry point ───────────────────────────────────────────────────────

def ask_agent(query: str, context: dict[str, Any] | None = None) -> str:
    """
    Route to LLM agent if configured, otherwise stub agent.
    """
    cfg = load_config()
    use_llm = cfg.get("agent", {}).get("use_llm", False)
    if use_llm:
        return run_llm_agent(query, context)
    return run_stub_agent(query, context)
