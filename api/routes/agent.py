"""api/routes/agent.py – /agent endpoint for natural-language queries."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from agent.agent import ask_agent

router = APIRouter()


class AgentRequest(BaseModel):
    query: str
    context: dict[str, Any] | None = None


@router.post("/agent", tags=["agent"])
def agent_query(req: AgentRequest) -> dict[str, Any]:
    """
    Send a natural-language query to the prediction agent.

    In stub mode the agent uses keyword routing.
    With OPENAI_API_KEY set and `agent.use_llm=true` in config, it uses GPT.

    Example queries:
      - "Who will win the Monaco Grand Prix?"
      - "Explain why Verstappen is predicted P1."
      - "Generate a race preview for Silverstone."
      - "Retrain the model with 2024 data."
    """
    logger.info(f"Agent query: {req.query!r}")
    response = ask_agent(req.query, req.context)
    return {"query": req.query, "response": response}
