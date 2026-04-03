"""
api/main.py – FastAPI application entry point.

Endpoints:
  GET  /           → minimal HTML UI (Jinja2 template)
  GET  /health     → system health check
  POST /predict    → predict finishing positions for a race
  POST /explain    → explain a single driver's prediction
  POST /preview    → generate a race preview
  POST /agent      → send a natural-language query to the agent
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.routes.explain import router as explain_router
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from api.routes.preview import router as preview_router
from api.routes.agent import router as agent_router
from src.utils import configure_logging

configure_logging()

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="F1 Race Prediction API",
    description=(
        "Predict F1 race finishing positions, explain predictions, "
        "generate race previews, and interact with the prediction agent."
    ),
    version="0.1.0",
)

# Templates
_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Include routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(preview_router)
app.include_router(agent_router)


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the minimal HTML prediction UI."""
    return templates.TemplateResponse("index.html", {"request": request})
