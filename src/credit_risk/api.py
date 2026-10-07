"""Optional FastAPI service: run `uvicorn credit_risk.api:app --reload`."""
from pathlib import Path
from typing import Any
from .pipeline import score_application
try:
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="RiskIntelligence", version="0.1.0")
    @app.get("/health")
    def health() -> dict[str, str]: return {"status": "ok"}
    @app.post("/score")
    def score(payload: dict[str, Any]) -> dict[str, Any]:
        try: return score_application(payload)
        except FileNotFoundError: raise HTTPException(503, "Model not trained; run risk-train first")
        except (KeyError, ValueError) as exc: raise HTTPException(422, str(exc))
except ImportError:
    app = None
