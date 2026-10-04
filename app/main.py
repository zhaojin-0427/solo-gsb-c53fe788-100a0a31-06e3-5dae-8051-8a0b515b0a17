"""HTTP API for the inference-graph buffer planner."""
from __future__ import annotations

from fastapi import FastAPI

from .models import PlanRequest, PlanResponse
from .planner import plan

app = FastAPI(
    title="Inference Graph Buffer Planner",
    summary="Schedules mutually exclusive execution paths and assigns "
            "tensor buffer addresses under a deadline and a memory limit.",
    version="1.0.0",
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/plans", response_model=PlanResponse)
def create_plan(request: PlanRequest) -> PlanResponse:
    return plan(request)
