"""FastAPI application exposing POST /plans."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .planner import PlannerError, plan
from .schemas import PlanRequest, PlanResponse

app = FastAPI(
    title="Inference Graph Buffer Planner",
    version="1.0.0",
    description=(
        "Plans operator schedules and tensor buffer placement for mutually "
        "exclusive inference paths."
    ),
)


@app.exception_handler(PlannerError)
async def planner_error_handler(_request, exc: PlannerError) -> JSONResponse:
    # Semantic validation errors share the 422 contract used for schema errors.
    return JSONResponse(
        status_code=422,
        content={"detail": [{"msg": str(exc), "type": "value_error.semantic"}]},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/plans", response_model=PlanResponse)
async def create_plan(request: PlanRequest) -> PlanResponse:
    return plan(request)
