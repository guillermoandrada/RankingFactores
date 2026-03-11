"""FastAPI application for financial data import and ranking."""

from __future__ import annotations

from fastapi import FastAPI

from api.routers.db_metrics import router as db_metrics_router
from api.routers.metrics import router as metrics_router
from api.routers.periods import router as periods_router
from api.routers.scorings import router as scorings_router
from api.routers.reference import router as reference_router
from api.routers.scoring_profiles import router as scoring_profiles_router

app = FastAPI(
    title="RankingFactores API",
    description="Upload financial data, manage ranking profiles, and compute rankings.",
    version="2.0.0",
)

app.include_router(metrics_router)
app.include_router(db_metrics_router)
app.include_router(periods_router)
app.include_router(scorings_router)
app.include_router(reference_router)
app.include_router(scoring_profiles_router)
