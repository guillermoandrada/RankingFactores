"""
FastAPI application for financial data import and ranking.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from modules.analytics import RankingEngine, ZScoreCalculator
from modules.db import FinancialDatabase
from modules.ingestion import DataImporter

app = FastAPI(
    title="RankingFactores API",
    description="Upload financial data and compute security rankings.",
    version="1.0.0",
)

# Shared instances (initialized on first use)
_db: Optional[FinancialDatabase] = None
_importer: Optional[DataImporter] = None


def get_db() -> FinancialDatabase:
    global _db
    if _db is None:
        _db = FinancialDatabase()
    return _db


def get_importer() -> DataImporter:
    global _importer
    if _importer is None:
        _importer = DataImporter()
    return _importer


# --- Request/Response models ---


class RankingRequest(BaseModel):
    """Request body for the ranking endpoint."""

    quarter: str = Field(..., description="Period, e.g. '2023 Q3'")
    industry: str = Field(
        default="",
        description="Filter by industry (e.g. 'Banks'). Leave empty for all companies.",
    )
    method: str = Field(
        ...,
        description="Aggregation method: 'linear' or 'softplus'",
    )
    weights: dict[str, float] = Field(
        ...,
        description="Metric names and their weights, e.g. {'Current Book to Price': 1.0}",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "quarter": "2023 Q3",
                    "industry": "",
                    "method": "linear",
                    "weights": {
                        "Current Book to Price": 1.0,
                        "5Y Average Book to Price": 0.5,
                        "Estimated Book to Price": 0.5,
                    },
                },
                {
                    "quarter": "2023 Q3",
                    "industry": "Banks",
                    "method": "softplus",
                    "weights": {"Current Book to Price": 1.0},
                },
            ]
        }
    }


class MetricUpdateRequest(BaseModel):
    """Request body for updating a metric's higher_is_better."""

    higher_is_better: bool = Field(..., description="True if higher values are better.")


# --- Endpoints ---


@app.get("/metrics")
async def list_metrics():
    """
    Return all metrics available in the database.
    Each metric includes metric_id, metric_name, and higher_is_better.
    """
    db = get_db()
    return {"metrics": db.list_metrics()}


@app.patch("/metrics/{metric_id}")
async def update_metric(metric_id: int, request: MetricUpdateRequest):
    """
    Update the higher_is_better column for a metric.
    """
    db = get_db()
    try:
        db.update_metric_higher_is_better(metric_id, request.higher_is_better)
        return {"success": True, "metric_id": metric_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an Excel file and import its data into the database.
    The period is inferred from cell A2 (mm/dd/yy format).
    """
    if not file.filename or not (
        file.filename.endswith(".xlsx") or file.filename.endswith(".xls")
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be an Excel file (.xlsx or .xls)",
        )

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".xlsx"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        importer = get_importer()
        result = importer.import_file(tmp_path, verbose=False)

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "success": True,
            "period": result.period,
            "companies_count": result.companies_count,
            "metrics_count": result.metrics_count,
            "records_count": result.records_count,
            "index_code": result.index_code,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ranking")
async def get_ranking(request: RankingRequest):
    """
    Compute security ranking for a given quarter.
    Optionally filter by industry. Pass metrics and weights for the score.
    """
    if request.method not in ("linear", "softplus"):
        raise HTTPException(
            status_code=400,
            detail="method must be 'linear' or 'softplus'",
        )
    if not request.weights:
        raise HTTPException(
            status_code=400,
            detail="weights cannot be empty",
        )

    db = get_db()
    try:
        metric_ids = db.get_metric_ids_by_names(list(request.weights.keys()))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    calculator = ZScoreCalculator(engine=db.engine)
    engine = RankingEngine()

    try:
        industry = request.industry.strip() or None
        df_z, direction_map = calculator.compute(
            period=request.quarter,
            metric_ids=list(metric_ids.values()),
            index_name=None,
            industry_name=industry,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        df_scored = engine.compute(
            df_z=df_z,
            weights=request.weights,
            direction_map=direction_map,
            method=request.method,
            out_col="score",
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Metric not found in data: {e}",
        )

    df_ranked = df_scored.sort_values("score", ascending=False).reset_index()

    # Convert to JSON-serializable list (handles numpy types, NaN)
    records = json.loads(df_ranked.to_json(orient="records", date_format="iso"))

    return {
        "quarter": request.quarter,
        "industry": industry,
        "method": request.method,
        "count": len(records),
        "ranking": records,
    }
