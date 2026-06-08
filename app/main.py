"""FastAPI application — serves the champion model over HTTP.

Endpoints:
    GET  /health         liveness + readiness (is the model loaded?)
    POST /predict        score one employee
    POST /predict/batch  score many employees in one call

Run locally:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status

from app.config import settings
from app.model_service import ModelService, model_service
from app.schemas import (
    BatchRequest,
    BatchResponse,
    EmployeeFeatures,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE when the app boots, not on every request.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown. If the model fails to load we log it and start anyway so /health
    can report ``model_loaded=false`` instead of the whole process crash-looping.
    """
    try:
        model_service.load()
    except Exception:  # noqa: BLE001 — log and degrade rather than crash-loop
        logger.exception("Model failed to load at startup; /health will report not-ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title=settings.app_title, version=settings.app_version, lifespan=lifespan)


def get_service() -> ModelService:
    """Dependency that hands handlers a *ready* model service or a 503.

    Centralising the readiness check here keeps each endpoint to a single line
    of guard-free logic.
    """
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded — check the tracking URI and that the champion is registered.",
        )
    return model_service


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report liveness and whether the model is loaded. Never raises."""
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_ready,
        model_uri=settings.model_uri,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(
    employee: EmployeeFeatures, service: ModelService = Depends(get_service)
) -> PredictionResponse:
    """Score a single employee."""
    return service.predict([employee])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(
    request: BatchRequest, service: ModelService = Depends(get_service)
) -> BatchResponse:
    """Score a batch of employees in one request."""
    predictions = service.predict(request.employees)
    return BatchResponse(count=len(predictions), predictions=predictions)
