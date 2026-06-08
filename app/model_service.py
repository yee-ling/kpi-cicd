"""Model loading + inference, isolated from the web layer.

Keeping this separate from ``main.py`` means the FastAPI handlers stay thin and
the model logic is unit-testable on its own. The service loads the champion
*once* (at app startup) and reuses it for every request — loading a model per
request would be slow and wasteful.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from app.config import settings
from app.schemas import FEATURE_ORDER, EmployeeFeatures, PredictionResponse

logger = logging.getLogger(__name__)

_LABELS = {0: "Unlikely to meet >80% of KPIs", 1: "Likely to meet >80% of KPIs"}


class ModelService:
    """Holds the loaded champion Pipeline and turns requests into predictions."""

    def __init__(self) -> None:
        self._model = None  # populated by load(); None means "not ready"

    @property
    def is_ready(self) -> bool:
        """True once a model has been loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the champion from the registry. Called once at startup.

        Raises:
            Exception: if the registry/model cannot be reached — we let it
                propagate so a misconfigured deployment fails loudly at boot
                rather than silently serving nothing.
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        logger.info("Loading model %s (tracking=%s)", settings.model_uri, settings.mlflow_tracking_uri)
        self._model = mlflow.sklearn.load_model(settings.model_uri)
        logger.info("Model loaded and ready")

    def predict(self, employees: list[EmployeeFeatures]) -> list[PredictionResponse]:
        """Score a list of employees.

        Args:
            employees: validated request objects.

        Returns:
            One PredictionResponse per input, in the same order.

        Raises:
            RuntimeError: if called before the model is loaded.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        # Pydantic objects -> DataFrame with the exact column order the model expects.
        frame = pd.DataFrame([e.model_dump() for e in employees])[FEATURE_ORDER]

        preds = self._model.predict(frame)
        proba = self._model.predict_proba(frame)[:, 1]

        return [
            PredictionResponse(
                meets_kpi=int(p),
                probability_meets_kpi=round(float(pr), 4),
                label=_LABELS[int(p)],
            )
            for p, pr in zip(preds, proba)
        ]


# Single shared instance — wired up in main.py's lifespan.
model_service = ModelService()
