"""Request/response schemas — the API's typed contract.

Every field mirrors a column the champion Pipeline was trained on (the same 11
features as lesson_02). FastAPI validates incoming JSON against these models and
returns a 422 automatically when something is wrong — no manual checking needed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Column order the model expects. The service builds its DataFrame from this.
FEATURE_ORDER = [
    "department",
    "region",
    "education",
    "gender",
    "recruitment_channel",
    "no_of_trainings",
    "age",
    "previous_year_rating",
    "length_of_service",
    "awards_won",
    "avg_training_score",
]


class EmployeeFeatures(BaseModel):
    """One employee record to score.

    Numeric fields carry range constraints so obviously-bad input is rejected at
    the boundary. ``previous_year_rating`` is optional because it is genuinely
    missing for new hires — the model's imputer fills it in.
    """

    department: str = Field(..., examples=["Technology"])
    region: str = Field(..., examples=["region_26"])
    education: str = Field(..., examples=["Bachelors"])
    gender: str = Field(..., pattern="^[mf]$", examples=["m"])
    recruitment_channel: str = Field(..., examples=["sourcing"])

    no_of_trainings: int = Field(..., ge=1, le=20, examples=[1])
    age: int = Field(..., ge=16, le=100, examples=[30])
    previous_year_rating: float | None = Field(None, ge=1, le=5, examples=[3.0])
    length_of_service: int = Field(..., ge=0, le=50, examples=[5])
    awards_won: int = Field(..., ge=0, le=1, examples=[0])
    avg_training_score: float = Field(..., ge=0, le=100, examples=[77])

    model_config = {
        "json_schema_extra": {
            "example": {
                "department": "Technology",
                "region": "region_26",
                "education": "Bachelors",
                "gender": "m",
                "recruitment_channel": "sourcing",
                "no_of_trainings": 1,
                "age": 30,
                "previous_year_rating": 3.0,
                "length_of_service": 5,
                "awards_won": 0,
                "avg_training_score": 77,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Model output for a single employee."""

    meets_kpi: int = Field(..., description="1 if predicted to meet >80% of KPIs, else 0")
    probability_meets_kpi: float = Field(..., ge=0, le=1)
    label: str = Field(..., description="Human-readable label")


class BatchRequest(BaseModel):
    """A batch of employees to score in one call."""

    employees: list[EmployeeFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    """Predictions for a batch, in the same order as the request."""

    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Liveness/readiness signal — used by curl, tests, and the Docker HEALTHCHECK."""

    status: str
    model_loaded: bool
    model_uri: str

    model_config = {"protected_namespaces": ()}
