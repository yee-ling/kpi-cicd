"""Instructor smoke test for the prediction API.

Not a student deliverable — this is the instructor's confidence check that the
app the class just built actually works. Run with:  pytest -v

Requires the champion to be registered first:  python bootstrap/train_and_register.py
FastAPI's TestClient drives the app in-process (no running server needed) and
triggers the lifespan, so the model is loaded exactly as in production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_EMPLOYEE = {
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


@pytest.fixture(scope="module")
def client():
    """TestClient as a context manager so startup/shutdown (lifespan) fire."""
    with TestClient(app) as c:
        yield c


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True  # bootstrap must have run first
    assert body["model_uri"].endswith("@champion")


def test_predict_returns_valid_prediction(client):
    resp = client.post("/predict", json=VALID_EMPLOYEE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["meets_kpi"] in (0, 1)
    assert 0.0 <= body["probability_meets_kpi"] <= 1.0
    assert isinstance(body["label"], str)


def test_predict_rejects_bad_input_with_422(client):
    bad = {**VALID_EMPLOYEE, "age": 5}  # below ge=16
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # FastAPI validation, no handler code needed


def test_predict_rejects_bad_gender_pattern(client):
    bad = {**VALID_EMPLOYEE, "gender": "male"}  # fails ^[mf]$
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_batch_prediction(client):
    payload = {"employees": [VALID_EMPLOYEE, {**VALID_EMPLOYEE, "age": 45}]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
