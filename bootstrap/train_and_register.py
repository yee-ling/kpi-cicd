"""Bootstrap the champion model for Lesson 03 — self-contained.

This reproduces the lesson_02 result so Lesson 03 (FastAPI serving) can run
WITHOUT a finished lesson_02 on disk. It trains two pipelines (RandomForest and
XGBoost), logs the FULL sklearn Pipeline to MLflow with a signature, registers
both in the Model Registry, and sets aliases:

    models:/employee-kpi-classifier@champion    <- the better val ROC-AUC
    models:/employee-kpi-classifier@challenger   <- the runner-up

Why a *direct* sqlite tracking URI (sqlite:///mlflow.db) and not ./mlruns?
    The MLflow Model Registry does NOT work on the default file store. A sqlite
    backend enables the registry without needing a running `mlflow server`.

Usage:
    python bootstrap/train_and_register.py
    MLFLOW_TRACKING_URI=http://mlflow:5000 python bootstrap/train_and_register.py  # lesson_04 / compose

Re-running is safe (idempotent-ish): it adds new model versions and re-points
the aliases at the freshly trained versions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("bootstrap")

# --- Registry contract (must match what the API loads in app/config.py) ------
REGISTERED_MODEL_NAME = "employee-kpi-classifier"
EXPERIMENT_NAME = "employee-kpi-classification"
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"

# --- Case-study schema (same 11 features as lesson_02) ------------------------
TARGET = "KPIs_met_more_than_80"
CATEGORICAL_COLS = ["department", "region", "education", "gender", "recruitment_channel"]
NUMERIC_COLS = [
    "no_of_trainings",
    "age",
    "previous_year_rating",
    "length_of_service",
    "awards_won",
    "avg_training_score",
]
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS
RANDOM_STATE = 42

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"


def build_preprocessor() -> ColumnTransformer:
    """Leak-safe preprocessing: impute INSIDE the pipeline, then encode/scale.

    Returns:
        A ColumnTransformer with a numeric branch (median impute + missing-flag +
        scale) and a categorical branch (mode impute + one-hot).
    """
    numeric_branch = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_branch, NUMERIC_COLS),
            ("cat", categorical_branch, CATEGORICAL_COLS),
        ]
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load + lightly clean the data, then 70/15/15 stratified split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = pd.read_csv(DATA_PATH)
    # Pure data hygiene (safe outside the pipeline — not learned from the data)
    df["education"] = df["education"].replace({"BACHELOR": "Bachelors"})
    df["gender"] = df["gender"].replace({"female": "f"})

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    logger.info("Train %s | Val %s | Test %s", X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_log(
    name: str,
    estimator,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[str, float]:
    """Fit a full Pipeline, log it to MLflow with a signature, return (model_uri, val_roc_auc).

    We return the ``model_uri`` from ``log_model`` (rather than building
    ``runs:/<id>/model`` by hand) so this works on both MLflow 2.x — where the
    model is a run artifact — and MLflow 3.x — where it is a first-class logged
    model. The URI is whatever the running MLflow version considers canonical.
    """
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_val)[:, 1]
        val_roc_auc = roc_auc_score(y_val, proba)
        val_pr_auc = average_precision_score(y_val, proba)
        val_f1 = f1_score(y_val, pipeline.predict(X_val))

        mlflow.log_params(params)
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.set_tags({"author": os.environ.get("STUDENT_ID", "instructor"), "purpose": "baseline"})

        # Log the WHOLE pipeline (preprocess + model) so it predicts on raw rows.
        signature = infer_signature(X_train, pipeline.predict(X_train))
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
        )
        logger.info("%s | val_roc_auc=%.4f | model_uri=%s", name, val_roc_auc, info.model_uri)
        return info.model_uri, float(val_roc_auc)


def register(model_uri: str, alias: str) -> int:
    """Register a logged model into the Model Registry and set `alias`.

    Returns the new model-version number.
    """
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, alias, mv.version)
    logger.info("Registered v%s -> @%s", mv.version, alias)
    return int(mv.version)


def main() -> int:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("Tracking URI: %s", tracking_uri)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    rf_uri, rf_auc = train_and_log(
        "rf-baseline",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        {"model_family": "RandomForestClassifier", "n_estimators": 100},
        X_train, y_train, X_val, y_val,
    )
    xgb_uri, xgb_auc = train_and_log(
        "xgb-baseline",
        XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
        ),
        {"model_family": "XGBClassifier", "n_estimators": 200, "max_depth": 4},
        X_train, y_train, X_val, y_val,
    )

    # The better validation ROC-AUC becomes the champion; the other, the challenger.
    if xgb_auc >= rf_auc:
        champion_uri, challenger_uri = xgb_uri, rf_uri
    else:
        champion_uri, challenger_uri = rf_uri, xgb_uri

    register(champion_uri, ALIAS_CHAMPION)
    register(challenger_uri, ALIAS_CHALLENGER)

    logger.info(
        "Done. Champion = %s (roc_auc=%.4f). Load it with "
        "models:/%s@champion",
        "XGB" if xgb_auc >= rf_auc else "RF",
        max(rf_auc, xgb_auc),
        REGISTERED_MODEL_NAME,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
