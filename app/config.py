"""Application configuration — immutable settings read from the environment.

Using ``pydantic_settings.BaseSettings`` means every setting is validated and
can be overridden by an environment variable (handy in Docker — lesson_04).
Nothing here is read from module-level globals scattered around the code; the
single ``settings`` object is imported where needed.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the prediction service.

    Attributes:
        mlflow_tracking_uri: Where the Model Registry lives. A direct
            ``sqlite:///mlflow.db`` URI is enough locally; in compose we point
            this at the MLflow service (e.g. ``http://mlflow:5000``).
        model_name: Registered model name (must match the bootstrap script).
        model_alias: Which alias to serve — ``champion`` in production.
        app_title / app_version: Surfaced in the OpenAPI docs.
    """

    # protected_namespaces=() lets us use model_name / model_alias without
    # Pydantic warning about its reserved "model_" namespace.
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", protected_namespaces=()
    )

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "employee-kpi-classifier"
    model_alias: str = "champion"

    app_title: str = "Employee KPI Prediction API"
    app_version: str = "1.0.0"

    @property
    def model_uri(self) -> str:
        """The registry URI the service loads, e.g. ``models:/...@champion``."""
        return f"models:/{self.model_name}@{self.model_alias}"


settings = Settings()
