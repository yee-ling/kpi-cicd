    # =============================================================================
# Lesson 04 — production image for the Employee KPI Prediction API (lesson_03).
#
# This is the FINAL form the TEACHING_GUIDE builds up to. It is:
#   * multi-stage   — build deps stay out of the shipped image (smaller, safer)
#   * non-root      — runs as an unprivileged user
#   * self-contained— bakes the champion model at build time, so
#                     `docker run -p 8000:8000 <image>` works with nothing else
#   * health-aware  — HEALTHCHECK lets Docker/compose know when it's ready
# =============================================================================

# ---- Stage 1: builder — install Python deps into an isolated venv ----------
FROM python:3.11-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime — copy only the venv + app, no build tooling ---------
FROM python:3.11-slim AS runtime

# libgomp1 is the OpenMP runtime XGBoost needs at import/predict time.
# Without it: "libgomp.so.1: cannot open shared object file" — a classic gotcha.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Application code + the bootstrap and data needed to bake the champion.
COPY app ./app
COPY bootstrap ./bootstrap
COPY data ./data

# Bake the champion into the image. Absolute artifact paths become /app/mlruns/...
# (container paths), so the model is portable inside the image.
RUN python bootstrap/train_and_register.py

# Drop root: create an unprivileged user and hand it ownership of /app.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness probe — pure-Python so we don't need curl in the image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
