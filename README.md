---
title: KPI CICD
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# KPI : CICD WORKFLOW

[![CI](https://github.com/yee-ling/kpi-cicd/actions/workflows/ci.yml/badge.svg)](https://github.com/yee-ling/kpi-cicd/actions/workflows/ci.yml)
[![CD - Publish image to ghcr.io](https://github.com/yee-ling/kpi-cicd/actions/workflows/cd-ghcr.yml/badge.svg)](https://github.com/yee-ling/kpi-cicd/actions/workflows/cd-ghcr.yml)
[![Deploy to Hugging Face Space](https://github.com/yee-ling/kpi-cicd/actions/workflows/deploy-hf.yml/badge.svg)](https://github.com/yee-ling/kpi-cicd/actions/workflows/deploy-hf.yml)

Employee KPI Prediction API — auto-tested, packaged, and deployed by GitHub Actions
on every push to `main`. Open `/docs` for the interactive Swagger UI.

## Pipeline

| Badge | Workflow | What it does |
|-------|----------|--------------|
| CI | `ci.yml` | lint (flake8) + tests (pytest) |
| CD - ghcr.io | `cd-ghcr.yml` | builds the Docker image and pushes it to GitHub Container Registry |
| Deploy to HF | `deploy-hf.yml` | deploys the service to a live Hugging Face Space |
