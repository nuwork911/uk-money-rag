# syntax=docker/dockerfile:1

# ---- build: install pinned, hash-checked deps into a venv --------------------
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY requirements.lock ./
RUN pip install --require-hashes -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# ---- runtime: no compilers, no source tree, non-root -----------------------
FROM python:3.12-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UKMONEY_DATA_DIR=/app/data/processed \
    PORT=8080
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
# The corpus is baked into the image: the image IS the versioned (code + data) artefact.
COPY --chown=app:app data/processed/chunks.jsonl data/processed/manifest.json ./data/processed/
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}
USER app
EXPOSE 8080
# Used by `docker run` locally; Cloud Run uses its own startup probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]
CMD ["ukmoney-serve"]
