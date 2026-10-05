# syntax=docker/dockerfile:1

# ---- deps: pinned, hash-checked dependencies -------------------------------
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY requirements.lock ./
RUN pip install --require-hashes -r requirements.lock

# ---- build: install the package -------------------------------------------
FROM deps AS build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && pip uninstall -y pip

# ---- index: download the pinned model and build the dense index IN the image
# The image, not a developer laptop, produces the served index (ADR-0002:
# embeddings differ across machines in the 4th decimal).
FROM build AS index
COPY data/snapshot /app/data/snapshot
RUN ukmoney-build-index \
      --data-dir /app/data/snapshot \
      --out /opt/indexes \
      --model-cache /opt/models \
 && test -f /opt/indexes/LATEST \
 && echo "built index: $(cat /opt/indexes/LATEST)"

# ---- runtime: no compilers, no source tree, non-root, offline --------------
FROM python:3.12-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    UKMONEY_DATA_DIR=/app/data/snapshot \
    UKMONEY_INDEX_ROOT=/app/indexes \
    UKMONEY_MODEL_CACHE=/opt/models \
    UKMONEY_DENSE=1 \
    HF_HUB_OFFLINE=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=index --chown=app:app /opt/models /opt/models
COPY --from=index --chown=app:app /opt/indexes /app/indexes
COPY --chown=app:app data/snapshot ./data/snapshot
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}
USER app
EXPOSE 8080
# Used by `docker run` locally; Cloud Run uses its own startup probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]
CMD ["ukmoney-serve"]
