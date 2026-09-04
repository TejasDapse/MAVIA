# MAVIA - Multi-Agent Visual Inspection & Audit System
#
# Multi-stage: dependencies are resolved in a builder layer and the resulting
# virtualenv is copied into a slim runtime, so the image does not ship uv, build
# toolchains, or the package cache.
#
# Build:  docker build -f docker/Dockerfile -t mavia:latest .
# Run:    docker run -p 8501:8501 \
#           -v "$PWD/data:/app/data:ro" \
#           -v "$PWD/models:/app/models:ro" \
#           -v "$PWD/artifacts:/app/artifacts" \
#           --env-file .env mavia:latest

# ---------------------------------------------------------------- builder

FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are resolved from the lockfile before the source is copied, so a
# code change does not invalidate the (slow) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra vision --extra memory --extra agents --extra report --extra dashboard

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
        --extra vision --extra memory --extra agents --extra report --extra dashboard

# ---------------------------------------------------------------- runtime

FROM python:3.11-slim-bookworm AS runtime

# libpango/libcairo: WeasyPrint (PDF reports). libgl/libglib: OpenCV.
# Installed without recommends and with the apt lists removed to keep the layer small.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user: the container mounts a writable artifacts volume and
# nothing here needs privileges.
RUN useradd --create-home --uid 1000 mavia

WORKDIR /app
COPY --from=builder --chown=mavia:mavia /app/.venv /app/.venv
COPY --chown=mavia:mavia src ./src
COPY --chown=mavia:mavia scripts ./scripts
COPY --chown=mavia:mavia pyproject.toml README.md ./

RUN mkdir -p /app/artifacts /app/data /app/models && chown -R mavia:mavia /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model weights land here rather than in the layer, so the mounted volume
    # survives a container restart.
    HF_HOME=/app/artifacts/hf \
    TORCH_HOME=/app/artifacts/torch \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER mavia
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "src/mavia/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
