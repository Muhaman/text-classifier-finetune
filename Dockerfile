FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /src
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home clf
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 MODEL_DIR=/models HF_HOME=/tmp/hf
EXPOSE 8000
USER clf
CMD ["uvicorn", "clf.api:app", "--host", "0.0.0.0", "--port", "8000"]
