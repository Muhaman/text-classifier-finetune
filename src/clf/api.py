"""FastAPI service exposing the fine-tuned classifier."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from clf.config import configure_logging
from clf.predict import LABEL_NAMES_FILE, Prediction, Predictor

log = structlog.get_logger(__name__)

Text = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class PredictRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=64)


class PredictResponse(BaseModel):
    predictions: list[Prediction]


def load_predictor_from_env() -> Predictor:
    model_dir = Path(os.environ.get("MODEL_DIR", "artifacts/model"))
    if not (model_dir / LABEL_NAMES_FILE).is_file():
        raise RuntimeError(
            f"MODEL_DIR={model_dir} does not contain a trained model "
            f"(missing {LABEL_NAMES_FILE}). Train one with `python -m clf.train`."
        )
    return Predictor.from_dir(model_dir)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.predictor = predictor or load_predictor_from_env()
        log.info("model_loaded")
        yield

    app = FastAPI(title="text-classifier-finetune", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": f"{location}: {first['msg']}"}},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        # Sync endpoint: inference is CPU-bound, so FastAPI runs it in its threadpool.
        predictor_: Predictor = request.app.state.predictor
        return PredictResponse(predictions=predictor_.predict(body.texts))

    return app


app = create_app()
