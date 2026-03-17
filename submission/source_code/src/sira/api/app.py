import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sira.api.routes import router
from sira.core.paths import DEFAULT_MODEL_PATH
from sira.services.inference_service import InferenceService


def _resolve_model_path() -> str:
    return os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = InferenceService.from_model_path(_resolve_model_path())
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="SIRA Inference API", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()