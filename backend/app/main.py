"""FastAPI application factory for the Harmony audio processing engine."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import analyze, batch, health, export
from app.api.websocket import router as ws_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application startup and shutdown lifecycle."""
    logger.info("harmony.startup", host=settings.host, port=settings.port)

    # Ensure cache directory exists
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load ML models if not in debug mode
    if not settings.debug:
        from app.ml.model_registry import registry
        await registry.preload()
        logger.info("harmony.models_loaded")

    yield

    logger.info("harmony.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Harmony Audio Engine",
        description="World's most accurate BPM & key detection API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for Tauri frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["tauri://localhost", "http://localhost:*", "https://localhost:*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router, tags=["health"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analysis"])
    app.include_router(batch.router, prefix="/api/v1", tags=["batch"])
    app.include_router(export.router, prefix="/api/v1", tags=["export"])
    app.include_router(ws_router, tags=["websocket"])

    return app


app = create_app()
