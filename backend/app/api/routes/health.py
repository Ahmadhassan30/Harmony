"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Basic health check."""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, str | bool]:
    """Readiness check — verifies models are loaded."""
    try:
        from app.ml.model_registry import registry
        models_ready = registry.is_loaded
    except Exception:
        models_ready = False

    return {
        "status": "ready" if models_ready else "loading",
        "models_loaded": models_ready,
    }
