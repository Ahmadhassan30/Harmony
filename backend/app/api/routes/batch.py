"""Batch processing endpoints."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter

from app.api.schemas import BatchRequest

logger = structlog.get_logger()
router = APIRouter()


@router.post("/batch")
async def batch_analyze(request: BatchRequest) -> dict[str, str | int]:
    """Queue a batch of files for analysis. Progress via WebSocket."""
    batch_id = str(uuid.uuid4())

    logger.info(
        "batch.queued",
        batch_id=batch_id,
        file_count=len(request.file_paths),
    )

    # TODO: Enqueue via ARQ worker
    # await arq_pool.enqueue_job(
    #     "process_batch", batch_id, request.file_paths, ...
    # )

    return {
        "batch_id": batch_id,
        "status": "queued",
        "file_count": len(request.file_paths),
    }
