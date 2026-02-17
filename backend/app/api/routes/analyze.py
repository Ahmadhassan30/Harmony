"""Single-file analysis endpoints."""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import ORJSONResponse

from app.api.schemas import AnalyzeRequest, AnalysisResponse, ErrorResponse
from app.config import settings
from app.core.orchestrator import AnalysisOrchestrator

logger = structlog.get_logger()
router = APIRouter()
orchestrator = AnalysisOrchestrator()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze_audio(request: AnalyzeRequest) -> AnalysisResponse:
    """Run full audio analysis on a single file."""
    job_id = str(uuid.uuid4())
    start_time = time.monotonic()

    log = logger.bind(job_id=job_id, file_path=request.file_path)
    log.info("analysis.started")

    try:
        result = await orchestrator.analyze(
            file_path=request.file_path,
            job_id=job_id,
            enable_separation=request.enable_separation,
            enable_extended=request.enable_extended,
        )

        elapsed = time.monotonic() - start_time
        result.processing_time_seconds = round(elapsed, 2)

        log.info("analysis.complete", elapsed=elapsed, bpm=result.bpm.bpm if result.bpm else None)
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Audio file not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("analysis.failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.post("/upload")
async def upload_file(audio: UploadFile = File(...)) -> dict[str, str]:
    """Upload an audio file and return a temporary file path."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = audio.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '.{ext}'. Supported: {settings.supported_formats}",
        )

    upload_dir = settings.cache_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}.{ext}"

    content = await audio.read()
    file_path.write_bytes(content)

    logger.info("file.uploaded", file_id=file_id, filename=audio.filename, size=len(content))

    return {"file_id": file_id, "file_path": str(file_path)}
