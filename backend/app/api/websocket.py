"""WebSocket connection manager for real-time progress updates."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger()
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections keyed by job_id."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[job_id] = websocket
        logger.info("ws.connected", job_id=job_id)

    def disconnect(self, job_id: str) -> None:
        self._connections.pop(job_id, None)
        logger.info("ws.disconnected", job_id=job_id)

    async def send_progress(
        self, job_id: str, progress: float, stage: str, message: str = ""
    ) -> None:
        ws = self._connections.get(job_id)
        if ws:
            await ws.send_json({
                "type": "progress",
                "job_id": job_id,
                "progress": progress,
                "stage": stage,
                "message": message,
            })

    async def send_result(self, job_id: str, result: dict) -> None:
        ws = self._connections.get(job_id)
        if ws:
            await ws.send_json({"type": "complete", "job_id": job_id, "result": result})

    async def send_error(self, job_id: str, error: str) -> None:
        ws = self._connections.get(job_id)
        if ws:
            await ws.send_json({"type": "error", "job_id": job_id, "error": error})


manager = ConnectionManager()


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time analysis progress."""
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(job_id)
