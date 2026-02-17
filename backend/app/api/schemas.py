"""Pydantic v2 request/response schemas for the Harmony API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for single-file analysis."""
    file_path: str
    enable_separation: bool = True
    enable_extended: bool = True


class BatchRequest(BaseModel):
    """Request body for batch analysis."""
    file_paths: list[str]
    enable_separation: bool = False
    enable_extended: bool = True


# ── Result Schemas ───────────────────────────────────────────────────────────


class AlgorithmResult(BaseModel):
    """Result from a single algorithm in the ensemble."""
    algorithm: str
    value: float | str
    confidence: float = Field(ge=0, le=1)
    method: str = ""


class BPMResult(BaseModel):
    """BPM detection result."""
    bpm: float = Field(ge=20, le=400)
    confidence: float = Field(ge=0, le=1)
    tempo_stable: bool = True
    algorithm_results: list[AlgorithmResult] = []
    tempo_curve: list[float] | None = None


class SecondaryKey(BaseModel):
    """A secondary key detected in the track (modulation)."""
    key: str
    mode: str
    camelot: str
    proportion: float = Field(ge=0, le=1, description="Fraction of track in this key")


class KeyResult(BaseModel):
    """Key detection result."""
    key: str
    mode: str
    camelot: str
    confidence: float = Field(ge=0, le=1)
    secondary_keys: list[SecondaryKey] | None = None
    algorithm_results: list[AlgorithmResult] = []


class LoudnessResult(BaseModel):
    """EBU R128 loudness analysis."""
    integrated_lufs: float
    short_term_lufs: float
    momentary_lufs: float
    loudness_range: float
    true_peak_dbtp: float


class InstrumentResult(BaseModel):
    """Per-instrument analysis result."""
    bpm: float | None = None
    key: str | None = None
    mode: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.0)


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    id: str
    status: str = "complete"
    file_name: str = ""
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    bpm: BPMResult | None = None
    key: KeyResult | None = None
    loudness: LoudnessResult | None = None
    time_signature: str | None = None
    instruments: dict[str, InstrumentResult] | None = None


class ProgressUpdate(BaseModel):
    """WebSocket progress update."""
    job_id: str
    progress: float = Field(ge=0, le=1)
    stage: str
    message: str = ""


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
