"""Universal audio loader with format detection and preprocessing."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import structlog

from app.config import settings

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = set(settings.supported_formats)


def load_audio(
    file_path: str | Path,
    sr: int | None = None,
    mono: bool = True,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and resample.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate (default: from settings).
        mono: Convert to mono if True.
        duration: Max duration in seconds to load.

    Returns:
        Tuple of (audio array, sample rate).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the format is unsupported or file is too long.
    """
    path = Path(file_path)
    target_sr = sr or settings.sample_rate

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: .{ext}")

    logger.debug("audio.loading", path=str(path), target_sr=target_sr)

    try:
        # librosa handles most formats via soundfile/audioread
        audio, file_sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            duration=duration or settings.max_audio_duration_seconds,
        )
    except Exception as e:
        logger.error("audio.load_failed", path=str(path), error=str(e))
        raise ValueError(f"Failed to load audio: {e}") from e

    file_duration = len(audio) / target_sr
    if file_duration > settings.max_audio_duration_seconds:
        raise ValueError(
            f"Audio too long ({file_duration:.0f}s). Max: {settings.max_audio_duration_seconds}s"
        )

    logger.info(
        "audio.loaded",
        path=path.name,
        duration=round(file_duration, 2),
        sr=target_sr,
        samples=len(audio),
    )

    return audio, target_sr


def get_audio_info(file_path: str | Path) -> dict:
    """Get audio file metadata without loading the full file."""
    path = Path(file_path)
    info = sf.info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }
