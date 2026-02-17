"""Audio preprocessing pipeline — normalization, noise reduction, feature extraction."""

from __future__ import annotations

import librosa
import numpy as np
import structlog

logger = structlog.get_logger()


def preprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """Standard preprocessing pipeline for analysis.

    Steps:
    1. Trim leading/trailing silence
    2. Normalize peak amplitude to -1 dBFS
    3. Apply pre-emphasis filter (optional, for vocal clarity)
    """
    # Trim silence (threshold = 20 dB below peak)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

    # Peak normalization
    peak = np.max(np.abs(audio_trimmed))
    if peak > 0:
        audio_trimmed = audio_trimmed / peak * 0.9  # -1 dBFS headroom

    logger.debug(
        "audio.preprocessed",
        original_len=len(audio),
        trimmed_len=len(audio_trimmed),
        peak=float(peak),
    )

    return audio_trimmed


def extract_harmonic_percussive(
    audio: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Separate audio into harmonic and percussive components.

    Uses librosa's HPSS (Harmonic-Percussive Source Separation).
    Harmonic is used for key detection, percussive for BPM.
    """
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive
