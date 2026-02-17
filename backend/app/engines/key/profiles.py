"""Key detection via chroma-profile correlation with modern profiles."""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Sha'ath (2011) — optimized for pop/rock/electronic
MAJOR_PROFILE = np.array([6.6, 2.0, 3.5, 2.2, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 2.9])
MINOR_PROFILE = np.array([6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2])

# Normalize
MAJOR_PROFILE = MAJOR_PROFILE / np.sum(MAJOR_PROFILE)
MINOR_PROFILE = MINOR_PROFILE / np.sum(MINOR_PROFILE)


def chroma_profile_match(chroma: np.ndarray | None) -> dict[str, float | str]:
    """Match averaged chroma vector against modern key profiles.

    Uses Sha'ath (2011) profiles which outperform Krumhansl-Kessler
    on modern pop/rock/electronic music.

    Args:
        chroma: (12, T) chromagram array from CQT.

    Returns:
        dict with 'key', 'scale', 'confidence'.
    """
    if chroma is None or chroma.size == 0:
        return {"key": "C", "scale": "major", "confidence": 0.0}

    # Average chroma over time
    chroma_avg = np.mean(chroma, axis=1)
    chroma_norm = chroma_avg / (np.sum(chroma_avg) + 1e-8)

    best_corr = -1.0
    best_key = "C"
    best_scale = "major"

    for shift in range(12):
        major_shifted = np.roll(MAJOR_PROFILE, shift)
        minor_shifted = np.roll(MINOR_PROFILE, shift)

        major_corr = float(np.corrcoef(chroma_norm, major_shifted)[0, 1])
        minor_corr = float(np.corrcoef(chroma_norm, minor_shifted)[0, 1])

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = NOTE_NAMES[shift]
            best_scale = "major"

        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = NOTE_NAMES[shift]
            best_scale = "minor"

    return {
        "key": best_key,
        "scale": best_scale,
        "confidence": max(0.0, min(best_corr, 1.0)),
    }
