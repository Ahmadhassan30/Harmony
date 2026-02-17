"""Essentia multi-feature rhythm extractor."""

from __future__ import annotations

import numpy as np


def essentia_rhythm_extract(
    audio: np.ndarray, sr: int, features: dict
) -> dict[str, float | str]:
    """BPM detection using Essentia's RhythmExtractor2013.

    Uses multi-scale beat detection with BPM histogram analysis.
    """
    try:
        import essentia.standard as es
    except ImportError:
        return {"bpm": 0.0, "confidence": 0.0, "method": "essentia (not installed)"}

    # Essentia expects float32
    audio_32 = audio.astype(np.float32)

    rhythm_extractor = es.RhythmExtractor2013()
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio_32)

    # Additional validation: beat interval consistency
    if len(beats_intervals) > 0:
        interval_std = float(np.std(beats_intervals))
        interval_mean = float(np.mean(beats_intervals))
        consistency = 1.0 - min(interval_std / max(interval_mean, 1e-6), 1.0)
        confidence = float(beats_confidence) * consistency
    else:
        confidence = float(beats_confidence) * 0.5

    return {
        "bpm": round(float(bpm), 2),
        "confidence": min(max(confidence, 0.0), 1.0),
        "method": "Essentia RhythmExtractor2013",
    }
