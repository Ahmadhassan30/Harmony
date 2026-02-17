"""Librosa Ellis DP beat tracker."""

from __future__ import annotations

import librosa
import numpy as np


def librosa_beat_track(
    audio: np.ndarray, sr: int, features: dict
) -> dict[str, float | str]:
    """BPM detection using Librosa's Ellis dynamic programming beat tracker.

    Reference: Ellis, D.P.W. (2007). Beat Tracking by Dynamic Programming.
    """
    onset_env = features.get("onset_envelope")
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, units="time"
    )

    # Refine tempo from actual beat intervals
    if len(beats) > 1:
        intervals = np.diff(beats)
        refined_bpm = 60.0 / float(np.median(intervals))
        # Confidence from interval consistency
        mad = float(np.median(np.abs(intervals - np.median(intervals))))
        confidence = 1.0 / (1.0 + 10 * mad)
    else:
        refined_bpm = float(np.atleast_1d(tempo)[0])
        confidence = 0.3

    return {
        "bpm": round(refined_bpm, 2),
        "confidence": min(max(confidence, 0.0), 1.0),
        "method": "Ellis DP beat tracker",
    }
