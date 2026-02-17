"""Spectral flux onset-based BPM detection."""

from __future__ import annotations

import librosa
import numpy as np


def onset_based_bpm(
    audio: np.ndarray, sr: int, features: dict
) -> dict[str, float | str]:
    """BPM detection from onset peak analysis.

    Detects onsets via spectral flux, then estimates tempo from
    inter-onset intervals using robust statistics.
    """
    onset_env = features.get("onset_envelope")
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    # Peak detection with tuned parameters
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.5,
        wait=10,
    )

    if len(peaks) < 2:
        return {"bpm": 0.0, "confidence": 0.0, "method": "spectral flux onset"}

    # Convert to time
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
    intervals = np.diff(times)

    # Filter outlier intervals (keep 10th-90th percentile)
    p10, p90 = np.percentile(intervals, [10, 90])
    filtered = intervals[(intervals >= p10) & (intervals <= p90)]

    if len(filtered) == 0:
        filtered = intervals

    median_interval = float(np.median(filtered))
    bpm = 60.0 / median_interval if median_interval > 0 else 0.0

    # Confidence from interval consistency
    mad = float(np.median(np.abs(filtered - median_interval)))
    confidence = 1.0 / (1.0 + 10 * mad)

    return {
        "bpm": round(bpm, 2),
        "confidence": min(max(confidence, 0.0), 1.0),
        "method": "Spectral flux onset detection",
    }
