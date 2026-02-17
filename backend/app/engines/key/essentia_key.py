"""Essentia multi-profile key detection."""

from __future__ import annotations

import numpy as np


def essentia_key_detect(audio: np.ndarray, sr: int) -> dict[str, float | str]:
    """Detect key using Essentia with multiple key profiles.

    Runs three profiles (Krumhansl, Temperley, EDMA) and picks
    the result with highest strength.
    """
    try:
        import essentia.standard as es
    except ImportError:
        return {"key": "C", "scale": "major", "confidence": 0.0}

    audio_32 = audio.astype(np.float32)

    profiles = ["krumhansl", "temperley", "edma"]
    best_result = {"key": "C", "scale": "major", "confidence": 0.0}

    for profile in profiles:
        try:
            key_extractor = es.Key(profileType=profile)
            key, scale, strength = key_extractor(audio_32)[:3]

            if float(strength) > best_result["confidence"]:
                best_result = {
                    "key": str(key),
                    "scale": str(scale),
                    "confidence": float(strength),
                }
        except Exception:
            continue

    return best_result
