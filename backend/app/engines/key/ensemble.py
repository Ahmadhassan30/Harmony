"""Key detection ensemble — multi-algorithm fusion for musical key."""

from __future__ import annotations

from collections import Counter

import numpy as np
import structlog

from app.api.schemas import AlgorithmResult, KeyResult
from app.engines.key.essentia_key import essentia_key_detect
from app.engines.key.profiles import chroma_profile_match
from app.utils.music_theory import key_to_camelot

logger = structlog.get_logger()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class KeyEnsemble:
    """5-algorithm key detection ensemble with majority voting."""

    def detect(self, audio: np.ndarray, sr: int, features: dict) -> KeyResult:
        """Run all key detection algorithms and combine results."""
        chroma = features.get("chroma_cqt")
        harmonic = features.get("harmonic")

        algorithms = [
            ("essentia_multi_profile", lambda: essentia_key_detect(audio, sr)),
            ("cqt_profile_correlation", lambda: chroma_profile_match(chroma)),
            # ("deep_chroma", lambda: deep_chroma_key(audio, sr)),       # TODO
            # ("ast_key_detector", lambda: ast_key_detect(audio, sr)),   # TODO
            # ("harmonic_analysis", lambda: chord_based_key(audio, sr)), # TODO
        ]

        results: list[AlgorithmResult] = []
        key_votes: list[tuple[str, str]] = []  # (key, mode)

        for name, fn in algorithms:
            try:
                res = fn()
                results.append(AlgorithmResult(
                    algorithm=name,
                    value=f"{res['key']} {res['scale']}",
                    confidence=res["confidence"],
                    method=name,
                ))
                key_votes.append((res["key"], res["scale"]))
            except Exception as e:
                logger.warning("key.algorithm_failed", algorithm=name, error=str(e))

        if not key_votes:
            return KeyResult(
                key="C", mode="major", camelot="8B",
                confidence=0.0, algorithm_results=results,
            )

        # Majority voting
        vote_counter = Counter(key_votes)
        (best_key, best_mode), best_count = vote_counter.most_common(1)[0]

        # Confidence = proportion of agreeing votes × average confidence
        agreement = best_count / len(key_votes)
        avg_conf = np.mean([r.confidence for r in results])
        confidence = 0.5 * agreement + 0.5 * avg_conf

        camelot = key_to_camelot(best_key, best_mode)

        return KeyResult(
            key=best_key,
            mode=best_mode,
            camelot=camelot,
            confidence=round(float(confidence), 3),
            algorithm_results=results,
        )
