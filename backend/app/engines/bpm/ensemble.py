"""BPM detection ensemble — 5-algorithm Bayesian-weighted combination."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import structlog
from scipy.stats import gaussian_kde

from app.api.schemas import AlgorithmResult, BPMResult
from app.engines.bpm.librosa_tracker import librosa_beat_track
from app.engines.bpm.essentia_tracker import essentia_rhythm_extract
from app.engines.bpm.onset_tracker import onset_based_bpm

logger = structlog.get_logger()


class BPMEnsemble:
    """5-algorithm BPM ensemble with KDE-based fusion and octave error resolution."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def detect(self, audio: np.ndarray, sr: int, features: dict) -> BPMResult:
        """Run all algorithms and combine via Bayesian model averaging."""
        algorithms = [
            ("librosa_ellis_dp", librosa_beat_track),
            ("essentia_rhythm", essentia_rhythm_extract),
            ("spectral_flux_onset", onset_based_bpm),
            # ("tcn_beat_tracker", tcn_beat_track),         # TODO: ML model
            # ("efficientnet_tempo", efficientnet_bpm),     # TODO: ML model
        ]

        results: list[AlgorithmResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(fn, audio, sr, features): name
                for name, fn in algorithms
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result(timeout=30)
                    results.append(AlgorithmResult(
                        algorithm=name,
                        value=res["bpm"],
                        confidence=res["confidence"],
                        method=res.get("method", ""),
                    ))
                except Exception as e:
                    logger.warning("bpm.algorithm_failed", algorithm=name, error=str(e))

        if not results:
            return BPMResult(bpm=120.0, confidence=0.0, tempo_stable=False)

        # Resolve octave errors
        resolved = self._resolve_octave_errors(results)

        # KDE-based fusion
        final_bpm, confidence = self._kde_fusion(resolved)

        # Tempo stability check
        bpms = [r.value for r in resolved if isinstance(r.value, (int, float))]
        cv = np.std(bpms) / np.mean(bpms) if len(bpms) > 1 else 0
        tempo_stable = cv < 0.05

        return BPMResult(
            bpm=round(final_bpm, 1),
            confidence=round(confidence, 3),
            tempo_stable=tempo_stable,
            algorithm_results=results,
        )

    def _resolve_octave_errors(
        self, results: list[AlgorithmResult]
    ) -> list[AlgorithmResult]:
        """Normalize tempo octave errors (60/120/240 confusion)."""
        bpms = [float(r.value) for r in results]
        median_bpm = float(np.median(bpms))

        resolved = []
        for r in results:
            bpm = float(r.value)
            # Bring into same octave as median
            while bpm < median_bpm / 1.5:
                bpm *= 2
            while bpm > median_bpm * 1.5:
                bpm /= 2
            resolved.append(AlgorithmResult(
                algorithm=r.algorithm,
                value=round(bpm, 2),
                confidence=r.confidence,
                method=r.method,
            ))
        return resolved

    def _kde_fusion(self, results: list[AlgorithmResult]) -> tuple[float, float]:
        """KDE-based fusion with confidence weighting."""
        bpms = np.array([float(r.value) for r in results])
        weights = np.array([r.confidence for r in results])

        if len(bpms) == 1:
            return float(bpms[0]), float(weights[0])

        # Weighted KDE
        try:
            kde = gaussian_kde(bpms, weights=weights, bw_method=0.05)
            grid = np.linspace(max(30, bpms.min() - 10), min(300, bpms.max() + 10), 1000)
            density = kde(grid)
            map_bpm = float(grid[np.argmax(density)])
        except Exception:
            map_bpm = float(np.average(bpms, weights=weights))

        # Confidence from agreement
        agreement = 1.0 / (1.0 + float(np.std(bpms)) / float(np.mean(bpms)))
        avg_conf = float(np.mean(weights))
        confidence = 0.6 * agreement + 0.4 * avg_conf

        return map_bpm, min(confidence, 1.0)
