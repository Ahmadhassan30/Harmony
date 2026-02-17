"""Analysis orchestrator — coordinates all engines for a complete analysis."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import structlog

from app.api.schemas import AnalysisResponse, BPMResult, KeyResult
from app.core.audio_loader import load_audio
from app.core.preprocessing import preprocess
from app.core.feature_extractor import FeatureExtractor

logger = structlog.get_logger()


class AnalysisOrchestrator:
    """Coordinates BPM, key, separation, and extended analysis."""

    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()

    async def analyze(
        self,
        file_path: str,
        job_id: str | None = None,
        enable_separation: bool = True,
        enable_extended: bool = True,
    ) -> AnalysisResponse:
        """Run complete analysis pipeline on an audio file."""
        job_id = job_id or str(uuid.uuid4())
        log = logger.bind(job_id=job_id)

        # Stage 1: Load audio
        log.info("orchestrator.loading")
        audio, sr = load_audio(file_path)
        duration = len(audio) / sr

        # Stage 2: Preprocess
        log.info("orchestrator.preprocessing")
        audio = preprocess(audio, sr)

        # Stage 3: Extract shared features
        log.info("orchestrator.extracting_features")
        features = self.feature_extractor.extract_all(audio, sr)

        # Stage 4: BPM Detection
        log.info("orchestrator.detecting_bpm")
        bpm_result = await self._detect_bpm(audio, sr, features)

        # Stage 5: Key Detection
        log.info("orchestrator.detecting_key")
        key_result = await self._detect_key(audio, sr, features)

        # Stage 6: Source Separation (optional)
        instruments = None
        if enable_separation:
            log.info("orchestrator.separating")
            # TODO: instruments = await self._separate(file_path, sr)

        # Stage 7: Extended analysis (optional)
        loudness = None
        time_sig = None
        if enable_extended:
            log.info("orchestrator.extended_analysis")
            # TODO: loudness, time_sig = await self._extended(audio, sr, features)

        return AnalysisResponse(
            id=job_id,
            status="complete",
            file_name=Path(file_path).name,
            duration_seconds=round(duration, 2),
            bpm=bpm_result,
            key=key_result,
            loudness=loudness,
            time_signature=time_sig,
            instruments=instruments,
        )

    async def _detect_bpm(self, audio, sr, features) -> BPMResult:
        """Run BPM ensemble."""
        from app.engines.bpm.ensemble import BPMEnsemble

        ensemble = BPMEnsemble()
        return ensemble.detect(audio, sr, features)

    async def _detect_key(self, audio, sr, features) -> KeyResult:
        """Run key detection ensemble."""
        from app.engines.key.ensemble import KeyEnsemble

        ensemble = KeyEnsemble()
        return ensemble.detect(audio, sr, features)
